#!/usr/bin/env python3
"""Compare BehaviorGraph outputs from three graph-building methods.

Runs the graph-building sub-pipeline for:
  1. ``cupid``       — influence embeddings → UMAP → KMeans → Markov chain
  2. ``enap_custom`` — visual encoder → HDBSCAN → GRU → Extended L* (custom)
  3. ``enap``        — visual encoder → HDBSCAN → PretrainRNN + PMM (faithful)

Each method is executed in its own run directory under ``data/compare_runs/``.
Where possible, shared upstream steps (e.g. ``train_enap_perception``) are
re-used via symlinks or ``skip_if_done=True``.

Usage::

    # Run everything from scratch:
    python scripts/compare_graph_methods.py

    # Override task or clustering:
    python scripts/compare_graph_methods.py \\
        --task-config transport_mh_jan28 \\
        --base-run-dir data/compare_graph_methods \\
        --methods cupid enap_custom enap

    # Data lives in a separate repo (e.g. the original cupid checkout):
    python scripts/compare_graph_methods.py \\
        --task-config transport_mh_jan28 \\
        --repo-root /home/erbauer/cupid \\
        --methods cupid

    # Resume (skip completed steps):
    python scripts/compare_graph_methods.py --resume

    # Dry-run (print what would run, don't execute):
    python scripts/compare_graph_methods.py --dry-run

Exit code 0 on success; non-zero if any method fails.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap
import traceback
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup path so the script works from the repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_base_cfg(
    task_config: str,
    config_root: str,
    repo_root: pathlib.Path,
    code_root: pathlib.Path | None = None,
):
    """Load and merge pipeline + task configs via OmegaConf.

    Args:
        task_config: Task config name (e.g. ``transport_mh_jan28``).
        config_root: ``"iv"`` or ``"pd"`` — where IV task configs live.
        repo_root: Root that contains ``data/outputs/...`` (may differ from code_root).
        code_root: Root that contains ``policy_doctor/configs/`` (defaults to repo_root).
    """
    from omegaconf import OmegaConf

    if code_root is None:
        code_root = repo_root

    pipeline_cfg_path = (
        _REPO_ROOT / "policy_doctor" / "configs" / "pipeline" / "config.yaml"
    )
    cfg = OmegaConf.load(str(pipeline_cfg_path))
    OmegaConf.update(cfg, "task_config", task_config, merge=True)
    OmegaConf.update(cfg, "config_root", config_root, merge=True)
    # repo_root is used by pipeline steps to resolve data/outputs/* paths
    OmegaConf.update(cfg, "repo_root", str(repo_root), merge=True)

    # Try to merge task-specific config if it exists
    task_cfg_path = (
        _REPO_ROOT / "policy_doctor" / "configs" / task_config / "task.yaml"
    )
    if task_cfg_path.exists():
        task_cfg = OmegaConf.load(str(task_cfg_path))
        cfg = OmegaConf.merge(cfg, task_cfg)

    return cfg


def _run_method(
    method: str,
    base_cfg,
    base_run_dir: pathlib.Path,
    skip_if_done: bool,
    dry_run: bool,
) -> Dict:
    """Run the graph-building sub-pipeline for *method* in its own subdirectory."""
    from omegaconf import OmegaConf
    from policy_doctor.curation_pipeline.pipeline import CurationPipeline

    run_dir = base_run_dir / method
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    OmegaConf.update(cfg, "graph_building.method", method, merge=True)
    OmegaConf.update(cfg, "run_dir", str(run_dir), merge=True)
    OmegaConf.update(cfg, "run_name", method, merge=True)
    if dry_run:
        OmegaConf.update(cfg, "dry_run", True, merge=True)

    pipeline = CurationPipeline(cfg)

    # For ENAP variants, we can save time by symlinking the shared
    # train_enap_perception output from the first ENAP run.
    if method in ("enap", "enap_custom"):
        _symlink_perception_step(base_run_dir, method, run_dir)

    results = pipeline.run(steps=["graph_building"], skip_if_done=skip_if_done)
    return results


def _symlink_perception_step(
    base_run_dir: pathlib.Path,
    current_method: str,
    current_run_dir: pathlib.Path,
) -> None:
    """Symlink train_enap_perception from the first ENAP run that completed it.

    This avoids re-running the visual encoder + HDBSCAN for every ENAP variant.
    Only creates the symlink when the perception step has already been completed
    in a sibling run directory.
    """
    for donor_method in ("enap_custom", "enap"):
        if donor_method == current_method:
            continue
        donor_dir = base_run_dir / donor_method / "train_enap_perception"
        done_flag = donor_dir / "done"
        target = current_run_dir / "train_enap_perception"
        if done_flag.exists() and not target.exists():
            try:
                target.symlink_to(donor_dir.resolve())
                print(
                    f"  [share] Symlinked train_enap_perception from "
                    f"{donor_method} → {current_method}"
                )
            except OSError:
                pass  # non-fatal — step will just re-run
            return


def _load_summary(run_dir: pathlib.Path, method: str) -> Optional[Dict]:
    """Load BehaviorGraph summary from build_behavior_graph result.json."""
    result_path = run_dir / method / "build_behavior_graph" / "result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def _print_comparison(
    methods: List[str],
    base_run_dir: pathlib.Path,
    statuses: Dict[str, str],
) -> None:
    """Print a side-by-side comparison table."""
    rows = []
    for method in methods:
        status = statuses.get(method, "unknown")
        summary = _load_summary(base_run_dir, method)
        if summary and status == "ok":
            seeds = summary.get("seeds", {})
            first_seed = next(iter(seeds.values()), summary)
            rows.append(
                {
                    "method": method,
                    "status": "ok",
                    "nodes": first_seed.get("num_cluster_nodes", "?"),
                    "episodes": first_seed.get("num_episodes", "?"),
                    "builder": summary.get("builder", method),
                    "level": first_seed.get("level", "?"),
                }
            )
        else:
            rows.append(
                {
                    "method": method,
                    "status": status,
                    "nodes": "—",
                    "episodes": "—",
                    "builder": "—",
                    "level": "—",
                }
            )

    col_w = {"method": 14, "status": 8, "nodes": 7, "episodes": 9, "level": 10}
    header = (
        f"{'Method':<{col_w['method']}} "
        f"{'Status':<{col_w['status']}} "
        f"{'Nodes':>{col_w['nodes']}} "
        f"{'Episodes':>{col_w['episodes']}} "
        f"{'Level':<{col_w['level']}}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  BehaviorGraph Comparison")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['method']:<{col_w['method']}} "
            f"{r['status']:<{col_w['status']}} "
            f"{str(r['nodes']):>{col_w['nodes']}} "
            f"{str(r['episodes']):>{col_w['episodes']}} "
            f"{str(r['level']):<{col_w['level']}}"
        )
    print(sep)
    print()
    for method in methods:
        run_dir = base_run_dir / method / "build_behavior_graph"
        if run_dir.exists():
            print(f"  [{method}] outputs → {run_dir}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Task config name (default: from pipeline config.yaml)",
    )
    parser.add_argument(
        "--base-run-dir",
        default="data/compare_graph_methods",
        help="Base directory for per-method run folders",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cupid", "enap_custom", "enap"],
        choices=["cupid", "enap_custom", "enap"],
        help="Methods to compare (default: all three)",
    )
    parser.add_argument(
        "--config-root",
        default="iv",
        help="Config root: 'iv' (influence_visualizer) or 'pd' (standalone)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Repo root containing data/outputs/... "
            "(default: policy_doctor_enap repo root). "
            "Override when data lives in a separate checkout, "
            "e.g. --repo-root /home/erbauer/cupid"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for ENAP neural steps (e.g. 'cuda:0', 'cpu'). "
             "Default: auto-detect (cuda if available).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for in-process training steps.",
    )
    parser.add_argument(
        "--wandb-project",
        default="policy_doctor_enap",
        help="W&B project name (default: policy_doctor_enap)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed steps (default: False = re-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing heavy computation",
    )
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT

    base_run_dir = pathlib.Path(args.base_run_dir)
    if not base_run_dir.is_absolute():
        base_run_dir = (_REPO_ROOT / base_run_dir).resolve()
    base_run_dir.mkdir(parents=True, exist_ok=True)

    # Load base config — code always comes from _REPO_ROOT; data from repo_root
    base_cfg = _load_base_cfg(
        task_config=args.task_config or "transport_mh_jan28",
        config_root=args.config_root,
        repo_root=repo_root,
    )

    # Apply optional overrides
    from omegaconf import OmegaConf
    if args.device:
        OmegaConf.update(base_cfg, "device", args.device, merge=True)
    if args.wandb:
        OmegaConf.update(base_cfg, "wandb.enabled", True, merge=True)
        OmegaConf.update(base_cfg, "wandb.project", args.wandb_project, merge=True)

    print(f"Data repo root:  {repo_root}")
    print(f"Code repo root:  {_REPO_ROOT}")

    skip_if_done = args.resume
    methods = args.methods
    statuses: Dict[str, str] = {}

    print(f"\nComparing graph methods: {methods}")
    print(f"Base run dir: {base_run_dir}\n")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        try:
            _run_method(
                method=method,
                base_cfg=base_cfg,
                base_run_dir=base_run_dir,
                skip_if_done=skip_if_done,
                dry_run=args.dry_run,
            )
            statuses[method] = "ok"
        except Exception as e:
            print(f"\n  [ERROR] {method} failed: {e}")
            traceback.print_exc()
            statuses[method] = f"error: {type(e).__name__}"

    _print_comparison(methods, base_run_dir, statuses)

    n_failed = sum(1 for s in statuses.values() if s != "ok")
    if n_failed:
        print(f"  {n_failed}/{len(methods)} method(s) failed.")
        return 1
    print("  All methods completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
