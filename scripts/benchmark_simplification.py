"""Run all simplification methods across all lever values on all benchmark tasks.

Saves per-task JSON to docs/simplification_results/{task}__{clustering}.json.
Each JSON contains, for each method, a list of FrontierPoint dicts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running from the worktree root.
_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

from policy_doctor.behaviors.behavior_graph import BehaviorGraph  # noqa: E402
from policy_doctor.behaviors.simplification import (  # noqa: E402
    METHODS,
    LEVER_GRIDS,
)
from policy_doctor.behaviors.simplification.frontier import sweep_method  # noqa: E402


# Clusterings to benchmark. (task_name, clustering_dir, k_clusters)
# The clusterings live in the sibling worktree at this path:
_DATA_ROOT = Path(
    "/Users/erik/stanford/asl_rotation/policy_doctor/.claude/worktrees/"
    "graph-simplification/third_party/influence_visualizer/configs"
)

DEFAULT_BENCHMARKS = [
    # (task, rep, k) - we'll use w=5, s=1
    ("transport_mh_jan28", "policy_emb_bottleneck_plan_t0", 5),
    ("transport_mh_jan28", "policy_emb_bottleneck_plan_t0", 15),
    ("square_mh_feb5",     "policy_emb_bottleneck_plan_t0", 5),
    ("square_mh_feb5",     "policy_emb_bottleneck_plan_t0", 15),
    ("lift_mh_jan26",      "policy_emb_bottleneck_plan_t0", 5),
    ("lift_mh_jan26",      "policy_emb_bottleneck_plan_t0", 15),
]


def find_clustering_dir(task: str, rep: str, k: int, w: int = 5, s: int = 1) -> Path:
    slug = f"{rep}_w{w}_s{s}_seed0_kmeans_k{k}"
    candidates = [
        _DATA_ROOT / task / "clustering" / slug,
        _DATA_ROOT / task / "clustering" / f"{rep}_seed0_kmeans_k{k}",
    ]
    for c in candidates:
        if (c / "cluster_labels.npy").exists():
            return c
    raise FileNotFoundError(f"No clustering for {task} / {rep} k={k}: tried {candidates}")


def load_clustering(d: Path) -> tuple[np.ndarray, list[dict], dict]:
    labels = np.load(d / "cluster_labels.npy").astype(np.int64)
    with (d / "metadata.json").open() as f:
        meta = json.load(f)
    manifest = {}
    mf = d / "manifest.yaml"
    if mf.exists():
        import yaml
        with mf.open() as f:
            manifest = yaml.safe_load(f) or {}
    return labels, meta, manifest


def run_one(
    task: str, rep: str, k: int, methods: List[str], n_folds: int, out_dir: Path,
    with_bootstrap: bool = False, n_bootstrap: int = 20,
) -> None:
    clu_dir = find_clustering_dir(task, rep, k)
    print(f"\n=== {task} | {clu_dir.name}", flush=True)
    labels, meta, manifest = load_clustering(clu_dir)
    level = manifest.get("level", "rollout")
    n_eps = len({m["rollout_idx" if level == "rollout" else "demo_idx"] for m in meta})
    print(f"    {len(labels)} samples, {n_eps} episodes, k={k}, level={level}", flush=True)

    graph = BehaviorGraph.from_cluster_assignments(labels, meta, level=level)
    n_cluster_nodes = len(graph.cluster_nodes)
    print(f"    raw graph: {n_cluster_nodes} cluster nodes", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}__{clu_dir.name}.json"

    results: Dict[str, List[dict]] = {}
    for m in methods:
        print(f"      sweeping {m} ...", end="", flush=True)
        t0 = time.time()
        # Cap PCCA+ / Markov-stability grid at n_cluster_nodes - 1
        grid = LEVER_GRIDS.get(m, np.array([0.0]))
        if m == "pcca_plus":
            grid = np.arange(2, max(3, n_cluster_nodes)).astype(float)
        try:
            pts = sweep_method(
                m, graph, labels, meta,
                lever_grid=grid,
                with_heldout=True, n_folds=n_folds,
                with_bootstrap=with_bootstrap, n_bootstrap=n_bootstrap,
            )
            results[m] = [p.as_dict() for p in pts]
            dt = time.time() - t0
            print(f" {len(pts)} pts in {dt:.1f}s", flush=True)
        except Exception as e:
            print(f" FAILED: {type(e).__name__}: {e}", flush=True)
            results[m] = [{"error": f"{type(e).__name__}: {e}"}]

    summary = {
        "task": task,
        "rep": rep,
        "k": k,
        "clustering_dir": str(clu_dir),
        "level": level,
        "n_samples": int(len(labels)),
        "n_episodes": int(n_eps),
        "n_cluster_nodes_raw": int(n_cluster_nodes),
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"    -> {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default="docs/simplification_results",
        help="Output directory (default: docs/simplification_results)",
    )
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--methods", nargs="*", default=None,
                   help="Methods to run (default: all registered)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Restrict to a subset of tasks (by name).")
    p.add_argument("--k", type=int, default=None,
                   help="Restrict to a specific k.")
    p.add_argument("--bootstrap", action="store_true",
                   help="Compute episode-bootstrap CIs (slow).")
    p.add_argument("--n_bootstrap", type=int, default=20)
    args = p.parse_args()

    methods = args.methods if args.methods else sorted(METHODS.keys())
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _WORKTREE / out_dir

    benchmarks = DEFAULT_BENCHMARKS
    if args.tasks:
        benchmarks = [b for b in benchmarks if b[0] in args.tasks]
    if args.k is not None:
        benchmarks = [b for b in benchmarks if b[2] == args.k]

    for task, rep, k in benchmarks:
        try:
            run_one(task, rep, k, methods, args.n_folds, out_dir,
                    with_bootstrap=args.bootstrap, n_bootstrap=args.n_bootstrap)
        except Exception as e:
            print(f"FAILED {task} k={k}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
