"""Select a MimicGen seed rollout from the behavior graph — pipeline step.

For each of the top-k paths to success in the behavior graph, this step finds
policy rollouts whose collapsed cluster-label sequence exactly matches that
path.  The first path with a matching rollout is used as the MimicGen seed.

If **none** of the top-k paths has a matching rollout a loud warning is printed
and the step raises ``RuntimeError``.

The materialised seed HDF5 (``step_dir/seed.hdf5``) is automatically picked up
by :class:`~policy_doctor.curation_pipeline.steps.generate_mimicgen_demos.GenerateMimicgenDemosStep`
when both steps are present in the same pipeline run.

Config keys (under ``mimicgen_datagen``):
    top_k_paths         Top-k paths to try, ranked by probability (default 5).
    min_path_probability  Minimum path probability to consider (default 0.0).
    policy_seed         Which policy seed's clustering result to use.
                        Default: the first key in RunClusteringStep's result.
    success_only        Only consider successful rollout episodes (default True).

Also reads standard pipeline config keys used by ``RunClusteringStep``:
    task_config, config_root, reference_seed, seeds (to resolve eval dir).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from policy_doctor.behaviors.behavior_graph import BehaviorGraph, SUCCESS_NODE_ID
from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.data.clustering_loader import load_clustering_result_from_path
from policy_doctor.mimicgen.graph_seed import top_paths_with_rollouts
from policy_doctor.mimicgen.materializer import RobomimicSeedMaterializer
from policy_doctor.mimicgen.seed_trajectory import MimicGenSeedTrajectory
from policy_doctor.paths import PACKAGE_ROOT, iv_task_configs_base


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _warn_box(lines: list[str]) -> str:
    """Return a boxed warning string for highly-visible console output."""
    width = max(len(l) for l in lines) + 4
    border = "╔" + "═" * width + "╗"
    footer = "╚" + "═" * width + "╝"
    body = "\n".join("║  " + l.ljust(width - 2) + "  ║" for l in lines)
    return f"\n{border}\n{body}\n{footer}\n"


def _fmt_path(path: list[int]) -> str:
    """Pretty-print a behavior graph path using node names."""
    labels = {
        -2: "START",
        -3: "END",
        -4: "SUCCESS",
        -5: "FAILURE",
    }
    return " → ".join(labels.get(n, str(n)) for n in path)


# ---------------------------------------------------------------------------
# Eval dir resolution
# ---------------------------------------------------------------------------

def _resolve_rollouts_hdf5(
    cfg: Any,
    repo_root: Path,
    seed: str,
) -> Path:
    """Locate the ``rollouts.hdf5`` for a given policy *seed*.

    Eval-dir resolution mirrors ``RunClusteringStep`` — three-level override chain:

    1. ``clustering_eval_dir`` explicit override (highest priority)
    2. ``evaluation.train_date`` + ``evaluation.task`` + ``evaluation.policy``
       → constructs path via :func:`~policy_doctor.curation_pipeline.paths.get_eval_dir`
    3. ``task_cfg["eval_dir"]`` from the static ``task_config`` YAML (lowest priority)

    The third fallback is a last resort only — it does not respect runtime
    ``evaluation.train_date`` overrides and can silently point to a different
    experiment's rollouts when task_config was written for a different date.
    A loud warning is printed when the fallback is used.
    """
    from influence_visualizer.data_loader import get_eval_dir_for_seed
    from policy_doctor.curation_pipeline.paths import get_eval_dir

    config_root = OmegaConf.select(cfg, "config_root") or "iv"
    task_config = OmegaConf.select(cfg, "task_config")
    if not task_config:
        raise ValueError("task_config is required in config to locate rollouts.hdf5")

    if config_root == "iv":
        base = iv_task_configs_base(repo_root)
    else:
        base = PACKAGE_ROOT / "configs"

    task_yaml = base / f"{task_config}.yaml"
    with open(task_yaml) as f:
        task_cfg = yaml.safe_load(f)

    # --- Three-level eval_dir_base resolution (mirrors RunClusteringStep) ---
    clustering_eval_dir_override = OmegaConf.select(cfg, "clustering_eval_dir")
    evaluation = OmegaConf.select(cfg, "evaluation") or {}
    eval_date = (
        OmegaConf.select(evaluation, "train_date")
        or OmegaConf.select(cfg, "evaluation.eval_date")
        or OmegaConf.select(cfg, "train_date")
    )
    eval_task = OmegaConf.select(evaluation, "task")
    eval_policy = OmegaConf.select(evaluation, "policy")
    eval_output_dir = OmegaConf.select(evaluation, "eval_output_dir") or "data/outputs/eval_save_episodes"

    if clustering_eval_dir_override:
        eval_dir_base: str = clustering_eval_dir_override
        print(f"  [_resolve_rollouts_hdf5] using clustering_eval_dir override: {eval_dir_base!r}")
    elif eval_date and eval_task and eval_policy:
        eval_dir_base = get_eval_dir(eval_output_dir, eval_date, eval_task, eval_policy, 0)
        print(f"  [_resolve_rollouts_hdf5] using evaluation.train_date={eval_date!r}: {eval_dir_base!r}")
    else:
        raise ValueError(
            f"Cannot resolve rollouts.hdf5 path: evaluation.train_date, evaluation.task, and "
            f"evaluation.policy must all be set to locate the correct experiment's rollouts.\n"
            f"  Got: eval_date={eval_date!r}, eval_task={eval_task!r}, eval_policy={eval_policy!r}\n"
            f"  task_config YAML ({task_config!r}) eval_dir={task_cfg.get('eval_dir')!r} is NOT "
            f"used as a fallback — it may point to a different experiment and cause silent data "
            f"contamination.\n"
            f"  Fix: set evaluation.train_date in your experiment YAML or pass it as a CLI override."
        )

    reference_seed = str(OmegaConf.select(cfg, "reference_seed") or 0)
    eval_dir_seed = get_eval_dir_for_seed(eval_dir_base, seed, reference_seed)
    eval_dir_abs: Path = repo_root / eval_dir_seed

    # Standard robomimic/cupid layout written by eval_save_episodes
    candidates = [
        eval_dir_abs / "episodes" / "rollouts.hdf5",
        eval_dir_abs / "rollouts.hdf5",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"rollouts.hdf5 not found for seed {seed}.\n"
        f"Tried: {candidates}\n"
        f"Eval dir resolved to: {eval_dir_abs}"
    )


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

class SelectMimicgenSeedFromGraphStep(PipelineStep[dict]):
    """Select a seed rollout for MimicGen from the behavior graph.

    Reads from :class:`~policy_doctor.curation_pipeline.steps.run_clustering
    .RunClusteringStep` and writes ``step_dir/seed.hdf5`` plus a result JSON
    that is consumed by
    :class:`~policy_doctor.curation_pipeline.steps.generate_mimicgen_demos
    .GenerateMimicgenDemosStep`.
    """

    name = "select_mimicgen_seed_from_graph"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

        cfg_mg = OmegaConf.select(self.cfg, "mimicgen_datagen") or {}

        top_k: int = int(OmegaConf.select(cfg_mg, "top_k_paths") or 5)
        min_prob: float = float(OmegaConf.select(cfg_mg, "min_path_probability") or 0.0)
        success_only: bool = bool(OmegaConf.select(cfg_mg, "success_only") if OmegaConf.select(cfg_mg, "success_only") is not None else True)

        # --- Load clustering result ---
        prior = RunClusteringStep(self.cfg, self.run_dir).load()
        clustering_dirs: dict[str, str] = {}
        if prior and isinstance(prior.get("clustering_dirs"), dict):
            clustering_dirs = {str(k): str(v) for k, v in prior["clustering_dirs"].items()}

        explicit = OmegaConf.select(self.cfg, "clustering_dir")
        if explicit and not clustering_dirs:
            # Accept a single explicit override (compatible with ExportMarkovReportStep)
            clustering_dirs["0"] = str(explicit)

        if not clustering_dirs:
            raise ValueError(
                "No clustering directories found.\n"
                "Run run_clustering first, or set clustering_dir in config."
            )

        # --- Select policy seed ---
        policy_seed_cfg = OmegaConf.select(cfg_mg, "policy_seed")
        if policy_seed_cfg is not None:
            seed = str(policy_seed_cfg)
            if seed not in clustering_dirs:
                raise KeyError(
                    f"mimicgen_datagen.policy_seed={seed!r} not in clustering results. "
                    f"Available seeds: {list(clustering_dirs.keys())}"
                )
        else:
            seed = sorted(clustering_dirs.keys())[0]
            print(
                f"  [select_mimicgen_seed_from_graph] WARNING: mimicgen_datagen.policy_seed not set; "
                f"defaulting to first available seed: {seed!r}  "
                f"(available: {sorted(clustering_dirs.keys())})"
            )

        cdir = Path(clustering_dirs[seed])
        if not cdir.is_absolute():
            cdir = (self.repo_root / cdir).resolve()

        print(f"  [select_mimicgen_seed_from_graph] seed={seed}  clustering_dir={cdir}")

        labels, metadata, manifest = load_clustering_result_from_path(cdir)
        level: str = manifest.get("level") or "rollout"

        # --- Build behavior graph ---
        graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)

        if SUCCESS_NODE_ID not in graph.nodes:
            raise RuntimeError(
                "Behavior graph has no SUCCESS node — all rollouts may have failed. "
                "Cannot select a success path for MimicGen generation."
            )

        # --- Rank top-k paths + find matching rollouts ---
        ranked = top_paths_with_rollouts(
            graph, labels, metadata,
            top_k=top_k,
            min_path_probability=min_prob,
            success_only=success_only,
            level=level,
        )

        if not ranked:
            raise RuntimeError(
                "enumerate_paths_to_terminal returned no paths to SUCCESS. "
                f"Graph has {graph.num_episodes} episodes and "
                f"{len(graph.nodes)} nodes."
            )

        warnings: list[str] = []
        selected: dict[str, Any] | None = None

        print(f"  [select_mimicgen_seed_from_graph] Top-{top_k} paths to SUCCESS:")
        for rank, entry in enumerate(ranked):
            path = entry["path"]
            prob = entry["path_prob"]
            idxs = entry["rollout_idxs"]
            has = entry["has_match"]
            seq = entry["cluster_seq"]
            print(
                f"    {rank+1}. {_fmt_path(path)}  "
                f"prob={prob:.3f}  seq={seq}  rollouts={idxs or '(none)'}"
            )
            if not has:
                msg = _warn_box([
                    f"WARNING: No rollout found for path #{rank+1}",
                    f"  Path: {_fmt_path(path)}",
                    f"  Cluster sequence: {seq}",
                    f"  Path probability: {prob:.3f}",
                    "  Trying next path...",
                ])
                print(msg)
                warnings.append(
                    f"No rollout found for path {path} (prob={prob:.3f}, seq={seq})"
                )
            else:
                selected = entry
                print(
                    f"  [select_mimicgen_seed_from_graph] "
                    f"Selected path #{rank+1}: rollout_idx={idxs[0]}"
                )
                break

        if selected is None:
            msg = _warn_box([
                "FATAL: No rollout matches any of the top-k paths to SUCCESS.",
                f"  top_k={top_k}  success_only={success_only}  seed={seed}",
                "  Paths tried:",
                *[f"    {e['cluster_seq']}  (prob={e['path_prob']:.3f})" for e in ranked],
                "",
                "  Possible causes:",
                "  - The clustering is too fine-grained (many clusters, few episodes).",
                "  - success_only=True but no rollout has the right sequence.",
                "  - top_k too small. Try increasing mimicgen_datagen.top_k_paths.",
            ])
            print(msg)
            raise RuntimeError(
                "SelectMimicgenSeedFromGraphStep: no rollout found for any top-k path to SUCCESS.\n"
                + "\n".join(warnings)
            )

        rollout_idx: int = selected["rollout_idxs"][0]

        # --- Load rollout from HDF5 ---
        rollouts_hdf5 = _resolve_rollouts_hdf5(self.cfg, self.repo_root, seed)
        print(f"  [select_mimicgen_seed_from_graph] loading rollout {rollout_idx} from {rollouts_hdf5}")

        demo_key = f"demo_{rollout_idx}"
        traj = MimicGenSeedTrajectory.from_rollout_hdf5(rollouts_hdf5, demo_key=demo_key)

        # --- Materialise seed HDF5 ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        seed_hdf5 = self.step_dir / "seed.hdf5"
        mat = RobomimicSeedMaterializer()
        mat.write_source_dataset(
            states=traj.states,
            actions=traj.actions,
            env_meta=traj.env_meta,
            output_path=seed_hdf5,
            model_file=traj.model_file,
        )
        print(f"  [select_mimicgen_seed_from_graph] seed HDF5 written: {seed_hdf5}")
        print(
            f"  [select_mimicgen_seed_from_graph] done. "
            f"path={selected['cluster_seq']}  prob={selected['path_prob']:.3f}  "
            f"rollout={rollout_idx}  T={traj.states.shape[0]}"
        )

        return {
            "selected_path": selected["path"],
            "selected_path_prob": selected["path_prob"],
            "selected_cluster_seq": selected["cluster_seq"],
            "selected_rollout_idx": rollout_idx,
            "rollouts_hdf5": str(rollouts_hdf5),
            "seed_hdf5_path": str(seed_hdf5.resolve()),
            "policy_seed": seed,
            "clustering_dir": str(cdir),
            "top_paths": [
                {
                    "path": e["path"],
                    "path_prob": e["path_prob"],
                    "cluster_seq": e["cluster_seq"],
                    "rollout_idxs": e["rollout_idxs"],
                    "has_match": e["has_match"],
                }
                for e in ranked
            ],
            "warnings": warnings,
        }
