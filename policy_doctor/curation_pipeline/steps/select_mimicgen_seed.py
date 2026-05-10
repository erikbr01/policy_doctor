"""Select a MimicGen seed rollout using a configurable heuristic — pipeline step.

This step selects a single rollout trajectory from the policy's eval episodes
to serve as the MimicGen seed, then materialises it as ``step_dir/seed.hdf5``
so that :class:`~policy_doctor.curation_pipeline.steps.generate_mimicgen_demos
.GenerateMimicgenDemosStep` can pick it up automatically.

Two selection strategies are supported via ``mimicgen_datagen.seed_selection_heuristic``:

``behavior_graph`` (default / proposed method)
    Builds a behavior graph from the clustering result, ranks paths to the
    SUCCESS node by probability, and returns the first rollout whose collapsed
    cluster sequence exactly matches the top path.  This informed selection
    should yield higher-quality MimicGen-generated data.

``random`` (baseline)
    Picks a successful rollout uniformly at random.  Used to isolate the
    contribution of informed seed selection vs. just running MimicGen at all.

Config keys (under ``mimicgen_datagen``):
    seed_selection_heuristic   ``"behavior_graph"`` | ``"random"`` (default: ``"behavior_graph"``).
    num_seeds                  How many seed trajectories to write into seed.hdf5 (default 1).
    top_k_paths                Number of candidate paths to try for ``behavior_graph`` (default 5).
    min_path_probability       Min path probability for ``behavior_graph`` (default 0.0).
    success_only               Only consider successful rollouts (default True).
    random_seed                RNG seed for ``random`` heuristic (default None).
    policy_seed                Which policy seed's clustering to use (default: first available).

Also reads standard pipeline config keys:
    reference_seed  (to resolve eval dir via evaluation.train_date / evaluation.task /
    evaluation.policy or the clustering_eval_dir override).

Result JSON:
    seed_hdf5_path       Absolute path to the materialised ``seed.hdf5``.
    num_seeds            Number of seed trajectories written to seed.hdf5.
    rollout_idxs         List of episode indices of the selected rollouts.
    heuristic            Name of the heuristic used.
    selection_info       List of heuristic-specific metadata dicts (one per seed).
    policy_seed          Which policy seed's clustering was used.
    clustering_dir       Resolved clustering directory.
    rollouts_hdf5        Path to the source rollout HDF5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.data.clustering_loader import load_clustering_result_from_path
from policy_doctor.mimicgen.heuristics import SeedSelectionResult, build_heuristic
from policy_doctor.mimicgen.materializer import RobomimicSeedMaterializer

# Re-use the eval-dir / rollout-HDF5 resolver from the existing graph step.
from policy_doctor.curation_pipeline.steps.select_mimicgen_seed_from_graph import (
    _resolve_rollouts_hdf5,
)


class SelectMimicgenSeedStep(PipelineStep[dict]):
    """Select a MimicGen seed rollout using a configurable heuristic.

    Writes ``step_dir/seed.hdf5`` — automatically consumed by
    :class:`~policy_doctor.curation_pipeline.steps.generate_mimicgen_demos
    .GenerateMimicgenDemosStep` when present.
    """

    name = "select_mimicgen_seed"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

        cfg_mg = OmegaConf.select(self.cfg, "mimicgen_datagen") or {}

        heuristic_name: str = OmegaConf.select(cfg_mg, "seed_selection_heuristic") or "behavior_graph"
        num_seeds: int = int(OmegaConf.select(cfg_mg, "num_seeds") or 1)
        top_k_paths: int = int(OmegaConf.select(cfg_mg, "top_k_paths") or 5)
        min_path_probability: float = float(OmegaConf.select(cfg_mg, "min_path_probability") or 0.0)
        success_only_raw = OmegaConf.select(cfg_mg, "success_only")
        success_only: bool = bool(success_only_raw) if success_only_raw is not None else True
        random_seed_raw = OmegaConf.select(cfg_mg, "random_seed")
        random_seed: int | None = int(random_seed_raw) if random_seed_raw is not None else None

        print(
            f"  [select_mimicgen_seed] heuristic={heuristic_name!r}  num_seeds={num_seeds}  "
            f"success_only={success_only}  top_k_paths={top_k_paths}"
        )

        # --- Load clustering result ---
        # Use parent_run_dir so this step finds RunClusteringStep when running
        # inside a CompositeStep (where run_dir is the composite's sub-directory).
        prior = RunClusteringStep(self.cfg, self.parent_run_dir).load()
        clustering_dirs: dict[str, str] = {}
        if prior and isinstance(prior.get("clustering_dirs"), dict):
            clustering_dirs = {str(k): str(v) for k, v in prior["clustering_dirs"].items()}

        explicit = OmegaConf.select(self.cfg, "clustering_dir")
        if explicit and not clustering_dirs:
            clustering_dirs["0"] = str(explicit)

        if not clustering_dirs:
            raise ValueError(
                "No clustering directories found.  "
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

        cdir = Path(clustering_dirs[seed])
        if not cdir.is_absolute():
            cdir = (self.repo_root / cdir).resolve()

        print(f"  [select_mimicgen_seed] seed={seed}  clustering_dir={cdir}")

        labels, metadata, manifest = load_clustering_result_from_path(cdir)
        level: str = manifest.get("level") or "rollout"

        # --- Resolve rollout HDF5 ---
        rollouts_hdf5 = _resolve_rollouts_hdf5(self.cfg, self.repo_root, seed)
        print(f"  [select_mimicgen_seed] rollouts_hdf5={rollouts_hdf5}")

        # --- Failure analysis: check if AnalyzeFailureStatesStep ran ---
        # When enabled, override num_seeds and select per-cluster using NearFailurePathHeuristic.
        from policy_doctor.curation_pipeline.steps.analyze_failure_states import (
            AnalyzeFailureStatesStep,
        )
        fa_step = AnalyzeFailureStatesStep(self.cfg, self.parent_run_dir)
        fa_result = fa_step.load() if fa_step.is_done() else None
        use_failure_analysis = (
            fa_result is not None
            and fa_result.get("enabled", False)
            and fa_result.get("clusters")
        )

        results: list[SeedSelectionResult] = []
        # per_seed_object_pose_ranges[i] and per_seed_subtask_constraints[i] carry
        # per-seed constraint suggestions derived from the failure cluster of that seed.
        per_seed_object_pose_ranges: list[dict | None] = []
        per_seed_subtask_constraints: list[dict | None] = []

        if use_failure_analysis:
            clusters = fa_result["clusters"]
            print(
                f"  [select_mimicgen_seed] failure analysis active: "
                f"{len(clusters)} clusters, selecting {heuristic_name} seeds per cluster"
            )
            for cluster_info in clusters:
                ci = cluster_info["cluster_idx"]
                budget = int(cluster_info.get("budget_per_cluster", 1))
                eligible = cluster_info.get("eligible_rollout_idxs") or None

                heuristic = build_heuristic(
                    heuristic_name,
                    top_k_paths=top_k_paths,
                    min_path_probability=min_path_probability,
                    success_only=success_only,
                    random_seed=random_seed,
                    eligible_rollout_idxs=eligible,
                )
                try:
                    cluster_results = heuristic.select_multiple(
                        n=budget,
                        cluster_labels=labels,
                        metadata=metadata,
                        rollout_hdf5_path=str(rollouts_hdf5),
                        level=level,
                    )
                except RuntimeError as e:
                    print(f"  [select_mimicgen_seed] cluster {ci}: selection failed: {e}")
                    continue

                opr = cluster_info.get("suggested_object_pose_ranges")
                sc = cluster_info.get("suggested_subtask_constraints")
                for r in cluster_results:
                    results.append(r)
                    per_seed_object_pose_ranges.append(opr)
                    per_seed_subtask_constraints.append(sc)
                print(
                    f"    cluster {ci}: selected {len(cluster_results)} seeds "
                    f"from {len(eligible or [])} eligible rollouts"
                )
        else:
            # Standard path: single heuristic, flat seed list.
            heuristic = build_heuristic(
                heuristic_name,
                top_k_paths=top_k_paths,
                min_path_probability=min_path_probability,
                success_only=success_only,
                random_seed=random_seed,
            )
            results = heuristic.select_multiple(
                n=num_seeds,
                cluster_labels=labels,
                metadata=metadata,
                rollout_hdf5_path=str(rollouts_hdf5),
                level=level,
            )
            per_seed_object_pose_ranges = [None] * len(results)
            per_seed_subtask_constraints = [None] * len(results)

        for r in results:
            print(
                f"  [select_mimicgen_seed] selected rollout_idx={r.rollout_idx}  "
                f"T={r.trajectory.states.shape[0]}  info={r.info}"
            )

        # --- Materialise seed HDF5 (one demo per trajectory) ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        seed_hdf5 = self.step_dir / "seed.hdf5"
        mat = RobomimicSeedMaterializer()
        if len(results) == 1:
            mat.write_source_dataset(
                states=results[0].trajectory.states,
                actions=results[0].trajectory.actions,
                env_meta=results[0].trajectory.env_meta,
                output_path=seed_hdf5,
                model_file=results[0].trajectory.model_file,
            )
        else:
            mat.write_multi_demo_source_dataset(
                trajectories=[r.trajectory for r in results],
                output_path=seed_hdf5,
            )
        print(f"  [select_mimicgen_seed] seed HDF5 written ({len(results)} demos): {seed_hdf5}")

        result: dict[str, Any] = {
            "seed_hdf5_path": str(seed_hdf5.resolve()),
            "num_seeds": len(results),
            "rollout_idxs": [r.rollout_idx for r in results],
            "heuristic": heuristic_name,
            "selection_info": [r.info for r in results],
            "policy_seed": seed,
            "clustering_dir": str(cdir),
            "rollouts_hdf5": str(rollouts_hdf5),
        }
        # Only include per-seed constraints in result when at least one is non-null.
        if any(v is not None for v in per_seed_object_pose_ranges):
            result["per_seed_object_pose_ranges"] = per_seed_object_pose_ranges
        if any(v is not None for v in per_seed_subtask_constraints):
            result["per_seed_subtask_constraints"] = per_seed_subtask_constraints
        return result
