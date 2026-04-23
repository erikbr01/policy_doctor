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
    top_k_paths                Number of candidate paths to try for ``behavior_graph`` (default 5).
    min_path_probability       Min path probability for ``behavior_graph`` (default 0.0).
    success_only               Only consider successful rollouts (default True).
    random_seed                RNG seed for ``random`` heuristic (default None).
    policy_seed                Which policy seed's clustering to use (default: first available).

Also reads standard pipeline config keys used by ``RunClusteringStep``:
    task_config, config_root, reference_seed  (to resolve eval dir and rollout HDF5).

Result JSON:
    seed_hdf5_path       Absolute path to the materialised ``seed.hdf5``.
    rollout_idx          Episode index of the selected rollout.
    heuristic            Name of the heuristic used.
    selection_info       Heuristic-specific metadata (path, probability, etc.).
    policy_seed          Which policy seed's clustering was used.
    clustering_dir       Resolved clustering directory.
    rollouts_hdf5        Path to the source rollout HDF5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        top_k_paths: int = int(OmegaConf.select(cfg_mg, "top_k_paths") or 5)
        min_path_probability: float = float(OmegaConf.select(cfg_mg, "min_path_probability") or 0.0)
        success_only_raw = OmegaConf.select(cfg_mg, "success_only")
        success_only: bool = bool(success_only_raw) if success_only_raw is not None else True
        random_seed_raw = OmegaConf.select(cfg_mg, "random_seed")
        random_seed: int | None = int(random_seed_raw) if random_seed_raw is not None else None

        print(
            f"  [select_mimicgen_seed] heuristic={heuristic_name!r}  "
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

        # --- Build and run heuristic ---
        heuristic = build_heuristic(
            heuristic_name,
            top_k_paths=top_k_paths,
            min_path_probability=min_path_probability,
            success_only=success_only,
            random_seed=random_seed,
        )

        result: SeedSelectionResult = heuristic.select(
            cluster_labels=labels,
            metadata=metadata,
            rollout_hdf5_path=str(rollouts_hdf5),
            level=level,
        )

        traj = result.trajectory
        rollout_idx = result.rollout_idx
        print(
            f"  [select_mimicgen_seed] selected rollout_idx={rollout_idx}  "
            f"T={traj.states.shape[0]}  info={result.info}"
        )

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
        print(f"  [select_mimicgen_seed] seed HDF5 written: {seed_hdf5}")

        return {
            "seed_hdf5_path": str(seed_hdf5.resolve()),
            "rollout_idx": rollout_idx,
            "heuristic": heuristic_name,
            "selection_info": result.info,
            "policy_seed": seed,
            "clustering_dir": str(cdir),
            "rollouts_hdf5": str(rollouts_hdf5),
        }
