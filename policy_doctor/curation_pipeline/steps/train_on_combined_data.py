"""Train a policy on original demonstrations plus MimicGen-generated data — pipeline step.

This step closes the MimicGen experiment loop:

1. Load the MimicGen-generated HDF5 from
   :class:`~policy_doctor.curation_pipeline.steps.generate_mimicgen_demos
   .GenerateMimicgenDemosStep`.
2. Resolve the original training dataset path (same as :class:`TrainCuratedStep`).
3. Combine the two into ``step_dir/combined.hdf5`` via
   :func:`~policy_doctor.mimicgen.combine_datasets.combine_hdf5_datasets`.
4. Train a policy on the combined dataset using the same Hydra/cupid machinery
   as the baseline and curated training steps.

The ``heuristic`` field from
:class:`~policy_doctor.curation_pipeline.steps.select_mimicgen_seed
.SelectMimicgenSeedStep` is embedded in the training run name so that runs
using different seed-selection strategies live in separate output directories
and can be compared side-by-side.

Config keys (under ``baseline``): same as :class:`TrainBaselineStep`
    config_dir, config_name, task, policy, seeds, num_epochs, train_ratio,
    val_ratio, checkpoint_topk, checkpoint_every.

Config keys (under ``mimicgen_datagen``):
    seed_selection_heuristic  Embedded in the run name (e.g. ``behavior_graph`` or
                              ``random``).  Used to label the run; no functional effect
                              here since the generated HDF5 is loaded from
                              :class:`GenerateMimicgenDemosStep`.

Result JSON:
    combined_hdf5_path     Absolute path to the combined HDF5.
    original_hdf5_path     Path to the original (unmodified) dataset.
    generated_hdf5_path    Path to the MimicGen-generated HDF5.
    num_combined_demos     Total demos in the combined dataset.
    num_generated_demos    Number of generated demos appended.
    heuristic              Seed-selection heuristic name.
    train_dirs             List of output directories, one per seed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.diffusion_overrides import baseline_diffusion_extra_overrides
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_name
from policy_doctor.mimicgen.combine_datasets import combine_hdf5_datasets


def _train_combined_worker(
    run_output_dir: str,
    config_dir_str: str,
    config_name: str,
    overrides: list,
) -> None:
    """Hydra-compose combined-data training in an isolated child process."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir_str, version_base=None)
    try:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        OmegaConf.resolve(cfg)
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=run_output_dir)
        workspace.run()
    finally:
        GlobalHydra.instance().clear()


class TrainOnCombinedDataStep(PipelineStep[dict]):
    """Train on original + MimicGen-generated demonstrations.

    Depends on :class:`~policy_doctor.curation_pipeline.steps.generate_mimicgen_demos
    .GenerateMimicgenDemosStep` having been run and produced ``generated_hdf5_path``
    in its result.
    """

    name = "train_on_combined_data"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.generate_mimicgen_demos import (
            GenerateMimicgenDemosStep,
        )

        cfg = self.cfg
        cfg_mg = OmegaConf.select(cfg, "mimicgen_datagen") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        # --- Resolve heuristic name (for labelling) ---
        heuristic_name: str = OmegaConf.select(cfg_mg, "seed_selection_heuristic") or "behavior_graph"

        # --- Load generated HDF5 from prior step ---
        gen_step = GenerateMimicgenDemosStep(cfg, self.run_dir)
        if not gen_step.is_done():
            raise RuntimeError(
                "TrainOnCombinedDataStep requires GenerateMimicgenDemosStep to have run first."
            )
        gen_result = gen_step.load()
        generated_hdf5_path = pathlib.Path(gen_result["generated_hdf5_path"])
        if not generated_hdf5_path.exists():
            raise FileNotFoundError(
                f"Generated HDF5 not found: {generated_hdf5_path}\n"
                f"GenerateMimicgenDemosStep may have failed or produced no successful demos."
            )

        # --- Resolve original dataset path ---
        original_hdf5_path = self._resolve_original_dataset_path(baseline)

        # --- Count generated demos ---
        import h5py
        with h5py.File(generated_hdf5_path, "r") as gen_f:
            num_generated = sum(1 for k in gen_f["data"].keys() if k.startswith("demo_"))
        with h5py.File(original_hdf5_path, "r") as orig_f:
            num_original = sum(1 for k in orig_f["data"].keys() if k.startswith("demo_"))

        print(
            f"  [train_on_combined_data] original demos={num_original}  "
            f"generated demos={num_generated}  heuristic={heuristic_name!r}"
        )

        # --- Combine datasets ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        combined_hdf5_path = self.step_dir / "combined.hdf5"
        num_combined = combine_hdf5_datasets(
            original_path=original_hdf5_path,
            generated_path=generated_hdf5_path,
            output_path=combined_hdf5_path,
        )
        print(
            f"  [train_on_combined_data] combined dataset written: {combined_hdf5_path}  "
            f"total_demos={num_combined}"
        )

        # --- Training config ---
        config_dir = (
            OmegaConf.select(baseline, "config_dir")
            or OmegaConf.select(cfg, "config_dir")
        )
        if not config_dir:
            raise ValueError("baseline.config_dir is required for training")
        config_dir_abs = str((self.repo_root / config_dir).resolve())
        if not pathlib.Path(config_dir_abs).exists():
            raise FileNotFoundError(f"Config dir not found: {config_dir_abs}")

        config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy") or "diffusion_unet_lowdim"
        seeds = expand_seeds(
            OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        num_epochs = int(OmegaConf.select(baseline, "num_epochs") or OmegaConf.select(cfg, "num_epochs") or 1001)
        checkpoint_topk = int(OmegaConf.select(baseline, "checkpoint_topk") or 3)
        checkpoint_every = int(OmegaConf.select(baseline, "checkpoint_every") or 50)
        train_ratio = float(OmegaConf.select(baseline, "train_ratio") or 0.64)
        val_ratio = float(OmegaConf.select(baseline, "val_ratio") or 0.04)
        uniform_quality = OmegaConf.select(baseline, "uniform_quality")
        if uniform_quality is None:
            uniform_quality = True
        output_dir = OmegaConf.select(cfg, "output_dir") or "data/outputs/train"
        project = OmegaConf.select(cfg, "project") or "cupid"
        wandb_tags = OmegaConf.select(cfg, "wandb_tags") or []
        if isinstance(wandb_tags, str):
            wandb_tags = [wandb_tags]
        train_date = (
            OmegaConf.select(cfg, "train_date")
            or OmegaConf.select(baseline, "train_date")
            or "default"
        )
        device = OmegaConf.select(cfg, "device") or "cuda:0"
        run_tag = OmegaConf.select(cfg, "run_tag")

        import multiprocessing as mp
        mp_ctx = mp.get_context("spawn")

        train_dirs: list[str] = []
        procs: list[tuple] = []

        for seed in seeds:
            base_name = get_train_name(train_date, task, policy, seed)
            train_name = f"{base_name}-mimicgen_combined-{heuristic_name}"
            if run_tag:
                train_name = f"{train_name}-{run_tag}"
            run_output_dir = str(self.repo_root / output_dir / train_date / train_name)
            train_dirs.append(run_output_dir)

            overrides = [
                f"training.seed={seed}",
                f"training.num_epochs={num_epochs}",
                f"checkpoint.topk.k={checkpoint_topk}",
                f"training.checkpoint_every={checkpoint_every}",
                f"training.rollout_every={checkpoint_every}",
                f"task.dataset.seed={seed}",
                f"task.dataset.val_ratio={val_ratio}",
                f"training.device={device}",
                f"+task.dataset.dataset_mask_kwargs.train_ratio={train_ratio}",
                f"+task.dataset.dataset_mask_kwargs.uniform_quality={uniform_quality}",
                f"logging.name={train_name}",
                f"logging.group={train_date}_train_{policy}_{task}",
                f"logging.project={project}",
                f"multi_run.wandb_name_base={train_name}",
                f"multi_run.run_dir={run_output_dir}",
                f"++task.dataset.dataset_path={combined_hdf5_path}",
                f"++task.env_runner.dataset_path={combined_hdf5_path}",
            ]
            if wandb_tags:
                tags_str = "[" + ",".join(str(t) for t in wandb_tags) + "]"
                overrides.append(f"logging.tags={tags_str}")
            overrides.extend(baseline_diffusion_extra_overrides(baseline))

            if self.dry_run:
                print(f"[dry_run] TrainOnCombinedDataStep seed={seed}  output_dir={run_output_dir}")
                print(f"[dry_run] overrides={overrides}")
                continue

            pathlib.Path(run_output_dir).mkdir(parents=True, exist_ok=True)
            p = mp_ctx.Process(
                target=_train_combined_worker,
                args=(run_output_dir, config_dir_abs, config_name, overrides),
                daemon=False,
            )
            p.start()
            procs.append((p, seed, run_output_dir))
            print(f"  [launched] pid={p.pid}  output_dir={run_output_dir}")

        for p, seed, run_output_dir in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"[train_on_combined_data] seed={seed} process (pid={p.pid}) "
                    f"exited with code {p.exitcode}"
                )

        return {
            "combined_hdf5_path": str(combined_hdf5_path.resolve()),
            "original_hdf5_path": str(original_hdf5_path),
            "generated_hdf5_path": str(generated_hdf5_path),
            "num_combined_demos": num_combined,
            "num_generated_demos": num_generated,
            "heuristic": heuristic_name,
            "train_dirs": train_dirs,
        }

    def _resolve_original_dataset_path(self, baseline: Any) -> pathlib.Path:
        """Locate the original training HDF5 from config or Hydra defaults.

        Priority:
        1. ``mimicgen_datagen.original_dataset_path`` (explicit override — preferred)
        2. ``task.dataset.dataset_path`` / ``dataset_path`` (pipeline task config)
        3. Hydra training config file (``baseline.config_dir/config_name``)
        """
        from policy_doctor.paths import PROJECT_ROOT

        cfg = self.cfg

        # Priority 1: explicit original dataset path (e.g. 60-demo subset)
        dataset_path = OmegaConf.select(cfg, "mimicgen_datagen.original_dataset_path")

        if not dataset_path:
            dataset_path = (
                OmegaConf.select(cfg, "task.dataset.dataset_path")
                or OmegaConf.select(cfg, "dataset_path")
            )
        if dataset_path is None:
            config_dir = (
                OmegaConf.select(baseline, "config_dir")
                or OmegaConf.select(cfg, "config_dir")
            )
            config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
            if config_dir:
                hydra_config_path = self.repo_root / config_dir / config_name
                if hydra_config_path.exists():
                    with open(hydra_config_path) as f:
                        hydra_cfg = yaml.safe_load(f) or {}
                    dataset_path = (
                        hydra_cfg.get("task", {}).get("dataset", {}).get("dataset_path")
                    )

        if dataset_path is None:
            raise ValueError(
                "Cannot determine original dataset path.  "
                "Set task.dataset.dataset_path in config, or ensure the Hydra "
                "config file has task.dataset.dataset_path defined."
            )

        p = pathlib.Path(dataset_path)
        if not p.is_absolute():
            # Try repo_root first (third_party/cupid), then project root
            candidate = (self.repo_root / p).resolve()
            if not candidate.exists():
                candidate = (PROJECT_ROOT / p).resolve()
            p = candidate
        if not p.exists():
            raise FileNotFoundError(f"Original dataset not found: {p}")
        return p
