"""Train baseline policy — pipeline step."""

from __future__ import annotations

import pathlib

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.diffusion_overrides import baseline_diffusion_extra_overrides
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_name


def _train_baseline_worker(
    run_output_dir: str,
    config_dir_str: str,
    config_name: str,
    overrides: list,
) -> None:
    """Hydra-compose training in an isolated child process.

    Must run in a separate process because ``hydra.initialize_config_dir``
    manages a global singleton that cannot be re-initialized in-process.
    """
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


class TrainBaselineStep(PipelineStep[None]):
    """Train the baseline policy for each seed using Hydra compose."""

    name = "train_baseline"

    def compute(self) -> None:
        cfg = self.cfg
        baseline = OmegaConf.select(cfg, "baseline") or {}

        config_dir = OmegaConf.select(baseline, "config_dir") or OmegaConf.select(cfg, "config_dir")
        if not config_dir:
            raise ValueError("baseline.config_dir is required")
        config_dir_abs = str((self.repo_root / config_dir).resolve())
        if not pathlib.Path(config_dir_abs).exists():
            raise FileNotFoundError(f"Config dir not found: {config_dir_abs}")

        config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy")
        # Prefer top-level cfg.seeds (CLI / experiment) over baseline.seeds from task YAML.
        seeds_raw = OmegaConf.select(cfg, "seeds")
        if seeds_raw is None:
            seeds_raw = OmegaConf.select(baseline, "seeds")
        seeds = expand_seeds(seeds_raw or [0])
        num_epochs = OmegaConf.select(baseline, "num_epochs") or 1001
        checkpoint_topk = OmegaConf.select(baseline, "checkpoint_topk") or 3
        checkpoint_every = OmegaConf.select(baseline, "checkpoint_every") or 50
        train_ratio = OmegaConf.select(baseline, "train_ratio") or 0.64
        val_ratio = OmegaConf.select(baseline, "val_ratio") or 0.04
        uniform_quality = OmegaConf.select(baseline, "uniform_quality")
        if uniform_quality is None:
            uniform_quality = True
        output_dir = OmegaConf.select(baseline, "output_dir") or "data/outputs/train"
        # Prefer top-level cfg.project (experiment / CLI) over baseline.project from task YAML.
        project = OmegaConf.select(cfg, "project") or OmegaConf.select(baseline, "project") or "influence-clustering"
        train_date = OmegaConf.select(baseline, "train_date") or OmegaConf.select(cfg, "train_date") or "default"
        script_name = OmegaConf.select(baseline, "script_name") or "train"
        exp_name = OmegaConf.select(baseline, "exp_name") or f"{script_name}_{policy}"
        device = OmegaConf.select(cfg, "device") or "cuda:0"

        for seed in seeds:
            train_name = get_train_name(train_date, task, policy, seed)
            run_output_dir = str(self.repo_root / output_dir / train_date / train_name)

            overrides = [
                f"name={exp_name}",
                f"training.device={device}",
                f"training.seed={seed}",
                f"training.num_epochs={num_epochs}",
                f"checkpoint.topk.k={checkpoint_topk}",
                f"training.checkpoint_every={checkpoint_every}",
                f"training.rollout_every={checkpoint_every}",
                f"task.dataset.seed={seed}",
                f"task.dataset.val_ratio={val_ratio}",
                f"+task.dataset.dataset_mask_kwargs.train_ratio={train_ratio}",
                f"+task.dataset.dataset_mask_kwargs.uniform_quality={uniform_quality}",
                f"logging.name={train_name}",
                f"logging.group={train_date}_{exp_name}_{task}",
                f"logging.project={project}",
                f"multi_run.wandb_name_base={train_name}",
                f"multi_run.run_dir={run_output_dir}",
            ]
            overrides.extend(baseline_diffusion_extra_overrides(baseline))

            if self.dry_run:
                print(f"[dry_run] TrainBaselineStep seed={seed}")
                print(f"[dry_run]   config_dir={config_dir_abs}  config_name={config_name}")
                print(f"[dry_run]   output_dir={run_output_dir}")
                print(f"[dry_run]   overrides={overrides}")
                continue

            print(f"  [train_baseline] seed={seed}  output_dir={run_output_dir}")
            self._run_in_process(
                _train_baseline_worker,
                {
                    "run_output_dir": run_output_dir,
                    "config_dir_str": config_dir_abs,
                    "config_name": config_name,
                    "overrides": overrides,
                },
            )
