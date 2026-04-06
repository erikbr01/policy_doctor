"""Train curated policy using Hydra compose — pipeline step class."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_name


def _train_curated_worker(
    run_output_dir: str,
    config_dir_str: str,
    config_name: str,
    overrides: list,
) -> None:
    """Hydra-compose curated training in an isolated child process."""
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


class TrainCuratedStep(PipelineStep[None]):
    """Train with a curation dataset using Hydra compose (no subprocess).

    Dependency resolution: if ``cfg.curation_config_path`` is not set, the
    step loads saved paths from :class:`RunCurationConfigStep`.
    """

    name = "train_curated"

    def compute(self) -> None:
        cfg = self.cfg
        baseline = OmegaConf.select(cfg, "baseline") or {}
        curation_mode = OmegaConf.select(cfg, "curation_mode") or "curation_selection"
        # Normalise shorthand aliases
        if curation_mode == "filter":
            curation_mode = "curation_filtering"
        elif curation_mode == "selection":
            curation_mode = "curation_selection"

        curated = (
            OmegaConf.select(cfg, f"curation_filtering" if curation_mode == "curation_filtering" else "curation_selection")
            or {}
        )

        config_dir = (
            OmegaConf.select(baseline, "config_dir")
            or OmegaConf.select(curated, "config_dir")
            or OmegaConf.select(cfg, "config_dir")
        )
        if not config_dir:
            raise ValueError("config_dir required (from baseline or curation config)")
        config_dir_abs = str((self.repo_root / config_dir).resolve())
        if not pathlib.Path(config_dir_abs).exists():
            raise FileNotFoundError(f"Config dir not found: {config_dir_abs}")

        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy") or "diffusion_unet_lowdim"
        config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
        seeds = expand_seeds(
            OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        num_epochs = OmegaConf.select(cfg, "num_epochs") or OmegaConf.select(curated, "num_epochs") or 2501
        checkpoint_topk = OmegaConf.select(curated, "checkpoint_topk") or 1
        checkpoint_every = OmegaConf.select(curated, "checkpoint_every") or 50
        train_ratio = OmegaConf.select(baseline, "train_ratio") or 0.16
        val_ratio = OmegaConf.select(baseline, "val_ratio") or 0.04
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

        curation_config_path = OmegaConf.select(cfg, "curation_config_path")
        if not curation_config_path:
            from policy_doctor.curation_pipeline.steps.run_curation_config import RunCurationConfigStep

            prior = RunCurationConfigStep(cfg, self.run_dir).load()
            if prior:
                curation_config_path = prior[0]
        if not curation_config_path:
            raise ValueError("curation_config_path required for curated training")

        curation_config_abs = pathlib.Path(curation_config_path)
        if not curation_config_abs.is_absolute():
            curation_config_abs = (self.repo_root / curation_config_abs).resolve()
        if not curation_config_abs.exists():
            raise FileNotFoundError(f"Curation config not found: {curation_config_abs}")

        curation_config_dir = str(curation_config_abs.parent)
        stem = curation_config_abs.stem
        curation_config_base = stem.rsplit("_seed", 1)[0] if "_seed" in stem else stem

        curation_method = OmegaConf.select(cfg, "curation_method") or "influence_sum_official"
        filter_ratio = OmegaConf.select(cfg, "filter_ratio") or 0.0
        select_ratio = OmegaConf.select(cfg, "select_ratio") or 0.10

        dataset_path_override = (
            OmegaConf.select(cfg, "task.dataset.dataset_path")
            or OmegaConf.select(cfg, "dataset_path")
        )
        if dataset_path_override is None:
            hydra_config_path = self.repo_root / config_dir / config_name
            if hydra_config_path.exists():
                with open(hydra_config_path) as f:
                    hydra_cfg = yaml.safe_load(f) or {}
                default_path = hydra_cfg.get("task", {}).get("dataset", {}).get("dataset_path")
                if isinstance(default_path, str):
                    dataset_path_override = default_path
        if dataset_path_override is not None:
            p = pathlib.Path(dataset_path_override)
            if not p.is_absolute():
                p = (self.repo_root / p).resolve()
            dataset_path_override = str(p)

        import multiprocessing as mp

        # Use 'spawn' to avoid fork-after-multithreaded-parent deadlocks (CUDA, logging, WandB).
        mp_ctx = mp.get_context("spawn")

        procs = []
        for seed in seeds:
            train_name = get_train_name(train_date, task, policy, seed)
            train_name = f"{train_name}-curation_{curation_method}-filter_{filter_ratio}-select_{select_ratio}"
            if run_tag:
                train_name = f"{train_name}-{run_tag}"
            run_output_dir = str(self.repo_root / output_dir / train_date / train_name)

            sample_curation_path = curation_config_abs.parent / f"{curation_config_base}_seed{seed}.yaml"
            if not sample_curation_path.exists():
                sample_curation_path = curation_config_abs
            sample_curation_str = str(sample_curation_path.resolve())

            print(f"    filter config: {sample_curation_str}")

            overrides = [
                f"training.seed={seed}",
                f"training.num_epochs={num_epochs}",
                f"checkpoint.topk.k={checkpoint_topk}",
                f"training.checkpoint_every={checkpoint_every}",
                f"training.rollout_every={checkpoint_every}",
                f"task.dataset.seed={seed}",
                f"task.dataset.val_ratio={val_ratio}",
                f"training.device={device}",
                f"+task.dataset.holdout_selection_config={sample_curation_str}",
                f"+task.dataset.dataset_mask_kwargs.train_ratio={train_ratio}",
                f"+task.dataset.dataset_mask_kwargs.uniform_quality={uniform_quality}",
                f"+task.dataset.dataset_mask_kwargs.curate_dataset=false",
                f"+task.dataset.dataset_mask_kwargs.curation_config_dir={curation_config_dir}",
                f"+task.dataset.dataset_mask_kwargs.curation_method={curation_method}",
                f"+task.dataset.dataset_mask_kwargs.filter_ratio={filter_ratio}",
                f"+task.dataset.dataset_mask_kwargs.select_ratio={select_ratio}",
                f"logging.name={train_name}",
                f"logging.group={train_date}_train_{policy}_{task}",
                f"logging.project={project}",
                f"multi_run.wandb_name_base={train_name}",
                f"multi_run.run_dir={run_output_dir}",
            ]
            if wandb_tags:
                tags_str = "[" + ",".join(str(t) for t in wandb_tags) + "]"
                overrides.append(f"logging.tags={tags_str}")
            if dataset_path_override:
                overrides.append(f"++task.dataset.dataset_path={dataset_path_override}")
                overrides.append(f"++task.env_runner.dataset_path={dataset_path_override}")

            if self.dry_run:
                print(f"[dry_run] TrainCuratedStep seed={seed}  output_dir={run_output_dir}")
                print(f"[dry_run] overrides={overrides}")
                continue

            pathlib.Path(run_output_dir).mkdir(parents=True, exist_ok=True)
            p = mp_ctx.Process(
                target=_train_curated_worker,
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
                    f"[train_curated] seed={seed} process (pid={p.pid}) exited with code {p.exitcode}"
                )


