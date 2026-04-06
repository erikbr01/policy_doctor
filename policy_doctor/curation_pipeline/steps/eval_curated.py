"""Evaluate curated (retrained) policy — pipeline step."""

from __future__ import annotations

import pathlib

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.steps.eval_policies import _call_eval_save_episodes
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir, get_train_name


class EvalCuratedStep(PipelineStep[None]):
    """Evaluate a curated retrained policy by collecting rollout episodes."""

    name = "eval_curated"

    def compute(self) -> None:
        cfg = self.cfg
        baseline = OmegaConf.select(cfg, "baseline") or {}
        curation_mode = OmegaConf.select(cfg, "curation_mode") or "curation_selection"
        if curation_mode == "filter":
            curation_mode = "curation_filtering"
        elif curation_mode == "selection":
            curation_mode = "curation_selection"

        curated = OmegaConf.select(cfg, curation_mode) or {}

        train_date = (
            OmegaConf.select(cfg, "train_date")
            or OmegaConf.select(baseline, "train_date")
            or "default"
        )
        eval_date = OmegaConf.select(cfg, "eval_date") or train_date
        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy") or "diffusion_unet_lowdim"
        seeds = expand_seeds(
            OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        train_ckpt = OmegaConf.select(cfg, "train_ckpt") or OmegaConf.select(curated, "train_ckpt") or "latest"
        num_episodes = OmegaConf.select(cfg, "num_episodes") or OmegaConf.select(curated, "num_episodes") or 100
        eval_output_dir = OmegaConf.select(cfg, "eval_output_dir") or "data/outputs/eval_save_episodes"
        train_output_dir = OmegaConf.select(cfg, "train_output_dir") or "data/outputs/train"
        eval_as_train_seed = OmegaConf.select(cfg, "eval_as_train_seed")
        if eval_as_train_seed is None:
            eval_as_train_seed = True
        test_start_seed = OmegaConf.select(cfg, "test_start_seed") or 100000
        overwrite = OmegaConf.select(cfg, "overwrite") or False
        device = OmegaConf.select(cfg, "device") or "cuda:0"

        curation_method = OmegaConf.select(cfg, "curation_method") or "influence_sum_official"
        filter_ratio_val = OmegaConf.select(cfg, "filter_ratio")
        filter_ratio = filter_ratio_val if filter_ratio_val is not None else 0.0
        select_ratio_val = OmegaConf.select(cfg, "select_ratio")
        select_ratio = select_ratio_val if select_ratio_val is not None else 0.10
        run_tag = OmegaConf.select(cfg, "run_tag")

        for seed in seeds:
            base_name = get_train_name(train_date, task, policy, seed)
            curated_name = f"{base_name}-curation_{curation_method}-filter_{filter_ratio}-select_{select_ratio}"
            if run_tag:
                curated_name = f"{curated_name}-{run_tag}"

            train_dir = str(self.repo_root / train_output_dir / train_date / curated_name)
            output_dir = str(self.repo_root / eval_output_dir / eval_date / curated_name / train_ckpt)

            if self.dry_run:
                print(f"[dry_run] EvalCuratedStep seed={seed}  train_dir={train_dir}")
                print(f"[dry_run] EvalCuratedStep seed={seed}  output_dir={output_dir}")
                continue

            if not pathlib.Path(train_dir).exists():
                raise FileNotFoundError(f"Curated train dir not found: {train_dir}")

            print(f"  [eval_curated] seed={seed}  output_dir={output_dir}")
            _call_eval_save_episodes(
                output_dir=output_dir,
                train_dir=train_dir,
                train_ckpt=train_ckpt,
                num_episodes=num_episodes,
                test_start_seed=test_start_seed,
                overwrite=overwrite,
                device=device,
            )
