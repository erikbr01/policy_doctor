"""Compute InfEmbed embeddings — pipeline step."""

from __future__ import annotations

import pathlib

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir


def _call_compute_infembed_embeddings(cmd_args: list) -> None:
    from compute_infembed_embeddings import main  # noqa: PLC0415

    main(cmd_args, standalone_mode=False)


class ComputeInfembedStep(PipelineStep[None]):
    """Compute InfEmbed embeddings for each seed."""

    name = "compute_infembed"

    def compute(self) -> None:
        cfg = self.cfg
        attribution = OmegaConf.select(cfg, "attribution") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        train_date = OmegaConf.select(attribution, "train_date") or OmegaConf.select(cfg, "train_date")
        eval_date = OmegaConf.select(attribution, "eval_date") or OmegaConf.select(cfg, "eval_date") or train_date
        task = OmegaConf.select(attribution, "task") or OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(attribution, "policy") or OmegaConf.select(baseline, "policy")
        seeds = expand_seeds(
            OmegaConf.select(attribution, "seeds")
            or OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        train_ckpt = OmegaConf.select(attribution, "train_ckpt") or "latest"
        eval_ckpt = OmegaConf.select(attribution, "eval_ckpt") or "latest"
        eval_as_train_seed = OmegaConf.select(attribution, "eval_as_train_seed")
        if eval_as_train_seed is None:
            eval_as_train_seed = True
        train_output_dir = OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        eval_output_dir = OmegaConf.select(attribution, "eval_output_dir") or "data/outputs/eval_save_episodes"

        modelout_fn = OmegaConf.select(attribution, "modelout_fn") or "DiffusionLowdimFunctionalModelOutput"
        loss_fn = OmegaConf.select(attribution, "loss_fn") or "square"
        num_timesteps = OmegaConf.select(attribution, "num_timesteps") or 64
        batch_size = OmegaConf.select(attribution, "infembed_batch_size") or OmegaConf.select(attribution, "batch_size") or 32
        device = OmegaConf.select(attribution, "device") or OmegaConf.select(cfg, "device") or "cuda:0"
        exp_seed = OmegaConf.select(attribution, "seed") or 0
        featurize_holdout = OmegaConf.select(attribution, "featurize_holdout")
        if featurize_holdout is None:
            featurize_holdout = True
        projection_dim = OmegaConf.select(attribution, "projection_dim") or 100
        arnoldi_dim = OmegaConf.select(attribution, "arnoldi_dim") or 200
        overwrite = OmegaConf.select(attribution, "overwrite") or False
        model_keys = OmegaConf.select(attribution, "model_keys") or "model."

        for seed in seeds:
            train_dir = str(self.repo_root / get_train_dir(train_output_dir, train_date, task, policy, seed))
            eval_dir = str(self.repo_root / get_eval_dir(
                eval_output_dir, eval_date, task, policy, seed,
                train_ckpt=eval_ckpt, eval_as_train_seed=eval_as_train_seed,
            ))

            if not pathlib.Path(train_dir).exists():
                raise FileNotFoundError(f"Train dir not found: {train_dir}")
            if not pathlib.Path(eval_dir).exists():
                raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

            cmd_args = [
                "--exp_name=auto",
                f"--eval_dir={eval_dir}",
                f"--train_dir={train_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--modelout_fn={modelout_fn}",
                f"--loss_fn={loss_fn}",
                f"--num_timesteps={num_timesteps}",
                f"--batch_size={batch_size}",
                f"--device={device}",
                f"--seed={exp_seed}",
                f"--projection_dim={projection_dim}",
                f"--arnoldi_dim={arnoldi_dim}",
            ]
            if model_keys:
                cmd_args.append(f"--model_keys={model_keys}")
            if featurize_holdout:
                cmd_args.append("--featurize_holdout")
            if overwrite:
                cmd_args.append("--overwrite")
            # Optional: override dataset path for MimicGen / RoboCasa when the checkpoint's
            # stored path is stale (machine migration, renamed directory, fresh generation run).
            attribution_dataset_path = OmegaConf.select(attribution, "dataset_path")
            if attribution_dataset_path:
                cmd_args.append(f"--dataset_path={attribution_dataset_path}")

            if self.dry_run:
                print(f"[dry_run] ComputeInfembedStep seed={seed}")
                print(f"[dry_run]   {' '.join(cmd_args)}")
                continue

            print(f"  [compute_infembed] seed={seed}")
            _call_compute_infembed_embeddings(cmd_args=cmd_args)
