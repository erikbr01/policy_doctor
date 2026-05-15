"""Finalize multi-checkpoint TRAK attribution — pipeline step."""

from __future__ import annotations

import subprocess

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir
from policy_doctor.paths import CUPID_ROOT


def _call_finalize_trak(cmd_args: list, conda_env: str | None = None) -> None:
    if conda_env:
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", str(CUPID_ROOT / "finalize_trak.py"),
            *cmd_args,
        ]
        result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"[finalize_attribution] subprocess (conda_env={conda_env}) failed with exit code {result.returncode}"
            )
    else:
        from finalize_trak import main  # noqa: PLC0415
        main(cmd_args, standalone_mode=False)


class FinalizeAttributionStep(PipelineStep[None]):
    """Ensemble TRAK scores across multiple checkpoints (skipped when num_ckpts=1)."""

    name = "finalize_attribution"

    def compute(self) -> None:
        cfg = self.cfg
        attribution = OmegaConf.select(cfg, "attribution") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        num_ckpts = OmegaConf.select(attribution, "num_ckpts") or 1
        if num_ckpts <= 1:
            print(f"  [{self.name}] skipped (num_ckpts={num_ckpts} ≤ 1)")
            return

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

        conda_env = (
            OmegaConf.select(attribution, "conda_env")
            or OmegaConf.select(cfg, "data_source.conda_env_train")
        )
        result_date = OmegaConf.select(attribution, "result_date") or "default"
        exp_seed = OmegaConf.select(attribution, "seed") or 0
        proj_dim = OmegaConf.select(attribution, "proj_dim") or 4000
        lambda_reg = OmegaConf.select(attribution, "lambda_reg") or 0.0
        loss_fn = OmegaConf.select(attribution, "loss_fn") or "square"
        num_timesteps = OmegaConf.select(attribution, "num_timesteps") or 64
        exp_name = (
            f"{result_date}_trak_results"
            f"-proj_dim={proj_dim}"
            f"-lambda_reg={lambda_reg}"
            f"-num_ckpts={num_ckpts}"
            f"-seed={exp_seed}"
            f"-loss_fn={loss_fn}"
            f"-num_timesteps={num_timesteps}"
        )

        for seed in seeds:
            train_dir = str(self.repo_root / get_train_dir(train_output_dir, train_date, task, policy, seed))
            eval_dir = str(self.repo_root / get_eval_dir(
                eval_output_dir, eval_date, task, policy, seed,
                train_ckpt=eval_ckpt, eval_as_train_seed=eval_as_train_seed,
            ))

            cmd_args = [
                f"--num_ckpts={num_ckpts}",
                f"--exp_name={exp_name}",
                f"--eval_dir={eval_dir}",
                f"--train_dir={train_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--model_keys={OmegaConf.select(attribution, 'model_keys') or 'model.'}",
                f"--modelout_fn={OmegaConf.select(attribution, 'modelout_fn')}",
                f"--gradient_co={OmegaConf.select(attribution, 'gradient_co')}",
                f"--proj_dim={proj_dim}",
                f"--proj_max_batch_size={OmegaConf.select(attribution, 'proj_max_batch_size') or 32}",
                f"--lambda_reg={lambda_reg}",
                f"--use_half_precision={OmegaConf.select(attribution, 'use_half_precision') or False}",
                f"--batch_size={OmegaConf.select(attribution, 'batch_size') or 32}",
                f"--device={OmegaConf.select(attribution, 'device') or OmegaConf.select(cfg, 'device') or 'cuda:0'}",
                f"--seed={exp_seed}",
                f"--featurize_holdout={OmegaConf.select(attribution, 'featurize_holdout') or False}",
            ]

            if self.dry_run:
                print(f"[dry_run] FinalizeAttributionStep seed={seed}")
                print(f"[dry_run]   {' '.join(cmd_args)}")
                continue

            print(f"  [finalize_attribution] seed={seed}  conda_env={conda_env}")
            _call_finalize_trak(cmd_args=cmd_args, conda_env=conda_env)
