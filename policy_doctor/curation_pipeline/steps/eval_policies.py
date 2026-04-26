"""Evaluate trained policies by collecting rollout episodes — pipeline step."""

from __future__ import annotations

import pathlib
import subprocess

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir
from policy_doctor.paths import CUPID_ROOT


def _call_eval_save_episodes(
    output_dir: str,
    train_dir: str,
    train_ckpt: str,
    num_episodes: int,
    test_start_seed: int,
    overwrite: bool,
    device: str,
    conda_env: str | None = None,
) -> None:
    args = [
        f"--output_dir={output_dir}",
        f"--train_dir={train_dir}",
        f"--train_ckpt={train_ckpt}",
        f"--num_episodes={num_episodes}",
        f"--test_start_seed={test_start_seed}",
        f"--overwrite={overwrite}",
        f"--device={device}",
        "--n_test_vis=0",   # disable video recording — offscreen renderer unavailable in headless training env
    ]
    if conda_env:
        # Dispatch as a subprocess in the specified conda env so that env-specific
        # packages (e.g. pytorch3d in mimicgen_torch2) are available.
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", str(CUPID_ROOT / "eval_save_episodes.py"),
            *args,
        ]
        result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"[eval_policies] subprocess (conda_env={conda_env}) failed with exit code {result.returncode}"
            )
    else:
        from eval_save_episodes import main  # noqa: PLC0415
        main(args, standalone_mode=False)


class EvalPoliciesStep(PipelineStep[None]):
    """Run rollout evaluation for each seed."""

    name = "eval_policies"

    def compute(self) -> None:
        cfg = self.cfg
        evaluation = OmegaConf.select(cfg, "evaluation") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        train_date = (
            OmegaConf.select(evaluation, "train_date")
            or OmegaConf.select(cfg, "train_date")
            or OmegaConf.select(baseline, "train_date")
        )
        eval_date = OmegaConf.select(evaluation, "eval_date") or OmegaConf.select(cfg, "eval_date") or train_date
        task = OmegaConf.select(evaluation, "task") or OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(evaluation, "policy") or OmegaConf.select(baseline, "policy")
        seeds = expand_seeds(
            OmegaConf.select(evaluation, "seeds")
            or OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        train_ckpt = OmegaConf.select(evaluation, "train_ckpt") or "latest"
        num_episodes = OmegaConf.select(evaluation, "num_episodes") or 100
        test_start_seed = OmegaConf.select(evaluation, "test_start_seed") or 100000
        overwrite = OmegaConf.select(evaluation, "overwrite") or False
        device = OmegaConf.select(evaluation, "device") or OmegaConf.select(cfg, "device") or "cuda:0"
        eval_output_dir = OmegaConf.select(evaluation, "eval_output_dir") or "data/outputs/eval_save_episodes"
        train_output_dir = OmegaConf.select(evaluation, "train_output_dir") or "data/outputs/train"
        eval_as_train_seed = OmegaConf.select(evaluation, "eval_as_train_seed")
        if eval_as_train_seed is None:
            eval_as_train_seed = True
        conda_env = (
            OmegaConf.select(evaluation, "conda_env")
            or OmegaConf.select(cfg, "data_source.conda_env_train")
        )

        for seed in seeds:
            train_dir = str(self.repo_root / get_train_dir(train_output_dir, train_date, task, policy, seed))
            if not pathlib.Path(train_dir).exists():
                raise FileNotFoundError(f"Train dir not found: {train_dir}")
            output_dir = str(self.repo_root / get_eval_dir(
                eval_output_dir, eval_date, task, policy, seed,
                train_ckpt=train_ckpt, eval_as_train_seed=eval_as_train_seed,
            ))

            if self.dry_run:
                print(f"[dry_run] EvalPoliciesStep seed={seed}  output_dir={output_dir}")
                continue

            print(f"  [eval_policies] seed={seed}  conda_env={conda_env}  output_dir={output_dir}")
            _call_eval_save_episodes(
                output_dir=output_dir,
                train_dir=train_dir,
                train_ckpt=train_ckpt,
                num_episodes=num_episodes,
                test_start_seed=test_start_seed,
                overwrite=overwrite,
                device=device,
                conda_env=conda_env,
            )
