"""Compute policy embeddings for rollout episodes — pipeline step."""

from __future__ import annotations

import pathlib
import subprocess

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir


class ComputePolicyEmbeddingsStep(PipelineStep[None]):
    """Compute policy embeddings for each seed's eval episodes.

    Calls ``compute_policy_embeddings.py`` (third_party/cupid) in the
    mimicgen_torch2 env (requires GPU).  Embeddings are saved to
    ``<eval_dir>/policy_embeddings/<layer>.npz``.
    """

    name = "compute_policy_embeddings"

    def compute(self) -> None:
        cfg = self.cfg
        attribution = OmegaConf.select(cfg, "attribution") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        train_date = (
            OmegaConf.select(attribution, "train_date")
            or OmegaConf.select(cfg, "train_date")
        )
        eval_date = (
            OmegaConf.select(attribution, "eval_date")
            or OmegaConf.select(cfg, "eval_date")
            or train_date
        )
        task = (
            OmegaConf.select(attribution, "task")
            or OmegaConf.select(baseline, "task")
            or OmegaConf.select(cfg, "task")
        )
        policy = OmegaConf.select(attribution, "policy") or OmegaConf.select(baseline, "policy")
        seeds = expand_seeds(
            OmegaConf.select(attribution, "seeds")
            or OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        train_ckpt = OmegaConf.select(attribution, "train_ckpt") or "latest"
        eval_as_train_seed = OmegaConf.select(attribution, "eval_as_train_seed")
        if eval_as_train_seed is None:
            eval_as_train_seed = True
        train_output_dir = OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        eval_output_dir = (
            OmegaConf.select(attribution, "eval_output_dir") or "data/outputs/eval_save_episodes"
        )
        layer = OmegaConf.select(cfg, "clustering_policy_emb_layer") or "plan_bottleneck"
        device = OmegaConf.select(attribution, "device") or OmegaConf.select(cfg, "device") or "cuda:0"
        overwrite = bool(OmegaConf.select(attribution, "overwrite") or False)
        conda_env = (
            OmegaConf.select(cfg, "data_source.conda_env_train")
            or OmegaConf.select(baseline, "conda_env")
            or "mimicgen_torch2"
        )

        script = self.repo_root / "compute_policy_embeddings.py"

        for seed in seeds:
            train_dir = str(
                self.repo_root / get_train_dir(train_output_dir, train_date, task, policy, seed)
            )
            _eval_ckpt = OmegaConf.select(attribution, "eval_ckpt") or "latest"
            eval_dir = str(
                self.repo_root
                / get_eval_dir(
                    eval_output_dir,
                    eval_date,
                    task,
                    policy,
                    seed,
                    train_ckpt=_eval_ckpt,
                    eval_as_train_seed=eval_as_train_seed,
                )
            )

            cmd = [
                "conda", "run", "-n", conda_env, "--no-capture-output",
                "python", str(script),
                f"--train_dir={train_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--eval_dir={eval_dir}",
                f"--layer={layer}",
                f"--device={device}",
            ]
            if overwrite:
                cmd.append("--overwrite")

            if self.dry_run:
                print(f"[dry_run] ComputePolicyEmbeddingsStep seed={seed} env={conda_env}")
                print(f"[dry_run]   {' '.join(cmd)}")
                continue

            print(f"  [compute_policy_embeddings] seed={seed} env={conda_env}")
            if not pathlib.Path(train_dir).exists():
                raise FileNotFoundError(f"Train dir not found: {train_dir}")
            if not pathlib.Path(eval_dir).exists():
                raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"[compute_policy_embeddings] subprocess failed (exit={result.returncode}) "
                    f"for seed={seed}"
                )
