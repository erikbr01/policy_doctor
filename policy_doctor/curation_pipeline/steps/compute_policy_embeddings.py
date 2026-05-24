"""Compute policy embeddings — pipeline step.

Wraps ``third_party/cupid/compute_policy_embeddings.py`` (mimicgen_torch2 env).
Saves per-rollout-timestep embeddings to ``<eval_dir>/policy_embeddings/<layer>.npz``
under key ``rollout_embeddings``.

Config keys (under ``policy_emb`` block on the top-level cfg):
    layer            Default ``"bottleneck_plan_t0"``. Format
                     ``{bottleneck|decoder|encoder}_{plan|plan8|exec}_t{0..N}``.
    n_noise_levels   Override the default (policy.num_inference_steps).
                     Ignored when ``layer`` contains an explicit ``_t{N}``.
    batch_size       Default 128 (rollout timesteps per forward pass).
    device           Default falls back to cfg.device or ``cuda:0``.
    overwrite        Default false; skip if ``<layer>.npz`` exists.

Train/eval dirs are resolved the same way as ``compute_infembed`` — from
``attribution.train_date``/``baseline``/``cfg`` with per-seed substitution.
"""

from __future__ import annotations

import pathlib
import subprocess

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir
from policy_doctor.paths import CUPID_ROOT


class ComputePolicyEmbeddingsStep(PipelineStep[None]):
    """Compute policy embeddings for each seed's saved rollout episodes."""

    name = "compute_policy_embeddings"

    def compute(self) -> None:
        cfg = self.cfg
        attribution = OmegaConf.select(cfg, "attribution") or OmegaConf.create({})
        baseline = OmegaConf.select(cfg, "baseline") or OmegaConf.create({})
        policy_emb = OmegaConf.select(cfg, "policy_emb") or OmegaConf.create({})

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
        policy = (
            OmegaConf.select(attribution, "policy")
            or OmegaConf.select(baseline, "policy")
        )
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
        train_output_dir = (
            OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        )
        eval_output_dir = (
            OmegaConf.select(attribution, "eval_output_dir")
            or "data/outputs/eval_save_episodes"
        )

        layer: str = OmegaConf.select(policy_emb, "layer") or "bottleneck_plan_t0"
        n_noise_levels = OmegaConf.select(policy_emb, "n_noise_levels")
        batch_size = int(OmegaConf.select(policy_emb, "batch_size") or 128)
        device = (
            OmegaConf.select(policy_emb, "device")
            or OmegaConf.select(cfg, "device")
            or "cuda:0"
        )
        overwrite = bool(OmegaConf.select(policy_emb, "overwrite") or False)
        conda_env = (
            OmegaConf.select(cfg, "data_source.conda_env_train")
            or "mimicgen_torch2"
        )

        for seed in seeds:
            train_dir = str(
                self.repo_root
                / get_train_dir(train_output_dir, train_date, task, policy, seed)
            )
            eval_dir = str(
                self.repo_root
                / get_eval_dir(
                    eval_output_dir, eval_date, task, policy, seed,
                    train_ckpt=eval_ckpt, eval_as_train_seed=eval_as_train_seed,
                )
            )

            if not pathlib.Path(train_dir).exists():
                raise FileNotFoundError(f"Train dir not found: {train_dir}")
            if not pathlib.Path(eval_dir).exists():
                raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

            out_path = pathlib.Path(eval_dir) / "policy_embeddings" / f"{layer}.npz"
            if out_path.exists() and not overwrite:
                print(
                    f"  [compute_policy_embeddings] seed={seed} layer={layer} "
                    f"[cached] {out_path}"
                )
                continue

            cmd = [
                "conda", "run", "-n", conda_env, "--no-capture-output",
                "python", str(CUPID_ROOT / "compute_policy_embeddings.py"),
                f"--train_dir={train_dir}",
                f"--eval_dir={eval_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--layer={layer}",
                f"--batch_size={batch_size}",
                f"--device={device}",
            ]
            if n_noise_levels is not None:
                cmd.append(f"--n_noise_levels={int(n_noise_levels)}")

            if self.dry_run:
                print(f"[dry_run] ComputePolicyEmbeddingsStep seed={seed}  layer={layer}")
                print(f"[dry_run]   {' '.join(cmd)}")
                continue

            print(f"  [compute_policy_embeddings] seed={seed}  layer={layer}")
            result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
            if result.returncode != 0:
                raise RuntimeError(
                    f"[compute_policy_embeddings] subprocess failed "
                    f"(exit={result.returncode}) for seed={seed}"
                )
