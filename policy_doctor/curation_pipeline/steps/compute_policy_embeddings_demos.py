"""Compute per-timestep policy embeddings for training demos — pipeline step.

Sibling of ``compute_infembed``: extracts diffusion U-Net activations from
the training demonstrations and persists them to
``<train_dir>/policy_embeddings_demos/<layer>.npz``.  The ``compute_data_support``
step consumes these alongside the rollout-side embeddings to build a shared
demo+rollout UMAP space.

The step runs in ``mimicgen_torch2`` env (same as ``compute_infembed``) so it
has access to the diffusion_policy training datasets + obs preprocessing.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Dict, List

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_dir
from policy_doctor.paths import CUPID_ROOT


def _call_compute_policy_embeddings_demos(cmd_args: list, conda_env: str | None) -> None:
    if conda_env:
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", str(CUPID_ROOT / "compute_policy_embeddings_demos.py"),
            *cmd_args,
        ]
        result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"[compute_policy_embeddings_demos] subprocess (conda_env={conda_env}) "
                f"failed with exit code {result.returncode}"
            )
    else:
        # In-process call: the script lives in third_party/cupid which is on
        # sys.path when CUPID_ROOT is the working tree.
        import sys
        sys.path.insert(0, str(CUPID_ROOT))
        from compute_policy_embeddings_demos import main  # noqa: PLC0415
        old_argv = sys.argv
        try:
            sys.argv = ["compute_policy_embeddings_demos.py", *cmd_args]
            main()
        finally:
            sys.argv = old_argv


class ComputePolicyEmbeddingsDemosStep(PipelineStep[Dict[str, List[str]]]):
    """Extract policy embeddings for each seed's training demos.

    Result: ``{"demo_embeddings_paths": {seed: path, ...}}``
    """

    name = "compute_policy_embeddings_demos"

    def compute(self) -> Dict[str, Dict[str, str]]:
        cfg = self.cfg

        attribution = OmegaConf.select(cfg, "attribution") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        train_date = (
            OmegaConf.select(attribution, "train_date")
            or OmegaConf.select(cfg, "train_date")
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
        train_ckpt = OmegaConf.select(attribution, "train_ckpt") or "best"
        train_output_dir = (
            OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        )
        device = (
            OmegaConf.select(cfg, "policy_emb_demos_device")
            or OmegaConf.select(attribution, "device")
            or OmegaConf.select(cfg, "device")
            or "cuda:0"
        )
        batch_size = int(OmegaConf.select(cfg, "policy_emb_demos_batch_size") or 128)
        layer = (
            OmegaConf.select(cfg, "clustering_policy_emb_layer")
            or "bottleneck_plan_t0"
        )
        include_holdout = bool(
            OmegaConf.select(cfg, "policy_emb_demos_include_holdout") or False
        )
        overwrite = bool(OmegaConf.select(cfg, "policy_emb_demos_overwrite") or False)
        dataset_path_override = OmegaConf.select(attribution, "dataset_path")
        conda_env = (
            OmegaConf.select(cfg, "policy_emb_demos_conda_env")
            or OmegaConf.select(attribution, "conda_env")
            or OmegaConf.select(cfg, "data_source.conda_env_train")
        )

        result: Dict[str, Dict[str, str]] = {"demo_embeddings_paths": {}}

        for seed in seeds:
            train_dir = self.repo_root / get_train_dir(
                train_output_dir, train_date, task, policy, seed
            )
            if not pathlib.Path(train_dir).exists():
                raise FileNotFoundError(f"Train dir not found: {train_dir}")

            out_path = pathlib.Path(train_dir) / "policy_embeddings_demos" / f"{layer}.npz"

            cmd_args = [
                f"--train_dir={train_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--layer={layer}",
                f"--batch_size={batch_size}",
                f"--device={device}",
            ]
            if include_holdout:
                cmd_args.append("--include_holdout")
            if overwrite:
                cmd_args.append("--overwrite")
            if dataset_path_override:
                cmd_args.append(f"--dataset_path={dataset_path_override}")

            if self.dry_run:
                print(f"[dry_run] ComputePolicyEmbeddingsDemosStep seed={seed}")
                print(f"[dry_run]   {' '.join(cmd_args)}")
                result["demo_embeddings_paths"][str(seed)] = str(out_path)
                continue

            print(f"  [compute_policy_embeddings_demos] seed={seed}  conda_env={conda_env}")
            _call_compute_policy_embeddings_demos(cmd_args=cmd_args, conda_env=conda_env)
            result["demo_embeddings_paths"][str(seed)] = str(out_path)

        return result
