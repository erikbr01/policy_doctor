"""Evaluate all top-k checkpoints from TrainBaselineStep — pipeline step.

Mirrors :class:`~policy_doctor.curation_pipeline.steps.eval_mimicgen_combined
.EvalMimicgenCombinedStep` but loads training directories from ``TrainBaselineStep``
config (which does not write a ``result.json``).

Config keys (under ``evaluation``):
    num_episodes      Episodes per checkpoint (default 500).
    test_start_seed   RNG seed for the rollout environment (default 100000).
    overwrite         Re-run even if eval output already exists (default false).
    device            CUDA device string (default ``cuda:0``).
    eval_output_dir   Base output dir (default ``data/outputs/eval_save_episodes``).

Result JSON:
    train_dirs             List of training directories evaluated.
    checkpoints            List of {checkpoint, num_episodes, num_success, success_rate}.
    mean_success_rate      Mean success rate across all evaluated checkpoints.
    best_success_rate      Best success rate across all evaluated checkpoints.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_name
from policy_doctor.paths import CUPID_ROOT


class EvalBaselineStep(PipelineStep[dict]):
    """Evaluate every top-k checkpoint produced by TrainBaselineStep."""

    name = "eval_baseline"

    def compute(self) -> dict[str, Any]:
        cfg = self.cfg
        evaluation = OmegaConf.select(cfg, "evaluation") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        num_episodes: int = int(OmegaConf.select(evaluation, "num_episodes") or 500)
        test_start_seed: int = int(OmegaConf.select(evaluation, "test_start_seed") or 100000)
        overwrite: bool = bool(OmegaConf.select(evaluation, "overwrite") or False)
        device: str = OmegaConf.select(cfg, "device") or "cuda:0"
        eval_output_dir: str = (
            OmegaConf.select(evaluation, "eval_output_dir")
            or "data/outputs/eval_save_episodes"
        )
        conda_env: str = (
            OmegaConf.select(baseline, "conda_env")
            or OmegaConf.select(cfg, "data_source.conda_env_train")
            or "mimicgen_torch2"
        )

        # --- Derive train dirs from config (same logic as TrainBaselineStep) ---
        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy") or "diffusion_unet_lowdim"
        seeds = expand_seeds(
            OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        train_date = (
            OmegaConf.select(cfg, "train_date")
            or OmegaConf.select(baseline, "train_date")
            or "default"
        )
        output_dir = OmegaConf.select(baseline, "output_dir") or "data/outputs/train"

        train_dirs: list[pathlib.Path] = []
        for seed in seeds:
            train_name = get_train_name(train_date, task, policy, seed)
            train_dirs.append(self.repo_root / output_dir / train_date / train_name)

        all_checkpoint_results: list[dict] = []

        for train_dir in train_dirs:
            if not train_dir.exists():
                print(f"  [eval_baseline] WARNING: train dir not found, skipping: {train_dir}")
                continue

            ckpt_dir = train_dir / "checkpoints"
            ckpt_files = sorted(
                p for p in ckpt_dir.iterdir()
                if p.suffix == ".ckpt" and p.stem != "latest"
            )
            if not ckpt_files:
                print(f"  [eval_baseline] WARNING: no checkpoints in {ckpt_dir}")
                continue

            print(
                f"  [eval_baseline] evaluating {len(ckpt_files)} checkpoints × {num_episodes} episodes"
                f"  train_dir={train_dir.name}"
            )

            for ckpt_path in ckpt_files:
                ckpt_stem = ckpt_path.stem
                output_dir_eval = str(
                    self.repo_root / eval_output_dir
                    / f"{train_dir.name}"
                    / ckpt_stem
                )

                if self.dry_run:
                    print(f"[dry_run] EvalBaselineStep  ckpt={ckpt_stem}  output={output_dir_eval}")
                    all_checkpoint_results.append({
                        "checkpoint": ckpt_stem,
                        "output_dir": output_dir_eval,
                        "num_episodes": num_episodes,
                        "num_success": 0,
                        "success_rate": 0.0,
                    })
                    continue

                print(f"    ckpt={ckpt_stem}")
                cmd = [
                    "conda", "run", "-n", conda_env, "--no-capture-output",
                    "python", str(CUPID_ROOT / "eval_save_episodes.py"),
                    f"--output_dir={output_dir_eval}",
                    f"--train_dir={train_dir}",
                    f"--train_ckpt={ckpt_stem}",
                    f"--num_episodes={num_episodes}",
                    f"--test_start_seed={test_start_seed}",
                    f"--overwrite={overwrite}",
                    f"--device={device}",
                    "--save_episodes=False",
                ]
                result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
                if result.returncode != 0:
                    raise RuntimeError(
                        f"[eval_baseline] eval subprocess failed "
                        f"(exit={result.returncode}) for ckpt={ckpt_stem}"
                    )

                rate = _read_mean_score(pathlib.Path(output_dir_eval))
                num_success = round(rate * num_episodes)
                print(
                    f"    ckpt={ckpt_stem}  "
                    f"successes~{num_success}/{num_episodes}  "
                    f"rate={rate:.3f}"
                )
                all_checkpoint_results.append({
                    "checkpoint": ckpt_stem,
                    "output_dir": output_dir_eval,
                    "num_episodes": num_episodes,
                    "num_success": num_success,
                    "success_rate": round(rate, 4),
                })

        if not all_checkpoint_results:
            mean_rate = 0.0
            best_rate = 0.0
        else:
            rates = [r["success_rate"] for r in all_checkpoint_results]
            mean_rate = round(sum(rates) / len(rates), 4)
            best_rate = round(max(rates), 4)

        print(
            f"  [eval_baseline]  mean_success_rate={mean_rate:.3f}  best={best_rate:.3f}"
        )

        return {
            "train_dirs": [str(d) for d in train_dirs],
            "checkpoints": all_checkpoint_results,
            "mean_success_rate": mean_rate,
            "best_success_rate": best_rate,
        }


def _read_mean_score(output_dir: pathlib.Path) -> float:
    """Read ``test/mean_score`` from the ``eval_log.json`` written by eval_save_episodes."""
    import json

    log_path = output_dir / "eval_log.json"
    if not log_path.exists():
        return 0.0
    data = json.loads(log_path.read_text())
    return float(data.get("test/mean_score", 0.0))
