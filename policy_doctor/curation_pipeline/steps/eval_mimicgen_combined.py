"""Evaluate all top-k checkpoints from TrainOnCombinedDataStep — pipeline step.

Runs ``num_episodes`` rollouts for every saved checkpoint (all non-``latest``
``.ckpt`` files in the training directory), then reports per-checkpoint success
rates and their mean.  The mean across checkpoints is a lower-variance estimator
than any single checkpoint, and guards against cherry-picking.

Config keys (under ``evaluation``):
    num_episodes      Episodes per checkpoint (default 500).
    test_start_seed   RNG seed for the rollout environment (default 100000).
    overwrite         Re-run even if eval output already exists (default false).
    device            CUDA device string (default ``cuda:0``).
    eval_output_dir   Base output dir (default ``data/outputs/eval_save_episodes``).

Result JSON:
    heuristic              Seed-selection heuristic name.
    train_dir              Path to the training directory evaluated.
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
from policy_doctor.paths import CUPID_ROOT


class EvalMimicgenCombinedStep(PipelineStep[dict]):
    """Evaluate every top-k checkpoint produced by TrainOnCombinedDataStep."""

    name = "eval_mimicgen_combined"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.train_on_combined_data import (
            TrainOnCombinedDataStep,
        )

        cfg = self.cfg
        evaluation = OmegaConf.select(cfg, "evaluation") or {}

        num_episodes: int = int(
            OmegaConf.select(evaluation, "num_episodes")
            or 500
        )
        test_start_seed: int = int(
            OmegaConf.select(evaluation, "test_start_seed") or 100000
        )
        overwrite: bool = bool(OmegaConf.select(evaluation, "overwrite") or False)
        device: str = OmegaConf.select(cfg, "device") or "cuda:0"
        eval_output_dir: str = (
            OmegaConf.select(evaluation, "eval_output_dir")
            or "data/outputs/eval_save_episodes"
        )
        conda_env: str = (
            OmegaConf.select(cfg, "data_source.conda_env_train")
            or "mimicgen_torch2"
        )

        # --- Load train dirs from TrainOnCombinedDataStep ---
        train_step = TrainOnCombinedDataStep(cfg, self.run_dir)
        if not train_step.is_done():
            raise RuntimeError(
                "EvalMimicgenCombinedStep requires TrainOnCombinedDataStep to have run first."
            )
        train_result = train_step.load()
        heuristic: str = train_result.get("heuristic", "unknown")
        train_dirs: list[str] = train_result.get("train_dirs", [])
        if not train_dirs:
            raise RuntimeError("TrainOnCombinedDataStep result has no train_dirs.")

        all_checkpoint_results: list[dict] = []

        for train_dir_str in train_dirs:
            train_dir = pathlib.Path(train_dir_str)
            if not train_dir.exists():
                raise FileNotFoundError(f"Train dir not found: {train_dir}")

            ckpt_dir = train_dir / "checkpoints"
            # All saved checkpoints except latest
            ckpt_files = sorted(
                p for p in ckpt_dir.iterdir()
                if p.suffix == ".ckpt" and p.stem != "latest"
            )
            if not ckpt_files:
                print(f"  [eval_mimicgen_combined] WARNING: no checkpoints in {ckpt_dir}")
                continue

            print(
                f"  [eval_mimicgen_combined] heuristic={heuristic!r}  "
                f"evaluating {len(ckpt_files)} checkpoints × {num_episodes} episodes"
            )

            for ckpt_path in ckpt_files:
                ckpt_stem = ckpt_path.stem  # e.g. "epoch=0850-test_mean_score=0.440"
                output_dir = str(
                    self.repo_root / eval_output_dir
                    / f"{train_dir.name}"
                    / ckpt_stem
                )

                if self.dry_run:
                    print(
                        f"[dry_run] EvalMimicgenCombinedStep  "
                        f"ckpt={ckpt_stem}  output={output_dir}"
                    )
                    all_checkpoint_results.append({
                        "checkpoint": ckpt_stem,
                        "output_dir": output_dir,
                        "num_episodes": num_episodes,
                        "num_success": 0,
                        "success_rate": 0.0,
                    })
                    continue

                print(f"    ckpt={ckpt_stem}")
                cmd = [
                    "conda", "run", "-n", conda_env, "--no-capture-output",
                    "python", str(CUPID_ROOT / "eval_save_episodes.py"),
                    f"--output_dir={output_dir}",
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
                        f"[eval_mimicgen_combined] eval subprocess failed "
                        f"(exit={result.returncode}) for ckpt={ckpt_stem}"
                    )

                # Read success rate from eval_log.json written by eval_save_episodes
                rate = _read_mean_score(pathlib.Path(output_dir))
                num_success = round(rate * num_episodes)
                print(
                    f"    ckpt={ckpt_stem}  "
                    f"successes~{num_success}/{num_episodes}  "
                    f"rate={rate:.3f}"
                )
                all_checkpoint_results.append({
                    "checkpoint": ckpt_stem,
                    "output_dir": output_dir,
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
            f"  [eval_mimicgen_combined] heuristic={heuristic!r}  "
            f"mean_success_rate={mean_rate:.3f}  best={best_rate:.3f}"
        )

        return {
            "heuristic": heuristic,
            "train_dir": train_dirs[0] if train_dirs else "",
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
