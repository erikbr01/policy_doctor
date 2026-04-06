"""
Policy Doctor pipeline entrypoint — powered by Hydra.

Run a single step or a sequence of steps, with config composed from:
  - policy_doctor package ``configs/config.yaml``  (base)
  - +experiment=<name>                 (experiment overrides)
  - key=value                          (ad-hoc CLI overrides)

Examples:
  # Full pipeline for an experiment
  python -m policy_doctor.scripts.run_pipeline \\
    +experiment=trak_filtering_mar13_p96 \\
    steps=[run_clustering,run_curation_config,train_curated,eval_curated]

  # Resume — skips already-completed steps
  python -m policy_doctor.scripts.run_pipeline \\
    +experiment=trak_filtering_mar13_p96 \\
    run_dir=data/pipeline_runs/myrun \\
    steps=[train_curated,eval_curated]

  # Dry-run
  python -m policy_doctor.scripts.run_pipeline \\
    +experiment=trak_filtering_mar13_p96 \\
    steps=[train_curated] dry_run=true
"""

from __future__ import annotations

import os

if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

from policy_doctor.paths import CONFIGS_DIR, REPO_ROOT

_REPO_ROOT = REPO_ROOT
_CONFIGS_DIR = CONFIGS_DIR

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path=str(_CONFIGS_DIR),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    steps = list(cfg.get("steps") or [])
    skip_if_done = bool(cfg.get("skip_if_done", True))

    # Inject repo_root so the Pipeline knows where the repo lives
    if not cfg.get("repo_root"):
        OmegaConf.update(cfg, "repo_root", str(_REPO_ROOT), merge=True)

    from policy_doctor.curation_pipeline.pipeline import CurationPipeline

    pipeline = CurationPipeline(cfg)
    print(f"[run_pipeline] run_dir: {pipeline.run_dir}")

    pipeline.run(steps=steps or None, skip_if_done=skip_if_done)


if __name__ == "__main__":
    main()
