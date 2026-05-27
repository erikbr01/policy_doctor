"""Experiment-centric artifact layout (REFACTOR_PLAN.md §3.1).

An :class:`Experiment` is a self-contained directory under
``$POLICY_DOCTOR_DATA/experiments/<name>/`` containing:

  - ``manifest.yaml``  — name, created_at, baseline_from, ...
  - ``config/``        — append-only Hydra snapshots (one per invocation)
  - ``shared/``        — baseline checkpoints + dataset symlinks shared across arms
  - ``artifacts/``     — pipeline step outputs (shared upstream + per-arm)
  - ``logs/``          — per-invocation log files
"""

from policy_doctor.experiment.experiment import Experiment
from policy_doctor.experiment.paths import (
    data_root,
    datasets_dir,
    experiment_dir,
    experiments_dir,
)

__all__ = [
    "Experiment",
    "data_root",
    "datasets_dir",
    "experiment_dir",
    "experiments_dir",
]
