"""Task-specific configuration for DAgger / E2 data collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from policy_doctor.paths import CONFIGS_DIR, REPO_ROOT


def get_data_collection_task_dir() -> Path:
    """Return the directory containing task-specific collection configs."""
    return CONFIGS_DIR / "data_collection" / "tasks"


def available_data_collection_tasks() -> list[str]:
    """Return available task config names."""
    task_dir = get_data_collection_task_dir()
    return sorted(p.stem for p in task_dir.glob("*.yaml"))


def load_data_collection_task_config(task: str) -> dict[str, Any]:
    """Load task-specific environment and recording config.

    Relative ``dataset_path`` values are resolved under ``REPO_ROOT`` so callers
    can run from either the project root or ``third_party/cupid``.
    """
    cfg_path = get_data_collection_task_dir() / f"{task}.yaml"
    if not cfg_path.exists():
        available = ", ".join(available_data_collection_tasks())
        raise FileNotFoundError(
            f"Data collection task config {task!r} not found at {cfg_path}. "
            f"Available: {available}"
        )

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    dataset_path = cfg.get("dataset_path")
    if dataset_path:
        p = Path(str(dataset_path))
        cfg["dataset_path"] = str(p if p.is_absolute() else REPO_ROOT / p)

    recording = cfg.get("recording") or {}
    obs_keys = recording.get("obs_keys")
    if not obs_keys:
        raise ValueError(
            f"Data collection task config {task!r} must define recording.obs_keys"
        )
    cfg["recording"] = recording
    return cfg
