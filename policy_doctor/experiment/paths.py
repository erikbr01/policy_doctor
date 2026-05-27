"""Data-root path resolution.

The canonical data root is ``$POLICY_DOCTOR_DATA`` if set, otherwise
``<repo_root>/data/``. All lookups are lazy so environment changes (e.g. in
pytest fixtures via ``monkeypatch.setenv``) are picked up.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Resolve the canonical data root.

    Resolution order:
      1. ``$POLICY_DOCTOR_DATA`` environment variable
      2. ``<repo_root>/data/`` (default; gitignored)
    """
    env = os.environ.get("POLICY_DOCTOR_DATA")
    return Path(env).resolve() if env else _REPO_ROOT / "data"


def datasets_dir() -> Path:
    """Where source HDF5 datasets live, shared across experiments."""
    return data_root() / "datasets"


def experiments_dir() -> Path:
    """Root for all self-contained experiment directories."""
    return data_root() / "experiments"


def experiment_dir(name: str) -> Path:
    """Path to a specific experiment's directory (may not exist yet)."""
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"Invalid experiment name: {name!r}")
    return experiments_dir() / name
