"""Third-party sim / dataset stacks: path helpers (no heavy imports at import time)."""

from __future__ import annotations

import sys
from pathlib import Path

from policy_doctor.paths import MIMICGEN_ROOT, ROBOCASA_ROOT


def ensure_mimicgen_on_path() -> Path:
    """Insert vendored MimicGen repo on ``sys.path`` (same idea as tests support)."""
    root = MIMICGEN_ROOT.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def ensure_robocasa_on_path() -> Path:
    """Insert vendored RoboCasa repo on ``sys.path`` for scripts and optional imports."""
    root = ROBOCASA_ROOT.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
