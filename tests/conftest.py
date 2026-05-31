"""Shared fixtures for policy_doctor tests.

Install the repo with editable packages (see scripts/install_policy_doctor_env.sh) so
``policy_doctor``, ``influence_visualizer``, and cupid top-level modules resolve without PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prepend the venv's lib/ to LD_LIBRARY_PATH so that subprocesses spawned from
# tests find the pip-installed libtbb.so (TBB_INTERFACE_VERSION >= 12060) before
# the system libtbb.so (typically TBB 2021.5, interface version 12050) which
# numba rejects. This mirrors what `uv run` does when dispatching commands but is
# not inherited by subprocess.run() calls that bypass uv.
_venv_lib = Path(sys.executable).parent.parent / "lib"
if _venv_lib.is_dir() and (_venv_lib / "libtbb.so.12").exists():
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_venv_lib}:{_existing}" if _existing else str(_venv_lib)
