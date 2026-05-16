"""Mirror of policy_doctor/streamlit_app/user_study/app_group_b.py as a page."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[4]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

_TARGET = _WORKTREE / "policy_doctor" / "streamlit_app" / "user_study" / "app_group_b.py"
runpy.run_path(str(_TARGET), run_name="__main__")
