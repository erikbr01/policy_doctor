"""Standalone user-study app — randomly assigns each participant to Group A or B.

Run with:
    conda activate policy_doctor
    streamlit run policy_doctor/streamlit_app/survey_app/Home.py

Environment variables:
    APP_PASSWORD_SHA256   SHA-256 hex of the access password (omit for open access)
    SURVEY_GCS_BUCKET     GCS bucket name for persisting responses (omit to save locally)
    SURVEY_LOCAL_DIR      Local response dir when not using GCS (default: ./survey_responses)

The group assignment is random and stable per session (stored in st.session_state).
Participants see no indication of which group they are in.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import runpy
import sys
from pathlib import Path

import streamlit as st

_WORKTREE = Path(__file__).resolve().parents[3]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))
for _m in [k for k in list(sys.modules.keys()) if k.startswith("policy_doctor")]:
    _f = getattr(sys.modules.get(_m), "__file__", None) or ""
    if _f and str(_WORKTREE) not in _f:
        del sys.modules[_m]

st.set_page_config(
    page_title="User Study — Teaching Robots from Examples",
    layout="wide",
    page_icon="🤖",
)


# ── Optional password gate (same SHA-256 scheme as demo app) ─────────────────

def _gate_on_password() -> bool:
    expected = os.environ.get("APP_PASSWORD_SHA256", "").strip().lower()
    if not expected:
        return True
    if st.session_state.get("_authed"):
        return True
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Access password", type="password")
        if st.form_submit_button("Enter"):
            got = hashlib.sha256(pw.encode()).hexdigest()
            if hmac.compare_digest(got, expected):
                st.session_state["_authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not _gate_on_password():
    st.stop()


# ── Stable random group assignment ───────────────────────────────────────────
# Assigned once per session and kept in session state so page reloads don't flip.

if "study_group" not in st.session_state:
    import random as _random
    st.session_state["study_group"] = _random.choice(["A", "B"])

group: str = st.session_state["study_group"]

# ── Delegate to the group app ────────────────────────────────────────────────
# The group apps each call st.set_page_config at their top level.  We've already
# configured the page above, so we temporarily replace it with a no-op to
# prevent Streamlit from raising "set_page_config() can only be called once."

import streamlit as _st  # noqa: E402 — must come after set_page_config above
_orig_cfg = _st.set_page_config
_st.set_page_config = lambda *a, **kw: None  # type: ignore[method-assign]

_US_DIR = _WORKTREE / "policy_doctor" / "streamlit_app" / "user_study"

try:
    runpy.run_path(
        str(_US_DIR / ("app_group_a.py" if group == "A" else "app_group_b.py")),
        run_name="__main__",
    )
finally:
    _st.set_page_config = _orig_cfg  # type: ignore[method-assign]
