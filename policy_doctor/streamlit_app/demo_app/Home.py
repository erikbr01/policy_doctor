"""Combined demo app: User study (Group A + B) + Graph testing demo.

Run with:
    streamlit run policy_doctor/streamlit_app/demo_app/Home.py

Uses st.navigation to decouple sidebar display label from the URL slug,
so URLs stay lowercase while labels stay nicely cased.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sys
from pathlib import Path

import streamlit as st


# Make sure the worktree's policy_doctor is on sys.path (the demo bundle
# may run inside docker without an editable install of the main repo).
_WORKTREE = Path(__file__).resolve().parents[3]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


def _gate_on_password() -> bool:
    """Return True if the request is authorized.

    If APP_PASSWORD_SHA256 is unset/empty, the app is open (local dev).
    Otherwise, show a login form and return False until the user enters a
    password whose SHA-256 hex digest matches the env var. The plaintext
    password never appears on the server. Comparison is constant-time.
    """
    expected_hash = os.environ.get("APP_PASSWORD_SHA256", "").strip().lower()
    if not expected_hash:
        return True
    if st.session_state.get("_authed"):
        return True

    st.title("Policy Doctor demo")
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            got = hashlib.sha256(pw.encode("utf-8")).hexdigest()
            if hmac.compare_digest(got, expected_hash):
                st.session_state["_authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not _gate_on_password():
    st.stop()


_HERE = Path(__file__).resolve().parent

pages = [
    st.Page(_HERE / "_pages" / "3_graph_demo.py",
            title="Graph Demo", url_path="graph_demo", default=True),
    st.Page(_HERE / "_pages" / "4_sweep_analysis.py",
            title="Sweep Analysis", url_path="sweep_analysis"),
    st.Page(_HERE / "_pages" / "1_user_study_a.py",
            title="User Study A", url_path="user_study_a"),
    st.Page(_HERE / "_pages" / "2_user_study_b.py",
            title="User Study B", url_path="user_study_b"),
]

pg = st.navigation(pages)
pg.run()
