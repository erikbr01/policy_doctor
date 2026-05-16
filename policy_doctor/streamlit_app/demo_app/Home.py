"""Combined demo app: User study (Group A + B) + Graph testing demo.

Run with:
    streamlit run policy_doctor/streamlit_app/demo_app/Home.py

Uses st.navigation to decouple sidebar display label from the URL slug,
so URLs stay lowercase while labels stay nicely cased.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# Make sure the worktree's policy_doctor is on sys.path (the demo bundle
# may run inside docker without an editable install of the main repo).
_WORKTREE = Path(__file__).resolve().parents[3]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


_HERE = Path(__file__).parent

pages = [
    st.Page(str(_HERE / "_pages" / "3_graph_demo.py"),
            title="Graph Demo", url_path="graph_demo", default=True),
    st.Page(str(_HERE / "_pages" / "1_user_study_a.py"),
            title="User Study A", url_path="user_study_a"),
    st.Page(str(_HERE / "_pages" / "2_user_study_b.py"),
            title="User Study B", url_path="user_study_b"),
]

pg = st.navigation(pages)
pg.run()
