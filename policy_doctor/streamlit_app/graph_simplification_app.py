"""Graph Simplification + Sweep Analysis app.

Two pages, sidebar-navigable via ``st.navigation``:

  1. **Simplification** — side-by-side comparison of graph-cleanup methods
     applied to a single source clustering.
  2. **Sweep Analysis** — per-task winners and Pareto frontiers over the
     full ``data/demo_sweep/`` results.

Run with:
    conda activate policy_doctor && streamlit run \\
        policy_doctor/streamlit_app/graph_simplification_app.py --server.port 8530
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure this worktree's policy_doctor is imported (not the editable install
# pointing at the main repo).
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))
# Drop any cached main-repo policy_doctor modules
for _mod in [k for k in list(sys.modules.keys()) if k.startswith("policy_doctor")]:
    if hasattr(sys.modules[_mod], "__file__") and sys.modules[_mod].__file__:
        if str(_WORKTREE_ROOT) not in sys.modules[_mod].__file__:
            del sys.modules[_mod]

import streamlit as st

st.set_page_config(page_title="Graph Simplification + Sweep Analysis", layout="wide")

_HERE = Path(__file__).resolve().parent
_PAGES = _HERE / "_simplification_pages"

pages = [
    st.Page(
        _PAGES / "simplification.py",
        title="Simplification",
        url_path="simplification",
        default=True,
    ),
    st.Page(
        _PAGES / "sweep_analysis.py",
        title="Sweep Analysis",
        url_path="sweep_analysis",
    ),
]

pg = st.navigation(pages)
pg.run()
