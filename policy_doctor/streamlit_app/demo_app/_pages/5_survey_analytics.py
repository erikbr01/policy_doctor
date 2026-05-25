"""Survey analytics — visualize collected user-study responses.

Reads from GCS (if SURVEY_GCS_BUCKET is set) or scans all local
``data/study_mp4s/*/study_responses/`` directories.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_WORKTREE = Path(__file__).resolve().parents[4]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from policy_doctor.streamlit_app.user_study.response_store import (
    GCSResponseStore,
    LocalResponseStore,
)

st.set_page_config(page_title="Survey Analytics — Policy Doctor", layout="wide")
st.title("Survey Analytics")
st.caption("Live view of collected user-study responses.")

# ── Load responses ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner="Loading responses…")
def _load_responses() -> list[dict]:
    bucket = os.environ.get("SURVEY_GCS_BUCKET", "").strip()
    if bucket:
        try:
            return GCSResponseStore(bucket).list_all()
        except Exception as e:
            st.error(f"GCS error: {e}")
            return []

    # Scan all local task dirs
    results: list[dict] = []
    mp4_root = _WORKTREE / "data" / "study_mp4s"
    if mp4_root.exists():
        for task_dir in sorted(mp4_root.iterdir()):
            resp_dir = task_dir / "study_responses"
            results.extend(LocalResponseStore(resp_dir).list_all())

    # Also check SURVEY_LOCAL_DIR env override
    local_dir = os.environ.get("SURVEY_LOCAL_DIR", "").strip()
    if local_dir:
        results.extend(LocalResponseStore(local_dir).list_all())

    return results


col_refresh, col_info = st.columns([1, 8])
with col_refresh:
    if st.button("Refresh"):
        st.cache_data.clear()

responses = _load_responses()

if not responses:
    st.info(
        "No responses found yet.  "
        "Set `SURVEY_GCS_BUCKET` or place response JSON files under "
        "`data/study_mp4s/<task>/study_responses/`."
    )
    st.stop()

# ── Overview metrics ──────────────────────────────────────────────────────────

n_total = len(responses)
n_a = sum(1 for r in responses if r.get("group") == "A")
n_b = sum(1 for r in responses if r.get("group") == "B")

m1, m2, m3 = st.columns(3)
m1.metric("Total responses", n_total)
m2.metric("Group A (no graph)", n_a)
m3.metric("Group B (with graph)", n_b)

st.divider()

# ── Submission timeline ───────────────────────────────────────────────────────

st.subheader("Submission timeline")
timestamps = []
for r in responses:
    ts = r.get("timestamp", "")
    if len(ts) >= 15:
        try:
            from datetime import datetime
            timestamps.append({
                "time": datetime.strptime(ts[:15], "%Y%m%d_%H%M%S"),
                "group": r.get("group", "?"),
            })
        except Exception:
            pass

if timestamps:
    import pandas as pd
    df_ts = pd.DataFrame(timestamps).sort_values("time")
    df_ts["count"] = 1
    df_ts["cumulative"] = df_ts.groupby("group")["count"].cumsum()
    fig_tl = px.line(
        df_ts, x="time", y="cumulative", color="group",
        color_discrete_map={"A": "#1f77b4", "B": "#ff7f0e"},
        labels={"cumulative": "Cumulative submissions", "time": "Time", "group": "Group"},
        title="Cumulative submissions over time",
    )
    fig_tl.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_tl, use_container_width=True, key="sa_timeline")

st.divider()

# ── Allocation distributions ──────────────────────────────────────────────────

st.subheader("Data allocation choices")
st.caption("How participants distributed their demo-collection budget across strategies.")

all_strategy_ids: set[str] = set()
for r in responses:
    all_strategy_ids.update(r.get("allocations", {}).keys())

if all_strategy_ids:
    alloc_rows = []
    for r in responses:
        alloc = r.get("allocations", {})
        for sid, val in alloc.items():
            alloc_rows.append({
                "group": r.get("group", "?"),
                "strategy": sid,
                "allocation": val,
            })

    import pandas as pd
    df_alloc = pd.DataFrame(alloc_rows)

    fig_alloc = px.box(
        df_alloc, x="strategy", y="allocation", color="group",
        color_discrete_map={"A": "#1f77b4", "B": "#ff7f0e"},
        labels={"allocation": "Demos allocated", "strategy": "Strategy"},
        title="Allocation distributions by strategy and group",
        points="all",
    )
    fig_alloc.update_layout(
        height=400, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_alloc, use_container_width=True, key="sa_alloc")
else:
    st.info("No allocation data found in responses.")

st.divider()

# ── NASA TLX ──────────────────────────────────────────────────────────────────

st.subheader("NASA Task Load Index")

TLX_DIMS = [
    "mental_demand", "physical_demand", "temporal_demand",
    "performance", "effort", "frustration",
]
TLX_LABELS = {
    "mental_demand": "Mental demand",
    "physical_demand": "Physical demand",
    "temporal_demand": "Temporal demand",
    "performance": "Performance",
    "effort": "Effort",
    "frustration": "Frustration",
}

tlx_rows = []
for r in responses:
    tlx = r.get("nasa_tlx", {})
    if not tlx:
        continue
    for dim in TLX_DIMS:
        val = tlx.get(dim)
        if val is not None:
            tlx_rows.append({
                "group": r.get("group", "?"),
                "dimension": TLX_LABELS.get(dim, dim),
                "score": float(val),
            })

if tlx_rows:
    import pandas as pd
    df_tlx = pd.DataFrame(tlx_rows)
    fig_tlx = px.box(
        df_tlx, x="dimension", y="score", color="group",
        color_discrete_map={"A": "#1f77b4", "B": "#ff7f0e"},
        labels={"score": "TLX score (0–100)", "dimension": "Dimension"},
        title="NASA TLX scores by dimension and group",
        points="all",
    )
    fig_tlx.update_layout(
        height=420, margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, 100]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_tlx, use_container_width=True, key="sa_tlx")
else:
    st.info("No NASA TLX data found in responses.")

st.divider()

# ── Likert survey ─────────────────────────────────────────────────────────────

st.subheader("Likert survey responses")

def _render_likert_block(key: str, label: str) -> None:
    rows = []
    for r in responses:
        block = r.get(key, {})
        if not block:
            continue
        grp = r.get("group", "?")
        for q, val in block.items():
            if val is not None:
                rows.append({"group": grp, "question": str(q), "score": float(val)})
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    fig = px.box(
        df, x="question", y="score", color="group",
        color_discrete_map={"A": "#1f77b4", "B": "#ff7f0e"},
        title=label, points="all",
        labels={"score": "Score", "question": "Question"},
    )
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"sa_likert_{key}")


_render_likert_block("likert_strategy", "Strategy-selection survey")
_render_likert_block("likert_final", "Final assessment")
_render_likert_block("likert_graph", "Graph interaction survey (Group B only)")

st.divider()

# ── Raw responses table ───────────────────────────────────────────────────────

with st.expander("Raw responses (JSON table)"):
    import pandas as pd
    flat = []
    for r in responses:
        flat.append({
            "participant_id": r.get("participant_id", ""),
            "group": r.get("group", ""),
            "task": r.get("task") or r.get("session", ""),
            "timestamp": r.get("timestamp", ""),
            "notes": (r.get("notes", "") or "")[:80],
        })
    st.dataframe(pd.DataFrame(flat), use_container_width=True)
