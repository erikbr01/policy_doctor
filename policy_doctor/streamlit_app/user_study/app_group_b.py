from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.streamlit_app.user_study.graph_explorer import render_graph_full_width
from policy_doctor.streamlit_app.user_study.path_explorer import render_path_explorer
from policy_doctor.streamlit_app.user_study.strategies import (
    load_study_config,
    render_strategy_allocator,
    render_strategy_summary,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group B", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

participant_id = st.sidebar.text_input("Participant ID", value="anonymous")
mp4_dir_input = st.sidebar.text_input("MP4 directory", value="")
study_config_input = st.sidebar.text_input("Study config YAML", value="")
clustering_dir_input = st.sidebar.text_input(
    "Clustering directory",
    value="",
    help="Directory containing cluster_labels.npy, metadata.json, and optionally embeddings_reduced.npy",
)

if st.sidebar.button("Load"):
    errors: list[str] = []

    mp4_dir = Path(mp4_dir_input)
    index_path = mp4_dir / "index.json"
    if not mp4_dir.is_dir():
        errors.append(f"MP4 directory not found: {mp4_dir}")
    elif not index_path.exists():
        errors.append(f"index.json not found in {mp4_dir}")

    config_path = Path(study_config_input)
    if not config_path.exists():
        errors.append(f"Study config not found: {config_path}")

    clust_dir = Path(clustering_dir_input)
    labels_path = clust_dir / "cluster_labels.npy"
    meta_path = clust_dir / "metadata.json"
    if not clust_dir.is_dir():
        errors.append(f"Clustering directory not found: {clust_dir}")
    elif not labels_path.exists() or not meta_path.exists():
        errors.append("cluster_labels.npy or metadata.json missing in clustering directory.")

    if errors:
        for e in errors:
            st.sidebar.error(e)
    else:
        with open(index_path) as f:
            st.session_state["gb_index"] = json.load(f)

        cfg = load_study_config(config_path)
        st.session_state["gb_strategies"] = cfg["strategies"]
        st.session_state["gb_budget"] = cfg.get("budget", {}).get("total_demos", 500)
        st.session_state["gb_alloc_step"] = cfg.get("budget", {}).get("allocation_step", 25)
        st.session_state["gb_mp4_dir"] = str(mp4_dir)

        labels = np.load(str(labels_path))
        with open(meta_path) as f:
            metadata = json.load(f)
        st.session_state["gb_labels"] = labels
        st.session_state["gb_metadata"] = metadata

        coords_path = clust_dir / "embeddings_reduced.npy"
        st.session_state["gb_coords"] = np.load(str(coords_path)) if coords_path.exists() else None

        graph = BehaviorGraph.from_cluster_assignments(
            labels,
            metadata,
            level="rollout" if any("rollout_idx" in m for m in metadata) else "demo",
        )
        st.session_state["gb_graph"] = graph
        st.sidebar.success("Loaded.")

if "gb_index" in st.session_state:
    n_ep = len(st.session_state["gb_index"]["episodes"])
    n_cl = len(set(st.session_state["gb_labels"].tolist()) - {-1})
    st.sidebar.caption(f"{n_ep} episodes | {n_cl} clusters")

index = st.session_state.get("gb_index")
strategies = st.session_state.get("gb_strategies")
mp4_dir_str = st.session_state.get("gb_mp4_dir")
labels = st.session_state.get("gb_labels")
metadata = st.session_state.get("gb_metadata")
graph: BehaviorGraph | None = st.session_state.get("gb_graph")

if index is None or strategies is None or graph is None:
    st.title("User Study: Data Collection Strategy Design")
    st.markdown(
        "**Group B** — You have access to rollout videos and a behavior graph. "
        "Use the sidebar to load data, then follow the guided steps below."
    )
    st.info("Use the sidebar to load an MP4 directory, study config, and clustering directory, then click **Load**.")
    st.stop()

mp4_dir = Path(mp4_dir_str)
total_budget = st.session_state.get("gb_budget", 500)
alloc_step = st.session_state.get("gb_alloc_step", 25)

# Compute overall success rate for header stats
n_success = sum(1 for ep in index["episodes"] if ep.get("success") is True)
n_total = len(index["episodes"])

# ── Guided flow ───────────────────────────────────────────────────────────────
st.title("User Study: Data Collection Strategy Design")
st.markdown(
    "**Group B** — Work through the sections below in order. Each section gives you a "
    "different lens on the robot's current behavior. At the end, allocate your data "
    "collection budget across the available strategies."
)

# ── Section 1: Task & base policy overview ────────────────────────────────────
st.header("Step 1 — Understand the Task & Base Policy")
st.markdown(
    "The robot must pick up an object and transport it to a goal location. "
    "The videos below show rollouts from the **base policy** — the starting point "
    "we want to improve with additional data."
)

ov_c1, ov_c2, ov_c3 = st.columns(3)
ov_c1.metric("Total rollouts", n_total)
ov_c2.metric("Successes", n_success)
ov_c3.metric("Success rate", f"{n_success / n_total:.0%}" if n_total else "—")

st.markdown("Browse a sample of rollouts to get a feel for what the policy does and where it struggles.")
render_video_browser(mp4_dir, index, page_size=4, key_prefix="gb_vbrow")

# ── Section 2: Behavior graph ────────────────────────────────────────────────
st.divider()
st.header("Step 2 — Behavior Graph")
st.markdown(
    "We clustered the policy's rollouts into **behavioral modes** — recurring movement "
    "patterns that appear across many episodes. The graph below shows how the policy "
    "transitions between these modes, and which transitions tend to lead to success or failure."
)
st.markdown(
    "> **How to read it:** Each circle is a behavioral mode. "
    "Arrows show how often the policy moves from one mode to another. "
    "Use the selector below the graph to pick a node and watch example videos from that mode."
)

highlighted_path = st.session_state.get("gb_pex_highlighted_path")

render_graph_full_width(
    graph=graph,
    labels=labels,
    metadata=metadata,
    mp4_dir=mp4_dir,
    mp4_index=index,
    key_prefix="gb_gex",
    highlighted_path=highlighted_path,
)

# ── Section 3: Path explorer ──────────────────────────────────────────────────
st.divider()
st.header("Step 3 — Common Paths")
st.markdown(
    "Each path through the graph represents a sequence of behavioral modes that episodes "
    "tend to follow. **Selecting a path highlights it in the graph above.** "
    "Select a path to watch episodes that follow it."
)

render_path_explorer(
    graph=graph,
    labels=labels,
    metadata=metadata,
    mp4_dir=mp4_dir,
    mp4_index=index,
    key_prefix="gb_pex",
)

# ── Section 4: Strategy design ────────────────────────────────────────────────
st.divider()
st.header("Step 4 — Design Your Data Collection Strategy")
st.markdown(
    "Based on what you observed above, allocate your **{} demo budget** across the "
    "strategies below. Each strategy targets a specific data collection protocol. "
    "You can leave some budget unallocated.".format(total_budget)
)

allocations = render_strategy_allocator(
    strategies,
    total_budget=total_budget,
    allocation_step=alloc_step,
    key_prefix="gb_alloc",
)
render_strategy_summary(allocations, strategies, total_budget)

# ── Section 5: Submit ─────────────────────────────────────────────────────────
st.divider()
st.header("Step 5 — Submit")

notes = st.text_area(
    "Any additional notes or reasoning about your choices",
    value="",
    height=120,
    placeholder="e.g. I noticed failures often occur when the arm approaches from the left, so I focused on...",
)

if st.button("Submit", type="primary"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = mp4_dir / "study_responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"group_b_{timestamp}.json"

    result = {
        "participant_id": participant_id,
        "group": "B",
        "allocations": allocations,
        "notes": notes,
        "timestamp": timestamp,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    st.success(f"Response saved to {out_path}")
    st.download_button(
        label="Download your response",
        data=json.dumps(result, indent=2),
        file_name=f"group_b_{timestamp}.json",
        mime="application/json",
    )
