from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.streamlit_app.user_study.graph_explorer import render_graph_explorer
from policy_doctor.streamlit_app.user_study.initial_conditions import render_initial_conditions
from policy_doctor.streamlit_app.user_study.path_explorer import render_path_explorer
from policy_doctor.streamlit_app.user_study.strategies import (
    load_study_config,
    render_strategy_allocator,
    render_strategy_summary,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group B", layout="wide")

st.title("User Study: Data Collection Strategy Design")
st.markdown(
    "**Group B** — You have access to a behavior graph that shows how the robot's policy "
    "clusters into distinct behavioral modes, alongside rollout videos. Use both to inform "
    "how you would allocate a data collection budget."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

participant_id = st.sidebar.text_input("Participant ID", value="anonymous")
mp4_dir_input = st.sidebar.text_input("MP4 directory", value="")
study_config_input = st.sidebar.text_input("Study config YAML", value="")
clustering_dir_input = st.sidebar.text_input(
    "Clustering directory",
    value="",
    help="Directory containing cluster_labels.npy, metadata.json, and optionally coords.npy",
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
        st.session_state["gb_config"] = str(config_path)

        labels = np.load(str(labels_path))
        with open(meta_path) as f:
            metadata = json.load(f)
        st.session_state["gb_labels"] = labels
        st.session_state["gb_metadata"] = metadata

        coords_path = clust_dir / "embeddings_reduced.npy"
        if coords_path.exists():
            st.session_state["gb_coords"] = np.load(str(coords_path))
        else:
            st.session_state.pop("gb_coords", None)

        graph = BehaviorGraph.from_cluster_assignments(
            labels,
            metadata,
            level="rollout" if any("rollout_idx" in m for m in metadata) else "demo",
        )
        st.session_state["gb_graph"] = graph
        st.sidebar.success("Loaded.")

if "gb_index" in st.session_state:
    st.sidebar.caption(
        f"{len(st.session_state['gb_index']['episodes'])} episodes | "
        f"{len(set(st.session_state['gb_labels'].tolist()) - {-1})} clusters"
    )

index = st.session_state.get("gb_index")
strategies = st.session_state.get("gb_strategies")
mp4_dir_str = st.session_state.get("gb_mp4_dir")
labels = st.session_state.get("gb_labels")
metadata = st.session_state.get("gb_metadata")
graph: BehaviorGraph | None = st.session_state.get("gb_graph")
coords = st.session_state.get("gb_coords")

if index is None or strategies is None or graph is None:
    st.info(
        "Use the sidebar to load an MP4 directory, study config, and clustering directory, "
        "then click **Load**."
    )
    st.stop()

mp4_dir = Path(mp4_dir_str)
total_budget = st.session_state.get("gb_budget", 500)
alloc_step = st.session_state.get("gb_alloc_step", 25)

# ── Exploration tabs ──────────────────────────────────────────────────────────
tab_graph, tab_paths, tab_ic, tab_videos, tab_strategy = st.tabs([
    "Graph Explorer",
    "Path Explorer",
    "Initial Conditions",
    "Rollout Videos",
    "Design Strategy",
])

with tab_graph:
    st.markdown(
        "Explore the behavior graph. Select a node to see which rollout episodes pass "
        "through it and watch video clips from those episodes."
    )
    render_graph_explorer(
        graph=graph,
        labels=labels,
        metadata=metadata,
        mp4_dir=mp4_dir,
        mp4_index=index,
        key_prefix="gb_gex",
    )

with tab_paths:
    st.markdown(
        "View the most common paths through the graph from START to SUCCESS or FAILURE. "
        "Select a path to see matching episode videos."
    )
    render_path_explorer(
        graph=graph,
        labels=labels,
        metadata=metadata,
        mp4_dir=mp4_dir,
        mp4_index=index,
        key_prefix="gb_pex",
    )

with tab_ic:
    st.markdown(
        "Each episode is plotted at its first timestep's embedding coordinate. "
        "This shows how the distribution of initial conditions relates to behavioral "
        "clusters and outcomes."
    )
    render_initial_conditions(
        graph=graph,
        labels=labels,
        metadata=metadata,
        coords=coords,
        mp4_index=index,
        key_prefix="gb_ic",
    )

with tab_videos:
    st.markdown("Browse all rollout videos from the base policy.")
    render_video_browser(mp4_dir, index, page_size=5, key_prefix="gb_vbrow")

with tab_strategy:
    st.header("Design Your Data Collection Strategy")
    st.markdown(
        """
Allocate your data collection budget across the strategies below.
Use what you learned from the graph and video explorations above to guide your choices.
"""
    )
    allocations = render_strategy_allocator(
        strategies,
        total_budget=total_budget,
        allocation_step=alloc_step,
        key_prefix="gb_alloc",
    )
    render_strategy_summary(allocations, strategies, total_budget)

    st.divider()
    st.header("Submit")

    notes = st.text_area("Any additional notes or reasoning", value="", height=120)

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
