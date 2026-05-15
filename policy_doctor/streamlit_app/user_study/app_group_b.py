from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    FAILURE_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.streamlit_app.user_study.graph_explorer import render_graph_full_width
from policy_doctor.streamlit_app.user_study.initial_conditions import (
    _initial_slice_per_episode,
    _success_per_episode,
)
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

# Discover available session configs
_REPO_ROOT = Path(__file__).parents[3]
_SESSIONS_DIR = _REPO_ROOT / "policy_doctor" / "configs" / "user_study" / "sessions"
_session_files = sorted(_SESSIONS_DIR.glob("*.yaml")) if _SESSIONS_DIR.is_dir() else []
_session_labels = {f.stem: yaml.safe_load(f.read_text()).get("label", f.stem) for f in _session_files}

participant_id = st.sidebar.text_input("Participant ID", value="anonymous")

if not _session_files:
    st.sidebar.warning("No session configs found in configs/user_study/sessions/")
    st.stop()

session_choice = st.sidebar.selectbox(
    "Session",
    options=[f.stem for f in _session_files],
    format_func=lambda k: _session_labels.get(k, k),
)

if st.sidebar.button("Load"):
    sess_path = _SESSIONS_DIR / f"{session_choice}.yaml"
    sess = yaml.safe_load(sess_path.read_text())

    # Resolve relative paths from repo root
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _REPO_ROOT / path

    mp4_dir = _resolve(sess["mp4_dir"])
    config_path = _resolve(sess["study_config"])
    clust_dir = _resolve(sess["clustering_dir"])

    errors: list[str] = []
    index_path = mp4_dir / "index.json"
    if not mp4_dir.is_dir():
        errors.append(f"MP4 directory not found: {mp4_dir}")
    elif not index_path.exists():
        errors.append("index.json not found in MP4 directory")
    if not config_path.exists():
        errors.append(f"Study config not found: {config_path}")
    labels_path = clust_dir / "cluster_labels.npy"
    meta_path = clust_dir / "metadata.json"
    if not clust_dir.is_dir():
        errors.append(f"Clustering directory not found: {clust_dir}")
    elif not labels_path.exists() or not meta_path.exists():
        errors.append("cluster_labels.npy or metadata.json missing")

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

# ── Section 4: Failure analysis ───────────────────────────────────────────────
st.divider()
st.header("Step 4 — Where Does the Policy Struggle?")
st.markdown(
    "This step highlights the graph nodes and initial conditions most associated "
    "with failure, to help you identify which behaviors need more data."
)

fa_left, fa_right = st.columns(2)

with fa_left:
    _SPECIAL = frozenset({SUCCESS_NODE_ID, FAILURE_NODE_ID})
    node_failure_rates: list[tuple[int, str, float]] = []
    for nid, node in graph.nodes.items():
        if nid in _SPECIAL:
            continue
        outgoing = graph.transition_probs.get(nid, {})
        total_weight = sum(outgoing.values())
        if total_weight == 0:
            continue
        fail_weight = outgoing.get(FAILURE_NODE_ID, 0.0)
        node_failure_rates.append((nid, node.name, fail_weight / total_weight))

    node_failure_rates.sort(key=lambda t: -t[2])
    top5 = node_failure_rates[:5]

    if top5:
        fig_fail = go.Figure(go.Bar(
            x=[r for _, _, r in top5],
            y=[f"Node {nid}: {name}" for nid, name, _ in top5],
            orientation="h",
            marker_color="#d62728",
        ))
        fig_fail.update_layout(
            title="Nodes most likely to lead to failure",
            height=280,
            margin=dict(l=0, r=20, t=40, b=20),
            xaxis=dict(title="Failure rate", range=[0, 1]),
            yaxis=dict(title=None, autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fail, use_container_width=True, key="gb_fa_fail_nodes")
    else:
        st.info("No cluster nodes with outgoing failure transitions found.")

with fa_right:
    st.metric("Overall success rate", f"{n_success / n_total:.0%}" if n_total else "—")
    if n_total:
        fig_pie = go.Figure(go.Pie(
            labels=["Success", "Failure"],
            values=[n_success, n_total - n_success],
            marker=dict(colors=["#2ca02c", "#d62728"]),
            hole=0.3,
            textinfo="percent+label",
        ))
        fig_pie.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="gb_fa_pie")

coords = st.session_state.get("gb_coords")
if coords is not None:
    ep_first = _initial_slice_per_episode(labels, metadata, coords)
    ep_success_map = _success_per_episode(metadata)
    all_ep_ids = sorted(ep_first.keys())
    xs = [ep_first[e][3] for e in all_ep_ids]
    ys = [ep_first[e][4] for e in all_ep_ids]
    successes = [ep_success_map.get(e) for e in all_ep_ids]

    fig_ic = go.Figure()
    for outcome, color, name in [
        (True, "#2ca02c", "Success"),
        (False, "#d62728", "Failure"),
        (None, "#aaaaaa", "Unknown"),
    ]:
        mask = [i for i, s in enumerate(successes) if s is outcome]
        if not mask:
            continue
        fig_ic.add_trace(go.Scatter(
            x=[xs[i] for i in mask],
            y=[ys[i] for i in mask],
            mode="markers",
            marker=dict(color=color, size=7),
            name=name,
            text=[f"Episode {all_ep_ids[i]}" for i in mask],
            hoverinfo="text",
        ))
    fig_ic.update_layout(
        title="Initial conditions (2D embedding of first timestep)",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(title="Embedding dim 1", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Embedding dim 2", showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ic, use_container_width=True, key="gb_fa_ic_scatter")
    st.caption(
        "Each point is one episode, plotted at its start embedding. "
        "Clusters of red points indicate initial conditions where the policy consistently fails."
    )
else:
    st.info("Run UMAP reduction to enable initial conditions view.")

# ── Section 5: Strategy design ────────────────────────────────────────────────
st.divider()
st.header("Step 5 — Design Your Data Collection Strategy")
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

# ── Section 6: Submit ─────────────────────────────────────────────────────────
st.divider()
st.header("Step 6 — Submit")

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
