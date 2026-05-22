from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.streamlit_app.components.mp4_player import mp4_player
from policy_doctor.streamlit_app.user_study.graph_explorer import render_graph_full_width
from policy_doctor.streamlit_app.user_study.graph_plot import compute_pruned_graph_nodes
from policy_doctor.streamlit_app.user_study.initial_conditions import (
    _episodes_for_path as _ic_episodes_for_path,
    _initial_slice_per_episode,
    _success_per_episode,
)
from policy_doctor.streamlit_app.user_study.likert_survey import (
    render_block1_graph_interaction,
    render_block2_strategy,
    render_block3_final,
)
from policy_doctor.streamlit_app.user_study.nasa_tlx import render_nasa_tlx
from policy_doctor.streamlit_app.user_study.path_explorer import render_path_explorer
from policy_doctor.streamlit_app.user_study.response_store import get_store
from policy_doctor.streamlit_app.user_study.strategies import (
    load_study_config,
    render_strategy_allocator,
    render_strategy_summary,
)
from policy_doctor.streamlit_app.user_study.survey_steps import (
    N_STEPS,
    STEP_LABELS,
    advance_step,
    get_step_durations,
    record_step_entry,
    render_progress_bar,
    render_rollout_timer,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group B", layout="wide")

# ── Sidebar: config + session loading ────────────────────────────────────────

st.sidebar.header("Study configuration")

_REPO_ROOT = Path(__file__).parents[3]
_SESSIONS_DIR = _REPO_ROOT / "policy_doctor" / "configs" / "user_study" / "sessions"
_session_files = sorted(_SESSIONS_DIR.glob("*.yaml")) if _SESSIONS_DIR.is_dir() else []
_session_labels = {f.stem: yaml.safe_load(f.read_text()).get("label", f.stem) for f in _session_files}

participant_id = st.sidebar.text_input("Participant ID", value="anonymous")

colorblind_mode = st.sidebar.toggle(
    "Colorblind mode",
    value=st.session_state.get("colorblind_mode", False),
    key="colorblind_mode",
)

if not _session_files:
    st.sidebar.warning("No session configs found in configs/user_study/sessions/")
    st.stop()

session_choice = st.sidebar.selectbox(
    "Session",
    options=[f.stem for f in _session_files],
    format_func=lambda k: _session_labels.get(k, k),
)

PFX = "gb"
STEP_KEY = f"{PFX}_step"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


if st.session_state.get(f"{PFX}_loaded_session") != session_choice:
    sess_path = _SESSIONS_DIR / f"{session_choice}.yaml"
    sess = yaml.safe_load(sess_path.read_text())

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
            st.session_state[f"{PFX}_index"] = json.load(f)

        cfg = load_study_config(config_path)
        st.session_state[f"{PFX}_strategies"] = cfg["strategies"]
        st.session_state[f"{PFX}_budget"] = cfg.get("budget", {}).get("total_demos", 500)
        st.session_state[f"{PFX}_alloc_step"] = cfg.get("budget", {}).get("allocation_step", 25)
        st.session_state[f"{PFX}_mp4_dir"] = str(mp4_dir)
        st.session_state[f"{PFX}_rollout_limit"] = sess.get("rollout_time_limit_seconds", 600)
        _dvd = sess.get("demo_videos_dir")
        st.session_state[f"{PFX}_demo_videos_dir"] = (
            str(_resolve(_dvd)) if _dvd else str(mp4_dir / "demo_videos")
        )

        labels = np.load(str(labels_path))
        with open(meta_path) as f:
            metadata = json.load(f)
        st.session_state[f"{PFX}_labels"] = labels
        st.session_state[f"{PFX}_metadata"] = metadata

        coords_path = clust_dir / "embeddings_reduced.npy"
        st.session_state[f"{PFX}_coords"] = np.load(str(coords_path)) if coords_path.exists() else None

        graph = BehaviorGraph.from_cluster_assignments(
            labels,
            metadata,
            level="rollout" if any("rollout_idx" in m for m in metadata) else "demo",
        )
        st.session_state[f"{PFX}_graph"] = graph
        st.session_state[f"{PFX}_loaded_session"] = session_choice

if f"{PFX}_index" in st.session_state:
    n_ep = len(st.session_state[f"{PFX}_index"]["episodes"])
    n_cl = len(set(st.session_state[f"{PFX}_labels"].tolist()) - {-1})
    st.sidebar.caption(f"{n_ep} episodes | {n_cl} clusters")

index = st.session_state.get(f"{PFX}_index")
strategies = st.session_state.get(f"{PFX}_strategies")
mp4_dir_str = st.session_state.get(f"{PFX}_mp4_dir")
labels = st.session_state.get(f"{PFX}_labels")
metadata = st.session_state.get(f"{PFX}_metadata")
graph: BehaviorGraph | None = st.session_state.get(f"{PFX}_graph")

if index is None or strategies is None or graph is None:
    st.title("User Study: Data Collection Strategy Design")
    st.error("Failed to load session — check the sidebar for errors.")
    st.stop()

mp4_dir = Path(mp4_dir_str)
total_budget = st.session_state.get(f"{PFX}_budget", 500)
alloc_step = st.session_state.get(f"{PFX}_alloc_step", 25)
rollout_limit = st.session_state.get(f"{PFX}_rollout_limit", 600)
_dvd_str = st.session_state.get(f"{PFX}_demo_videos_dir")
demo_videos_dir = Path(_dvd_str) if _dvd_str else None

# ── Step routing ──────────────────────────────────────────────────────────────

step = st.session_state.get(STEP_KEY, 0)

render_progress_bar(step, STEP_LABELS)
record_step_entry(step, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Consent
# ─────────────────────────────────────────────────────────────────────────────
if step == 0:
    st.title("Invitation to Participate in a Research Study")
    st.markdown("""
We would like to invite you to participate in a research study on **robot learning and data collection**.

The research investigates whether a visual representation of a robot policy's behavioral modes helps
human planners make more targeted and effective data-collection decisions.

**What you will do:**
- Watch videos of a robot attempting a manipulation task
- Explore an interactive **behavior graph** showing the robot's movement patterns
- Decide how to allocate a data collection budget across different strategies
- Complete a brief survey about your experience

**Time commitment:** approximately **25–35 minutes** in total.

**Participation is voluntary.** You may withdraw at any time without any disadvantage.
You will not receive compensation for participation.

**Group assignment.** Participants are randomly assigned to one of two conditions.
You have been assigned to the condition with the behavior graph visualization.

**Data privacy.** Your responses are recorded anonymously with your chosen participant ID.
""")

    agreed = st.checkbox(
        "I have read the above information and voluntarily agree to participate.",
        key=f"{PFX}_consent_agreed",
    )
    if st.button("Continue →", type="primary", disabled=not agreed):
        advance_step(0, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Introduction
# ─────────────────────────────────────────────────────────────────────────────
elif step == 1:
    st.title("Study Introduction")

    st.markdown("""
### Background: How does the robot learn?

Instead of programming the robot with explicit rules, we show it many examples of the task being done correctly.
The robot learns a *policy* — a mapping from what it sees (camera images, joint positions) to what action to take next.

This approach is called **Learning from Demonstrations (LfD)**.

### The task

The robot must pick up an object from a table and transport it to a goal location.
- ✓ **Success** — the object reaches the goal
- ✗ **Failure** — the robot drops it, misses, or runs out of time

### Your role

You will watch videos of the robot's current behavior and explore a **behavior graph** — an
automatic grouping of the robot's movement patterns into labeled behavioral modes.
Then you will allocate a **demo collection budget** across different data collection strategies.

### About the behavior graph

The behavior graph clusters the robot's rollouts into recurring movement patterns ("behaviors").
Arrows in the graph show how often the robot transitions from one behavior to another, and which
transitions tend to lead to success or failure.
**You can click any node or edge** to see example video clips for that behavior or transition.
""")

    limit_min = rollout_limit // 60
    limit_sec = rollout_limit % 60
    limit_str = f"{limit_min} minute{'s' if limit_min != 1 else ''}" + (
        f" {limit_sec}s" if limit_sec else ""
    )

    st.warning(
        f"**About the next section — Rollout Info:**  \n"
        f"You will have **{limit_str}** to explore the robot videos and behavior graph. "
        f"A countdown timer will be shown at the top of the page. "
        f"Once the timer expires (or you click Proceed), you will move on automatically "
        f"and **cannot return** to this page. "
        f"Use your time wisely — browse videos, explore the graph, and note patterns "
        f"that will inform your data collection decisions."
    )

    if st.button("Continue →", type="primary"):
        advance_step(1, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Rollout Info  (timer-gated)
# ─────────────────────────────────────────────────────────────────────────────
elif step == 2:
    st.title("Explore the Robot's Behavior")

    # Timer setup
    start_key = f"{PFX}_rollout_start"
    if start_key not in st.session_state:
        st.session_state[start_key] = st.session_state.get(
            f"{PFX}_times", {}).get(2, {}).get("start")
        if st.session_state[start_key] is None:
            import time as _time
            st.session_state[start_key] = _time.time()

    remaining, expired = render_rollout_timer(
        st.session_state[start_key],
        rollout_limit,
        key=f"{PFX}_rtimer",
    )

    # ── Section A: Videos ────────────────────────────────────────────────────
    n_total = len(index["episodes"])
    n_success = sum(1 for ep in index["episodes"] if ep.get("success") is True)
    ov_c1, ov_c2, ov_c3 = st.columns(3)
    ov_c1.metric("Total rollouts", n_total)
    ov_c2.metric("Successes", n_success)
    ov_c3.metric("Success rate", f"{n_success / n_total:.0%}" if n_total else "—")

    st.subheader("Watch the Robot in Action")
    st.markdown(
        "These videos show **rollouts** — the robot attempting the task from scratch. "
        "Watch several to understand what it does well and where it struggles."
    )
    render_video_browser(mp4_dir, index, page_size=4, key_prefix=f"{PFX}_vbrow")

    # ── Section B: Behavior graph ─────────────────────────────────────────────
    st.divider()
    st.subheader("Explore the Behavior Graph")
    st.markdown(
        "The robot's rollouts have been automatically grouped into **behavioral modes** — "
        "recurring movement patterns. "
        "The graph shows how often the robot transitions between modes and which lead to "
        "**success ✓** or **failure ✗**."
    )
    with st.expander("❓ How to read this graph", expanded=False):
        st.markdown("""
- **Each circle** is a behavioral mode
- **Arrows** show transitions (thickness = probability)
- **Larger circles** = more episodes passed through this mode
- **Click any node or edge** to see example clips
- **Click the background** to deselect
""")

    highlighted_path = st.session_state.get(f"{PFX}_pex_highlighted_path")

    _VIZ_OPTIONS = [
        "tree_native_svg", "tree_sunburst", "tree_icicle",
        "markov_svg_bfs", "markov_svg_temporal",
    ]
    _VIZ_LABELS = {
        "tree_native_svg": "Trajectory tree",
        "tree_sunburst": "Sunburst",
        "tree_icicle": "Icicle",
        "markov_svg_bfs": "Markov graph — BFS-layered",
        "markov_svg_temporal": "Markov graph — temporal mean",
    }

    _c_viz, _c_color = st.columns([2, 1])
    with _c_viz:
        _viz_type = st.selectbox(
            "Visualization",
            options=_VIZ_OPTIONS,
            format_func=lambda v: _VIZ_LABELS[v],
            index=0,
            key=f"{PFX}_viz_type",
        )
    _is_tree = _viz_type.startswith("tree_")
    with _c_color:
        _color_opts = ["outcome", "id"] if _is_tree else ["id", "timesteps"]
        _color_labels = {
            "outcome": "Outcome (success rate)", "id": "Cluster ID",
            "timesteps": "Timestep count",
        }
        _color_by = st.selectbox(
            "Color nodes by",
            options=_color_opts,
            format_func=lambda v: _color_labels[v],
            index=0,
            key=f"{PFX}_color_by",
        )

    _min_branch = st.slider(
        "Hide transitions where count < N",
        1, 50, 2,
        key=f"{PFX}_min_branch",
    )

    _active_graph, _active_labels = graph, labels
    _excluded_nodes = compute_pruned_graph_nodes(
        _active_graph, min_visit_prob=0.0, n_total=_active_graph.num_episodes,
        min_edge_count=int(_min_branch),
    )
    _n_edges_total = sum(len(t) for t in _active_graph.transition_counts.values())
    _n_edges_hidden = sum(
        1
        for src, targets in _active_graph.transition_counts.items()
        for tgt, cnt in targets.items()
        if src in _excluded_nodes or tgt in _excluded_nodes or cnt < int(_min_branch)
    )
    st.caption(
        f"Hidden {_n_edges_hidden} / {_n_edges_total} edges  ·  "
        f"{len(_excluded_nodes)} / {len(_active_graph.nodes)} nodes pruned"
    )

    if _is_tree:
        from policy_doctor.streamlit_app.components.trajectory_tree_view import (
            render_trajectory_tree,
        )
        render_trajectory_tree(
            labels=_active_labels,
            metadata=metadata,
            view_mode=_viz_type.replace("tree_", ""),
            min_branch=int(_min_branch),
            max_depth_cap=500,
            color_mode=_color_by,
            node_values={},
            cluster_names=None,
            mp4_dir=mp4_dir,
            mp4_index=index,
            height=600,
            level=getattr(_active_graph, "level", "rollout"),
            key_prefix=f"{PFX}_tree",
            edge_style="lines",
            edge_width_slope=5.0,
            node_size_slope=24.0,
        )
    else:
        _SUCCESS_COL = "#0072B2" if colorblind_mode else "#2ca02c"
        _FAILURE_COL = "#D55E00" if colorblind_mode else "#d62728"
        _SPECIAL_TERM = {START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID}
        _color_override = {
            nid: (_SUCCESS_COL if node.cluster_id == SUCCESS_NODE_ID else _FAILURE_COL)
            for nid, node in _active_graph.nodes.items()
            if node.cluster_id in (SUCCESS_NODE_ID, FAILURE_NODE_ID)
        }
        if _color_by == "timesteps":
            try:
                import plotly.express as _px
                _viridis = _px.colors.sequential.Viridis
            except Exception:
                _viridis = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
            _ts_max = max(
                (np.log1p(_active_graph.nodes[nid].num_timesteps)
                 for nid in _active_graph.nodes if nid not in _SPECIAL_TERM),
                default=1.0,
            ) or 1.0
            _color_override = {}
            for nid in _active_graph.nodes:
                if nid in _SPECIAL_TERM:
                    continue
                _t = np.log1p(_active_graph.nodes[nid].num_timesteps) / _ts_max
                _idx = int(min(len(_viridis) - 1, max(0, _t * (len(_viridis) - 1))))
                _color_override[nid] = _viridis[_idx]

        _pos = None
        if _viz_type == "markov_svg_temporal":
            from policy_doctor.behaviors import graph_simplification as _gs
            try:
                _pos = _gs.temporal_layout(
                    _active_graph, _active_labels, metadata,
                    level=getattr(_active_graph, "level", "rollout"),
                )
            except Exception as _e:
                st.warning(f"Temporal layout failed ({_e}); falling back to BFS-layered.")

        render_graph_full_width(
            graph=_active_graph,
            labels=_active_labels,
            metadata=metadata,
            mp4_dir=mp4_dir,
            mp4_index=index,
            key_prefix=f"{PFX}_gex",
            highlighted_path=highlighted_path,
            excluded_node_ids=_excluded_nodes,
            min_edge_count=int(_min_branch),
            pos=_pos,
            color_override=_color_override,
        )

    # ── Section C: Path explorer ──────────────────────────────────────────────
    st.divider()
    st.subheader("Common Paths")
    st.markdown(
        "Each path through the graph represents a sequence of behavioral modes. "
        "**Select a path to highlight it in the graph above** and watch matching episodes."
    )
    render_path_explorer(
        graph=_active_graph,
        labels=_active_labels,
        metadata=metadata,
        mp4_dir=mp4_dir,
        mp4_index=index,
        key_prefix=f"{PFX}_pex",
    )

    # ── Section D: Failure analysis ───────────────────────────────────────────
    st.divider()
    st.subheader("Where Does the Policy Struggle?")

    # Helper functions (scoped to step 2)
    ep_key_meta = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"

    def _failure_eps_for_node(node_id: int) -> list[tuple[int, int, int]]:
        ep_slices: dict[int, list[int]] = {}
        ep_success_map: dict[int, bool] = {}
        for i, m in enumerate(metadata):
            if int(_active_labels[i]) != node_id:
                continue
            ep_idx = m.get(ep_key_meta, -1)
            ts = m.get("window_start", m.get("timestep", 0))
            ep_slices.setdefault(ep_idx, []).append(ts)
            if ep_idx not in ep_success_map and "success" in m:
                ep_success_map[ep_idx] = bool(m["success"])
        result_list = []
        for ep_idx, tss in ep_slices.items():
            if not ep_success_map.get(ep_idx, True):
                result_list.append((ep_idx, min(tss), max(tss)))
        return sorted(result_list)

    def _mp4_entry(ep_idx: int) -> Optional[dict]:
        for ep in index.get("episodes", []):
            if ep.get("index") == ep_idx:
                return ep
        return None

    def _ep_behavior_range(ep_idx: int, node_id: int) -> Optional[tuple[int, int]]:
        tss = [
            m.get("window_start", m.get("timestep", 0))
            for i, m in enumerate(metadata)
            if m.get(ep_key_meta) == ep_idx and int(_active_labels[i]) == node_id
        ]
        return (min(tss), max(tss)) if tss else None

    _SPECIAL = frozenset({SUCCESS_NODE_ID, FAILURE_NODE_ID})
    node_failure_rates: list[tuple[int, str, float]] = []
    for nid, node in _active_graph.nodes.items():
        if nid in _SPECIAL:
            continue
        outgoing = _active_graph.transition_probs.get(nid, {})
        total_weight = sum(outgoing.values())
        if total_weight == 0:
            continue
        fail_weight = outgoing.get(FAILURE_NODE_ID, 0.0)
        node_failure_rates.append((nid, node.name, fail_weight / total_weight))
    node_failure_rates.sort(key=lambda t: -t[2])
    top_failure_nodes = [(name, nid, rate) for nid, name, rate in node_failure_rates]

    fa_left, fa_right = st.columns(2)
    with fa_left:
        top5 = top_failure_nodes[:5]
        if top5:
            fig_fail = go.Figure(go.Bar(
                x=[r for _, _, r in top5],
                y=[name for name, _, _ in top5],
                orientation="h",
                marker_color="#d62728",
            ))
            fig_fail.update_layout(
                title="Nodes most likely to lead to failure",
                height=280, margin=dict(l=0, r=20, t=40, b=20),
                xaxis=dict(title="Failure rate", range=[0, 1]),
                yaxis=dict(title=None, autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_fail, use_container_width=True, key=f"{PFX}_fa_fail_nodes")
    with fa_right:
        st.metric("Overall success rate", f"{n_success / n_total:.0%}" if n_total else "—")
        if n_total:
            fig_pie = go.Figure(go.Pie(
                labels=["Success", "Failure"],
                values=[n_success, n_total - n_success],
                marker=dict(colors=["#2ca02c", "#d62728"]),
                hole=0.3, textinfo="percent+label",
            ))
            fig_pie.update_layout(
                height=280, margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key=f"{PFX}_fa_pie")

    # 4A: High-failure-rate nodes
    st.markdown("#### Nodes Most Likely to Lead to Failure")
    _fa_nodes = [(name, nid, rate) for name, nid, rate in top_failure_nodes if rate > 0]
    _FA_PAGE_SIZE = 3
    _fa_page_key = f"{PFX}_fa_node_page"
    _fa_page = st.session_state.get(_fa_page_key, 0)
    _fa_n_pages = max(1, (len(_fa_nodes) + _FA_PAGE_SIZE - 1) // _FA_PAGE_SIZE)

    if _fa_nodes:
        if _fa_n_pages > 1:
            _fac1, _fac2, _fac3 = st.columns([1, 4, 1])
            with _fac1:
                if st.button("← Prev", disabled=(_fa_page == 0), key=f"{PFX}_fa_prev"):
                    st.session_state[_fa_page_key] = max(0, _fa_page - 1)
                    st.rerun()
            _fac2.caption(f"Nodes {_fa_page * _FA_PAGE_SIZE + 1}–{min((_fa_page + 1) * _FA_PAGE_SIZE, len(_fa_nodes))} of {len(_fa_nodes)}")
            with _fac3:
                if st.button("Next →", disabled=(_fa_page >= _fa_n_pages - 1), key=f"{PFX}_fa_next"):
                    st.session_state[_fa_page_key] = min(_fa_n_pages - 1, _fa_page + 1)
                    st.rerun()

        for node_id, node_name, fail_rate in _fa_nodes[_fa_page * _FA_PAGE_SIZE:(_fa_page + 1) * _FA_PAGE_SIZE]:
            with st.expander(f"**{node_name}** — {fail_rate:.0%} failure rate", expanded=False):
                _clips = _failure_eps_for_node(node_id)
                _CP_SIZE = 6
                _cp_key = f"{PFX}_fa_cp_{node_id}"
                _cp = st.session_state.get(_cp_key, 0)
                _cn_pages = max(1, (len(_clips) + _CP_SIZE - 1) // _CP_SIZE)
                _page_clips = _clips[_cp * _CP_SIZE:(_cp + 1) * _CP_SIZE]
                if _cn_pages > 1:
                    cc1, cc2, cc3 = st.columns([1, 4, 1])
                    with cc1:
                        if st.button("← Prev", disabled=(_cp == 0), key=f"{PFX}_cp_prev_{node_id}"):
                            st.session_state[_cp_key] = max(0, _cp - 1)
                            st.rerun()
                    cc2.caption(f"Clips {_cp * _CP_SIZE + 1}–{min((_cp + 1) * _CP_SIZE, len(_clips))} of {len(_clips)}")
                    with cc3:
                        if st.button("Next →", disabled=(_cp >= _cn_pages - 1), key=f"{PFX}_cp_next_{node_id}"):
                            st.session_state[_cp_key] = min(_cn_pages - 1, _cp + 1)
                            st.rerun()
                for row_start in range(0, len(_page_clips), 3):
                    row = _page_clips[row_start:row_start + 3]
                    cols = st.columns(3)
                    for col, (ep_idx, ts_start, ts_end) in zip(cols, row):
                        entry = _mp4_entry(ep_idx)
                        if entry is None:
                            continue
                        with col:
                            st.caption(f"Episode {ep_idx}")
                            mp4_player(
                                mp4_dir / entry["path"],
                                key=f"{PFX}_fa_vid_{node_id}_{ep_idx}",
                                max_height_px=180,
                                slice_start=ts_start,
                                slice_end=ts_end,
                                total_frames=entry.get("frame_count"),
                            )

    # 4B: Failure traces by initial behavior
    st.markdown("#### Which Initial Behaviors Lead to Failure?")
    _IC_SPECIAL = frozenset({START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})
    _fail_ep_ids: set[int] = {
        ep.get("index") for ep in index.get("episodes", []) if ep.get("success") is False
    }
    _all_graph_paths = _active_graph.enumerate_paths(max_paths=100)
    _failure_graph_paths = [
        (path, prob) for path, prob, _ in _all_graph_paths
        if path and path[-1] == FAILURE_NODE_ID
    ]
    _ic_groups: dict[int, dict] = {}
    for _path, _prob in _failure_graph_paths:
        _first_b = next((n for n in _path if n not in _IC_SPECIAL), None)
        if _first_b is None:
            continue
        _matching = _ic_episodes_for_path(_path, _active_labels, metadata)
        _fail_matching = [ep for ep in _matching if ep in _fail_ep_ids]
        if not _fail_matching:
            continue
        if _first_b not in _ic_groups:
            _ic_groups[_first_b] = {"paths": [], "episodes": set()}
        _ic_groups[_first_b]["paths"].append((_path, _prob, len(_fail_matching)))
        _ic_groups[_first_b]["episodes"].update(_fail_matching)

    _sorted_ic = sorted(_ic_groups.items(), key=lambda x: -len(x[1]["episodes"]))
    if not _sorted_ic:
        st.info("No failure paths with initial behavior grouping found.")
    else:
        for _fnid, _grp in _sorted_ic:
            _fnode = _active_graph.nodes.get(_fnid)
            _fnode_name = _fnode.name if _fnode else str(_fnid)
            _grp_eps = sorted(_grp["episodes"])
            _top_fps = sorted(_grp["paths"], key=lambda x: -x[2])
            with st.expander(f"**{_fnode_name}** — {len(_grp_eps)} failing episode(s)", expanded=False):
                st.caption("Failure paths through this starting behavior:")
                for _fp, _fprob, _fn_ep in _top_fps[:6]:
                    _fp_str = " → ".join(
                        ("START" if n == START_NODE_ID
                         else "✗ FAILURE" if n == FAILURE_NODE_ID
                         else _active_graph.nodes[n].name if n in _active_graph.nodes else str(n))
                        for n in _fp if n != END_NODE_ID
                    )
                    st.caption(f"  {_fp_str}  ({_fn_ep} ep)")
                _ic_show = _grp_eps[:9]
                _ic_cols = st.columns(3)
                for _ci, _ep_idx in enumerate(_ic_show):
                    _entry = _mp4_entry(_ep_idx)
                    if _entry is None:
                        continue
                    with _ic_cols[_ci % 3]:
                        _ts = _ep_behavior_range(_ep_idx, _fnid)
                        st.caption(f"Ep {_ep_idx}")
                        mp4_player(
                            mp4_dir / _entry["path"],
                            key=f"{PFX}_ic_vid_{_fnid}_{_ep_idx}",
                            max_height_px=180,
                            slice_start=_ts[0] if _ts else None,
                            slice_end=_ts[1] if _ts else None,
                            total_frames=_entry.get("frame_count"),
                        )

    st.info(
        "Use what you found above to guide your data collection choices. "
        "Nodes with high failure rates suggest targeted collection around those behaviors."
    )

    st.divider()
    if expired or st.button("Proceed to Data Collection →", type="primary"):
        advance_step(2, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Data Collection
# ─────────────────────────────────────────────────────────────────────────────
elif step == 3:
    st.title("Design Your Data Collection Strategy")
    st.markdown(
        "Based on what you observed, allocate your **{} demo budget** across the strategies below. "
        "Each strategy targets a specific data collection protocol.".format(total_budget)
    )

    allocations = render_strategy_allocator(
        strategies,
        total_budget=total_budget,
        allocation_step=alloc_step,
        key_prefix=f"{PFX}_alloc",
        demo_videos_dir=demo_videos_dir,
    )
    render_strategy_summary(allocations, strategies, total_budget)

    st.divider()
    if st.button("Continue to Survey →", type="primary"):
        st.session_state[f"{PFX}_allocations"] = allocations
        advance_step(3, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Survey + Submit
# ─────────────────────────────────────────────────────────────────────────────
elif step == 4:
    st.title("Survey")

    st.header("1 — Behavior-Graph Survey")
    likert_graph = render_block1_graph_interaction(key_prefix=f"{PFX}_likert")

    st.divider()
    st.header("2 — Strategy-Selection Survey")
    likert_strategy = render_block2_strategy(key_prefix=f"{PFX}_likert")

    st.divider()
    st.header("3 — NASA Task Load Index")
    tlx_responses = render_nasa_tlx(key_prefix=f"{PFX}_tlx")

    st.divider()
    st.header("4 — Final Assessment")
    likert_final = render_block3_final(
        key_prefix=f"{PFX}_likert", include_graph_questions=True,
    )

    st.divider()
    notes = st.text_area(
        "Any additional notes or reasoning about your choices",
        value="",
        height=120,
        placeholder="e.g. I noticed failures often occur when the arm approaches from the left, so I focused on...",
    )

    if st.button("Submit", type="primary"):
        record_step_exit(4, STEP_KEY)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "participant_id": participant_id,
            "group": "B",
            "session": session_choice,
            "allocations": st.session_state.get(f"{PFX}_allocations", {}),
            "nasa_tlx": tlx_responses,
            "likert_graph": likert_graph,
            "likert_strategy": likert_strategy,
            "likert_final": likert_final,
            "notes": notes,
            "timestamp": timestamp,
            "step_durations_seconds": get_step_durations(STEP_KEY),
        }

        store = get_store(mp4_dir / "study_responses")
        response_id = store.save(result)
        st.success(f"Response saved (ID: {response_id})")
        st.download_button(
            label="Download your response",
            data=json.dumps(result, indent=2),
            file_name=f"group_b_{timestamp}.json",
            mime="application/json",
        )
        st.balloons()
