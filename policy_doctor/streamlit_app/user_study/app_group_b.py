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
from policy_doctor.streamlit_app.user_study.nasa_tlx import render_nasa_tlx
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


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


# Auto-load on session change (no Load button click required).
if st.session_state.get("gb_loaded_session") != session_choice:
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
        st.session_state["gb_loaded_session"] = session_choice

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
    st.error("Failed to load session — check the sidebar for errors.")
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
    "**Click any node** to see example videos and outgoing transitions from that mode."
)

highlighted_path = st.session_state.get("gb_pex_highlighted_path")

# ── Visualization picker + filter ───────────────────────────────────────────
_VIZ_OPTIONS = [
    "tree_native_svg",
    "tree_sunburst",
    "tree_icicle",
    "markov_svg_bfs",
    "markov_svg_temporal",
]
_VIZ_LABELS = {
    "tree_native_svg":     "Trajectory tree",
    "tree_sunburst":       "Sunburst",
    "tree_icicle":         "Icicle",
    "markov_svg_bfs":      "Markov graph — BFS-layered",
    "markov_svg_temporal": "Markov graph — temporal mean",
}

_c_viz, _c_color = st.columns([2, 1])
with _c_viz:
    _viz_type = st.selectbox(
        "Visualization",
        options=_VIZ_OPTIONS,
        format_func=lambda v: _VIZ_LABELS[v],
        index=0,
        key="gb_viz_type",
    )
_is_tree = _viz_type.startswith("tree_")
with _c_color:
    if _is_tree:
        _color_opts = ["outcome", "id"]
        _color_labels = {"outcome": "Outcome (success rate)", "id": "Cluster ID"}
    else:
        _color_opts = ["id", "timesteps"]
        _color_labels = {"id": "Cluster ID", "timesteps": "Timestep count"}
    _color_by = st.selectbox(
        "Color nodes by",
        options=_color_opts,
        format_func=lambda v: _color_labels[v],
        index=0,
        key="gb_color_by",
    )

_min_branch = st.slider(
    "Hide transitions where count(s, s′) < N",
    1, 50, 2,
    key="gb_min_branch",
    help=(
        "count(s, s′) is the number of rollouts in which the transition s → s′ "
        "was observed. Edges with fewer than N observations are hidden, and "
        "any nodes that become unreachable from START are pruned as a "
        "consequence."
    ),
)

# Stats summary
_active_graph, _active_labels = graph, labels
_excluded_nodes = compute_pruned_graph_nodes(
    _active_graph, min_visit_prob=0.0, n_total=_active_graph.num_episodes,
    min_edge_count=int(_min_branch),
)
_n_nodes_total = len(_active_graph.nodes)
_n_edges_total = sum(len(t) for t in _active_graph.transition_counts.values())
_n_edges_hidden = sum(
    1
    for src, targets in _active_graph.transition_counts.items()
    for tgt, cnt in targets.items()
    if src in _excluded_nodes or tgt in _excluded_nodes or cnt < int(_min_branch)
)
st.caption(
    f"Hidden {_n_edges_hidden} / {_n_edges_total} edges  ·  "
    f"{len(_excluded_nodes)} / {_n_nodes_total} nodes pruned"
)

# ── Dispatch ─────────────────────────────────────────────────────────────────
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
        key_prefix="gb_tree",
    )
else:
    # Markov view with color override matching the demo's logic
    _color_override = None
    if _color_by == "timesteps":
        _SPECIAL = {START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID}
        try:
            import plotly.express as _px
            _viridis = _px.colors.sequential.Viridis
        except Exception:
            _viridis = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
        _ts_max = max(
            (np.log1p(_active_graph.nodes[nid].num_timesteps)
             for nid in _active_graph.nodes if nid not in _SPECIAL),
            default=1.0,
        ) or 1.0
        _color_override = {}
        for nid in _active_graph.nodes:
            if nid in _SPECIAL:
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
        key_prefix="gb_gex",
        highlighted_path=highlighted_path,
        excluded_node_ids=_excluded_nodes,
        min_edge_count=int(_min_branch),
        pos=_pos,
        color_override=_color_override,
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
    graph=_active_graph,
    labels=_active_labels,
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
    top5 = top_failure_nodes[:5]

    if top5:
        fig_fail = go.Figure(go.Bar(
            x=[r for _, _, r in top5],
            y=[f"{name}" for name, _, _ in top5],
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

ep_key_meta = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"

# ── Shared helpers ─────────────────────────────────────────────────────────────
def _failure_eps_for_node(node_id: int) -> list[tuple[int, int, int]]:
    """All (ep_idx, ts_start, ts_end) triples in this node that belong to failing episodes."""
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
    result = []
    for ep_idx, tss in ep_slices.items():
        if not ep_success_map.get(ep_idx, True):
            result.append((ep_idx, min(tss), max(tss)))
    return sorted(result)

def _mp4_entry(ep_idx: int) -> Optional[dict]:
    for ep in index.get("episodes", []):
        if ep.get("index") == ep_idx:
            return ep
    return None

def _ep_behavior_range(ep_idx: int, node_id: int) -> Optional[tuple[int, int]]:
    """Timestep range of a specific behavior node within a specific episode."""
    tss = [
        m.get("window_start", m.get("timestep", 0))
        for i, m in enumerate(metadata)
        if m.get(ep_key_meta) == ep_idx and int(_active_labels[i]) == node_id
    ]
    return (min(tss), max(tss)) if tss else None

# ── 4A: High-failure-rate nodes (paginated) ───────────────────────────────────
st.markdown("#### 4A — Nodes Most Likely to Lead to Failure")
st.caption(
    "Each panel shows a behavior node where exits frequently go to FAILURE. "
    "Watch the highlighted clips to understand what the robot is doing in these states."
)

_fa_nodes = [(name, nid, rate) for name, nid, rate in top_failure_nodes if rate > 0]
_FA_PAGE_SIZE = 3
_fa_page_key = "gb_fa_node_page"
_fa_page = st.session_state.get(_fa_page_key, 0)
_fa_n_pages = max(1, (len(_fa_nodes) + _FA_PAGE_SIZE - 1) // _FA_PAGE_SIZE)

if _fa_n_pages > 1:
    _pc, _pp, _pn = st.columns([1, 3, 1])
    with _pc:
        if st.button("← Prev", disabled=(_fa_page == 0), key="gb_fa_prev"):
            st.session_state[_fa_page_key] = max(0, _fa_page - 1)
            st.rerun()
    _pp.caption(f"Nodes {_fa_page * _FA_PAGE_SIZE + 1}–{min((_fa_page + 1) * _FA_PAGE_SIZE, len(_fa_nodes))} of {len(_fa_nodes)}")
    with _pn:
        if st.button("Next →", disabled=(_fa_page >= _fa_n_pages - 1), key="gb_fa_next"):
            st.session_state[_fa_page_key] = min(_fa_n_pages - 1, _fa_page + 1)
            st.rerun()

_page_fa_nodes = _fa_nodes[_fa_page * _FA_PAGE_SIZE:(_fa_page + 1) * _FA_PAGE_SIZE]
_CLIPS_PER_PAGE = 6

for node_name, node_id, rate in _page_fa_nodes:
    fail_clips = _failure_eps_for_node(node_id)
    if not fail_clips:
        continue
    n_clips = len(fail_clips)
    clip_word = "clip" if n_clips == 1 else "clips"
    with st.expander(
        f"**{node_name}** — {rate:.0%} of exits go to failure  ({n_clips} {clip_word})",
        expanded=False,
    ):
        _cp_key = f"gb_fa_clip_page_{node_id}"
        _cp = st.session_state.get(_cp_key, 0)
        _cn_pages = max(1, (n_clips + _CLIPS_PER_PAGE - 1) // _CLIPS_PER_PAGE)
        _page_clips = fail_clips[_cp * _CLIPS_PER_PAGE:(_cp + 1) * _CLIPS_PER_PAGE]

        if _cn_pages > 1:
            _cc1, _cc2, _cc3 = st.columns([1, 3, 1])
            with _cc1:
                if st.button("← Prev", disabled=(_cp == 0), key=f"gb_fa_cp_prev_{node_id}"):
                    st.session_state[_cp_key] = max(0, _cp - 1)
                    st.rerun()
            _cc2.caption(f"Clips {_cp * _CLIPS_PER_PAGE + 1}–{min((_cp + 1) * _CLIPS_PER_PAGE, n_clips)} of {n_clips}")
            with _cc3:
                if st.button("Next →", disabled=(_cp >= _cn_pages - 1), key=f"gb_fa_cp_next_{node_id}"):
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
                        key=f"gb_fa_vid_{node_id}_{ep_idx}",
                        max_height_px=180,
                        slice_start=ts_start,
                        slice_end=ts_end,
                        total_frames=entry.get("frame_count"),
                    )

# ── 4B: Failure traces by initial behavior ────────────────────────────────────
st.divider()
st.markdown("#### 4B — Which Initial Behaviors Lead to Failure?")
st.markdown(
    "Failing episodes are grouped by the **first behavior the robot exhibits after starting**. "
    "This reveals whether specific initial arm positions or object configurations "
    "consistently funnel episodes toward failure. "
    "The orange bar marks the window where the first behavior occurs in each episode."
)

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

        with st.expander(
            f"**{_fnode_name}** — {len(_grp_eps)} failing episode(s)",
            expanded=False,
        ):
            st.caption("Failure paths through this starting behavior (ranked by episode count):")
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
                        key=f"gb_ic_vid_{_fnid}_{_ep_idx}",
                        max_height_px=180,
                        slice_start=_ts[0] if _ts else None,
                        slice_end=_ts[1] if _ts else None,
                        total_frames=_entry.get("frame_count"),
                    )

st.info(
    "Use what you found above to guide your choices in Step 5 below. "
    "For example, nodes that frequently exit to failure suggest targeted data collection "
    "around those behaviors (e.g. constrained initial positions or recovery demos)."
)

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

# ── Section 6: NASA Task Load Index ──────────────────────────────────────────
st.divider()
st.header("Step 6 — NASA Task Load Index")
tlx_responses = render_nasa_tlx(key_prefix="gb_tlx")

# ── Section 7: Submit ─────────────────────────────────────────────────────────
st.divider()
st.header("Step 7 — Submit")

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
        "nasa_tlx": tlx_responses,
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
