"""Shared trajectory-tree rendering — used by both the comparison app and the
main Policy Doctor visualizer's Behavior Graph tab.

Renders one of {native SVG node-edge tree, Sunburst, Icicle, Treemap} given
a (labels, metadata) pair, plus a "Hide branches reaching fewer than N
episodes" slider and an effectively-uncapped max-depth control. The native
SVG variant reuses the existing clickable-graph component (with curved edges,
drag, reset, and the click-to-explore video panel).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base64
import io

import numpy as np
import streamlit as st
import streamlit.components.v1 as _components

_THUMB_STRIP = _components.declare_component(
    "thumb_strip",
    path=str(Path(__file__).parent / "_thumb_strip"),
)


@st.cache_data(show_spinner=False)
def _thumbnail_b64(video_path_str: str) -> str:
    """Extract first frame of a video as a base64 JPEG thumbnail."""
    try:
        import imageio
        from PIL import Image
        reader = imageio.get_reader(video_path_str)
        frame = reader.get_data(0)
        reader.close()
        img = Image.fromarray(frame).resize((96, 64), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    BehaviorNode,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.behaviors import graph_simplification as gs
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS


TREE_VIEW_OPTIONS = ["native_svg", "sunburst", "icicle"]
TREE_VIEW_LABELS = {
    "native_svg": "Trajectory tree",
    "sunburst": "Sunburst",
    "icicle": "Icicle",
}


def render_trajectory_tree(
    labels: np.ndarray,
    metadata: List[Dict],
    *,
    view_mode: str = "native_svg",
    min_branch: int = 2,
    max_depth_cap: int = 500,
    color_mode: str = "outcome",
    node_values: Optional[Dict[int, float]] = None,
    cluster_names: Optional[Dict[int, str]] = None,
    mp4_dir: Optional[Path] = None,
    mp4_index: Optional[Dict] = None,
    height: int = 600,
    level: str = "rollout",
    key_prefix: str = "tree",
    show_stats: bool = True,
    theme: str = "dark",
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 24.0,
) -> None:
    """Render the trajectory tree. Self-contained: produces controls (none —
    the caller is expected to provide the min_branch / view selectors)
    and the chart + optional stats below it.
    """
    cluster_names = cluster_names or {}

    # Build the trie
    nodes = gs.build_trajectory_tree(labels, metadata, level=level)
    for nd in nodes:
        cid = nd["cluster_id"]
        if cid >= 0 and cluster_names.get(int(cid)):
            nd["label"] = cluster_names[int(cid)]

    nodes_f = [
        nd for nd in nodes
        if nd["n_episodes"] >= int(min_branch) and nd["depth"] <= int(max_depth_cap)
    ]

    # Auto-prune unreachable + false-terminal nodes to a fixed point.
    _TERM = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID}
    n_pruned = 0
    while True:
        keep = {tuple(nd["path"]) for nd in nodes_f}
        children: Dict[Tuple, List[Tuple]] = defaultdict(list)
        for nd in nodes_f:
            if nd["parent_path"] is not None and tuple(nd["parent_path"]) in keep:
                children[tuple(nd["parent_path"])].append(tuple(nd["path"]))
        new_f = []
        by_p = {tuple(nd["path"]): nd for nd in nodes_f}
        for nd in nodes_f:
            p = tuple(nd["path"])
            # Reachable from root
            anc = nd["parent_path"]
            ok = True
            while anc is not None:
                if tuple(anc) not in keep:
                    ok = False
                    break
                anc = tuple(anc)[:-1] if len(anc) > 0 else None
                if anc == ():
                    break
            if not ok:
                continue
            # Reaches a terminal
            if nd["cluster_id"] in _TERM:
                new_f.append(nd)
                continue
            stack = [p]
            seen = set()
            found = False
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for cp in children.get(cur, []):
                    if cp not in by_p:
                        continue
                    if by_p[cp]["cluster_id"] in _TERM:
                        found = True
                        break
                    stack.append(cp)
                if found:
                    break
            if found:
                new_f.append(nd)
        if len(new_f) == len(nodes_f):
            break
        n_pruned += len(nodes_f) - len(new_f)
        nodes_f = new_f

    if n_pruned > 0:
        st.caption(f"{n_pruned} nodes pruned.")
    if not nodes_f:
        st.warning("No nodes match the filter; lower 'min branch'.")
        return

    if view_mode == "native_svg":
        # Path buttons + episode summary at the TOP, then graph | video columns.
        if show_stats:
            _render_stats(
                nodes, nodes_f, cluster_names,
                key_prefix=key_prefix,
                interactive=True,
            )
        _render_native_svg(
            nodes_f, labels, metadata,
            cluster_names=cluster_names,
            mp4_dir=mp4_dir, mp4_index=mp4_index,
            height=height, level=level, key_prefix=key_prefix,
            color_mode=color_mode, node_values=node_values or {},
            theme=theme, edge_style=edge_style,
            edge_width_slope=edge_width_slope,
            node_size_slope=node_size_slope,
        )
    else:
        _render_plotly(
            nodes_f, view_mode, color_mode, node_values or {},
            height, key=f"{key_prefix}_plotly",
        )
        if show_stats:
            _render_stats(
                nodes, nodes_f, cluster_names,
                key_prefix=key_prefix,
                interactive=False,
            )


# ── Native SVG branch ────────────────────────────────────────────────────────

def _render_native_svg(
    nodes_f: List[Dict],
    labels: np.ndarray,
    metadata: List[Dict],
    *,
    cluster_names: Dict[int, str],
    mp4_dir: Optional[Path],
    mp4_index: Optional[Dict],
    height: int,
    level: str,
    key_prefix: str,
    color_mode: str = "outcome",
    node_values: Optional[Dict[int, float]] = None,
    theme: str = "dark",
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 24.0,
) -> None:
    node_values = node_values or {}
    from policy_doctor.streamlit_app.user_study.graph_explorer import render_graph_full_width
    by_path_all: Dict[Tuple, Dict] = {tuple(nd["path"]): nd for nd in nodes_f}

    # Assign synthetic IDs. Each leaf — including each terminal — gets a
    # unique id; color/symbol overrides keep terminals visually distinct AND
    # keep same-cluster nodes the same color across the tree.
    path_to_id: Dict[Tuple, int] = {}
    symbol_override: Dict[int, str] = {}
    color_override: Dict[int, str] = {}
    next_id = 100_000

    # Helpers for value / outcome color modes.
    def _v_for_cluster(cid: int, n_success: int, n_episodes: int) -> float:
        if cid == SUCCESS_NODE_ID: return 1.0
        if cid == FAILURE_NODE_ID: return -1.0
        if cid in (END_NODE_ID, START_NODE_ID): return 0.0
        return float(node_values.get(int(cid), 0.0))
    _BINS = ["#d62728", "#ff7f0e", "#e8c32a", "#9dc95d", "#2ca02c"]
    def _diverging(t: float) -> str:
        return _BINS[min(4, int(max(0.0, min(1.0, t)) * 5))]
    # Pre-compute V range for value mode. Exclude SUCCESS / FAILURE /
    # END / START terminals — their hardcoded ±1 values dominate the
    # range and squash every behavioral cluster into the grey middle.
    v_range = 1.0
    if color_mode == "value":
        _terminal_ids = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID}
        v_range = max(
            (abs(_v_for_cluster(nd["cluster_id"], nd["n_success"], nd["n_episodes"]))
             for nd in nodes_f
             if nd["cluster_id"] not in _terminal_ids),
            default=1.0,
        ) or 1.0

    for nd in nodes_f:
        p = tuple(nd["path"])
        cid = nd["cluster_id"]
        if cid == START_NODE_ID:
            path_to_id[p] = START_NODE_ID
        else:
            path_to_id[p] = next_id
            # Symbol stays the same regardless of color mode.
            if cid == SUCCESS_NODE_ID: symbol_override[next_id] = "star"
            elif cid == FAILURE_NODE_ID: symbol_override[next_id] = "x"
            elif cid == END_NODE_ID:    symbol_override[next_id] = "square"
            # Color depends on color_mode.
            is_special = cid in (SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID)
            if color_mode == "outcome" and not is_special:
                rate = nd["n_success"] / max(1, nd["n_episodes"])
                color_override[next_id] = _diverging(rate)
            elif color_mode == "value" and not is_special:
                v = _v_for_cluster(cid, nd["n_success"], nd["n_episodes"])
                color_override[next_id] = _diverging(0.5 + v / (2 * v_range))
            elif cid == SUCCESS_NODE_ID: color_override[next_id] = "#2ca02c"
            elif cid == FAILURE_NODE_ID: color_override[next_id] = "#d62728"
            elif cid == END_NODE_ID:     color_override[next_id] = "#888888"
            else:
                # id mode
                color_override[next_id] = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            next_id += 1

    # Aggregate stats per synth id (with unique terminals per leaf, agg is
    # essentially a one-to-one map).
    agg: Dict[int, Dict] = {}
    for nd in nodes_f:
        p = tuple(nd["path"])
        nid = path_to_id[p]
        if nid not in agg:
            cid = nd["cluster_id"]
            if cid == SUCCESS_NODE_ID: name = "SUCCESS"
            elif cid == FAILURE_NODE_ID: name = "FAILURE"
            elif cid == END_NODE_ID: name = "END"
            elif cid == START_NODE_ID: name = "START"
            else:
                name = cluster_names.get(int(cid), f"Behavior {cid}")
            agg[nid] = {
                "name": name, "n_episodes": 0,
                "n_success": 0, "n_failure": 0,
                "episode_indices": set(),
            }
        agg[nid]["n_episodes"] += nd["n_episodes"]
        agg[nid]["n_success"] += nd["n_success"]
        agg[nid]["n_failure"] += nd["n_failure"]
        agg[nid]["episode_indices"].update(nd.get("episode_indices", []))

    synth_nodes: Dict[int, BehaviorNode] = {}
    for nid, info in agg.items():
        synth_nodes[nid] = BehaviorNode(
            cluster_id=nid,
            name=info["name"],
            num_timesteps=info["n_episodes"],
            num_episodes=len(info["episode_indices"]),
            episode_indices=sorted(info["episode_indices"]),
        )

    # Build transitions
    synth_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for nd in nodes_f:
        p = tuple(nd["path"])
        if nd["parent_path"] is None:
            continue
        parent = tuple(nd["parent_path"])
        if parent not in path_to_id or p not in path_to_id:
            continue
        src = path_to_id[parent]
        tgt = path_to_id[p]
        if src == tgt:
            continue
        synth_counts[src][tgt] += nd["n_episodes"]
    synth_probs: Dict[int, Dict[int, float]] = {}
    for src, tgts in synth_counts.items():
        total = sum(tgts.values()) or 1
        synth_probs[src] = {t: c / total for t, c in tgts.items()}

    synth_graph = BehaviorGraph(
        nodes=synth_nodes,
        transition_counts={k: dict(v) for k, v in synth_counts.items()},
        transition_probs=synth_probs,
        num_episodes=agg.get(START_NODE_ID, {}).get("n_episodes", 0),
        level=level,
    )

    # Tree layout: split [0,1] x-range among children weighted by subtree
    # size; y = -depth (root at top).
    children_of: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for nd in nodes_f:
        if nd["parent_path"] is not None and tuple(nd["parent_path"]) in by_path_all:
            children_of[tuple(nd["parent_path"])].append(tuple(nd["path"]))
    for k in children_of:
        children_of[k].sort(key=lambda pp: by_path_all[pp]["cluster_id"])

    pos_paths: Dict[Tuple, Tuple[float, float]] = {}
    def _assign(path: Tuple, x_lo: float, x_hi: float, depth: int) -> None:
        pos_paths[path] = ((x_lo + x_hi) / 2, -depth)
        ch = children_of.get(path, [])
        if not ch:
            return
        weights = [by_path_all[c]["n_episodes"] for c in ch]
        total = sum(weights) or 1
        cur = x_lo
        for c, w in zip(ch, weights):
            span = (x_hi - x_lo) * w / total
            _assign(c, cur, cur + span, depth + 1)
            cur += span
    _assign((), 0.0, 1.0, 0)

    # Collapse positions by synth_id (each path is now unique → 1:1 map)
    pos_synth: Dict[int, Tuple[float, float]] = {}
    for path, (x, y) in pos_paths.items():
        pos_synth[path_to_id[path]] = (x, y)

    # Map [0,1] x → [-2.5, 2.5]; depth → y in SVG coords (smaller = top).
    max_d = max(abs(y) for _, y in pos_synth.values()) or 1
    pos_final: Dict[int, Tuple[float, float]] = {}
    for nid, (x, y) in pos_synth.items():
        pos_final[nid] = (-2.5 + 5.0 * x, -2.5 + 5.0 * ((-y) / max_d))

    # Synthetic per-window labels: each window's assignment is the synth id
    # of the tree node at the corresponding depth of its episode's
    # run-length-collapsed sequence. So clicking a tree node shows videos
    # of episodes that took that specific prefix.
    synth_labels = np.full(len(labels), -1, dtype=np.int64)
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    ep_groups: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for i, m in enumerate(metadata):
        c = int(labels[i])
        if c == -1:
            continue
        ep = m.get(ep_key, 0)
        ts = m.get("window_start", m.get("timestep", 0))
        ep_groups[ep].append((ts, c, i))
    for ep, lst in ep_groups.items():
        lst.sort()
        phases: List[Tuple[int, List[int]]] = []
        for _, c, i in lst:
            if phases and phases[-1][0] == c:
                phases[-1][1].append(i)
            else:
                phases.append((c, [i]))
        running_path: List[int] = []
        for c, idxs in phases:
            running_path.append(c)
            p = tuple(running_path)
            if p in path_to_id:
                nid = path_to_id[p]
                for i in idxs:
                    synth_labels[i] = nid

    # Save mapping so `_render_stats` can convert a clicked path's
    # cluster-id sequence back into synthetic node ids to highlight.
    st.session_state[f"{key_prefix}_path_to_id"] = {
        tuple(p): nid for p, nid in path_to_id.items()
    }
    # Also save synth_id → "START → A → B → …" string so the click
    # panels can show the full prefix in their header.
    _id_to_prefix: Dict[int, str] = {}
    for p, nid in path_to_id.items():
        chain = ["START"]
        for j in range(1, len(p) + 1):
            sub = path_to_id.get(tuple(p[:j]))
            if sub is not None and sub in agg:
                chain.append(agg[sub]["name"])
        _id_to_prefix[nid] = " → ".join(chain)
    st.session_state[f"{key_prefix}_id_to_prefix"] = _id_to_prefix
    _highlighted_path = st.session_state.get(f"{key_prefix}_highlighted_path")
    _fps = int((mp4_index or {}).get("fps", 10))

    col_g, col_v = st.columns([2, 1])
    with col_g:
        render_graph_full_width(
            graph=synth_graph,
            labels=synth_labels,
            metadata=metadata,
            mp4_dir=mp4_dir if mp4_dir is not None else Path("/tmp/_nonexistent"),
            mp4_index=mp4_index or {"episodes": []},
            key_prefix=key_prefix,
            min_edge_prob=0.0,
            pos=pos_final,
            symbol_override=symbol_override,
            color_override=color_override,
            highlighted_path=_highlighted_path,
            theme=theme,
            edge_style=edge_style,
            edge_width_slope=edge_width_slope,
            node_size_slope=node_size_slope,
            suppress_video_panel=True,
        )
    with col_v:
        _render_right_video_panel(
            labels=synth_labels,
            metadata=metadata,
            mp4_dir=mp4_dir,
            mp4_index=mp4_index or {"episodes": []},
            key_prefix=key_prefix,
            fps=_fps,
            graph=synth_graph,
        )


# ── Plotly branches (sunburst / icicle / treemap) ────────────────────────────
# All Plotly figure construction is delegated to policy_doctor.plotting.plotly
# per the repo's Streamlit/plotting separation convention.

def _render_plotly(
    nodes_f: List[Dict],
    view_mode: str,
    color_mode: str,
    node_values: Dict[int, float],
    height: int,
    key: str,
) -> None:
    from policy_doctor.plotting.plotly.trajectory_tree import (
        create_trajectory_sunburst,
        create_trajectory_icicle,
        create_trajectory_treemap,
    )
    kwargs = dict(color_mode=color_mode, node_values=node_values, height=height)
    if view_mode == "sunburst":
        fig = create_trajectory_sunburst(nodes_f, **kwargs)
    elif view_mode == "icicle":
        fig = create_trajectory_icicle(nodes_f, **kwargs)
    else:  # treemap
        fig = create_trajectory_treemap(nodes_f, **kwargs)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Stats below the chart ────────────────────────────────────────────────────

def _render_stats(
    nodes_all: List[Dict],
    nodes_filtered: List[Dict],
    cluster_names: Dict[int, str],
    *,
    key_prefix: str = "ttree",
    interactive: bool = False,
) -> None:
    """Top success / failure paths summary below the chart."""
    root = nodes_all[0]
    st.caption(
        f"{root['n_episodes']} episodes  ·  "
        f"{root['n_success']} success ({root['n_success']/max(1,root['n_episodes']):.0%})  ·  "
        f"{root['n_failure']} failure ({root['n_failure']/max(1,root['n_episodes']):.0%})  ·  "
        f"{len(nodes_filtered)}/{len(nodes_all)} tree nodes shown after filtering"
    )
    # Source leaves from the FILTERED tree so the list reflects what's
    # actually visible above. Paths pruned by min_branch don't show up.
    leaves = [
        nd for nd in nodes_filtered
        if nd["cluster_id"] in (SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID)
    ]
    succ_paths = sorted(
        [nd for nd in leaves if nd["cluster_id"] == SUCCESS_NODE_ID],
        key=lambda nd: -nd["n_episodes"],
    )[:5]
    fail_paths = sorted(
        [nd for nd in leaves if nd["cluster_id"] == FAILURE_NODE_ID],
        key=lambda nd: -nd["n_episodes"],
    )[:5]

    def _format_path(nd):
        # nd["path"] omits the root START node (it lives in a separate
        # entry at path=()), but visually a path from START makes more
        # sense, so prepend it.
        disp = ["START"]
        for cid in nd["path"]:
            if cid == SUCCESS_NODE_ID: disp.append("✓")
            elif cid == FAILURE_NODE_ID: disp.append("✗")
            elif cid == END_NODE_ID: disp.append("END")
            else:
                nm = cluster_names.get(int(cid))
                disp.append(nm if nm else f"B{cid}")
        return " → ".join(disp)

    total_eps = max(1, root["n_episodes"])

    _current = st.session_state.get(f"{key_prefix}_highlighted_path")

    def _maybe_button(label: str, key: str, leaf_path: tuple, nd: dict) -> None:
        """Render a button when interactive, otherwise a markdown bullet.

        Clicking a path highlights it in the graph and reveals its videos in
        the right panel; clicking the active path again deselects it.
        """
        if not interactive:
            st.markdown(f"- {label}")
            return
        path_to_id = st.session_state.get(f"{key_prefix}_path_to_id", {})
        synth_path: list[int] = []
        start_nid = path_to_id.get(())
        if start_nid is not None:
            synth_path.append(start_nid)
        for i in range(1, len(leaf_path) + 1):
            nid = path_to_id.get(tuple(leaf_path[:i]))
            if nid is not None:
                synth_path.append(nid)
        is_active = (_current is not None and list(_current) == synth_path)
        if st.button(
            label, key=key, use_container_width=True,
            type=("primary" if is_active else "secondary"),
        ):
            if is_active:
                st.session_state.pop(f"{key_prefix}_highlighted_path", None)
                st.session_state.pop(f"{key_prefix}_path_ep_list", None)
                st.session_state.pop(f"{key_prefix}_path_label", None)
            elif synth_path:
                st.session_state[f"{key_prefix}_highlighted_path"] = synth_path
                st.session_state[f"{key_prefix}_path_ep_list"] = sorted(
                    nd.get("episode_indices", [])
                )
                st.session_state[f"{key_prefix}_path_label"] = _format_path(nd)
                # Store (synth_node_id, label) pairs for multi-segment annotation.
                # We use synth node IDs (not raw cluster IDs) because the right
                # panel receives synth_labels, not the original labels array.
                _term = {SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID, START_NODE_ID}
                _path_to_id = st.session_state.get(f"{key_prefix}_path_to_id", {})
                _path_synth_ids = []
                for _depth in range(1, len(nd["path"]) + 1):
                    _prefix = nd["path"][:_depth]
                    _cid = _prefix[-1]
                    if _cid in _term:
                        continue
                    _nid = _path_to_id.get(tuple(_prefix))
                    if _nid is not None:
                        _lbl = cluster_names.get(int(_cid), f"B{_cid}")
                        _path_synth_ids.append((_nid, _lbl))
                st.session_state[f"{key_prefix}_path_synth_ids"] = _path_synth_ids
                # Clear any node selection so the right panel shows path videos
                st.session_state.pop(f"{key_prefix}_graph_selected", None)
            st.rerun()

    c_s, c_f = st.columns(2)
    with c_s:
        st.markdown("**Top success paths**" + ("  (click to highlight)" if interactive else ""))
        for i, nd in enumerate(succ_paths):
            pct = nd["n_episodes"] / total_eps * 100
            label = f"( {nd['n_episodes']:>3} eps · {pct:.0f}% )  {_format_path(nd)}"
            _maybe_button(label, key=f"{key_prefix}_succ_path_{i}", leaf_path=tuple(nd["path"]), nd=nd)
    with c_f:
        st.markdown("**Top failure paths**" + ("  (click to highlight)" if interactive else ""))
        for i, nd in enumerate(fail_paths):
            pct = nd["n_episodes"] / total_eps * 100
            label = f"( {nd['n_episodes']:>3} eps · {pct:.0f}% )  {_format_path(nd)}"
            _maybe_button(label, key=f"{key_prefix}_fail_path_{i}", leaf_path=tuple(nd["path"]), nd=nd)


# ── Right-column single-video panel ──────────────────────────────────────────

_SEG_COLORS_DEFAULT = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]
# Okabe-Ito categorical — safe for deuteranopia/protanopia
_SEG_COLORS_CB = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]


def _seg_colors() -> List[str]:
    return (
        _SEG_COLORS_CB
        if st.session_state.get("colorblind_mode", False)
        else _SEG_COLORS_DEFAULT
    )


def _render_right_video_panel(
    labels: np.ndarray,
    metadata: List[Dict],
    mp4_dir: Optional[Path],
    mp4_index: Dict,
    key_prefix: str,
    fps: int = 10,
    graph=None,
) -> None:
    """One-video-at-a-time panel for the right column of the trajectory tree."""
    from policy_doctor.streamlit_app.user_study.graph_explorer import (
        _episodes_for_node,
        _find_mp4_episode,
        _render_edge_panel,
    )

    selected_node = st.session_state.get(f"{key_prefix}_graph_selected")
    selected_edge = st.session_state.get(f"{key_prefix}_graph_selected_edge")
    path_eps: Optional[List[int]] = st.session_state.get(f"{key_prefix}_path_ep_list")
    path_label: str = st.session_state.get(f"{key_prefix}_path_label", "")

    if selected_edge is not None and graph is not None and mp4_dir is not None:
        src_id, tgt_id = selected_edge
        _render_edge_panel(
            src_id, tgt_id, graph, labels, metadata,
            mp4_dir, mp4_index, key_prefix,
        )
    elif selected_node is not None:
        ep_slices = _episodes_for_node(int(selected_node), labels, metadata)
        ep_slices_by_idx: Dict[int, Tuple[int, int]] = {e[0]: (e[1], e[2]) for e in ep_slices}
        all_eps = sorted(ep_slices_by_idx.keys())
        # Terminal nodes (SUCCESS/FAILURE) are never assigned synth_labels so
        # ep_slices will be empty — fall back to the graph node's episode list.
        if not all_eps and graph is not None:
            gnode = graph.nodes.get(int(selected_node))
            if gnode is not None:
                all_eps = sorted(gnode.episode_indices)
        _id_to_prefix = st.session_state.get(f"{key_prefix}_id_to_prefix", {})
        title = _id_to_prefix.get(int(selected_node), f"Node {selected_node}")
        st.caption(f"**{title}**")
        _show_one_video_panel(
            all_eps, ep_slices_by_idx, mp4_dir, mp4_index,
            f"{key_prefix}_node_{selected_node}", fps,
        )
    elif path_eps:
        st.caption(f"**{path_label}**" if path_label else "**Selected path**")
        # Build per-episode multi-segment data using synth node IDs (which
        # match the synth_labels array passed as `labels` to this panel).
        path_synth_ids: List = st.session_state.get(f"{key_prefix}_path_synth_ids", [])
        ep_segs_by_idx: Dict[int, List] = {}
        path_eps_set = set(path_eps)
        for i, (nid, lbl) in enumerate(path_synth_ids):
            _sc = _seg_colors()
            col = _sc[i % len(_sc)]
            for ep_idx, ts_s, ts_e in _episodes_for_node(nid, labels, metadata):
                if ep_idx in path_eps_set:
                    ep_segs_by_idx.setdefault(ep_idx, []).append((ts_s, ts_e, lbl, col))
        _show_one_video_panel(
            path_eps, {}, mp4_dir, mp4_index,
            f"{key_prefix}_path", fps,
            ep_segs_by_idx=ep_segs_by_idx,
        )
    else:
        st.info("Click a node or a path button to see videos here.")


def _show_one_video_panel(
    ep_list: List[int],
    ep_slices_by_idx: Dict[int, Tuple[int, int]],
    mp4_dir: Optional[Path],
    mp4_index: Dict,
    key_prefix: str,
    fps: int,
    ep_segs_by_idx: Optional[Dict[int, List]] = None,
) -> None:
    """Render one episode video with prev / next controls below."""
    from policy_doctor.streamlit_app.user_study.graph_explorer import _find_mp4_episode
    from policy_doctor.streamlit_app.components.mp4_player import mp4_player

    n = len(ep_list)
    if n == 0:
        st.info("No episodes for this selection.")
        return

    vp_key = f"{key_prefix}_vid_page"
    # Reset to 0 when the episode list changes (different path/node selected)
    list_sig = (ep_list[0] if ep_list else -1, n)
    sig_key = f"{key_prefix}_vid_list_sig"
    if st.session_state.get(sig_key) != list_sig:
        st.session_state[sig_key] = list_sig
        st.session_state[vp_key] = 0
    vp = max(0, min(st.session_state.get(vp_key, 0), n - 1))
    ep_idx = ep_list[vp]
    ep_entry = _find_mp4_episode(ep_idx, mp4_index)

    if ep_entry and mp4_dir:
        ts_range = ep_slices_by_idx.get(ep_idx)
        segs = (ep_segs_by_idx or {}).get(ep_idx)
        # Extend the last segment to the end of the video so there's no gap.
        if segs:
            _fc = ep_entry.get("frame_count")
            if _fc:
                segs = [*segs[:-1], (segs[-1][0], _fc, segs[-1][2], segs[-1][3])]
        success = ep_entry.get("success")
        status = "✓ Success" if success is True else "✗ Failure" if success is False else ""
        st.caption(f"Episode {ep_idx} — {status}")
        # Extend single-behavior slice_end by a small buffer to cover the
        # last window (max_ts is window_start, actual coverage is +W-1 frames).
        _fc = ep_entry.get("frame_count")
        _slice_end = ts_range[1] if ts_range else None
        if _slice_end is not None and _fc:
            _slice_end = min(_slice_end + fps, _fc)
        mp4_player(
            mp4_dir / ep_entry["path"],
            key=f"{key_prefix}_vid_{ep_idx}",
            max_height_px=220,
            slice_start=ts_range[0] if ts_range else None,
            slice_end=_slice_end,
            total_frames=ep_entry.get("frame_count"),
            fps=fps,
            segments=segs,
        )
    else:
        st.warning(f"No video found for episode {ep_idx}.")

    # Horizontal thumbnail strip
    thumbs = []
    for i, ep in enumerate(ep_list):
        ep_e = _find_mp4_episode(ep, mp4_index)
        icon = ("✓" if ep_e and ep_e.get("success") is True
                else "✗" if ep_e and ep_e.get("success") is False else "•")
        b64 = ""
        if ep_e and mp4_dir:
            b64 = _thumbnail_b64(str(mp4_dir / ep_e["path"]))
        thumbs.append({"b64": b64, "label": f"{icon} {ep}"})

    clicked = _THUMB_STRIP(
        thumbs=thumbs,
        selected=vp,
        key=f"{key_prefix}_thumb_strip",
    )
    if clicked is not None:
        new_vp = int(clicked[0])
        if new_vp != vp:
            st.session_state[vp_key] = new_vp
            st.rerun()
