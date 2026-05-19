from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.streamlit_app.components.mp4_player import mp4_player
from policy_doctor.streamlit_app.user_study.graph_plot import render_graph_component


def _episodes_for_node(
    node_id: int,
    labels: np.ndarray,
    metadata: list[dict],
) -> list[tuple[int, int, int]]:
    ep_slices: dict[int, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        if int(labels[i]) == node_id:
            ep_idx = meta.get("rollout_idx", meta.get("demo_idx", -1))
            ts = meta.get("timestep", meta.get("window_start", 0))
            ep_slices[ep_idx].append(ts)
    result = []
    for ep_idx, timesteps in ep_slices.items():
        result.append((ep_idx, min(timesteps), max(timesteps)))
    return sorted(result, key=lambda x: x[0])


def _find_mp4_episode(ep_idx: int, mp4_index: dict) -> Optional[dict]:
    for ep in mp4_index.get("episodes", []):
        if ep.get("index") == ep_idx:
            return ep
    return None


def _episodes_for_edge(
    src_id: int,
    tgt_id: int,
    labels: np.ndarray,
    metadata: list[dict],
    graph: Optional[BehaviorGraph] = None,
) -> list[tuple[int, int, Optional[int], Optional[int]]]:
    """Find episodes containing the src -> tgt transition.

    Returns [(ep_idx, ts_src, ts_tgt, ts_tgt_end), ...] sorted by ep_idx.
      • ts_src: window_start of the src behavior
      • ts_tgt: window_start of the tgt behavior (None for X→FAILURE/SUCCESS,
        which extend the source bar to episode end and omit the target bar)
      • ts_tgt_end: window_start of the behavior AFTER tgt (None if tgt is last)
    """
    ep_key = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"

    ep_wins: dict[int, list[tuple[int, int]]] = {}
    ep_success: dict[int, Optional[bool]] = {}
    for i, m in enumerate(metadata):
        ep_idx = m.get(ep_key)
        if ep_idx is None:
            continue
        lab = int(labels[i])
        ts = m.get("window_start", m.get("timestep", 0))
        ep_wins.setdefault(ep_idx, []).append((ts, lab))
        if ep_idx not in ep_success:
            ep_success[ep_idx] = m.get("success")

    result: list[tuple[int, int, Optional[int], Optional[int]]] = []
    for ep_idx, wins in ep_wins.items():
        wins.sort()
        rle: list[tuple[int, int]] = []
        for ts, lab in wins:
            if lab == -1:
                continue
            if not rle or rle[-1][1] != lab:
                rle.append((ts, lab))
        if not rle:
            continue

        first_ts, first_lab = rle[0]
        last_ts, last_lab = rle[-1]
        success = ep_success.get(ep_idx)

        if src_id == START_NODE_ID:
            if tgt_id == first_lab:
                tgt_end = rle[1][0] if len(rle) > 1 else None
                result.append((ep_idx, first_ts, first_ts, tgt_end))
            continue
        if tgt_id == FAILURE_NODE_ID:
            if last_lab == src_id and success is False:
                result.append((ep_idx, last_ts, None, None))
            continue
        if tgt_id == SUCCESS_NODE_ID:
            if last_lab == src_id and success is True:
                result.append((ep_idx, last_ts, None, None))
            continue

        for i in range(len(rle) - 1):
            if rle[i][1] == src_id and rle[i + 1][1] == tgt_id:
                ts_tgt_end = rle[i + 2][0] if i + 2 < len(rle) else None
                result.append((ep_idx, rle[i][0], rle[i + 1][0], ts_tgt_end))
                break

    # Fallback for synthetic graphs (trajectory tree, etc.) where the
    # special START/SUCCESS/FAILURE checks don't match because terminals
    # have synthetic ids per branch.
    if not result and graph is not None:
        tgt_node = graph.nodes.get(tgt_id) if hasattr(graph, "nodes") else None
        if tgt_node is not None and getattr(tgt_node, "episode_indices", None):
            for ep_idx in tgt_node.episode_indices:
                last_ts_local = max(
                    (ts for ts, _ in ep_wins.get(ep_idx, [])), default=0,
                )
                result.append((ep_idx, last_ts_local, None, None))

    return sorted(result)


def _render_edge_panel(
    src_id: int,
    tgt_id: int,
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str,
) -> None:
    src_node = graph.nodes.get(src_id)
    tgt_node = graph.nodes.get(tgt_id)
    src_name = src_node.name if src_node else str(src_id)
    tgt_name = tgt_node.name if tgt_node else str(tgt_id)
    prob = graph.transition_probs.get(src_id, {}).get(tgt_id, 0.0)

    # If the tree dispatched this panel, it stored a synth_id → "START → … →
    # tgt" string; show that as a fuller subtitle so users see the prefix
    # they're inspecting.
    _id_to_prefix = st.session_state.get(f"{key_prefix}_id_to_prefix", {})
    _tgt_prefix = _id_to_prefix.get(tgt_id)

    with st.container(border=True):
        header_col, close_col = st.columns([20, 1])
        with header_col:
            st.subheader(f"{src_name}  →  {tgt_name}")
            if _tgt_prefix:
                st.caption(_tgt_prefix)
            st.caption(f"Transition probability: {prob:.1%}")
        with close_col:
            if st.button(
                "✕", key=f"{key_prefix}_edge_panel_close",
                help="Dismiss", use_container_width=True,
            ):
                st.session_state.pop(f"{key_prefix}_graph_selected_edge", None)
                _rt_key = f"{key_prefix}_graph_render_token"
                st.session_state[_rt_key] = st.session_state.get(_rt_key, 0) + 1
                st.rerun()

        all_ep_triples = _episodes_for_edge(src_id, tgt_id, labels, metadata, graph=graph)
        n_eps = len(all_ep_triples)

        if not all_ep_triples:
            st.info("No episodes found for this transition.")
            return

        _VIDS_PER_PAGE = 3
        _vp_key = f"{key_prefix}_edge_vid_page_{src_id}_{tgt_id}"
        _vp = st.session_state.get(_vp_key, 0)
        _vp_total = max(1, (n_eps + _VIDS_PER_PAGE - 1) // _VIDS_PER_PAGE)
        show_triples = all_ep_triples[_vp * _VIDS_PER_PAGE:(_vp + 1) * _VIDS_PER_PAGE]

        if _vp_total > 1:
            _vc1, _vc2, _vc3 = st.columns([1, 18, 1])
            with _vc1:
                if st.button(
                    "←", disabled=(_vp == 0),
                    key=f"{key_prefix}_ep_prev_{src_id}_{tgt_id}",
                    use_container_width=True,
                ):
                    st.session_state[_vp_key] = max(0, _vp - 1)
                    st.rerun()
            _vc2.markdown(
                f"<div style='text-align:center;padding-top:6px;color:#888;font-size:0.82em;'>"
                f"Episodes {_vp * _VIDS_PER_PAGE + 1}–{min((_vp + 1) * _VIDS_PER_PAGE, n_eps)} of {n_eps}"
                f"</div>", unsafe_allow_html=True)
            with _vc3:
                if st.button(
                    "→", disabled=(_vp >= _vp_total - 1),
                    key=f"{key_prefix}_ep_next_{src_id}_{tgt_id}",
                    use_container_width=True,
                ):
                    st.session_state[_vp_key] = min(_vp_total - 1, _vp + 1)
                    st.rerun()

        vid_cols = st.columns(min(3, len(show_triples)))
        for col, (ep_idx, ts_src, ts_tgt, ts_tgt_end) in zip(vid_cols, show_triples):
            ep_entry = _find_mp4_episode(ep_idx, mp4_index)
            if ep_entry is None:
                continue
            success = ep_entry.get("success")
            status = "✓" if success is True else "✗" if success is False else ""
            total_frames = ep_entry.get("frame_count")
            # Terminal transitions (X→FAILURE/SUCCESS) have ts_tgt=None.
            # Extend the source bar to episode end and skip the target bar.
            effective_tgt = ts_tgt if ts_tgt is not None else total_frames
            # If tgt is the LAST behavior in this episode, no follow-on
            # behavior gives us a ts_tgt_end — extend bar2 to episode end
            # so the target's full duration is still visible.
            if ts_tgt is not None:
                effective_tgt_end = ts_tgt_end if ts_tgt_end is not None else total_frames
            else:
                effective_tgt_end = None
            with col:
                st.caption(f"Ep {ep_idx} {status}")
                mp4_player(
                    mp4_dir / ep_entry["path"],
                    key=f"{key_prefix}_edge_vid_{src_id}_{tgt_id}_{ep_idx}",
                    max_height_px=220,
                    slice_start=ts_src,
                    slice_end=effective_tgt,
                    total_frames=total_frames,
                    slice2_start=ts_tgt if effective_tgt_end is not None else None,
                    slice2_end=effective_tgt_end,
                    bar1_label=src_name,
                    bar2_label=tgt_name if effective_tgt_end is not None else "",
                )


def render_graph_full_width(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str = "gex",
    highlighted_path: list[int] | None = None,
    excluded_node_ids: frozenset[int] = frozenset(),
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    pos: dict[int, tuple[float, float]] | None = None,
    symbol_override: dict[int, str] | None = None,
    color_override: dict[int, str] | None = None,
    theme: str = "dark",
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 24.0,
    suppress_video_panel: bool = False,
) -> None:
    """Full-width clickable behavior graph. Clicking a node opens a details panel."""

    encoding = (
        "Line width + grey level = transition probability"
        if edge_style == "lines"
        else "Arrow thickness = transition probability"
    )
    st.caption(
        f"**Click any node or edge** to explore it — larger circles = more episodes. "
        f"{encoding}. ★ = success, ✕ = failure; the percentage next to each "
        "terminal label is P(reach this terminal)."
    )

    clicked_node_id = render_graph_component(
        graph, height=650,
        key=f"{key_prefix}_graph",
        highlighted_path=highlighted_path,
        mp4_dir=mp4_dir,
        excluded_node_ids=excluded_node_ids,
        min_edge_prob=min_edge_prob,
        min_edge_count=min_edge_count,
        pos=pos,
        symbol_override=symbol_override,
        color_override=color_override,
        theme=theme,
        edge_style=edge_style,
        edge_width_slope=edge_width_slope,
        node_size_slope=node_size_slope,
    )

    selected_edge = st.session_state.get(f"{key_prefix}_graph_selected_edge")

    if suppress_video_panel:
        return

    if clicked_node_id is not None and clicked_node_id in graph.nodes:
        _render_node_panel(
            node_id=clicked_node_id,
            graph=graph,
            labels=labels,
            metadata=metadata,
            mp4_dir=mp4_dir,
            mp4_index=mp4_index,
            key_prefix=key_prefix,
        )
    elif selected_edge is not None:
        src_id, tgt_id = selected_edge
        if src_id in graph.nodes and tgt_id in graph.nodes:
            _render_edge_panel(
                src_id=src_id,
                tgt_id=tgt_id,
                graph=graph,
                labels=labels,
                metadata=metadata,
                mp4_dir=mp4_dir,
                mp4_index=mp4_index,
                key_prefix=key_prefix,
            )
        else:
            st.info("Click a node or edge in the graph above to explore it.")
    else:
        st.info("Click a node or edge in the graph above to explore it.")


def _render_node_panel(
    node_id: int,
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str,
) -> None:
    """Inline bordered panel showing node details + video clips below the graph."""
    node = graph.nodes[node_id]

    _id_to_prefix = st.session_state.get(f"{key_prefix}_id_to_prefix", {})
    _node_prefix = _id_to_prefix.get(node_id)

    with st.container(border=True):
        header_col, close_col = st.columns([20, 1])
        with header_col:
            st.subheader(node.name)
            if _node_prefix:
                st.caption(_node_prefix)
        with close_col:
            if st.button(
                "✕", key=f"{key_prefix}_panel_close",
                help="Dismiss", use_container_width=True,
            ):
                st.session_state.pop(f"{key_prefix}_graph_selected", None)
                _rt_key = f"{key_prefix}_graph_render_token"
                st.session_state[_rt_key] = st.session_state.get(_rt_key, 0) + 1
                st.rerun()

        # ── Videos first — they are the primary content ──────────────────────
        if node.is_special:
            st.info("No video clips for START / terminal nodes.")
        else:
            ep_slices = _episodes_for_node(node_id, labels, metadata)
            ep_slices_by_idx = {e[0]: (e[1], e[2]) for e in ep_slices}
            all_eps = node.episode_indices
            n_eps = len(all_eps)
            _VIDS_PER_PAGE = 3

            if not all_eps:
                st.info("No videos available for this node.")
            else:
                _vp_key = f"{key_prefix}_vid_page_{node_id}"
                _vp = st.session_state.get(_vp_key, 0)
                _vp_total = max(1, (n_eps + _VIDS_PER_PAGE - 1) // _VIDS_PER_PAGE)
                show_eps = all_eps[_vp * _VIDS_PER_PAGE:(_vp + 1) * _VIDS_PER_PAGE]

                if _vp_total > 1:
                    _vc1, _vc2, _vc3 = st.columns([1, 18, 1])
                    with _vc1:
                        if st.button(
                            "←", disabled=(_vp == 0),
                            key=f"{key_prefix}_vp_prev_{node_id}",
                            use_container_width=True,
                        ):
                            st.session_state[_vp_key] = max(0, _vp - 1)
                            st.rerun()
                    _vc2.markdown(
                        f"<div style='text-align:center;padding-top:6px;color:#888;font-size:0.82em;'>"
                        f"Episodes {_vp * _VIDS_PER_PAGE + 1}–{min((_vp + 1) * _VIDS_PER_PAGE, n_eps)} of {n_eps}"
                        f"</div>", unsafe_allow_html=True)
                    with _vc3:
                        if st.button(
                            "→", disabled=(_vp >= _vp_total - 1),
                            key=f"{key_prefix}_vp_next_{node_id}",
                            use_container_width=True,
                        ):
                            st.session_state[_vp_key] = min(_vp_total - 1, _vp + 1)
                            st.rerun()

                vid_cols = st.columns(min(3, len(show_eps)))
                available = 0
                for col, ep_idx in zip(vid_cols, show_eps):
                    ep_entry = _find_mp4_episode(ep_idx, mp4_index)
                    if ep_entry is None:
                        continue
                    available += 1
                    success = ep_entry.get("success")
                    status = "✓ Success" if success is True else "✗ Failure" if success is False else ""
                    ts_range = ep_slices_by_idx.get(ep_idx)
                    total_frames = ep_entry.get("frame_count")
                    effective_end = ts_range[1] if ts_range else None
                    with col:
                        st.caption(f"Episode {ep_idx} — {status}")
                        mp4_player(
                            mp4_dir / ep_entry["path"],
                            key=f"{key_prefix}_panel_vid_{node_id}_{ep_idx}",
                            max_height_px=220,
                            slice_start=ts_range[0] if ts_range else None,
                            slice_end=effective_end,
                            total_frames=total_frames,
                        )
                if available == 0:
                    st.info("No videos found in index for this node.")

        # ── Stats + outgoing transitions below ────────────────────────────────
        ep_key = "rollout_idx" if graph.level == "rollout" else "demo_idx"
        ep_success: dict[int, Optional[bool]] = {}
        for meta in metadata:
            eidx = meta.get(ep_key)
            if eidx is not None and eidx not in ep_success:
                ep_success[eidx] = meta.get("success")

        success_count = sum(1 for eidx in node.episode_indices if ep_success.get(eidx) is True)

        stats_col, trans_col = st.columns([1, 2])
        with stats_col:
            m1, m2, m3 = st.columns(3)
            m1.metric("Timesteps", node.num_timesteps)
            m2.metric("Episodes", node.num_episodes)
            m3.metric("Success rate", f"{success_count / node.num_episodes:.0%}" if node.num_episodes else "—")
            st.caption("Success rate = fraction of episodes that visit this node and ultimately succeed.")

        outgoing = graph.transition_probs.get(node_id, {})
        with trans_col:
            if outgoing and not node.is_special:
                tgt_labels = [graph.nodes[t].name if t in graph.nodes else str(t) for t in outgoing]
                tgt_probs = list(outgoing.values())
                bar_colors = [
                    "#2ca02c" if t == SUCCESS_NODE_ID else "#d62728" if t == FAILURE_NODE_ID else "#1f77b4"
                    for t in outgoing
                ]
                fig_trans = go.Figure(go.Bar(x=tgt_probs, y=tgt_labels, orientation="h", marker_color=bar_colors))
                fig_trans.update_layout(
                    title="Where does this node lead?",
                    height=max(100, 32 * len(tgt_labels) + 50),
                    margin=dict(l=0, r=0, t=36, b=0),
                    xaxis=dict(range=[0, 1], title="Probability"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_trans, use_container_width=True, key=f"{key_prefix}_panel_trans")
