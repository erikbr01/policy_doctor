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



def render_graph_full_width(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str = "gex",
    highlighted_path: list[int] | None = None,
) -> None:
    """Full-width clickable behavior graph. Clicking a node opens a details panel."""

    st.caption(
        "**Click any node** to explore it — larger circles = more episodes. "
        "Arrow thickness = transition probability. ★ = success, ✕ = failure."
    )

    clicked_node_id = render_graph_component(
        graph, height=650,
        key=f"{key_prefix}_graph",
        highlighted_path=highlighted_path,
    )

    # Inline node detail panel
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
    else:
        st.info("Click a node in the graph above to explore it.")


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

    with st.container(border=True):
        header_col, close_col = st.columns([10, 1])
        with header_col:
            st.subheader(f"🔍 {node.name}")
        with close_col:
            if st.button("✕", key=f"{key_prefix}_panel_close", help="Dismiss"):
                st.session_state.pop(f"{key_prefix}_clicked_node", None)
                st.rerun()

        # ── Videos first — they are the primary content ──────────────────────
        if node.is_special:
            st.info("No video clips for START / terminal nodes.")
        else:
            ep_slices = _episodes_for_node(node_id, labels, metadata)
            ep_slices_by_idx = {e[0]: (e[1], e[2]) for e in ep_slices}
            show_eps = node.episode_indices[:3]

            if not show_eps:
                st.info("No videos available for this node.")
            else:
                vid_cols = st.columns(min(3, len(show_eps)))
                available = 0
                for col, ep_idx in zip(vid_cols, show_eps):
                    ep_entry = _find_mp4_episode(ep_idx, mp4_index)
                    if ep_entry is None:
                        continue
                    available += 1
                    success = ep_entry.get("success")
                    status = "✓ Success" if success is True else "✗ Failure" if success is False else ""
                    with col:
                        st.caption(f"Episode {ep_idx} — {status}")
                        mp4_player(
                            mp4_dir / ep_entry["path"],
                            key=f"{key_prefix}_panel_vid_{node_id}_{ep_idx}",
                            max_height_px=220,
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
