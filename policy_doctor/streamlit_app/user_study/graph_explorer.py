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
    END_NODE_ID,
)
from policy_doctor.plotting import create_behavior_graph_plot
from policy_doctor.streamlit_app.components.mp4_player import mp4_player


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


def render_graph_explorer(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str = "gex",
) -> None:
    col_graph, col_inspector = st.columns([3, 2])

    with col_graph:
        st.subheader("Behavior Graph")
        fig = create_behavior_graph_plot(graph, height=500)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_graph")

    with col_inspector:
        st.subheader("Node Inspector")

        cluster_nodes = {
            nid: node
            for nid, node in graph.nodes.items()
            if not node.is_special
        }
        special_nodes = {
            nid: node
            for nid, node in graph.nodes.items()
            if node.is_special
        }
        ordered_node_ids = (
            [START_NODE_ID]
            + sorted(cluster_nodes.keys())
            + [nid for nid in [SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID] if nid in special_nodes]
        )

        def _node_label(nid: int) -> str:
            node = graph.nodes[nid]
            return f"Node {nid}: {node.name} ({node.num_episodes} episodes)"

        selected_label = st.selectbox(
            "Select behavior node",
            options=[_node_label(nid) for nid in ordered_node_ids],
            key=f"{key_prefix}_node_select",
        )
        selected_idx = [_node_label(nid) for nid in ordered_node_ids].index(selected_label)
        selected_id = ordered_node_ids[selected_idx]
        node = graph.nodes[selected_id]

        ep_key = "rollout_idx" if graph.level == "rollout" else "demo_idx"
        ep_success: dict[int, Optional[bool]] = {}
        for meta in metadata:
            eidx = meta.get(ep_key)
            if eidx is not None and eidx not in ep_success:
                ep_success[eidx] = meta.get("success")

        success_count = sum(
            1 for eidx in node.episode_indices if ep_success.get(eidx) is True
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Timesteps", node.num_timesteps)
        m2.metric("Episodes", node.num_episodes)
        if node.num_episodes > 0:
            m3.metric("Success rate", f"{success_count / node.num_episodes:.0%}")
        else:
            m3.metric("Success rate", "—")

        outgoing = graph.transition_probs.get(selected_id, {})
        if outgoing:
            st.markdown("**Outgoing transitions**")
            tgt_labels = [graph.nodes[t].name if t in graph.nodes else str(t) for t in outgoing]
            tgt_probs = list(outgoing.values())
            bar_colors = []
            for t in outgoing:
                if t == SUCCESS_NODE_ID:
                    bar_colors.append("#2ca02c")
                elif t == FAILURE_NODE_ID:
                    bar_colors.append("#d62728")
                else:
                    bar_colors.append("#1f77b4")
            fig_trans = go.Figure(
                go.Bar(
                    x=tgt_probs,
                    y=tgt_labels,
                    orientation="h",
                    marker_color=bar_colors,
                )
            )
            fig_trans.update_layout(
                height=max(100, 40 * len(tgt_labels) + 40),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 1], title="Probability"),
                yaxis=dict(title=None),
            )
            st.plotly_chart(fig_trans, use_container_width=True, key=f"{key_prefix}_trans_{selected_id}")

        if node.is_special:
            st.info("No video clips for START / terminal nodes.")
            return

        ep_slices = _episodes_for_node(selected_id, labels, metadata)
        ep_slices_by_idx = {e[0]: (e[1], e[2]) for e in ep_slices}

        show_eps = node.episode_indices[:3]
        if not show_eps:
            st.info("No videos available for this node.")
            return

        st.markdown("**Video clips**")
        available_count = 0
        for ep_idx in show_eps:
            ep_entry = _find_mp4_episode(ep_idx, mp4_index)
            if ep_entry is None:
                continue
            available_count += 1
            success = ep_entry.get("success")
            status = "✓ Success" if success is True else "✗ Failure" if success is False else "Unknown"
            with st.expander(f"Episode {ep_idx} — {status}"):
                video_path = mp4_dir / ep_entry["path"]
                ts_range = ep_slices_by_idx.get(ep_idx)
                total_frames = ep_entry.get("frame_count")
                mp4_player(
                    video_path,
                    label="",
                    slice_start=ts_range[0] if ts_range else None,
                    slice_end=ts_range[1] if ts_range else None,
                    total_frames=total_frames,
                    key=f"{key_prefix}_vid_{selected_id}_{ep_idx}",
                )

        if available_count == 0:
            st.info("No videos available for this node.")
