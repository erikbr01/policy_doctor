from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.streamlit_app.appearance import get_theme, plotly_layout_overrides
from policy_doctor.streamlit_app.components.mp4_player import mp4_player
from policy_doctor.streamlit_app.user_study.graph_plot import render_graph_component

_SPECIAL_IDS = frozenset({START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})


def _node_display_name(node_id: int, graph: BehaviorGraph) -> str:
    if node_id == START_NODE_ID:
        return "START"
    if node_id == SUCCESS_NODE_ID:
        return "SUCCESS"
    if node_id == FAILURE_NODE_ID:
        return "FAILURE"
    if node_id == END_NODE_ID:
        return "END"
    node = graph.nodes.get(node_id)
    return node.name if node else str(node_id)


def _format_path(path_nodes: list[int], graph: BehaviorGraph) -> str:
    parts = []
    for nid in path_nodes:
        if nid == START_NODE_ID:
            parts.append("START")
        elif nid == SUCCESS_NODE_ID:
            parts.append("✓ SUCCESS")
        elif nid == FAILURE_NODE_ID:
            parts.append("✗ FAILURE")
        elif nid == END_NODE_ID:
            parts.append("END")
        else:
            node = graph.nodes.get(nid)
            parts.append(node.name if node else str(nid))
    return " → ".join(parts)


def _get_episodes_for_path(
    path_nodes: list[int],
    labels: np.ndarray,
    metadata: list[dict],
) -> list[int]:
    required = [n for n in path_nodes if n not in _SPECIAL_IDS]
    if not required:
        return []

    ep_key = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"

    ep_sequences: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        ep_idx = meta.get(ep_key)
        if ep_idx is None:
            continue
        label = int(labels[i])
        if label == -1:
            continue
        ts = meta.get("timestep", meta.get("window_start", 0))
        ep_sequences[ep_idx].append((ts, label))

    for ep_idx in ep_sequences:
        ep_sequences[ep_idx].sort(key=lambda x: x[0])

    matching: list[int] = []
    for ep_idx, seq in ep_sequences.items():
        seq_labels = [lab for _, lab in seq]
        cursor = 0
        found_all = True
        for req in required:
            found = False
            while cursor < len(seq_labels):
                if seq_labels[cursor] == req:
                    cursor += 1
                    found = True
                    break
                cursor += 1
            if not found:
                found_all = False
                break
        if found_all:
            matching.append(ep_idx)

    return sorted(matching)


def _find_mp4_episode(ep_idx: int, mp4_index: dict) -> Optional[dict]:
    for ep in mp4_index.get("episodes", []):
        if ep.get("index") == ep_idx:
            return ep
    return None


def _episode_node_range(
    ep_idx: int,
    node_id: int,
    labels: np.ndarray,
    metadata: list[dict],
) -> Optional[tuple[int, int]]:
    ep_key = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"
    timesteps = [
        meta.get("timestep", meta.get("window_start", 0))
        for i, meta in enumerate(metadata)
        if meta.get(ep_key) == ep_idx and int(labels[i]) == node_id
    ]
    if not timesteps:
        return None
    return min(timesteps), max(timesteps)


def render_path_explorer(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    mp4_dir: Path,
    mp4_index: dict,
    key_prefix: str = "pex",
    theme: str | None = None,
) -> None:
    theme = theme or get_theme()
    start_bg = "#64748b" if theme == "light" else "#444"
    arrow_color = "#64748b" if theme == "light" else "#aaa"
    st.subheader("Top Paths Through Behavior Graph")

    max_paths = st.slider(
        "Max paths to show",
        min_value=1,
        max_value=20,
        value=10,
        key=f"{key_prefix}_max_paths",
    )

    all_paths = graph.enumerate_paths(max_paths=max_paths)
    terminal_paths = [
        (path, prob, loops)
        for path, prob, loops in all_paths
        if path and path[-1] in {SUCCESS_NODE_ID, FAILURE_NODE_ID}
    ]

    if not terminal_paths:
        st.info("No paths ending in SUCCESS or FAILURE found.")
        return

    # Precompute matching episode counts for each path
    ep_counts = [len(_get_episodes_for_path(path, labels, metadata)) for path, _, _ in terminal_paths]

    path_labels = []
    path_colors = []
    path_texts = []
    for (path, prob, _), n_ep in zip(terminal_paths, ep_counts):
        terminal = path[-1]
        label = _format_path(path, graph)
        path_labels.append(label)
        path_colors.append("#2ca02c" if terminal == SUCCESS_NODE_ID else "#d62728")
        path_texts.append(f"p={prob:.3f}")

    max_ep = max(ep_counts) if ep_counts else 1
    fig_paths = go.Figure(go.Bar(
        x=ep_counts,
        y=path_labels,
        orientation="h",
        marker_color=path_colors,
        text=path_texts,
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_paths.update_layout(
        height=max(120, 40 * len(path_labels) + 60),
        margin=dict(l=0, r=80, t=20, b=20),
        xaxis=dict(title="Episodes following this path", range=[0, max_ep * 1.45]),
        yaxis=dict(title=None, autorange="reversed"),
        **plotly_layout_overrides(theme),
    )
    st.plotly_chart(fig_paths, use_container_width=True, key=f"{key_prefix}_paths_chart")

    st.divider()

    def _path_select_label(i: int, path: list[int], prob: float, n_ep: int) -> str:
        terminal = path[-1]
        outcome = "SUCCESS" if terminal == SUCCESS_NODE_ID else "FAILURE"
        short = " → ".join(_node_display_name(n, graph) for n in path)
        return f"Path {i + 1} ({n_ep} ep, {outcome}) — {short}"

    _NO_SELECTION = "— Select a path to explore and highlight it in the graph —"
    select_options = [_NO_SELECTION] + [
        _path_select_label(i, path, prob, ep_counts[i])
        for i, (path, prob, _) in enumerate(terminal_paths)
    ]
    selected_label = st.selectbox(
        "Explore path:",
        options=select_options,
        key=f"{key_prefix}_path_select",
    )

    if selected_label == _NO_SELECTION:
        # Clear any existing highlight so the graph shows all paths
        st.session_state.pop(f"{key_prefix}_highlighted_path", None)
        return

    selected_path_idx = select_options.index(selected_label) - 1  # offset for NO_SELECTION
    selected_path, selected_prob, _ = terminal_paths[selected_path_idx]

    # Store for graph highlighting; rerun once if path changed so the graph above updates
    current = st.session_state.get(f"{key_prefix}_highlighted_path")
    st.session_state[f"{key_prefix}_highlighted_path"] = selected_path
    if current != selected_path:
        st.rerun()

    terminal_id = selected_path[-1]
    outcome_str = "SUCCESS" if terminal_id == SUCCESS_NODE_ID else "FAILURE"
    outcome_color = "#2ca02c" if terminal_id == SUCCESS_NODE_ID else "#d62728"

    flow_parts = []
    for nid in selected_path:
        if nid == START_NODE_ID:
            part = f"<span style='background:{start_bg};color:#fff;padding:4px 10px;border-radius:4px;font-weight:bold;'>START</span>"
        elif nid == SUCCESS_NODE_ID:
            part = f"<span style='background:{outcome_color};color:white;padding:4px 10px;border-radius:4px;font-weight:bold;'>✓ SUCCESS</span>"
        elif nid == FAILURE_NODE_ID:
            part = f"<span style='background:{outcome_color};color:white;padding:4px 10px;border-radius:4px;font-weight:bold;'>✗ FAILURE</span>"
        elif nid == END_NODE_ID:
            part = "<span style='background:#888;color:white;padding:4px 10px;border-radius:4px;'>END</span>"
        else:
            node = graph.nodes.get(nid)
            name = node.name if node else str(nid)
            part = f"<span style='background:#1f77b4;color:white;padding:4px 10px;border-radius:4px;'>{name}</span>"
        flow_parts.append(part)

    arrow = f" <span style='color:{arrow_color};font-size:1.2em;'>→</span> "
    st.markdown(arrow.join(flow_parts), unsafe_allow_html=True)
    st.caption(f"Path probability: {selected_prob:.4f} | Outcome: {outcome_str}")

    render_graph_component(
        graph,
        height=520,
        key=f"{key_prefix}_compact_graph",
        highlighted_path=selected_path,
        theme=theme,
    )

    matching_episodes = _get_episodes_for_path(selected_path, labels, metadata)
    show_eps = matching_episodes[:5]

    if not show_eps:
        st.info("No episodes found matching this path.")
        return

    st.markdown(f"**{len(matching_episodes)} matching episode(s)** (showing up to 5)")

    intermediate_nodes = [n for n in selected_path if n not in _SPECIAL_IDS]

    for ep_idx in show_eps:
        ep_entry = _find_mp4_episode(ep_idx, mp4_index)
        if ep_entry is None:
            st.caption(f"Episode {ep_idx} — video not found in index.")
            continue
        success = ep_entry.get("success")
        status = "✓ Success" if success is True else "✗ Failure" if success is False else "Unknown"
        with st.expander(f"Episode {ep_idx} — {status}"):
            video_path = mp4_dir / ep_entry["path"]
            total_frames = ep_entry.get("frame_count")
            ep_fps = int(round(ep_entry.get("fps") or 10))
            if intermediate_nodes:
                first_node = intermediate_nodes[0]
                last_node = intermediate_nodes[-1]
                first_range = _episode_node_range(ep_idx, first_node, labels, metadata)
                last_range = _episode_node_range(ep_idx, last_node, labels, metadata)
                slice_start = first_range[0] if first_range else None
                slice_end = last_range[1] if last_range else None
            else:
                slice_start = slice_end = None
            mp4_player(
                video_path,
                label="",
                slice_start=slice_start,
                slice_end=slice_end,
                total_frames=total_frames,
                fps=ep_fps,
                key=f"{key_prefix}_vid_{selected_path_idx}_{ep_idx}",
            )
