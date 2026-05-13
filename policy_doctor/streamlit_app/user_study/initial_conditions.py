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
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS, NOISE_COLOR

_SPECIAL_IDS = frozenset({START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})


def _ep_key(metadata: list[dict]) -> str:
    return "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"


def _initial_slice_per_episode(
    labels: np.ndarray,
    metadata: list[dict],
    coords: np.ndarray,
) -> dict[int, tuple[int, int, int, float, float]]:
    """Return {ep_idx: (slice_i, cluster_id, timestep, x, y)} for each episode's first slice."""
    key = _ep_key(metadata)
    ep_first: dict[int, tuple[int, int, int, float, float]] = {}
    for i, meta in enumerate(metadata):
        ep_idx = meta.get(key)
        if ep_idx is None:
            continue
        ts = meta.get("timestep", meta.get("window_start", 0))
        if ep_idx not in ep_first or ts < ep_first[ep_idx][2]:
            x, y = float(coords[i, 0]), float(coords[i, 1])
            ep_first[ep_idx] = (i, int(labels[i]), ts, x, y)
    return ep_first


def _success_per_episode(metadata: list[dict]) -> dict[int, Optional[bool]]:
    key = _ep_key(metadata)
    out: dict[int, Optional[bool]] = {}
    for meta in metadata:
        ep_idx = meta.get(key)
        if ep_idx is not None and ep_idx not in out:
            out[ep_idx] = meta.get("success")
    return out


def _episodes_for_node(node_id: int, labels: np.ndarray, metadata: list[dict]) -> set[int]:
    key = _ep_key(metadata)
    eps: set[int] = set()
    for i, meta in enumerate(metadata):
        if int(labels[i]) == node_id:
            ep = meta.get(key)
            if ep is not None:
                eps.add(ep)
    return eps


def _episodes_for_path(path_nodes: list[int], labels: np.ndarray, metadata: list[dict]) -> list[int]:
    required = [n for n in path_nodes if n not in _SPECIAL_IDS]
    if not required:
        return []
    key = _ep_key(metadata)
    ep_seqs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        ep_idx = meta.get(key)
        if ep_idx is None:
            continue
        lab = int(labels[i])
        if lab == -1:
            continue
        ts = meta.get("timestep", meta.get("window_start", 0))
        ep_seqs[ep_idx].append((ts, lab))
    for ep_idx in ep_seqs:
        ep_seqs[ep_idx].sort()
    matching = []
    for ep_idx, seq in ep_seqs.items():
        seq_labs = [lab for _, lab in seq]
        cursor = 0
        ok = True
        for req in required:
            found = False
            while cursor < len(seq_labs):
                if seq_labs[cursor] == req:
                    cursor += 1
                    found = True
                    break
                cursor += 1
            if not found:
                ok = False
                break
        if ok:
            matching.append(ep_idx)
    return sorted(matching)


def render_initial_conditions(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: list[dict],
    coords: np.ndarray,
    mp4_index: Optional[dict] = None,
    key_prefix: str = "ic",
) -> None:
    """Visualize initial conditions in the 2D embedding space.

    Shows where each episode starts in the behavioral embedding, colored by
    cluster / success / path membership. Useful for seeing how initial state
    distribution relates to behavioral mode and outcome.
    """
    if coords is None or coords.shape[0] != len(metadata):
        st.warning(
            "2D embedding coordinates not available. Load clustering with UMAP/PCA reduction "
            "to enable this view."
        )
        return

    st.subheader("Initial Conditions in Embedding Space")
    st.caption(
        "Each point is one episode, plotted at its first timestep's embedding coordinate. "
        "The embedding reflects behavioral similarity — nearby points behave similarly."
    )

    ep_first = _initial_slice_per_episode(labels, metadata, coords)
    ep_success = _success_per_episode(metadata)

    if not ep_first:
        st.info("No episode initial conditions found.")
        return

    color_mode = st.radio(
        "Color by",
        ["Cluster (initial)", "Success / Failure", "Path membership"],
        horizontal=True,
        key=f"{key_prefix}_color_mode",
    )

    all_ep_ids = sorted(ep_first.keys())
    xs = [ep_first[e][3] for e in all_ep_ids]
    ys = [ep_first[e][4] for e in all_ep_ids]
    cluster_ids = [ep_first[e][1] for e in all_ep_ids]
    successes = [ep_success.get(e) for e in all_ep_ids]

    fig = go.Figure()

    if color_mode == "Cluster (initial)":
        unique_clusters = sorted(set(cluster_ids))
        for cid in unique_clusters:
            mask = [i for i, c in enumerate(cluster_ids) if c == cid]
            if cid == -1:
                color = NOISE_COLOR
                name = "Noise"
            elif cid not in _SPECIAL_IDS:
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                node = graph.nodes.get(cid)
                name = f"Node {cid}: {node.name}" if node else f"Cluster {cid}"
            else:
                continue
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in mask],
                y=[ys[i] for i in mask],
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.8, line=dict(width=0.5, color="white")),
                name=name,
                text=[f"Episode {all_ep_ids[i]}" for i in mask],
                hovertemplate="%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            ))

    elif color_mode == "Success / Failure":
        for outcome, color, name in [
            (True, "#2ca02c", "Success"),
            (False, "#d62728", "Failure"),
            (None, "#aaaaaa", "Unknown"),
        ]:
            mask = [i for i, s in enumerate(successes) if s is outcome]
            if not mask:
                continue
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in mask],
                y=[ys[i] for i in mask],
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.8, line=dict(width=0.5, color="white")),
                name=name,
                text=[f"Episode {all_ep_ids[i]}" for i in mask],
                hovertemplate="%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            ))

    else:  # Path membership
        top_paths = graph.enumerate_paths(max_paths=8)
        terminal_paths = [
            (path, prob)
            for path, prob, _ in top_paths
            if path and path[-1] in {SUCCESS_NODE_ID, FAILURE_NODE_ID}
        ]

        path_options = [
            f"Path {i + 1} (p={prob:.3f}) — {' → '.join(graph.nodes[n].name if n in graph.nodes else str(n) for n in path if n not in _SPECIAL_IDS)}"
            for i, (path, prob) in enumerate(terminal_paths)
        ]

        if not path_options:
            st.info("No terminal paths found. Build the graph first.")
            return

        selected_path_label = st.selectbox(
            "Highlight path",
            options=path_options,
            key=f"{key_prefix}_path_sel",
        )
        selected_path_idx = path_options.index(selected_path_label)
        selected_path, _ = terminal_paths[selected_path_idx]

        path_eps = set(_episodes_for_path(selected_path, labels, metadata))

        in_mask = [i for i, e in enumerate(all_ep_ids) if e in path_eps]
        out_mask = [i for i, e in enumerate(all_ep_ids) if e not in path_eps]

        if out_mask:
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in out_mask],
                y=[ys[i] for i in out_mask],
                mode="markers",
                marker=dict(color="#cccccc", size=7, opacity=0.4),
                name="Other episodes",
                text=[f"Episode {all_ep_ids[i]}" for i in out_mask],
                hovertemplate="%{text}<extra></extra>",
            ))
        if in_mask:
            terminal = selected_path[-1]
            color = "#2ca02c" if terminal == SUCCESS_NODE_ID else "#d62728"
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in in_mask],
                y=[ys[i] for i in in_mask],
                mode="markers",
                marker=dict(color=color, size=11, opacity=0.9, line=dict(width=1, color="white")),
                name=f"Path episodes ({len(in_mask)})",
                text=[f"Episode {all_ep_ids[i]}" for i in in_mask],
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
        xaxis=dict(title="Embedding dim 1", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Embedding dim 2", showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_scatter")

    st.caption(
        f"Showing {len(all_ep_ids)} episodes. "
        f"Success: {sum(1 for s in successes if s is True)}, "
        f"Failure: {sum(1 for s in successes if s is False)}, "
        f"Unknown: {sum(1 for s in successes if s is None)}."
    )
