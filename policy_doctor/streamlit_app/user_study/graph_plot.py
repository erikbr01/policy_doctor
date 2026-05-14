from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS

_SPECIAL_IDS = frozenset({START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})

_NODE_COLOR = {
    START_NODE_ID: "#2ca02c",
    SUCCESS_NODE_ID: "#2ca02c",
    FAILURE_NODE_ID: "#d62728",
    END_NODE_ID: "#888888",
}

_COMPONENT_DIR = Path(__file__).parent / "_graph_component"
_graph_component = components.declare_component("behavior_graph", path=str(_COMPONENT_DIR))


def _compute_layout(graph: BehaviorGraph) -> dict[int, tuple[float, float]]:
    """Kamada-Kawai layout on the undirected version of the graph."""
    import networkx as nx

    G = nx.DiGraph()
    for nid in graph.nodes:
        G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if src in graph.nodes and tgt in graph.nodes and prob >= 0.01:
                G.add_edge(src, tgt, weight=prob)

    U = G.to_undirected()
    raw = nx.kamada_kawai_layout(U, weight=None, scale=2.0)

    # Orient so START is on the left of the cluster mass
    cluster_xs = [raw[n][0] for n in graph.nodes if n not in _SPECIAL_IDS and n in raw]
    if cluster_xs:
        median_x = float(np.median(cluster_xs))
        start_pos = raw.get(START_NODE_ID)
        if start_pos is not None and start_pos[0] > median_x:
            raw = {n: (-x, y) for n, (x, y) in raw.items()}

    return {nid: (float(xy[0]), float(xy[1])) for nid, xy in raw.items()}


def _build_graph_json(graph: BehaviorGraph, pos: dict[int, tuple[float, float]]) -> str:
    """Serialize graph data for the SVG component."""
    nodes = []
    for nid, node in graph.nodes.items():
        if nid not in pos:
            continue
        if nid in _SPECIAL_IDS:
            color = _NODE_COLOR.get(nid, "#888")
            symbol = (
                "star" if nid == SUCCESS_NODE_ID else
                "x" if nid == FAILURE_NODE_ID else
                "diamond" if nid == START_NODE_ID else
                "square"
            )
            tooltip = node.name
            label = node.name
        else:
            color = CLUSTER_COLORS[nid % len(CLUSTER_COLORS)]
            symbol = "circle"
            label = node.name
            outgoing = graph.transition_probs.get(nid, {})
            top_out = sorted(outgoing.items(), key=lambda kv: -kv[1])[:3]
            out_lines = "\n".join(
                f"  → {graph.nodes[t].name}: {p:.0%}"
                for t, p in top_out
                if t in graph.nodes
            )
            tooltip = (
                f"{node.name}\n"
                f"Episodes: {node.num_episodes}  •  Timesteps: {node.num_timesteps}\n"
                f"Top transitions:\n{out_lines}"
            )
        nodes.append({
            "id": nid,
            "label": label,
            "color": color,
            "symbol": symbol,
            "num_episodes": node.num_episodes,
            "num_timesteps": node.num_timesteps,
            "is_special": nid in _SPECIAL_IDS,
            "tooltip": tooltip,
        })

    edges = []
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob >= 0.01 and src in pos and tgt in pos:
                edges.append({"src": src, "tgt": tgt, "prob": round(prob, 4)})

    positions = {str(nid): list(xy) for nid, xy in pos.items()}

    return json.dumps({"nodes": nodes, "edges": edges, "positions": positions})


def render_graph_component(
    graph: BehaviorGraph,
    height: int = 580,
    key: str = "behavior_graph",
) -> Optional[int]:
    """Render the custom SVG behavior graph component.

    Returns the node_id that was clicked, or None if nothing was clicked yet.
    Persists the last clicked node in session state.
    """
    pos = _compute_layout(graph)
    graph_json = _build_graph_json(graph, pos)
    selected = st.session_state.get(f"{key}_selected")

    clicked = _graph_component(
        graph_json=graph_json,
        height=height,
        selected_node_id=selected,
        key=key,
        default=None,
    )

    if clicked is not None:
        try:
            node_id = int(clicked)
            if node_id in graph.nodes:
                st.session_state[f"{key}_selected"] = node_id
                return node_id
        except (TypeError, ValueError):
            pass

    return selected
