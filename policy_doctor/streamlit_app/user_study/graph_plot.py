from __future__ import annotations

import base64
import io
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
    """BFS-layered layout with barycenter refinement (adapted from the main plotting module).

    Nodes are columns by BFS depth from START; terminals pinned to the far right.
    10 passes of barycenter + spread give a clean, crossing-minimised layout.
    """
    import networkx as nx
    from collections import defaultdict

    G = nx.DiGraph()
    for nid in graph.nodes:
        G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if src in graph.nodes and tgt in graph.nodes and prob > 0:
                G.add_edge(src, tgt, weight=prob)

    terminal_ids = {n for n in G.nodes() if n in _SPECIAL_IDS and n != START_NODE_ID}

    try:
        distances = dict(nx.single_source_shortest_path_length(G, START_NODE_ID))
    except nx.NodeNotFound:
        distances = {START_NODE_ID: 0}

    # Reverse-BFS for unreachable nodes
    rev_distances: dict[int, int] = {}
    for term_id in terminal_ids:
        try:
            rd = dict(nx.single_source_shortest_path_length(G.reverse(), term_id))
            for n, d in rd.items():
                if n not in rev_distances or d < rev_distances[n]:
                    rev_distances[n] = d
        except (nx.NodeNotFound, nx.NetworkXError):
            pass

    max_fwd = max((d for n, d in distances.items() if n not in terminal_ids), default=1)
    for node in G.nodes():
        if node not in distances:
            distances[node] = (max_fwd + 1 - rev_distances[node]) if node in rev_distances else max_fwd // 2 + 1

    end_layer = max((d for n, d in distances.items() if n not in terminal_ids), default=1) + 1
    for term_id in terminal_ids:
        distances[term_id] = end_layer

    layers: dict[int, list] = defaultdict(list)
    for node, d in distances.items():
        layers[d].append(node)
    for layer_idx in layers:
        layers[layer_idx].sort(key=lambda n: (n in _SPECIAL_IDS, n))

    total_layers = max(layers.keys()) if layers else 1
    x_min, x_max = -2.5, 2.5

    pos: dict[int, tuple[float, float]] = {}
    for layer_idx, nodes in layers.items():
        x = x_min + (x_max - x_min) * layer_idx / max(total_layers, 1)
        n = len(nodes)
        y_spacing = max(1.0, 1.2 - 0.05 * n)
        for i, node in enumerate(nodes):
            pos[node] = (x, (i - (n - 1) / 2) * y_spacing)

    # Barycenter refinement
    min_gap = 0.8
    for _ in range(10):
        for layer_idx in sorted(layers.keys()):
            nodes = layers[layer_idx]
            if len(nodes) <= 1:
                continue
            for node in nodes:
                neighbours = list(G.predecessors(node)) + list(G.successors(node))
                ys = [pos[nb][1] for nb in neighbours if nb in pos]
                if ys:
                    pos[node] = (pos[node][0], 0.6 * float(np.mean(ys)) + 0.4 * pos[node][1])
            nodes_sorted = sorted(nodes, key=lambda n: pos[n][1])
            for i in range(1, len(nodes_sorted)):
                py = pos[nodes_sorted[i - 1]][1]
                cy = pos[nodes_sorted[i]][1]
                if cy - py < min_gap:
                    pos[nodes_sorted[i]] = (pos[nodes_sorted[i]][0], py + min_gap)
            ys = [pos[n][1] for n in nodes_sorted]
            center = float(np.mean(ys))
            for n in nodes_sorted:
                pos[n] = (pos[n][0], pos[n][1] - center)

    return pos


def _extract_node_thumbnails(
    graph: BehaviorGraph,
    mp4_dir: Path,
) -> dict[int, list[str]]:
    """Extract representative frame thumbnails for each cluster node.

    Returns a mapping from node_id to a list of up to 2 base64-encoded JPEG
    strings (each suitable as a ``data:image/jpeg;base64,...`` src).
    Missing or unreadable mp4 files are silently skipped.
    """
    try:
        import imageio  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return {}

    thumbnails: dict[int, list[str]] = {}
    for nid, node in graph.nodes.items():
        if nid in _SPECIAL_IDS:
            continue
        imgs: list[str] = []
        for ep_idx in node.episode_indices[:2]:
            mp4_path = mp4_dir / f"ep{ep_idx}.mp4"
            if not mp4_path.exists():
                continue
            try:
                reader = imageio.get_reader(str(mp4_path))
                n_frames = len(reader)
                frame = reader.get_data(n_frames // 2)
                reader.close()
                img = Image.fromarray(frame).resize((80, 80), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                b64 = base64.b64encode(buf.getvalue()).decode()
                imgs.append(b64)
            except Exception:
                continue
        if imgs:
            thumbnails[nid] = imgs
    return thumbnails


def _build_graph_json(
    graph: BehaviorGraph,
    pos: dict[int, tuple[float, float]],
    thumbnails: dict[int, list[str]] | None = None,
) -> str:
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
        node_dict: dict = {
            "id": nid,
            "label": label,
            "color": color,
            "symbol": symbol,
            "num_episodes": node.num_episodes,
            "num_timesteps": node.num_timesteps,
            "is_special": nid in _SPECIAL_IDS,
            "tooltip": tooltip,
        }
        if thumbnails and nid in thumbnails:
            node_dict["thumbnails"] = [
                f"data:image/jpeg;base64,{b64}" for b64 in thumbnails[nid]
            ]
        nodes.append(node_dict)

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
    highlighted_path: Optional[list[int]] = None,
    mp4_dir: Optional[Path] = None,
) -> Optional[int]:
    """Render the custom SVG behavior graph component.

    Args:
        highlighted_path: List of node IDs forming a path to highlight.
            Edges on the path are drawn in orange; off-path elements are dimmed.
        mp4_dir: Directory containing ``ep<idx>.mp4`` files. When provided,
            representative frame thumbnails are embedded in node hover tooltips.

    Returns:
        The node_id that was clicked, or the previously selected node.
    """
    pos = _compute_layout(graph)
    thumbnails: dict[int, list[str]] | None = None
    if mp4_dir is not None:
        thumbnails = _extract_node_thumbnails(graph, mp4_dir)
    graph_json = _build_graph_json(graph, pos, thumbnails=thumbnails)
    selected = st.session_state.get(f"{key}_selected")

    clicked = _graph_component(
        graph_json=graph_json,
        height=height,
        selected_node_id=selected,
        highlighted_path=highlighted_path,
        key=key,
        default=None,
    )

    if clicked is not None:
        try:
            node_id = int(clicked)
            if node_id == -1:
                st.session_state.pop(f"{key}_selected", None)
                return None
            if node_id in graph.nodes:
                st.session_state[f"{key}_selected"] = node_id
                return node_id
        except (TypeError, ValueError):
            pass

    return selected
