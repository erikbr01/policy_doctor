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

# Unique object used as a sentinel so `last_click` in session state is never
# equal to None or any real component value on the very first render.
_SENTINEL = object()

_NODE_COLOR = {
    START_NODE_ID: "#2ca02c",
    SUCCESS_NODE_ID: "#2ca02c",
    FAILURE_NODE_ID: "#d62728",
    END_NODE_ID: "#888888",
}

_COMPONENT_DIR = Path(__file__).parent / "_graph_component"
_graph_component = components.declare_component("behavior_graph", path=str(_COMPONENT_DIR))


def compute_pruned_graph_nodes(
    graph: BehaviorGraph,
    min_visit_prob: float,
    n_total: int,
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
) -> frozenset[int]:
    """Return the set of node IDs to EXCLUDE from the graph.

    Algorithm:
    1. Remove regular nodes whose visit frequency < min_visit_prob
       (or whose num_episodes < min_edge_count, when min_edge_count > 0).
    2. BFS from START following only edges with prob >= min_edge_prob
       (and transition_count >= min_edge_count when > 0) through surviving
       nodes; remove anything unreachable.
    Special nodes (START/SUCCESS/FAILURE) are never excluded by the threshold but
    may disappear if all paths to them are cut by node or edge pruning.
    """
    surviving = {
        nid for nid, node in graph.nodes.items()
        if nid in _SPECIAL_IDS or (
            (n_total == 0 or node.num_episodes / n_total >= min_visit_prob)
            and (min_edge_count <= 0 or node.num_episodes >= min_edge_count)
        )
    }

    reachable: set[int] = set()
    queue = [START_NODE_ID] if START_NODE_ID in surviving else []
    while queue:
        nid = queue.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        for tgt, prob in graph.transition_probs.get(nid, {}).items():
            if tgt not in surviving or tgt in reachable:
                continue
            if prob < min_edge_prob:
                continue
            if min_edge_count > 0:
                cnt = graph.transition_counts.get(nid, {}).get(tgt, 0)
                if cnt < min_edge_count:
                    continue
            queue.append(tgt)

    return frozenset(nid for nid in graph.nodes if nid not in reachable)


def _compute_layout(
    graph: BehaviorGraph,
    excluded: frozenset[int] = frozenset(),
) -> dict[int, tuple[float, float]]:
    """BFS-layered layout with barycenter refinement (adapted from the main plotting module).

    Nodes are columns by BFS depth from START; terminals pinned to the far right.
    10 passes of barycenter + spread give a clean, crossing-minimised layout.
    """
    import networkx as nx
    from collections import defaultdict

    G = nx.DiGraph()
    for nid in graph.nodes:
        if nid not in excluded:
            G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        if src in excluded:
            continue
        for tgt, prob in targets.items():
            if tgt in excluded:
                continue
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


@st.cache_resource(show_spinner=False)
def _extract_node_thumbnails_cached(
    cache_key: str,
    node_eps: tuple[tuple[int, tuple[int, ...]], ...],
    mp4_dir_str: str,
) -> dict[int, list[str]]:
    """Cached worker: ``cache_key`` is unused except to dedup the cache key
    by graph identity + mp4_dir; ``node_eps`` is the canonical hashable form
    of ``{node_id: episode_indices[:2]}`` and is what we actually iterate.
    """
    try:
        import imageio  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return {}

    mp4_dir = Path(mp4_dir_str)
    thumbnails: dict[int, list[str]] = {}
    for nid, ep_indices in node_eps:
        imgs: list[str] = []
        for ep_idx in ep_indices:
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


def _extract_node_thumbnails(
    graph: BehaviorGraph,
    mp4_dir: Path,
) -> dict[int, list[str]]:
    """Cached wrapper. Hashes the graph's non-special nodes' first-two
    episode_indices + mp4_dir so the imageio decode runs once per graph
    revision, not on every Streamlit rerun."""
    node_eps = tuple(
        (nid, tuple(node.episode_indices[:2]))
        for nid, node in sorted(graph.nodes.items())
        if nid not in _SPECIAL_IDS
    )
    cache_key = f"{mp4_dir}|{hash(node_eps)}"
    return _extract_node_thumbnails_cached(cache_key, node_eps, str(mp4_dir))


def _build_graph_json(
    graph: BehaviorGraph,
    pos: dict[int, tuple[float, float]],
    thumbnails: dict[int, list[str]] | None = None,
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    symbol_override: Optional[dict[int, str]] = None,
    color_override: Optional[dict[int, str]] = None,
) -> str:
    """Serialize graph data for the SVG component."""
    symbol_override = symbol_override or {}
    color_override = color_override or {}
    nodes = []
    for nid, node in graph.nodes.items():
        if nid not in pos:
            continue
        if nid in symbol_override:
            symbol = symbol_override[nid]
            color = color_override.get(nid, CLUSTER_COLORS[nid % len(CLUSTER_COLORS)])
            tooltip = node.name
            label = node.name
            is_special = True
        elif nid in _SPECIAL_IDS:
            color = color_override.get(nid, _NODE_COLOR.get(nid, "#888"))
            symbol = (
                "star" if nid == SUCCESS_NODE_ID else
                "x" if nid == FAILURE_NODE_ID else
                "diamond" if nid == START_NODE_ID else
                "square"
            )
            tooltip = node.name
            label = node.name
            is_special = True
        else:
            color = color_override.get(nid, CLUSTER_COLORS[nid % len(CLUSTER_COLORS)])
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
            is_special = False
        node_dict: dict = {
            "id": nid,
            "label": label,
            "color": color,
            "symbol": symbol,
            "num_episodes": node.num_episodes,
            "num_timesteps": node.num_timesteps,
            "is_special": is_special,
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
            if prob < max(0.01, min_edge_prob):
                continue
            if min_edge_count > 0:
                cnt = graph.transition_counts.get(src, {}).get(tgt, 0)
                if cnt < min_edge_count:
                    continue
            if src in pos and tgt in pos:
                edges.append({"src": src, "tgt": tgt, "prob": round(prob, 4)})

    positions = {str(nid): list(xy) for nid, xy in pos.items()}

    return json.dumps({"nodes": nodes, "edges": edges, "positions": positions})


def render_graph_component(
    graph: BehaviorGraph,
    height: int = 580,
    key: str = "behavior_graph",
    highlighted_path: Optional[list[int]] = None,
    mp4_dir: Optional[Path] = None,
    excluded_node_ids: frozenset[int] = frozenset(),
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    pos: Optional[dict[int, tuple[float, float]]] = None,
    symbol_override: Optional[dict[int, str]] = None,
    color_override: Optional[dict[int, str]] = None,
    theme: str = "dark",
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 24.0,
) -> Optional[int]:
    """Render the custom SVG behavior graph component.

    Args:
        highlighted_path: List of node IDs forming a path to highlight.
        mp4_dir: Directory containing MP4 files for thumbnail extraction.
        excluded_node_ids: Nodes to hide (computed by compute_pruned_graph_nodes).

    Returns:
        The node_id that was clicked, or the previously selected node.
    """
    if pos is None:
        pos = _compute_layout(graph, excluded=excluded_node_ids)
    else:
        # Caller-supplied layout: ensure every node has a position.
        pos = dict(pos)
        for nid in graph.nodes:
            if nid not in pos:
                pos[nid] = (0.0, 0.0)
    # Thumbnails on tooltip-hover were costing 3-10s of imageio frame
    # extraction on every Streamlit rerun for ~zero user-visible benefit
    # (only seen on hover; the panel below the graph shows the videos
    # anyway). Disabled.
    thumbnails: dict[int, list[str]] | None = None
    graph_json = _build_graph_json(
        graph, pos, thumbnails=thumbnails, min_edge_prob=min_edge_prob,
        min_edge_count=min_edge_count,
        symbol_override=symbol_override, color_override=color_override,
    )
    selected = st.session_state.get(f"{key}_selected")
    selected_edge = st.session_state.get(f"{key}_selected_edge")
    last_seq = st.session_state.get(f"{key}_last_seq", -1)

    # X-button handlers bump `{key}_render_token` so iframe args differ and
    # Streamlit pushes a fresh render (otherwise the iframe's local state
    # would not update). Click-sequence numbers (see JS clickSeq) make
    # repeat clicks on the same node distinct so the persisted sendValue
    # replay doesn't get treated as a fresh click after dismiss.
    render_token = st.session_state.get(f"{key}_render_token", 0)

    clicked = _graph_component(
        graph_json=graph_json,
        height=height,
        selected_node_id=selected,
        selected_edge=list(selected_edge) if selected_edge else None,
        highlighted_path=highlighted_path,
        render_token=render_token,
        theme=theme,
        edge_style=edge_style,
        edge_width_slope=float(edge_width_slope),
        node_size_slope=float(node_size_slope),
        key=key,
        default=None,
    )

    # Protocol from JS:
    #   {svg_export: xml}    — auto-published SVG snapshot for export
    #                          (dispatched first; dicts never look like
    #                          a click). The JS side already dedupes by
    #                          content; here we dedupe on the stored
    #                          value so st.rerun() doesn't fire when the
    #                          SVG hasn't actually changed.
    #   [node_id, seq]       — node click  (node_id == -1 means deselect)
    #   [src, tgt, seq]      — edge click
    if isinstance(clicked, dict) and "svg_export" in clicked:
        xml = str(clicked.get("svg_export", ""))
        if xml and xml != st.session_state.get("captured_svg"):
            st.session_state["captured_svg"] = xml
            st.session_state[f"{key}_svg_export"] = xml
            st.rerun()
        return selected

    if isinstance(clicked, list) and len(clicked) >= 2:
        try:
            seq = int(clicked[-1])
        except (TypeError, ValueError):
            seq = None
        if seq is not None and seq != last_seq:
            st.session_state[f"{key}_last_seq"] = seq
            if len(clicked) == 2:
                try:
                    node_id = int(clicked[0])
                except (TypeError, ValueError):
                    return selected
                if node_id == -1:
                    _cleared = bool(st.session_state.pop(f"{key}_selected", None) is not None
                                    or st.session_state.pop(f"{key}_selected_edge", None) is not None)
                    # key = f"{key_prefix}_graph" — also clear path selection
                    _kp = key[:-6] if key.endswith("_graph") else key
                    for _k in ("_path_ep_list", "_highlighted_path",
                               "_path_label", "_path_synth_ids"):
                        if st.session_state.pop(f"{_kp}{_k}", None) is not None:
                            _cleared = True
                    if _cleared:
                        st.rerun()
                if node_id in graph.nodes:
                    st.session_state[f"{key}_selected"] = node_id
                    st.session_state.pop(f"{key}_selected_edge", None)
                    # Force a second pass so the iframe args carry the
                    # newly-selected node id; otherwise the iframe message
                    # handler overwrites the JS-local halo back to the
                    # pre-click `selected`. The seq dedup prevents the
                    # second pass from re-processing the click.
                    st.rerun()
            elif len(clicked) == 3:
                try:
                    src_v, tgt_v = int(clicked[0]), int(clicked[1])
                except (TypeError, ValueError):
                    return selected
                st.session_state[f"{key}_selected_edge"] = (src_v, tgt_v)
                st.session_state.pop(f"{key}_selected", None)
                # Same reasoning: ensure args reflect the new edge selection
                # so the iframe halo lands on the correct edge.
                st.rerun()

    return selected
