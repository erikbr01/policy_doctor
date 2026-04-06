"""Pure plotting functions for behavior transition graph visualizations.

Provides network graph and transition matrix heatmap visualizations
for BehaviorGraph objects. No Streamlit imports allowed.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
    BehaviorNode,
)
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS, _get_cluster_color

START_COLOR = "#2ca02c"
END_COLOR = "#888888"
SUCCESS_COLOR = "#2ca02c"
FAILURE_COLOR = "#d62728"


def _get_node_color(node_id: int) -> str:
    if node_id == START_NODE_ID:
        return START_COLOR
    if node_id == SUCCESS_NODE_ID:
        return SUCCESS_COLOR
    if node_id == FAILURE_NODE_ID:
        return FAILURE_COLOR
    if node_id == END_NODE_ID:
        return END_COLOR
    return _get_cluster_color(node_id)


def _behavior_cluster_node_ids(graph: BehaviorGraph) -> List[int]:
    """Behavior / cluster node ids only (excludes START and terminal nodes)."""
    return sorted(nid for nid, node in graph.nodes.items() if not node.is_special)


def _quadratic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    n_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a quadratic Bezier curve through p0 -> control p1 -> p2."""
    t = np.linspace(0, 1, n_points)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return x, y


def _compute_layered_layout(
    G,
    graph: BehaviorGraph,
    x_min: float = -2.5,
    x_max: float = 2.5,
) -> Dict[int, Tuple[float, float]]:
    """Compute a layered left-to-right layout using BFS depth from START.

    Nodes are placed in columns by their shortest path distance from START.
    Vertical positions are refined using barycenter heuristic to reduce
    edge crossings.
    """
    import networkx as nx
    from collections import defaultdict

    terminal_ids = {n for n in G.nodes() if n in TERMINAL_NODE_IDS}

    # BFS shortest distance from START
    try:
        distances = dict(nx.single_source_shortest_path_length(G, START_NODE_ID))
    except nx.NodeNotFound:
        distances = {START_NODE_ID: 0}

    # For nodes not reachable from START, try reverse BFS from each terminal
    rev_distances: Dict[int, int] = {}
    for term_id in terminal_ids:
        try:
            rd = dict(
                nx.single_source_shortest_path_length(G.reverse(), term_id)
            )
            for n, d in rd.items():
                if n not in rev_distances or d < rev_distances[n]:
                    rev_distances[n] = d
        except (nx.NodeNotFound, nx.NetworkXError):
            pass

    max_fwd = max(
        (d for n, d in distances.items() if n not in terminal_ids), default=1
    )

    for node in G.nodes():
        if node not in distances:
            if node in rev_distances:
                distances[node] = max(1, max_fwd + 1 - rev_distances[node])
            else:
                distances[node] = max_fwd // 2 + 1

    # Force all terminal nodes to the rightmost layer
    end_layer = max(
        (d for n, d in distances.items() if n not in terminal_ids), default=1
    ) + 1
    for term_id in terminal_ids:
        distances[term_id] = end_layer

    # Group nodes by layer
    layers: Dict[int, list] = defaultdict(list)
    for node, d in distances.items():
        layers[d].append(node)

    # Sort nodes within each layer for deterministic initial placement
    for layer_idx in layers:
        layers[layer_idx].sort(key=lambda n: (n == END_NODE_ID, n))

    total_layers = max(layers.keys()) if layers else 1

    # Initial positions: evenly spaced within each layer
    pos: Dict[int, Tuple[float, float]] = {}
    for layer_idx, nodes in layers.items():
        x = x_min + (x_max - x_min) * layer_idx / max(total_layers, 1)
        n = len(nodes)
        y_spacing = max(1.0, 1.2 - 0.05 * n)
        for i, node in enumerate(nodes):
            y = (i - (n - 1) / 2) * y_spacing
            pos[node] = (x, y)

    # Barycenter refinement: adjust y-positions to average of neighbors,
    # then re-spread to maintain a minimum gap.  Several passes smooth
    # out crossings.
    min_gap = 0.8
    for _ in range(10):
        for layer_idx in sorted(layers.keys()):
            nodes = layers[layer_idx]
            if len(nodes) <= 1:
                continue
            # Compute barycenter target for each node
            for node in nodes:
                neighbors = list(G.predecessors(node)) + list(G.successors(node))
                neighbor_ys = [pos[nb][1] for nb in neighbors if nb in pos]
                if neighbor_ys:
                    target_y = float(np.mean(neighbor_ys))
                    # Blend toward barycenter
                    old_y = pos[node][1]
                    pos[node] = (pos[node][0], 0.6 * target_y + 0.4 * old_y)

            # Re-spread to avoid overlaps
            nodes_sorted = sorted(nodes, key=lambda n: pos[n][1])
            for i in range(1, len(nodes_sorted)):
                prev_y = pos[nodes_sorted[i - 1]][1]
                curr_y = pos[nodes_sorted[i]][1]
                if curr_y - prev_y < min_gap:
                    pos[nodes_sorted[i]] = (
                        pos[nodes_sorted[i]][0],
                        prev_y + min_gap,
                    )
            # Re-center the layer vertically
            ys = [pos[n][1] for n in nodes_sorted]
            center = float(np.mean(ys))
            for n in nodes_sorted:
                pos[n] = (pos[n][0], pos[n][1] - center)

    return pos


def create_behavior_graph_plot(
    graph: BehaviorGraph,
    min_probability: float = 0.0,
    height: int = 700,
    title: str = "Behavior Transition Graph",
    node_scale: float = 1.0,
) -> go.Figure:
    """Create a directed graph visualization of behavior transitions.

    Uses networkx for layout computation and Plotly for rendering.
    Nodes represent behavioral clusters (plus START/END sentinels).
    Directed edges show transition probabilities between behaviors.

    Args:
        graph: BehaviorGraph with nodes and transition probabilities.
        min_probability: Hide edges below this threshold for cleaner display.
        height: Plot height in pixels.
        title: Plot title.
        node_scale: Multiplier for node marker sizes.

    Returns:
        Plotly Figure object.
    """
    try:
        import networkx as nx
    except ImportError:
        fig = go.Figure()
        fig.add_annotation(
            text="Install networkx for graph visualization: pip install networkx",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig.update_layout(height=height, title=title)
        return fig

    # Build networkx directed graph
    G = nx.DiGraph()
    for node_id in graph.nodes:
        G.add_node(node_id)

    # Collect edges that pass the probability threshold
    edges: List[Tuple[int, int, float, int]] = []
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob >= min_probability and src in graph.nodes and tgt in graph.nodes:
                G.add_edge(src, tgt, weight=prob)
                edges.append((src, tgt, prob, graph.transition_counts[src][tgt]))

    # Layered left-to-right layout: BFS depth determines x, barycenter
    # refinement determines y.  START is pinned at the far left, END at
    # the far right, and cluster nodes are arranged by their shortest
    # path distance from START.
    pos = _compute_layered_layout(G, graph)

    # Identify bidirectional pairs for edge curving
    edge_set = {(s, t) for s, t, _, _ in edges}
    bidir_pairs = set()
    for src, tgt, _, _ in edges:
        if (tgt, src) in edge_set:
            bidir_pairs.add((min(src, tgt), max(src, tgt)))

    fig = go.Figure()

    # --- Draw edges ---
    for src, tgt, prob, count in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            continue

        ux, uy = dx / length, dy / length
        node_radius = 0.12

        # Shorten endpoints so lines don't overlap node markers
        x0s = x0 + ux * node_radius
        y0s = y0 + uy * node_radius
        x1s = x1 - ux * node_radius
        y1s = y1 - uy * node_radius

        line_width = max(1.0, prob * 6)
        opacity = max(0.3, min(0.85, 0.3 + prob * 0.7))
        edge_color = f"rgba(80, 80, 80, {opacity})"

        pair = (min(src, tgt), max(src, tgt))
        is_bidir = pair in bidir_pairs

        if is_bidir:
            perp_x, perp_y = -uy, ux
            direction = 1 if src < tgt else -1
            curve_offset = 0.2 * direction
            ctrl_x = (x0s + x1s) / 2 + perp_x * curve_offset
            ctrl_y = (y0s + y1s) / 2 + perp_y * curve_offset
            bx, by = _quadratic_bezier(
                (x0s, y0s), (ctrl_x, ctrl_y), (x1s, y1s), n_points=40,
            )
            mid_idx = len(bx) // 2
            mid_x, mid_y = float(bx[mid_idx]), float(by[mid_idx])

            # Arrow direction at the end of the curve
            arrow_dx = float(bx[-1] - bx[-3])
            arrow_dy = float(by[-1] - by[-3])

            fig.add_trace(
                go.Scatter(
                    x=bx.tolist() + [None],
                    y=by.tolist() + [None],
                    mode="lines",
                    line=dict(width=line_width, color=edge_color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            mid_x = (x0s + x1s) / 2
            mid_y = (y0s + y1s) / 2
            arrow_dx = ux
            arrow_dy = uy

            fig.add_trace(
                go.Scatter(
                    x=[x0s, x1s, None],
                    y=[y0s, y1s, None],
                    mode="lines",
                    line=dict(width=line_width, color=edge_color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Arrowhead annotation at the target
        arrow_len = np.sqrt(arrow_dx ** 2 + arrow_dy ** 2)
        if arrow_len > 1e-6:
            anx = arrow_dx / arrow_len
            any_ = arrow_dy / arrow_len
            fig.add_annotation(
                x=x1s, y=y1s,
                ax=x1s - anx * 0.08, ay=y1s - any_ * 0.08,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=max(1.5, line_width * 0.8),
                arrowcolor=edge_color,
                standoff=0,
            )

        # Edge label at midpoint
        src_name = graph.nodes[src].name if src in graph.nodes else str(src)
        tgt_name = graph.nodes[tgt].name if tgt in graph.nodes else str(tgt)

        # Offset label slightly perpendicular to edge to avoid overlap
        perp_label_x, perp_label_y = -uy * 0.04, ux * 0.04

        fig.add_trace(
            go.Scatter(
                x=[mid_x + perp_label_x],
                y=[mid_y + perp_label_y],
                mode="text",
                text=[f"{prob:.0%}"],
                textfont=dict(size=9, color="rgba(50,50,50,0.85)"),
                hovertext=[
                    f"{src_name} → {tgt_name}<br>"
                    f"Probability: {prob:.1%}<br>"
                    f"Count: {count}",
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

    # --- Draw nodes ---
    for node_id, node in graph.nodes.items():
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        color = _get_node_color(node_id)

        if node.is_start:
            symbol = "diamond"
            size = 30 * node_scale
        elif node.is_success:
            symbol = "star"
            size = 32 * node_scale
        elif node.is_failure:
            symbol = "x"
            size = 30 * node_scale
        elif node.is_end:
            symbol = "square"
            size = 30 * node_scale
        else:
            symbol = "circle"
            size = (22 + np.log1p(node.num_timesteps) * 3) * node_scale

        hover_parts = [f"<b>{node.name}</b>"]
        if node.is_end:
            hover_parts.append(f"Episodes: {node.num_episodes:,}")
        elif not node.is_special:
            hover_parts.append(f"Timesteps: {node.num_timesteps:,}")
            hover_parts.append(f"Episodes: {node.num_episodes:,}")

        outgoing = graph.get_outgoing_transitions(node_id)
        if outgoing:
            hover_parts.append("")
            hover_parts.append("<b>Outgoing:</b>")
            for tgt_id, cnt, p in outgoing[:6]:
                tgt_name = graph.nodes.get(tgt_id, None)
                tgt_label = tgt_name.name if tgt_name else str(tgt_id)
                hover_parts.append(f"  → {tgt_label}: {p:.0%} ({cnt})")
            if len(outgoing) > 6:
                hover_parts.append(f"  ... +{len(outgoing) - 6} more")

        incoming = graph.get_incoming_transitions(node_id)
        if incoming:
            hover_parts.append("")
            hover_parts.append("<b>Incoming:</b>")
            for src_id, cnt, p in incoming[:6]:
                src_name = graph.nodes.get(src_id, None)
                src_label = src_name.name if src_name else str(src_id)
                hover_parts.append(f"  ← {src_label}: {p:.0%} ({cnt})")
            if len(incoming) > 6:
                hover_parts.append(f"  ... +{len(incoming) - 6} more")

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color="white"),
                ),
                text=[node.name],
                textposition="top center",
                textfont=dict(size=11, color="black"),
                hovertext=["<br>".join(hover_parts)],
                hoverinfo="text",
                showlegend=False,
                name=node.name,
            )
        )

    # Compute axis ranges from positions
    all_x = [pos[n][0] for n in pos]
    all_y = [pos[n][1] for n in pos]
    x_margin = 0.5
    y_margin = 0.5

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(
            visible=False,
            range=[min(all_x) - x_margin, max(all_x) + x_margin],
        ),
        yaxis=dict(
            visible=False,
            scaleanchor="x",
            range=[min(all_y) - y_margin, max(all_y) + y_margin],
        ),
        plot_bgcolor="white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_transition_matrix_heatmap(
    graph: BehaviorGraph,
    height: int = 500,
    title: str = "Transition Probability Matrix",
) -> go.Figure:
    """Create a heatmap of transition probabilities between behaviors.

    Rows are source behaviors, columns are targets. Cell values are
    P(target | source). START and END nodes are included.

    Args:
        graph: BehaviorGraph with transition probabilities.
        height: Plot height in pixels.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    # Order: START, cluster nodes sorted, terminal nodes
    cluster_ids = _behavior_cluster_node_ids(graph)
    terminal_ids = sorted(graph.terminal_node_ids)
    node_ids = [START_NODE_ID] + cluster_ids + terminal_ids
    labels = [graph.nodes[nid].name for nid in node_ids]

    n = len(node_ids)
    matrix = np.zeros((n, n))
    for i, src in enumerate(node_ids):
        for j, tgt in enumerate(node_ids):
            matrix[i, j] = graph.transition_probs.get(src, {}).get(tgt, 0.0)

    # Text annotations with probability and count
    text = []
    for i, src in enumerate(node_ids):
        row_text = []
        for j, tgt in enumerate(node_ids):
            p = matrix[i, j]
            count = graph.transition_counts.get(src, {}).get(tgt, 0)
            if p > 0:
                row_text.append(f"{p:.0%}<br>({count})")
            else:
                row_text.append("")
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar_title="P(target|source)",
            hovertemplate=(
                "From: %{y}<br>To: %{x}<br>"
                "Probability: %{z:.1%}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=max(height, n * 45 + 120),
        xaxis_title="Target Behavior",
        yaxis_title="Source Behavior",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def create_behavior_paths_plot(
    graph: BehaviorGraph,
    paths: List[Tuple[List[int], float, List[Tuple[int, int, float]]]],
    height: int = 700,
    title: str = "Behavior Paths (START → END)",
) -> go.Figure:
    """Show all simple paths from START to END as parallel horizontal lanes.

    Each path is a row.  Nodes are colored circles connected by arrows.
    Loop-back edges are drawn as curved arcs above the lane.

    Args:
        graph: BehaviorGraph (used for node names / colors).
        paths: Output of ``BehaviorGraph.enumerate_paths()``.
        height: Plot height in pixels.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    if not paths:
        fig = go.Figure()
        fig.add_annotation(
            text="No paths found from START to END.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig.update_layout(height=300, title=title)
        return fig

    n_paths = len(paths)
    max_len = max(len(p) for p, _, _ in paths)

    # Grid: x = position in chain, y = path row (top = most probable)
    row_gap = 1.5
    node_radius = 0.18

    fig = go.Figure()

    # Collect all x/y for axis range
    all_x: List[float] = []
    all_y: List[float] = []

    for row, (path, prob, loops) in enumerate(paths):
        y_row = -row * row_gap  # top-to-bottom

        # --- Forward edges (arrows between consecutive nodes) ---
        for step in range(len(path) - 1):
            x0 = float(step)
            x1 = float(step + 1)
            fig.add_trace(
                go.Scatter(
                    x=[x0 + node_radius, x1 - node_radius, None],
                    y=[y_row, y_row, None],
                    mode="lines",
                    line=dict(width=2, color="rgba(80,80,80,0.5)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Arrowhead
            fig.add_annotation(
                x=x1 - node_radius, y=y_row,
                ax=x1 - node_radius - 0.12, ay=y_row,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor="rgba(80,80,80,0.5)",
                standoff=0,
            )
            # Edge probability label
            edge_prob = graph.transition_probs.get(
                path[step], {}
            ).get(path[step + 1], 0.0)
            fig.add_trace(
                go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[y_row + 0.22],
                    mode="text",
                    text=[f"{edge_prob:.0%}"],
                    textfont=dict(size=8, color="rgba(80,80,80,0.7)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # --- Loop-back arcs ---
        for src_node, tgt_node, loop_prob in loops:
            src_idx = path.index(src_node)
            tgt_idx = path.index(tgt_node)
            x_src = float(src_idx)
            x_tgt = float(tgt_idx)
            span = abs(src_idx - tgt_idx)
            arc_height = 0.35 + 0.12 * span
            ctrl_x = (x_src + x_tgt) / 2
            ctrl_y = y_row + arc_height

            bx, by = _quadratic_bezier(
                (x_src, y_row + node_radius * 0.8),
                (ctrl_x, ctrl_y),
                (x_tgt, y_row + node_radius * 0.8),
                n_points=30,
            )
            fig.add_trace(
                go.Scatter(
                    x=bx.tolist() + [None],
                    y=by.tolist() + [None],
                    mode="lines",
                    line=dict(
                        width=1.5,
                        color="rgba(200,60,60,0.55)",
                        dash="dot",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Arrow at loop target
            fig.add_annotation(
                x=float(bx[-1]), y=float(by[-1]),
                ax=float(bx[-3]), ay=float(by[-3]),
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=1.2,
                arrowcolor="rgba(200,60,60,0.55)",
                standoff=0,
            )
            # Loop label
            src_name = graph.nodes[src_node].name
            tgt_name = graph.nodes[tgt_node].name
            fig.add_trace(
                go.Scatter(
                    x=[ctrl_x],
                    y=[ctrl_y + 0.08],
                    mode="text",
                    text=[f"{loop_prob:.0%}"],
                    textfont=dict(size=8, color="rgba(200,60,60,0.75)"),
                    hovertext=[
                        f"Loop: {src_name} → {tgt_name}<br>"
                        f"Probability: {loop_prob:.1%}",
                    ],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            all_y.append(ctrl_y + 0.15)

        # --- Nodes ---
        for step, node_id in enumerate(path):
            x = float(step)
            color = _get_node_color(node_id)
            node = graph.nodes.get(node_id)
            name = node.name if node else str(node_id)

            if node_id == START_NODE_ID:
                symbol = "diamond"
                size = 22
            elif node_id == SUCCESS_NODE_ID:
                symbol = "star"
                size = 24
            elif node_id == FAILURE_NODE_ID:
                symbol = "x"
                size = 22
            elif node_id in TERMINAL_NODE_IDS:
                symbol = "square"
                size = 22
            else:
                symbol = "circle"
                size = 20

            hover = [f"<b>{name}</b>"]
            if node and not node.is_special:
                hover.append(f"Timesteps: {node.num_timesteps:,}")
                hover.append(f"Episodes: {node.num_episodes:,}")

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y_row],
                    mode="markers+text",
                    marker=dict(
                        size=size,
                        color=color,
                        symbol=symbol,
                        line=dict(width=1.5, color="white"),
                    ),
                    text=[name],
                    textposition="bottom center",
                    textfont=dict(size=9, color="black"),
                    hovertext=["<br>".join(hover)],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            all_x.append(x)
            all_y.append(y_row)

        # Path probability label on the left
        fig.add_annotation(
            x=-0.7,
            y=y_row,
            text=f"<b>Path {row + 1}</b>  ({prob:.1%})",
            showarrow=False,
            xanchor="right",
            font=dict(size=11),
            xref="x",
            yref="y",
        )

    # Layout
    x_margin = 1.2
    y_margin = 0.8
    fig.update_layout(
        title=title,
        height=max(height, n_paths * 80 + 120),
        xaxis=dict(
            visible=False,
            range=[min(all_x) - x_margin, max(all_x) + x_margin],
        ),
        yaxis=dict(
            visible=False,
            range=[min(all_y) - y_margin, max(all_y) + y_margin],
        ),
        plot_bgcolor="white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def _value_to_rgb(value: float, vmin: float, vmax: float) -> str:
    """Map a value to a diverging red-white-green color string."""
    if vmax == vmin:
        return "rgb(255,255,255)"
    # Normalize to [-1, 1]
    span = max(abs(vmin), abs(vmax))
    if span == 0:
        t = 0.0
    else:
        t = np.clip(value / span, -1.0, 1.0)
    if t >= 0:
        # White (255,255,255) -> Green (39,174,96)
        r = int(255 - t * (255 - 39))
        g = int(255 - t * (255 - 174))
        b = int(255 - t * (255 - 96))
    else:
        # White (255,255,255) -> Red (214,39,40)
        s = -t
        r = int(255 - s * (255 - 214))
        g = int(255 - s * (255 - 39))
        b = int(255 - s * (255 - 40))
    return f"rgb({r},{g},{b})"


def _sequential_viridis_rgb(t: float) -> str:
    """Map t in [0, 1] to an RGB string (viridis-like sequential scale)."""
    t = float(np.clip(t, 0.0, 1.0))
    stops = np.array(
        [
            [68, 1, 84],
            [72, 40, 120],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=float,
    )
    x = t * (stops.shape[0] - 1)
    i = int(np.floor(x))
    i = min(i, stops.shape[0] - 2)
    f = x - i
    c = stops[i] * (1.0 - f) + stops[i + 1] * f
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def _timestep_count_to_rgb(count: int, cmin: int, cmax: int) -> str:
    """Sequential color from low (dark purple) to high (yellow) by timestep count."""
    if cmax <= cmin:
        return _sequential_viridis_rgb(0.5)
    t = (float(count) - float(cmin)) / float(cmax - cmin)
    return _sequential_viridis_rgb(t)


def create_node_value_bar_chart(
    graph: BehaviorGraph,
    values: Dict[int, float],
    height: int = 400,
    title: str = "Node Values (Bellman)",
) -> go.Figure:
    """Horizontal bar chart of per-node values on a diverging red-green scale.

    Args:
        graph: BehaviorGraph for node names.
        values: Dict mapping node_id to V(s).
        height: Plot height in pixels.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    # Order: START, cluster nodes sorted by value desc, then terminals
    cluster_ids = sorted(
        _behavior_cluster_node_ids(graph),
        key=lambda nid: values.get(nid, 0),
        reverse=True,
    )
    terminal_ids = sorted(graph.terminal_node_ids)
    ordered = [START_NODE_ID] + cluster_ids + terminal_ids
    ordered = [nid for nid in ordered if nid in values]

    names = [graph.nodes[nid].name for nid in ordered]
    vals = [values[nid] for nid in ordered]

    vmin = min(vals) if vals else -1
    vmax = max(vals) if vals else 1
    colors = [_value_to_rgb(v, vmin, vmax) for v in vals]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=vals,
            orientation="h",
            marker_color=colors,
            marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(height, len(ordered) * 32 + 100),
        xaxis_title="V(s)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        margin=dict(l=120, r=60, t=50, b=40),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def create_advantage_matrix_heatmap(
    graph: BehaviorGraph,
    node_values: Dict[int, float],
    gamma: float = 0.99,
    height: int = 500,
    title: str = "Advantage matrix  A(s, s′) = Value(s→s′) − V(s),  Value(s→s′) = γ·V(s′)",
) -> go.Figure:
    """Heatmap of advantage A(s, s′) = Value(s→s′) − V(s) for each transition (Value = γ·V(s′)).

    Args:
        graph: BehaviorGraph for structure and transition data.
        node_values: Dict from ``compute_values()``.
        gamma: Discount factor.
        height: Plot height in pixels.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    cluster_ids = _behavior_cluster_node_ids(graph)
    terminal_ids = sorted(graph.terminal_node_ids)
    src_ids = [START_NODE_ID] + cluster_ids
    tgt_ids = cluster_ids + terminal_ids
    src_labels = [graph.nodes[n].name for n in src_ids]
    tgt_labels = [graph.nodes[n].name for n in tgt_ids]

    nr, nc = len(src_ids), len(tgt_ids)
    matrix = np.full((nr, nc), np.nan)
    text = [[""] * nc for _ in range(nr)]

    for i, src in enumerate(src_ids):
        v_s = node_values.get(src, 0.0)
        for j, tgt in enumerate(tgt_ids):
            prob = graph.transition_probs.get(src, {}).get(tgt, 0.0)
            if prob > 0:
                q = gamma * node_values.get(tgt, 0.0)
                adv = q - v_s
                matrix[i, j] = adv
                count = graph.transition_counts.get(src, {}).get(tgt, 0)
                text[i][j] = f"{adv:+.3f}<br>P={prob:.0%} ({count})"

    abs_max = np.nanmax(np.abs(matrix[np.isfinite(matrix)])) if np.any(np.isfinite(matrix)) else 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=tgt_labels,
            y=src_labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar_title="A(s,s')",
            hovertemplate="From: %{y}<br>To: %{x}<br>A = %{z:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(height, nr * 45 + 120),
        xaxis_title="Target (s')",
        yaxis_title="Source (s)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_q_value_distribution_plot(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    q_values: np.ndarray,
    node_values: Dict[int, float],
    height: int = 400,
    title: str = "Transition value distribution per cluster",
) -> go.Figure:
    """Box plot showing the spread of per-slice transition values within each cluster.

    Args:
        graph: BehaviorGraph for node names.
        cluster_labels: Per-slice cluster assignments.
        q_values: Per-slice transition values from ``compute_slice_values()`` (γ·V(s′) along the slice).
        node_values: Dict of V(s) for reference lines.
        height: Plot height in pixels.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    cluster_ids = _behavior_cluster_node_ids(graph)

    fig = go.Figure()
    for c_id in cluster_ids:
        mask = cluster_labels == c_id
        if not np.any(mask):
            continue
        qs = q_values[mask]
        name = graph.nodes[c_id].name
        v_s = node_values.get(c_id, 0.0)

        color = _get_cluster_color(c_id)
        fig.add_trace(
            go.Box(
                y=qs,
                name=name,
                marker_color=color,
                boxmean=True,
                hoverinfo="y+name",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        yaxis_title="Transition value Value(s→s′)",
        xaxis_title="Behavior cluster",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def create_episode_value_timeline(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    q_values: np.ndarray,
    level: str = "rollout",
    max_episodes: int = 30,
    height: int = 500,
    title: str = "Transition value timeline per episode",
) -> go.Figure:
    """Line plot of per-slice transition values along the timestep axis per episode.

    Episodes are sorted by their mean transition value (successful episodes tend to
    be at the top).

    Args:
        cluster_labels: Per-slice cluster assignments.
        metadata: Per-slice metadata dicts.
        q_values: Per-slice transition values (γ·V(s′) along the transition).
        level: "rollout" or "demo".
        max_episodes: Cap on the number of episodes to show.
        height: Plot height.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    episodes: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
    ep_outcomes: Dict[int, Optional[bool]] = {}
    for i, meta in enumerate(metadata):
        ep_idx = meta[ep_key]
        t = meta.get("timestep", meta.get("window_start", 0))
        episodes[ep_idx].append((t, float(q_values[i]), int(cluster_labels[i])))
        if "success" in meta and ep_idx not in ep_outcomes:
            ep_outcomes[ep_idx] = meta["success"]

    # Sort episodes by mean transition value descending
    ep_means = {
        ep: float(np.mean([q for _, q, lbl in slices if lbl != -1]))
        for ep, slices in episodes.items()
        if any(lbl != -1 for _, _, lbl in slices)
    }
    sorted_eps = sorted(ep_means, key=lambda e: -ep_means[e])[:max_episodes]

    fig = go.Figure()
    for ep_idx in sorted_eps:
        slices = sorted(episodes[ep_idx], key=lambda x: x[0])
        ts = [s[0] for s in slices if s[2] != -1]
        qs = [s[1] for s in slices if s[2] != -1]
        if not ts:
            continue

        outcome = ep_outcomes.get(ep_idx)
        if outcome is True:
            color = "rgba(39,174,96,0.6)"
            dash = "solid"
        elif outcome is False:
            color = "rgba(214,39,40,0.6)"
            dash = "solid"
        else:
            color = "rgba(128,128,128,0.4)"
            dash = "dot"

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=qs,
                mode="lines",
                line=dict(color=color, width=1.5, dash=dash),
                name=f"Ep {ep_idx}" + (
                    " ✓" if outcome is True else " ✗" if outcome is False else ""
                ),
                hovertemplate=f"Ep {ep_idx}<br>t=%{{x}}<br>Value=%{{y:+.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Timestep",
        yaxis_title="Transition value Value(s→s′)",
        plot_bgcolor="white",
        legend=dict(font=dict(size=9)),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def create_episode_q_advantage_timeline(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    q_values: np.ndarray,
    advantages: np.ndarray,
    ep_idx: int,
    level: str = "rollout",
    height: int = 360,
    title: Optional[str] = None,
) -> go.Figure:
    """Line plot of Value(s→s′) and Advantage over timesteps for a single episode.

    Filters to slices belonging to the given episode and excludes noise (cluster -1).
    Two traces: transition value and Advantage.

    Args:
        cluster_labels: Per-slice cluster assignments.
        metadata: Per-slice metadata dicts with episode index and timestep.
        q_values: Per-slice transition values (γ·V(s′) along the transition).
        advantages: Per-slice advantages.
        ep_idx: Episode index (rollout_idx or demo_idx).
        level: "rollout" or "demo".
        height: Plot height in pixels.
        title: Optional plot title.

    Returns:
        Plotly Figure with two lines (value and Advantage).
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    points: List[Tuple[int, float, float]] = []
    for i, meta in enumerate(metadata):
        if meta[ep_key] != ep_idx or int(cluster_labels[i]) == -1:
            continue
        t = meta.get("timestep", meta.get("window_start", 0))
        points.append((t, float(q_values[i]), float(advantages[i])))

    points.sort(key=lambda x: x[0])
    if not points:
        fig = go.Figure()
        fig.update_layout(
            title=title or f"Episode {ep_idx} — Value(s→s′) & Advantage",
            height=height,
        )
        return fig

    ts = [p[0] for p in points]
    qs = [p[1] for p in points]
    advs = [p[2] for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=qs,
            mode="lines",
            line=dict(color="rgb(31, 119, 180)", width=1.5),
            name="Value(s→s′)",
            hovertemplate="t=%{x}<br>Value=%{y:+.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=advs,
            mode="lines",
            line=dict(color="rgb(255, 127, 14)", width=1.5),
            name="Advantage",
            hovertemplate="t=%{x}<br>A=%{y:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or f"Episode {ep_idx} — Value(s→s′) & Advantage",
        height=height,
        xaxis_title="Timestep",
        yaxis_title="Value / Advantage",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def create_value_heatmap(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    q_values: np.ndarray,
    level: str = "rollout",
    height: int = 600,
    title: str = "Transition value heatmap (episodes × timesteps)",
) -> go.Figure:
    """Heatmap with episodes on y-axis, timesteps on x-axis, colored by transition value.

    Episodes are sorted by outcome (success first) then by mean transition value.

    Args:
        cluster_labels: Per-slice cluster assignments.
        metadata: Per-slice metadata dicts.
        q_values: Per-slice transition values.
        level: "rollout" or "demo".
        height: Plot height.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    episodes: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
    ep_outcomes: Dict[int, Optional[bool]] = {}
    for i, meta in enumerate(metadata):
        ep_idx = meta[ep_key]
        t = meta.get("timestep", meta.get("window_start", 0))
        episodes[ep_idx].append((t, float(q_values[i]), int(cluster_labels[i])))
        if "success" in meta and ep_idx not in ep_outcomes:
            ep_outcomes[ep_idx] = meta["success"]

    # Sort: success first, then by mean transition value desc
    def _sort_key(ep):
        outcome = ep_outcomes.get(ep)
        slices = episodes[ep]
        valid_qs = [q for _, q, lbl in slices if lbl != -1]
        mean_q = float(np.mean(valid_qs)) if valid_qs else 0.0
        outcome_rank = 0 if outcome is True else 1 if outcome is None else 2
        return (outcome_rank, -mean_q)

    sorted_eps = sorted(episodes.keys(), key=_sort_key)

    # Find global timestep range
    all_ts = set()
    for slices in episodes.values():
        for t, _, lbl in slices:
            if lbl != -1:
                all_ts.add(t)
    if not all_ts:
        fig = go.Figure()
        fig.update_layout(title=title, height=300)
        return fig

    t_min, t_max = min(all_ts), max(all_ts)
    t_range = list(range(t_min, t_max + 1))
    t_to_col = {t: j for j, t in enumerate(t_range)}
    n_eps = len(sorted_eps)
    n_ts = len(t_range)

    matrix = np.full((n_eps, n_ts), np.nan)
    for row, ep_idx in enumerate(sorted_eps):
        for t, q, lbl in episodes[ep_idx]:
            if lbl != -1 and t in t_to_col:
                matrix[row, t_to_col[t]] = q

    y_labels = []
    for ep in sorted_eps:
        outcome = ep_outcomes.get(ep)
        suffix = " ✓" if outcome is True else " ✗" if outcome is False else ""
        y_labels.append(f"Ep {ep}{suffix}")

    abs_max = np.nanmax(np.abs(matrix[np.isfinite(matrix)])) if np.any(np.isfinite(matrix)) else 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=t_range,
            y=y_labels,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar_title="Value(s→s′)",
            hovertemplate="Episode: %{y}<br>Timestep: %{x}<br>Value = %{z:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(height, n_eps * 18 + 120),
        xaxis_title="Timestep",
        yaxis_title="Episode",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _compute_node_levels(graph: BehaviorGraph) -> Dict[int, int]:
    """Compute hierarchical levels for all nodes via BFS from START.

    START gets level 0, terminal nodes get the max level, and cluster
    nodes are placed by their shortest-path distance from START.
    """
    from collections import deque

    levels: Dict[int, int] = {START_NODE_ID: 0}
    queue = deque([START_NODE_ID])
    terminal_ids = graph.terminal_node_ids

    while queue:
        node = queue.popleft()
        for tgt in graph.transition_probs.get(node, {}):
            if tgt not in levels and tgt in graph.nodes:
                if tgt in terminal_ids:
                    continue
                levels[tgt] = levels[node] + 1
                queue.append(tgt)

    max_level = max(levels.values(), default=0) + 1
    for tid in terminal_ids:
        if tid in graph.nodes:
            levels[tid] = max_level

    # Catch any unreachable nodes
    for nid in graph.nodes:
        if nid not in levels:
            levels[nid] = max_level // 2

    return levels


def _precompute_positions(
    graph: BehaviorGraph,
    min_probability: float = 0.0,
) -> Dict[int, Tuple[float, float]]:
    """Compute the same layered positions used by the Plotly graph view.

    Returns positions in abstract coordinates (will be scaled to pixels
    before passing to vis.js).
    """
    import networkx as nx

    G = nx.DiGraph()
    for node_id in graph.nodes:
        G.add_node(node_id)
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob >= min_probability and src in graph.nodes and tgt in graph.nodes:
                G.add_edge(src, tgt, weight=prob)

    return _compute_layered_layout(G, graph)


def _build_interactive_behavior_graph_html(
    graph: BehaviorGraph,
    min_probability: float = 0.0,
    height: str = "650px",
    width: str = "100%",
    physics_enabled: bool = True,
    layout_algorithm: str = "layeredStatic",
    cluster_node_color_fn: Optional[Callable[[int, BehaviorNode], str]] = None,
) -> str:
    """Shared Pyvis HTML builder (also used for timestep-colored graphs)."""
    import json

    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )

    use_hierarchical = layout_algorithm == "hierarchicalRepulsion"
    use_layered_static = layout_algorithm == "layeredStatic"
    node_levels = _compute_node_levels(graph) if use_hierarchical else {}
    precomputed_pos: Dict[int, Tuple[float, float]] = {}

    if use_layered_static:
        precomputed_pos = _precompute_positions(graph, min_probability)

    # Build options dict properly to avoid JSON formatting issues
    options: dict = {
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "tooltipDelay": 100,
        },
    }

    if use_layered_static:
        options["physics"] = {"enabled": False}
        options["edges"] = {"smooth": {"type": "curvedCW", "roundness": 0.15}}
    elif use_hierarchical:
        options["layout"] = {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "nodeSpacing": 180,
                "levelSeparation": 250,
                "treeSpacing": 200,
            },
        }
        options["physics"] = {
            "enabled": physics_enabled,
            "solver": "hierarchicalRepulsion",
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 200,
                "springConstant": 0.01,
                "nodeDistance": 200,
                "damping": 0.09,
                "avoidOverlap": 0.5,
            },
        }
        options["edges"] = {
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "horizontal",
            },
        }
    else:
        options["physics"] = {
            "enabled": physics_enabled,
            "solver": layout_algorithm,
            layout_algorithm: {
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.3,
            },
        }
        options["edges"] = {"smooth": {"type": "dynamic"}}

    net.set_options(json.dumps(options))

    # Use string node IDs to avoid potential vis.js issues with negatives
    def _nid(node_id: int) -> str:
        return str(node_id)

    # Add nodes
    for node_id, node in graph.nodes.items():
        if cluster_node_color_fn is not None and not node.is_special:
            color = cluster_node_color_fn(node_id, node)
        else:
            color = _get_node_color(node_id)

        if node.is_start:
            shape = "diamond"
            size = 35
        elif node.is_success:
            shape = "star"
            size = 38
        elif node.is_failure:
            shape = "triangleDown"
            size = 35
        elif node.is_end:
            shape = "square"
            size = 35
        else:
            shape = "dot"
            size = max(20, min(55, 18 + np.log1p(node.num_timesteps) * 4))

        title_parts = [node.name]
        if node.is_end:
            title_parts.append(f"Episodes: {node.num_episodes:,}")
        elif not node.is_special:
            title_parts.append(f"Timesteps: {node.num_timesteps:,}")
            title_parts.append(f"Episodes: {node.num_episodes:,}")

        outgoing = graph.get_outgoing_transitions(node_id)
        if outgoing:
            title_parts.append("")
            title_parts.append("Outgoing:")
            for tgt_id, cnt, p in outgoing[:8]:
                tgt_name = graph.nodes[tgt_id].name if tgt_id in graph.nodes else str(tgt_id)
                title_parts.append(f"  → {tgt_name}: {p:.0%} ({cnt})")

        incoming = graph.get_incoming_transitions(node_id)
        if incoming:
            title_parts.append("")
            title_parts.append("Incoming:")
            for src_id, cnt, p in incoming[:8]:
                src_name = graph.nodes[src_id].name if src_id in graph.nodes else str(src_id)
                title_parts.append(f"  ← {src_name}: {p:.0%} ({cnt})")

        kwargs: dict = dict(
            label=node.name,
            title="\n".join(title_parts),
            color=color,
            shape=shape,
            size=int(size),
            font={"size": 14, "color": "#333"},
            borderWidth=2,
            borderWidthSelected=3,
        )
        if use_hierarchical and node_id in node_levels:
            kwargs["level"] = node_levels[node_id]
        if use_layered_static and node_id in precomputed_pos:
            px, py = precomputed_pos[node_id]
            kwargs["x"] = int(px * 250)
            kwargs["y"] = int(py * 200)
            kwargs["physics"] = False

        net.add_node(_nid(node_id), **kwargs)

    # Add edges
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob < min_probability:
                continue
            if src not in graph.nodes or tgt not in graph.nodes:
                continue

            count = graph.transition_counts.get(src, {}).get(tgt, 0)
            edge_width = max(1.0, prob * 8)
            opacity = max(0.25, min(0.9, 0.2 + prob * 0.8))

            src_name = graph.nodes[src].name
            tgt_name = graph.nodes[tgt].name

            net.add_edge(
                _nid(src),
                _nid(tgt),
                value=edge_width,
                title=f"{src_name} → {tgt_name}\nP = {prob:.1%}\nCount: {count}",
                label=f"{prob:.0%}" if prob >= 0.10 else "",
                arrows="to",
                color={"color": f"rgba(80,80,80,{opacity})", "highlight": "#1f77b4"},
                font={"size": 10, "color": "#555", "align": "top"},
                smooth=True,
            )

    # Generate HTML string
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
    with open(f.name, "r") as f:
        html = f.read()

    return html


def create_interactive_behavior_graph(
    graph: BehaviorGraph,
    min_probability: float = 0.0,
    height: str = "650px",
    width: str = "100%",
    physics_enabled: bool = True,
    layout_algorithm: str = "layeredStatic",
    cluster_node_color_fn: Optional[Callable[[int, BehaviorNode], str]] = None,
) -> str:
    """Create an interactive, draggable graph using Pyvis (vis.js).

    Nodes can be dragged and repositioned. By default uses the same layered
    static layout as ``create_value_colored_interactive_graph`` (no physics).
    Other layout algorithms enable vis.js physics for auto-arrangement.

    Args:
        graph: BehaviorGraph with nodes and transition probabilities.
        min_probability: Hide edges below this threshold.
        height: CSS height for the graph container.
        width: CSS width for the graph container.
        physics_enabled: Whether physics simulation is active initially
            (ignored when ``layout_algorithm`` is ``layeredStatic``).
        layout_algorithm: ``layeredStatic`` (default): same fixed layout as the
            value-colored graph, drag freely. Or ``hierarchicalRepulsion``,
            ``barnesHut``, ``forceAtlas2Based``, ``repulsion`` for force-directed
            layouts with physics.
        cluster_node_color_fn: If set, non-special (behavior cluster) nodes use
            this fill color; START / SUCCESS / FAILURE / END keep semantic colors.

    Returns:
        HTML string that can be embedded via ``st.components.v1.html()``.
    """
    return _build_interactive_behavior_graph_html(
        graph=graph,
        min_probability=min_probability,
        height=height,
        width=width,
        physics_enabled=physics_enabled,
        layout_algorithm=layout_algorithm,
        cluster_node_color_fn=cluster_node_color_fn,
    )


def create_value_colored_interactive_graph(
    graph: BehaviorGraph,
    values: Dict[int, float],
    gamma: float = 0.99,
    min_probability: float = 0.0,
    height: str = "650px",
    width: str = "100%",
    highlight_edges_below_advantage: Optional[float] = None,
    highlight_edges_above_advantage: Optional[float] = None,
    highlight_nodes_below_value: Optional[float] = None,
    highlight_nodes_above_value: Optional[float] = None,
) -> str:
    """Pyvis interactive graph with nodes colored by V(s) and edges by transition value Value(s→s′).

    Edge color encodes γ·V(s′) (per-transition value in this MRP). Uses the layered static layout
    (same as the Plotly Graph View) with physics disabled so nodes can be freely dragged.

    Args:
        graph: BehaviorGraph with nodes and transition probabilities.
        values: Dict mapping node_id to V(s).
        gamma: Discount factor for transition values on edges (γ·V(s′)).
        min_probability: Hide edges below this threshold.
        height: CSS height for the container.
        width: CSS width for the container.
        highlight_edges_below_advantage: If set, edges with advantage A(s,s') below
            this value are styled differently (red, thicker) to mark "curate out" edges.
        highlight_edges_above_advantage: If set, edges with advantage A(s,s') above
            this value are styled differently (green, thicker) to mark "select" edges.
        highlight_nodes_below_value: If set, nodes with V(s) below this value are
            styled with a red border (for value-based "curate out").
        highlight_nodes_above_value: If set, nodes with V(s) above this value are
            styled with a green border (for value-based "select").

    Returns:
        HTML string for ``st.components.v1.html()``.
    """
    import json

    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )

    precomputed_pos = _precompute_positions(graph, min_probability)

    options: dict = {
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "tooltipDelay": 100,
        },
        "physics": {"enabled": False},
        "edges": {"smooth": {"type": "curvedCW", "roundness": 0.15}},
    }
    net.set_options(json.dumps(options))

    def _nid(node_id: int) -> str:
        return str(node_id)

    all_vals = [v for v in values.values()]
    vmin = min(all_vals) if all_vals else -1
    vmax = max(all_vals) if all_vals else 1

    for node_id, node in graph.nodes.items():
        v = values.get(node_id, 0.0)
        color = _value_to_rgb(v, vmin, vmax)

        # Value-based node highlighting (curate out = red border, select = green border)
        border_color = "rgba(0,0,0,0.4)"
        border_width = 2
        if highlight_nodes_below_value is not None and v < highlight_nodes_below_value:
            border_color = "#d62728"
            border_width = 4
        elif highlight_nodes_above_value is not None and v >= highlight_nodes_above_value:
            border_color = "#2ca02c"
            border_width = 4

        if node.is_start:
            shape = "diamond"
            size = 35
        elif node.is_success:
            shape = "star"
            size = 38
        elif node.is_failure:
            shape = "triangleDown"
            size = 35
        elif node.is_end:
            shape = "square"
            size = 35
        else:
            shape = "dot"
            size = max(20, min(55, 18 + np.log1p(node.num_timesteps) * 4))

        title_parts = [
            f"{node.name}",
            f"V(s) = {v:+.4f}",
        ]
        if not node.is_special:
            title_parts.append(f"Timesteps: {node.num_timesteps:,}")
            title_parts.append(f"Episodes: {node.num_episodes:,}")

        outgoing = graph.get_outgoing_transitions(node_id)
        if outgoing:
            title_parts.append("")
            title_parts.append("Outgoing:")
            for tgt_id, cnt, p in outgoing[:8]:
                tgt_name = graph.nodes[tgt_id].name if tgt_id in graph.nodes else str(tgt_id)
                tgt_v = values.get(tgt_id, 0.0)
                title_parts.append(
                    f"  -> {tgt_name}: {p:.0%} ({cnt})  V={tgt_v:+.3f}"
                )

        kwargs: dict = dict(
            label=f"{node.name}\n{v:+.3f}",
            title="\n".join(title_parts),
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#1f77b4"},
            },
            shape=shape,
            size=int(size),
            font={"size": 14, "color": "#333", "multi": "md"},
            borderWidth=border_width,
            borderWidthSelected=3,
        )
        if node_id in precomputed_pos:
            px, py = precomputed_pos[node_id]
            kwargs["x"] = int(px * 250)
            kwargs["y"] = int(py * 200)
            kwargs["physics"] = False

        net.add_node(_nid(node_id), **kwargs)

    # Collect transition values for all visible edges to set the color scale
    edge_q_vals: List[float] = []
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob >= min_probability and src in graph.nodes and tgt in graph.nodes:
                edge_q_vals.append(gamma * values.get(tgt, 0.0))
    q_abs_max = max(abs(v) for v in edge_q_vals) if edge_q_vals else 1.0

    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob < min_probability:
                continue
            if src not in graph.nodes or tgt not in graph.nodes:
                continue

            count = graph.transition_counts.get(src, {}).get(tgt, 0)
            edge_width = max(1.0, prob * 8)

            q_val = gamma * values.get(tgt, 0.0)
            v_src = values.get(src, 0.0)
            adv = q_val - v_src
            edge_color = _value_to_rgb(q_val, -q_abs_max, q_abs_max)

            # Highlight "curate out" edges when advantage is below threshold; "select" edges when above
            if highlight_edges_below_advantage is not None and adv < highlight_edges_below_advantage:
                edge_color = "#d62728"
                edge_width = max(edge_width, 3.0)
                dashes = True
            elif highlight_edges_above_advantage is not None and adv >= highlight_edges_above_advantage:
                edge_color = "#2ca02c"
                edge_width = max(edge_width, 3.0)
                dashes = True
            else:
                dashes = False

            src_name = graph.nodes[src].name
            tgt_name = graph.nodes[tgt].name

            net.add_edge(
                _nid(src),
                _nid(tgt),
                value=edge_width,
                title=(
                    f"{src_name} -> {tgt_name}\n"
                    f"P = {prob:.1%}  Count: {count}\n"
                    f"Value(s→s′) = {q_val:+.3f}  A = {adv:+.3f}"
                ),
                label=f"{prob:.0%}" if prob >= 0.10 else "",
                arrows="to",
                color={"color": edge_color, "highlight": "#1f77b4"},
                font={"size": 10, "color": "#555", "align": "top"},
                smooth=True,
                dashes=dashes,
            )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
    with open(f.name, "r") as f:
        html = f.read()

    return html
