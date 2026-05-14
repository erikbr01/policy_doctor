from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS

_SPECIAL_IDS = frozenset({START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})

# Colors for special nodes
_NODE_COLOR = {
    START_NODE_ID: "#2ca02c",
    SUCCESS_NODE_ID: "#2ca02c",
    FAILURE_NODE_ID: "#d62728",
    END_NODE_ID: "#888888",
}


def _layout(graph: BehaviorGraph) -> dict[int, tuple[float, float]]:
    """BFS-based layered layout: x = depth from START, y = barycenter-refined within layer."""
    import networkx as nx

    G = nx.DiGraph()
    for nid in graph.nodes:
        G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if src in graph.nodes and tgt in graph.nodes and prob > 0:
                G.add_edge(src, tgt, weight=prob)

    # BFS depth from START
    layers: dict[int, list[int]] = {}
    depth: dict[int, int] = {START_NODE_ID: 0}
    queue = [START_NODE_ID]
    while queue:
        node = queue.pop(0)
        d = depth[node]
        layers.setdefault(d, []).append(node)
        for tgt in G.successors(node):
            if tgt not in depth:
                depth[tgt] = d + 1
                queue.append(tgt)
    # Put unreachable nodes at the end
    max_d = max(depth.values(), default=0)
    for nid in graph.nodes:
        if nid not in depth:
            depth[nid] = max_d + 1
            layers.setdefault(max_d + 1, []).append(nid)

    # Assign y within each layer (evenly spaced, barycenter refined once)
    pos: dict[int, tuple[float, float]] = {}
    for d, nodes in sorted(layers.items()):
        for k, nid in enumerate(nodes):
            pos[nid] = (float(d), float(k) - len(nodes) / 2.0)

    # Barycenter pass (1 iteration)
    for d, nodes in sorted(layers.items()):
        if d == 0:
            continue
        new_ys = {}
        for nid in nodes:
            preds = [p for p in G.predecessors(nid) if p in pos]
            if preds:
                new_ys[nid] = float(np.mean([pos[p][1] for p in preds]))
        for nid, ny in new_ys.items():
            pos[nid] = (pos[nid][0], ny)

    return pos


def build_study_graph(
    graph: BehaviorGraph,
    height: int = 600,
    min_edge_prob: float = 0.01,
) -> go.Figure:
    """Build a clean, click-friendly behavior graph for the user study.

    Uses a single edge trace (with None separators) and one scatter trace
    per node, so Plotly click events map directly to node IDs via customdata.
    Node size scales with episode count; edge width scales with probability.
    """
    pos = _layout(graph)

    fig = go.Figure()

    # ── Edges ────────────────────────────────────────────────────────────────
    # Group edges by approximate width bucket to avoid one trace per edge while
    # still showing variable thickness. We use 4 width buckets.
    buckets: dict[int, list[tuple[float, float, float, float]]] = {1: [], 2: [], 3: [], 4: []}
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob < min_edge_prob:
                continue
            if src not in pos or tgt not in pos:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            # Slight offset for bidirectional edges
            if tgt in graph.transition_probs and src in graph.transition_probs.get(tgt, {}):
                dx, dy = x1 - x0, y1 - y0
                length = max((dx**2 + dy**2) ** 0.5, 1e-6)
                ox, oy = -dy / length * 0.07, dx / length * 0.07
            else:
                ox, oy = 0.0, 0.0
            bucket = max(1, min(4, int(prob * 10) + 1))
            buckets[bucket].append((x0 + ox, y0 + oy, x1 + ox, y1 + oy))

    width_map = {1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0}
    for bk, segs in buckets.items():
        if not segs:
            continue
        xs, ys = [], []
        for x0, y0, x1, y1 in segs:
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(100,100,100,0.35)", width=width_map[bk]),
            hoverinfo="skip", showlegend=False, name="",
        ))

    # ── Nodes ────────────────────────────────────────────────────────────────
    # Compute per-node success rate for coloring border
    ep_success: dict[int, Optional[bool]] = {}
    all_eps = set()
    for meta in []:  # populated at call site if needed
        pass

    max_ep = max((n.num_episodes for n in graph.nodes.values()), default=1) or 1

    for nid, node in graph.nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]

        # Special nodes
        if nid in _SPECIAL_IDS:
            color = _NODE_COLOR.get(nid, "#888888")
            symbol = "star" if nid == SUCCESS_NODE_ID else (
                "x" if nid == FAILURE_NODE_ID else (
                    "diamond" if nid == START_NODE_ID else "square"
                )
            )
            size = 24
            label = node.name
            hover = node.name
        else:
            color = CLUSTER_COLORS[nid % len(CLUSTER_COLORS)]
            symbol = "circle"
            size = int(20 + 18 * (node.num_episodes / max_ep))
            label = node.name
            hover = (
                f"<b>{node.name}</b><br>"
                f"Timesteps: {node.num_timesteps}<br>"
                f"Episodes: {node.num_episodes}<br>"
                f"<i>Click to explore</i>"
            )

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=size, color=color, symbol=symbol,
                line=dict(width=2, color="white"),
            ),
            text=[label],
            textposition="top center",
            textfont=dict(size=10, color="#333"),
            hovertext=[hover],
            hoverinfo="text",
            customdata=[[nid]],
            showlegend=False,
            name=node.name,
        ))

    all_x = [v[0] for v in pos.values()]
    all_y = [v[1] for v in pos.values()]
    margin = 0.8
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False, range=[min(all_x) - margin, max(all_x) + margin]),
        yaxis=dict(visible=False, scaleanchor="x", range=[min(all_y) - margin, max(all_y) + margin]),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        clickmode="event+select",
        dragmode="select",
    )
    return fig


def node_id_from_event(event, fig: go.Figure, pos: dict[int, tuple[float, float]]) -> Optional[int]:
    """Extract node_id from a Streamlit on_select event, with nearest-node fallback."""
    if not event:
        return None
    sel = getattr(event, "selection", None) or {}
    points = sel.get("points", []) if isinstance(sel, dict) else (getattr(sel, "points", []) or [])
    if not points:
        return None
    pt = points[0]

    def _get(obj, key, attr):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, attr, None)

    curve_num = _get(pt, "curve_number", "curve_number")
    click_x = _get(pt, "x", "x")
    click_y = _get(pt, "y", "y")

    # Direct hit: check customdata
    if curve_num is not None and 0 <= int(curve_num) < len(fig.data):
        cd = getattr(fig.data[int(curve_num)], "customdata", None)
        if cd is not None and len(cd) > 0:
            try:
                return int(cd[0][0])
            except (TypeError, ValueError, IndexError):
                pass

    # Fallback: nearest node by data-space distance
    if click_x is not None and click_y is not None and pos:
        try:
            cx, cy = float(click_x), float(click_y)
            nearest = min(pos.items(), key=lambda kv: (kv[1][0] - cx) ** 2 + (kv[1][1] - cy) ** 2)
            return nearest[0]
        except (TypeError, ValueError):
            pass

    return None
