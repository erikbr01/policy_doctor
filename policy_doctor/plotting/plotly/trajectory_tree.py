"""Pure Plotly figure builders for the trajectory tree.

Takes preprocessed tree nodes (the output of
:func:`policy_doctor.behaviors.graph_simplification.build_trajectory_tree`,
optionally filtered) and returns ``plotly.graph_objects.Figure`` objects.
No Streamlit imports — the streamlit-side code is expected to ``st.plotly_chart``
the result.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS


def _outcome_color_kwargs(succ_rate: List[float]) -> Dict[str, Any]:
    return dict(
        marker=dict(
            colors=succ_rate,
            colorscale=[[0.0, "#d62728"], [0.5, "#dddddd"], [1.0, "#2ca02c"]],
            cmid=0.5, cmin=0.0, cmax=1.0,
        ),
    )


def _cluster_color_kwargs(nodes_f: List[Dict]) -> Dict[str, Any]:
    def _color(nd):
        cid = nd["cluster_id"]
        if cid == SUCCESS_NODE_ID: return "#2ca02c"
        if cid == FAILURE_NODE_ID: return "#d62728"
        if cid == END_NODE_ID: return "#888"
        if cid == START_NODE_ID: return "#1a1a1a"
        return CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    return dict(marker=dict(colors=[_color(nd) for nd in nodes_f]))


def _value_color_kwargs(
    nodes_f: List[Dict],
    node_values: Dict[int, float],
) -> Dict[str, Any]:
    """Color by V(cluster) on a red→grey→green diverging scale.

    `node_values` maps cluster_id (NOT tree-node synthetic id) → V. Tree
    nodes share their underlying cluster's V; terminals get fixed rewards.
    """
    vs = []
    for nd in nodes_f:
        cid = nd["cluster_id"]
        if cid == SUCCESS_NODE_ID:
            v = 1.0
        elif cid == FAILURE_NODE_ID:
            v = -1.0
        elif cid == END_NODE_ID:
            v = 0.0
        elif cid == START_NODE_ID:
            v = 0.0
        else:
            v = float(node_values.get(int(cid), 0.0))
        vs.append(v)
    cmax = max(abs(min(vs)), abs(max(vs)), 1e-9)
    return dict(
        marker=dict(
            colors=vs,
            colorscale=[[0.0, "#d62728"], [0.5, "#dddddd"], [1.0, "#2ca02c"]],
            cmid=0.0, cmin=-cmax, cmax=cmax,
        ),
    )


def _common_kwargs(
    nodes_f: List[Dict],
    color_mode: str,
    node_values: Dict[int, float],
) -> Dict[str, Any]:
    def _id(nd):
        return "/".join(str(x) for x in nd["path"]) or "ROOT"
    def _parent(nd):
        if nd["parent_path"] is None:
            return ""
        return "/".join(str(x) for x in nd["parent_path"]) or "ROOT"

    ids = [_id(nd) for nd in nodes_f]
    parents = [_parent(nd) for nd in nodes_f]
    labels = [nd["label"] for nd in nodes_f]
    values = [nd["n_episodes"] for nd in nodes_f]
    succ_rate = [
        (nd["n_success"] / nd["n_episodes"]) if nd["n_episodes"] else 0.0
        for nd in nodes_f
    ]
    hovertext = [
        f"<b>{nd['label']}</b><br>"
        f"Depth: {nd['depth']}<br>"
        f"Episodes through here: {nd['n_episodes']}<br>"
        f"Success: {nd['n_success']} ({nd['n_success']/max(1,nd['n_episodes']):.0%})<br>"
        f"Failure: {nd['n_failure']} ({nd['n_failure']/max(1,nd['n_episodes']):.0%})"
        for nd in nodes_f
    ]
    if color_mode == "value":
        color = _value_color_kwargs(nodes_f, node_values)
    elif color_mode == "id":
        color = _cluster_color_kwargs(nodes_f)
    else:  # outcome
        color = _outcome_color_kwargs(succ_rate)
    return dict(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", hovertext=hovertext, hoverinfo="text",
        **color,
    )


def create_trajectory_sunburst(
    nodes_f: List[Dict],
    *,
    color_mode: str = "outcome",
    node_values: Dict[int, float] | None = None,
    height: int = 700,
) -> go.Figure:
    """Sunburst: radial, root at the center, leaves on the outer ring.

    `color_mode` ∈ {"outcome", "id", "value"}. For "value", supply
    `node_values` as a dict mapping cluster_id → V(cluster).
    """
    common = _common_kwargs(nodes_f, color_mode, node_values or {})
    fig = go.Figure(go.Sunburst(**common, insidetextorientation="radial"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def create_trajectory_icicle(
    nodes_f: List[Dict],
    *,
    color_mode: str = "outcome",
    node_values: Dict[int, float] | None = None,
    height: int = 700,
) -> go.Figure:
    """Icicle: horizontal flat tree, depth = x-axis, weight = vertical extent."""
    common = _common_kwargs(nodes_f, color_mode, node_values or {})
    fig = go.Figure(go.Icicle(**common, tiling=dict(orientation="h")))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def create_trajectory_treemap(
    nodes_f: List[Dict],
    *,
    color_mode: str = "outcome",
    node_values: Dict[int, float] | None = None,
    height: int = 700,
) -> go.Figure:
    """Treemap: nested rectangles, area = episode count."""
    common = _common_kwargs(nodes_f, color_mode, node_values or {})
    fig = go.Figure(go.Treemap(**common))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def create_trajectory_node_edge_plotly(
    nodes_f: List[Dict],
    *,
    color_mode: str = "outcome",
    node_values: Dict[int, float] | None = None,
    height: int = 700,
) -> go.Figure:
    """Top-down node-edge tree with subtree-weighted horizontal layout.

    Children of each node are laid out left-to-right in horizontal slots
    proportional to their subtree's episode count. Edges are vertical
    splines. `color_mode` ∈ {"outcome", "id", "value"} controls both node
    fill and edge color; for "value" pass `node_values` mapping cluster_id
    → V(cluster).
    """
    node_values = node_values or {}
    by_path: Dict[Tuple, Dict] = {tuple(nd["path"]): nd for nd in nodes_f}
    children_of: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for nd in nodes_f:
        if nd["parent_path"] is not None and tuple(nd["parent_path"]) in by_path:
            children_of[tuple(nd["parent_path"])].append(tuple(nd["path"]))
    for k in children_of:
        children_of[k].sort(key=lambda p: by_path[p]["cluster_id"])

    pos: Dict[Tuple, Tuple[float, float]] = {}
    def _assign(path: Tuple, x_lo: float, x_hi: float, depth: int) -> None:
        pos[path] = ((x_lo + x_hi) / 2, -depth)
        ch = children_of.get(path, [])
        if not ch:
            return
        weights = [by_path[c]["n_episodes"] for c in ch]
        total = sum(weights) or 1
        cur = x_lo
        for c, w in zip(ch, weights):
            span = (x_hi - x_lo) * w / total
            _assign(c, cur, cur + span, depth + 1)
            cur += span
    _assign((), 0.0, 1.0, 0)

    max_depth = max(d for _, d in pos.values()) or 1
    y_scale = 1.0 / max(1, abs(max_depth))

    def _diverging_color(t: float) -> str:
        """t ∈ [0, 1]; 0 = red, 0.5 = grey, 1 = green."""
        t = max(0.0, min(1.0, t))
        r = int(214 + (44 - 214) * t)
        g = int(39 + (160 - 39) * t)
        b = int(40 + (44 - 40) * t)
        return f"rgb({r},{g},{b})"

    def _node_color_id(nd: Dict) -> str:
        cid = nd["cluster_id"]
        if cid == SUCCESS_NODE_ID: return "#2ca02c"
        if cid == FAILURE_NODE_ID: return "#d62728"
        if cid == END_NODE_ID: return "#888"
        if cid == START_NODE_ID: return "#444"
        return CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

    # Pre-compute value range for normalization (value mode)
    def _v_for(nd: Dict) -> float:
        cid = nd["cluster_id"]
        if cid == SUCCESS_NODE_ID: return 1.0
        if cid == FAILURE_NODE_ID: return -1.0
        if cid in (END_NODE_ID, START_NODE_ID): return 0.0
        return float(node_values.get(int(cid), 0.0))
    v_range = max(abs(_v_for(nd)) for nd in nodes_f) or 1.0

    fig = go.Figure()

    # Edges first so they sit behind nodes
    for path, nd in by_path.items():
        if nd["parent_path"] is None:
            continue
        parent = tuple(nd["parent_path"])
        if parent not in pos:
            continue
        x0, y0 = pos[parent]
        x1, y1 = pos[path]
        y0s, y1s = y0 * y_scale, y1 * y_scale
        width = max(0.8, 0.6 + np.log1p(nd["n_episodes"]) * 0.9)
        rate = nd["n_success"] / max(1, nd["n_episodes"])
        if color_mode == "outcome":
            ec = _diverging_color(rate)
        elif color_mode == "value":
            # Edge advantage = V(child) - V(parent); map to [0, 1] for color.
            parent_v = _v_for(by_path[parent])
            child_v = _v_for(nd)
            adv = (child_v - parent_v) / (2 * v_range)  # in [-0.5, 0.5]
            ec = _diverging_color(0.5 + adv)
        else:  # id
            ec = "rgba(140,140,140,0.6)"
        cx, cy = (x0 + x1) / 2, (y0s + y1s) / 2
        fig.add_trace(go.Scatter(
            x=[x0, cx, x1], y=[y0s, cy, y1s],
            mode="lines",
            line=dict(width=width, color=ec, shape="spline", smoothing=1.0),
            hoverinfo="text",
            hovertext=(
                f"<b>{by_path[parent]['label']} → {nd['label']}</b><br>"
                f"Episodes: {nd['n_episodes']}<br>"
                f"Success rate of this branch: {rate:.0%}"
            ),
            showlegend=False,
        ))

    # Nodes
    xs, ys, txts, hovers, colors_, sizes = [], [], [], [], [], []
    for path, nd in by_path.items():
        x, y = pos[path]
        xs.append(x); ys.append(y * y_scale)
        txts.append(nd["label"])
        rate = nd["n_success"] / max(1, nd["n_episodes"])
        hovers.append(
            f"<b>{nd['label']}</b><br>Depth: {nd['depth']}<br>"
            f"Episodes: {nd['n_episodes']}<br>"
            f"Success rate: {rate:.0%}<br>"
            f"Path: {' → '.join(str(c) for c in path)}"
        )
        cid = nd["cluster_id"]
        is_special = cid in (START_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID)
        if color_mode == "outcome" and not is_special:
            colors_.append(_diverging_color(rate))
        elif color_mode == "value" and not is_special:
            v = _v_for(nd)
            colors_.append(_diverging_color(0.5 + v / (2 * v_range)))
        else:
            colors_.append(_node_color_id(nd))
        sizes.append(max(10, min(60, 8 + np.log1p(nd["n_episodes"]) * 8)))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=colors_, line=dict(width=1.5, color="white")),
        text=txts, textposition="middle right",
        textfont=dict(size=11, color="#ddd"),
        hovertext=hovers, hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(visible=False, range=[-0.05, 1.15]),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig
