"""Pyvis behavior graph colored by per-node timestep counts."""

from __future__ import annotations

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph, BehaviorNode
from policy_doctor.plotting.plotly.behavior_graph import (
    _build_interactive_behavior_graph_html,
)


def _cluster_node_ids(graph: BehaviorGraph) -> list[int]:
    """Behavior cluster node ids only (excludes START and terminal nodes)."""
    return sorted(nid for nid, node in graph.nodes.items() if not node.is_special)


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


def create_timestep_colored_interactive_graph(
    graph: BehaviorGraph,
    min_probability: float = 0.0,
    height: str = "650px",
    width: str = "100%",
) -> str:
    """Interactive graph with behavior cluster nodes colored by ``num_timesteps``.

    Uses the same layered static layout as the plain interactive graph. Scale is
    min–max over **behavior cluster** nodes only (START / terminals keep
    semantic colors). ``num_timesteps`` counts slice-level assignments (windows)
    belonging to each cluster, matching the graph construction in
    ``BehaviorGraph.from_cluster_assignments``.
    """
    cids = _cluster_node_ids(graph)
    counts = [graph.nodes[nid].num_timesteps for nid in cids] if cids else [0]
    cmin, cmax = int(min(counts)), int(max(counts))

    def _color_fn(_nid: int, node: BehaviorNode) -> str:
        return _timestep_count_to_rgb(node.num_timesteps, cmin, cmax)

    return _build_interactive_behavior_graph_html(
        graph=graph,
        min_probability=min_probability,
        height=height,
        width=width,
        physics_enabled=False,
        layout_algorithm="layeredStatic",
        cluster_node_color_fn=_color_fn,
    )
