"""Image-only graph representation: a single PNG, no text blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
)
from policy_doctor.vlm.proposals.graph_representation.base import (
    GraphRepresentation,
    VLMArtefact,
)
from policy_doctor.vlm.proposals.registry import register_graph_representation

if TYPE_CHECKING:
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.vlm.proposals.pool import RolloutPool


_START_COLOR = "#2ca02c"
_SUCCESS_COLOR = "#2ca02c"
_FAILURE_COLOR = "#d62728"
_END_COLOR = "#888888"
_CLUSTER_COLOR = "#1f77b4"


def _node_color(node_id: int) -> str:
    if node_id == START_NODE_ID:
        return _START_COLOR
    if node_id == SUCCESS_NODE_ID:
        return _SUCCESS_COLOR
    if node_id == FAILURE_NODE_ID:
        return _FAILURE_COLOR
    if node_id == END_NODE_ID:
        return _END_COLOR
    return _CLUSTER_COLOR


def _node_label(node_id: int, name: str) -> str:
    if node_id in TERMINAL_NODE_IDS or node_id == START_NODE_ID:
        return name
    return f"c{node_id}"


def render_behavior_graph_png(
    graph: "BehaviorGraph",
    output_path: Path,
    *,
    figsize: tuple = (10, 7),
    dpi: int = 140,
    min_probability: float = 0.0,
) -> Path:
    """Render *graph* as a PNG at *output_path*. Heavy deps imported here."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for node_id in graph.nodes:
        G.add_node(node_id)
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob < min_probability:
                continue
            if src in graph.nodes and tgt in graph.nodes:
                G.add_edge(src, tgt, weight=prob)

    # Layered left-to-right by BFS depth from START
    try:
        distances = dict(nx.single_source_shortest_path_length(G, START_NODE_ID))
    except nx.NodeNotFound:
        distances = {START_NODE_ID: 0}
    max_d = max((d for n, d in distances.items() if n not in TERMINAL_NODE_IDS), default=1)
    end_layer = max_d + 1
    for nid in graph.nodes:
        if nid in TERMINAL_NODE_IDS:
            distances[nid] = end_layer
        elif nid not in distances:
            distances[nid] = max_d // 2 + 1

    layers: Dict[int, list] = {}
    for nid, d in distances.items():
        layers.setdefault(d, []).append(nid)
    pos: Dict[int, tuple] = {}
    for d, ids in layers.items():
        ids_sorted = sorted(ids, key=lambda n: (n in TERMINAL_NODE_IDS, n != START_NODE_ID, n))
        n = len(ids_sorted)
        for i, nid in enumerate(ids_sorted):
            x = float(d)
            y = (i - (n - 1) / 2) * 1.2
            pos[nid] = (x, y)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    edge_widths = []
    edge_alphas = []
    edges = list(G.edges())
    for src, tgt in edges:
        prob = graph.transition_probs.get(src, {}).get(tgt, 0.0)
        edge_widths.append(max(0.6, prob * 4.5))
        edge_alphas.append(max(0.25, min(0.9, 0.25 + prob * 0.7)))

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=edges,
        width=edge_widths,
        edge_color="#555555",
        alpha=0.7,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        node_size=900,
    )

    edge_labels = {
        (s, t): f"{graph.transition_probs.get(s, {}).get(t, 0.0):.0%}"
        for s, t in edges
    }
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=edge_labels, font_size=7, label_pos=0.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
    )

    node_colors = [_node_color(n) for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=900,
        edgecolors="white", linewidths=1.5,
    )
    labels = {
        n: _node_label(n, graph.nodes[n].name) for n in G.nodes()
    }
    nx.draw_networkx_labels(
        G, pos, ax=ax, labels=labels, font_size=9, font_color="white",
        font_weight="bold",
    )

    ax.set_axis_off()
    ax.set_title("Behavior transition graph", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


class ImageOnlyGraphRepresentation(GraphRepresentation):
    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.figsize = tuple(params.get("figsize", (10, 7)))
        self.dpi = int(params.get("dpi", 140))
        self.min_probability = float(params.get("min_probability", 0.0))
        self.filename = str(params.get("filename", "behavior_graph.png"))

    def render(
        self,
        graph: "BehaviorGraph",
        pool: "RolloutPool",
        output_dir: Path,
    ) -> VLMArtefact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / self.filename
        render_behavior_graph_png(
            graph,
            png_path,
            figsize=self.figsize,
            dpi=self.dpi,
            min_probability=self.min_probability,
        )
        return VLMArtefact(
            images=[png_path.absolute()],
            text_blocks=[],
            metadata={"representation": "image_only", "n_nodes": len(graph.nodes)},
        )


def build_image_only(params: Dict[str, Any]) -> ImageOnlyGraphRepresentation:
    return ImageOnlyGraphRepresentation(params)


register_graph_representation("image_only", build_image_only)
