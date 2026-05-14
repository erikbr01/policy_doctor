"""Text-only graph representation: node summary + transition matrix + per-rollout paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

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


def _node_label(node_id: int, name: str) -> str:
    if node_id == START_NODE_ID:
        return "START"
    if node_id == SUCCESS_NODE_ID:
        return "SUCCESS"
    if node_id == FAILURE_NODE_ID:
        return "FAILURE"
    if node_id == END_NODE_ID:
        return "END"
    return f"c{node_id}"


def _format_path(path: Optional[List[int]], graph_nodes) -> str:
    if not path:
        return "(no path)"
    parts = []
    for nid in path:
        name = graph_nodes[nid].name if nid in graph_nodes else str(nid)
        parts.append(_node_label(nid, name))
    return " -> ".join(parts)


def _try_compute_values(graph: "BehaviorGraph") -> Optional[Dict[int, float]]:
    try:
        return graph.compute_values()
    except Exception:
        return None


def build_node_summary_block(graph: "BehaviorGraph") -> str:
    values = _try_compute_values(graph)
    lines = ["## Behavior nodes", ""]
    header = "| node | episodes | timesteps | V(s) |"
    sep = "|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    cluster_ids = sorted(nid for nid, n in graph.nodes.items() if not n.is_special)
    ordered = [START_NODE_ID] + cluster_ids + sorted(graph.terminal_node_ids)
    for nid in ordered:
        if nid not in graph.nodes:
            continue
        node = graph.nodes[nid]
        v = values.get(nid) if values is not None else None
        v_str = f"{v:+.3f}" if v is not None else "-"
        lines.append(
            f"| {_node_label(nid, node.name)} | {node.num_episodes} | "
            f"{node.num_timesteps} | {v_str} |"
        )
    return "\n".join(lines)


def build_transition_matrix_block(graph: "BehaviorGraph") -> str:
    cluster_ids = sorted(nid for nid, n in graph.nodes.items() if not n.is_special)
    src_ids = [START_NODE_ID] + cluster_ids
    tgt_ids = cluster_ids + sorted(graph.terminal_node_ids)
    src_labels = [_node_label(s, graph.nodes[s].name) for s in src_ids if s in graph.nodes]
    tgt_labels = [_node_label(t, graph.nodes[t].name) for t in tgt_ids if t in graph.nodes]
    lines = ["## Transition probabilities  P(target | source)", ""]
    lines.append("| from \\ to | " + " | ".join(tgt_labels) + " |")
    lines.append("|" + "---|" * (len(tgt_labels) + 1))
    for s, s_lbl in zip(src_ids, src_labels):
        row = [s_lbl]
        for t in tgt_ids:
            p = graph.transition_probs.get(s, {}).get(t, 0.0)
            row.append(f"{p:.0%}" if p > 0 else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_pool_paths_block(pool: "RolloutPool", graph: "BehaviorGraph") -> str:
    lines = ["## Rollout cluster paths", ""]
    nodes = graph.nodes
    for entry in pool.entries:
        outcome = (
            "success" if entry.success is True
            else "failure" if entry.success is False
            else "unknown"
        )
        path_str = _format_path(entry.cluster_path, nodes)
        lines.append(f"- {entry.rollout_id} [{outcome}]: {path_str}")
    return "\n".join(lines)


class TextTableGraphRepresentation(GraphRepresentation):
    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.include_paths = bool(params.get("include_paths", True))

    def render(
        self,
        graph: "BehaviorGraph",
        pool: "RolloutPool",
        output_dir: Path,
    ) -> VLMArtefact:
        text_blocks = [
            build_node_summary_block(graph),
            build_transition_matrix_block(graph),
        ]
        if self.include_paths:
            text_blocks.append(build_pool_paths_block(pool, graph))
        return VLMArtefact(
            images=[],
            text_blocks=text_blocks,
            metadata={
                "representation": "text_table",
                "n_nodes": len(graph.nodes),
                "n_rollouts": len(pool),
            },
        )


def build_text_table(params: Dict[str, Any]) -> TextTableGraphRepresentation:
    return TextTableGraphRepresentation(params)


register_graph_representation("text_table", build_text_table)
