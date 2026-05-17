"""Stationary-skeleton reduction.

Compute the stationary distribution π over cluster nodes (ignoring START /
terminals). Keep only nodes with π > threshold; absorb the rest into their
most-probable "anchor" neighbor among the retained set. This produces a
"main highway" view of the behavior graph — useful for visualization.

Single lever: π_min (fraction). Larger → fewer nodes retained.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    _SPECIAL_GRAPH_NODE_IDS,
    _apply_merge_map_to_labels,
)
from policy_doctor.behaviors.simplification.api import (
    SimplificationResult,
    register_method,
)
from policy_doctor.behaviors.simplification.metrics import compute_metrics
from policy_doctor.behaviors.simplification.spectral import (
    _stationary_distribution,
    _transition_matrix,
)


@register_method(
    name="stationary_skeleton",
    description=(
        "Keep cluster nodes whose stationary probability π(s) ≥ π_min. Other "
        "nodes are absorbed into their most-probable retained successor "
        "(reachable by greedy max-π walk). Produces a 'main road' graph."
    ),
    lever_label="π_min (stationary probability floor)",
)
def stationary_skeleton(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    pi_min = 0.05 if lever is None else float(lever)
    cluster_ids = sorted(
        nid for nid in graph.nodes if nid not in _SPECIAL_GRAPH_NODE_IDS
    )
    n = len(cluster_ids)
    if n <= 1:
        metrics = compute_metrics(graph, cluster_labels, metadata)
        return SimplificationResult(
            method="stationary_skeleton",
            lever_value=pi_min,
            graph=graph,
            new_labels=cluster_labels.copy(),
            node_mapping={},
            metrics=metrics,
        )

    P = _transition_matrix(graph, cluster_ids)
    pi = _stationary_distribution(P)

    keep_mask = pi >= pi_min
    if not keep_mask.any():
        # keep at least the highest-π node
        keep_mask = np.zeros_like(pi, dtype=bool)
        keep_mask[int(np.argmax(pi))] = True
    keep_ids = [cluster_ids[i] for i in range(n) if keep_mask[i]]
    drop_ids = [cluster_ids[i] for i in range(n) if not keep_mask[i]]

    # For each dropped node, route it to the highest-probability kept node
    # reachable via greedy walk in P.
    keep_set = set(keep_ids)
    merge_map: Dict[int, int] = {}
    for d in drop_ids:
        cur = cluster_ids.index(d)
        path_visited = {cur}
        anchor: Optional[int] = None
        for _ in range(n * 2):
            # Find best successor not yet visited
            row = P[cur].copy()
            row[list(path_visited)] = -1
            best = int(np.argmax(row))
            if row[best] <= 0:
                break
            cur = best
            path_visited.add(cur)
            if cluster_ids[cur] in keep_set:
                anchor = cluster_ids[cur]
                break
        if anchor is None:
            # fall back: highest-π retained node overall
            anchor = keep_ids[int(np.argmax(pi[[cluster_ids.index(k) for k in keep_ids]]))]
        merge_map[d] = anchor

    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_labels = _apply_merge_map_to_labels(cluster_labels, merge_map)
    new_graph = BehaviorGraph.from_cluster_assignments(
        new_labels, metadata, level=graph.level,
    )
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=merge_map,
    )
    return SimplificationResult(
        method="stationary_skeleton",
        lever_value=pi_min,
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=merge_map,
        metrics=metrics,
        extras={
            "n_kept": float(len(keep_ids)),
            "pi_min_observed": float(pi.min()),
            "pi_max_observed": float(pi.max()),
        },
    )
