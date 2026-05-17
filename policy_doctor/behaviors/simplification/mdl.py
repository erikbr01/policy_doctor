"""MDL-greedy merging.

Greedily merge the node pair whose merge reduces the MDL objective:
    L(graph) = NLL_train_bits + λ · |params|
where |params| = (n_nodes_with_outgoing) · (n_nodes_with_outgoing - 1)
counts free transition probabilities (rows sum to 1, so one per row is fixed).

Single lever: λ. Bigger λ → stronger compression pressure → fewer nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
from policy_doctor.behaviors.simplification.metrics import (
    compute_metrics,
    free_params_count,
    predictive_nll_bits,
)


def _mdl_score(
    graph: BehaviorGraph,
    original_labels: np.ndarray,
    metadata: List[Dict],
    node_mapping: Dict[int, int],
    lambda_: float,
) -> Tuple[float, float, int]:
    """MDL = predictive_NLL + (λ/2) · log₂(N_original) · n_free_params.

    Predictive NLL uses *original* transitions as the fixed reference data,
    so merging doesn't artificially shrink the loss by shortening the
    trajectory. This is the principled fix for the "1-node trivial winner"
    failure mode.
    """
    nll, _n_scored, n_orig = predictive_nll_bits(
        graph, original_labels, metadata, node_mapping,
    )
    n_params = free_params_count(graph)
    code_bits = lambda_ * 0.5 * float(np.log2(max(n_orig, 2))) * n_params
    return nll + code_bits, nll, n_params


@register_method(
    name="mdl_greedy",
    description=(
        "Greedy merge minimizing MDL objective: NLL + λ · (½ log₂ N) · |params|. "
        "**λ is the lever** (bits of penalty per free parameter). The method "
        "tries every merge candidate at each round and picks the one that "
        "reduces MDL the most. Stops when no merge improves the score."
    ),
    lever_label="λ (MDL penalty per free parameter)",
)
def mdl_greedy(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    max_iters: int = 200,
    **kwargs,
) -> SimplificationResult:
    lambda_ = 1.0 if lever is None else float(lever)
    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    labels = original_labels.copy()
    level = graph.level
    cur = graph
    overall_map: Dict[int, int] = {}
    cur_score, _, _ = _mdl_score(cur, original_labels, metadata, overall_map, lambda_)

    for _ in range(max_iters):
        cluster_ids = sorted(
            nid for nid in cur.nodes if nid not in _SPECIAL_GRAPH_NODE_IDS
        )
        if len(cluster_ids) < 2:
            break

        best_delta = 0.0
        best_pair: Optional[Tuple[int, int]] = None
        best_labels: Optional[np.ndarray] = None
        best_graph: Optional[BehaviorGraph] = None

        for i, a in enumerate(cluster_ids):
            for b in cluster_ids[i + 1 :]:
                trial_labels = _apply_merge_map_to_labels(labels, {b: a})
                trial_graph = BehaviorGraph.from_cluster_assignments(
                    trial_labels, metadata, level=level,
                )
                trial_map = dict(overall_map)
                for k, v in list(trial_map.items()):
                    if v == b:
                        trial_map[k] = a
                trial_map[b] = a
                trial_score, _, _ = _mdl_score(
                    trial_graph, original_labels, metadata, trial_map, lambda_,
                )
                delta = trial_score - cur_score
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_pair = (a, b)
                    best_labels = trial_labels
                    best_graph = trial_graph

        if best_pair is None or best_labels is None or best_graph is None:
            break
        a, b = best_pair
        for k, v in list(overall_map.items()):
            if v == b:
                overall_map[k] = a
        overall_map[b] = a
        labels = best_labels
        cur = best_graph
        cur_score += best_delta

    metrics = compute_metrics(
        cur, labels, metadata,
        original_labels=original_labels, node_mapping=overall_map,
    )
    return SimplificationResult(
        method="mdl_greedy",
        lever_value=lambda_,
        graph=cur,
        new_labels=labels,
        node_mapping=overall_map,
        metrics=metrics,
        extras={"final_mdl": float(cur_score)},
    )
