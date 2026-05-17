"""Successor-distribution merging.

Two cluster nodes are merged when their outgoing transition distributions are
statistically indistinguishable. Method "kl_merge" uses a JS-divergence
threshold; "hoeffding_merge" uses the Alergia (Carrasco-Oncina) test, which
explicitly accounts for sample size; "chi2_merge" uses the χ² independence
test on the (node × successor) contingency.

All variants iterate to a fixed point: merge the best candidate pair, rebuild
the graph, repeat.
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
    chi2_pvalue,
    compute_metrics,
    hoeffding_compatible,
    js_distance_bits,
)


def _cluster_node_ids(graph: BehaviorGraph) -> List[int]:
    return sorted(nid for nid in graph.nodes if nid not in _SPECIAL_GRAPH_NODE_IDS)


def _outgoing_counts(graph: BehaviorGraph, node_id: int) -> Dict[int, int]:
    return dict(graph.transition_counts.get(node_id, {}))


def _iterative_merge(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    compatible_fn,
    score_fn=None,
    max_iters: int = 500,
) -> Tuple[BehaviorGraph, np.ndarray, Dict[int, int]]:
    """Generic iterative-merge driver.

    At each round, find pairs of cluster nodes whose outgoing distributions
    are *compatible* (`compatible_fn(c1, c2) -> bool`). If multiple are
    compatible, pick the pair with the smallest `score_fn(c1, c2)` (or any
    pair if score_fn is None). Merge and repeat to fixed point.
    """
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    overall_map: Dict[int, int] = {}
    level = graph.level
    cur = graph

    for _ in range(max_iters):
        cluster_ids = _cluster_node_ids(cur)
        if len(cluster_ids) < 2:
            break
        best: Optional[Tuple[float, int, int]] = None
        for i, a in enumerate(cluster_ids):
            ca = _outgoing_counts(cur, a)
            if not ca:
                continue
            for b in cluster_ids[i + 1 :]:
                cb = _outgoing_counts(cur, b)
                if not cb:
                    continue
                if not compatible_fn(ca, cb):
                    continue
                s = score_fn(ca, cb) if score_fn is not None else 0.0
                if best is None or s < best[0]:
                    best = (s, a, b)
        if best is None:
            break
        _, a, b = best
        merge_map = {b: a}
        labels = _apply_merge_map_to_labels(labels, merge_map)
        # update overall_map (everything that was previously mapped to b now goes to a)
        for k, v in list(overall_map.items()):
            if v == b:
                overall_map[k] = a
        overall_map[b] = a
        cur = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
    return cur, labels, overall_map


@register_method(
    name="js_merge",
    description=(
        "Merge node pairs whose smoothed outgoing distributions have "
        "JS divergence ≤ τ bits. Greedy: smallest-JS pair first; iterate to "
        "fixed point. **Does not use sample size**, so the user has to pick τ "
        "with noise in mind."
    ),
    lever_label="τ (max JS distance, bits)",
)
def js_merge(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    tau = 0.05 if lever is None else float(lever)

    def compat(c1: Dict[int, int], c2: Dict[int, int]) -> bool:
        return js_distance_bits(c1, c2) <= tau

    def score(c1: Dict[int, int], c2: Dict[int, int]) -> float:
        return js_distance_bits(c1, c2)

    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_graph, new_labels, mp = _iterative_merge(
        graph, cluster_labels, metadata, compat, score,
    )
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=mp,
    )
    return SimplificationResult(
        method="js_merge",
        lever_value=tau,
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=mp,
        metrics=metrics,
    )


@register_method(
    name="hoeffding_merge",
    description=(
        "Alergia compatibility test (Carrasco-Oncina 1994): merge pairs whose "
        "outgoing distributions agree within a Hoeffding bound at confidence δ. "
        "**Sample-size aware** — low-count nodes get larger tolerance, so the "
        "method doesn't merge two noisy distributions just because they happen "
        "to look similar by chance."
    ),
    lever_label="δ (Hoeffding confidence; smaller δ → more merging)",
)
def hoeffding_merge(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    delta = 0.05 if lever is None else float(lever)

    def compat(c1: Dict[int, int], c2: Dict[int, int]) -> bool:
        return hoeffding_compatible(c1, c2, delta)

    def score(c1: Dict[int, int], c2: Dict[int, int]) -> float:
        return js_distance_bits(c1, c2)

    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_graph, new_labels, mp = _iterative_merge(
        graph, cluster_labels, metadata, compat, score,
    )
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=mp,
    )
    return SimplificationResult(
        method="hoeffding_merge",
        lever_value=delta,
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=mp,
        metrics=metrics,
    )


@register_method(
    name="chi2_merge",
    description=(
        "χ² test of homogeneity: merge node pairs whose outgoing-distribution "
        "contingency yields p-value ≥ α (no evidence of difference). Like "
        "Hoeffding it's sample-size aware, but uses asymptotic χ² instead of a "
        "concentration inequality."
    ),
    lever_label="α (χ² p-value threshold; larger α → less merging)",
)
def chi2_merge(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    alpha = 0.05 if lever is None else float(lever)

    def compat(c1: Dict[int, int], c2: Dict[int, int]) -> bool:
        return chi2_pvalue(c1, c2) >= alpha

    def score(c1: Dict[int, int], c2: Dict[int, int]) -> float:
        return -chi2_pvalue(c1, c2)  # prefer highest p-value pair first

    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_graph, new_labels, mp = _iterative_merge(
        graph, cluster_labels, metadata, compat, score,
    )
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=mp,
    )
    return SimplificationResult(
        method="chi2_merge",
        lever_value=alpha,
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=mp,
        metrics=metrics,
    )
