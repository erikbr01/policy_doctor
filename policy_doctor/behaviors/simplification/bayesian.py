"""Bayesian merging with Dirichlet posteriors.

Treats each row of the transition matrix as a Dirichlet posterior given
counts + symmetric Dirichlet(α) prior. Two nodes are merged when the
posterior probability that their successor distributions are "close" (in
Jensen-Shannon divergence) exceeds the lever threshold.

This is the *uncertainty-aware* counterpart of `js_merge`: instead of
thresholding the point-estimate JS divergence, it thresholds the *posterior
probability of similarity*. Nodes with few observations have wide posteriors
and easily fail / pass; nodes with many observations get a sharp test.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.behaviors.simplification.api import (
    SimplificationResult,
    register_method,
)
from policy_doctor.behaviors.simplification.merging import _iterative_merge
from policy_doctor.behaviors.simplification.metrics import (
    compute_metrics,
    js_distance_bits,
    posterior_overlap_prob,
)


@register_method(
    name="bayesian_merge",
    description=(
        "Bayesian successor merging: merge nodes whose Dirichlet posteriors "
        "give P(JS ≤ ε_tol) ≥ lever. Naturally robust to small sample sizes — "
        "low-count nodes have wide posteriors so the test rarely declares them "
        "equal *or* different. Slow (Monte Carlo per pair), so use small n."
    ),
    lever_label="P_min (min posterior similarity probability)",
)
def bayesian_merge(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    tol_bits: float = 0.05,
    n_samples: int = 200,
    **kwargs,
) -> SimplificationResult:
    p_min = 0.5 if lever is None else float(lever)

    def compat(c1: Dict[int, int], c2: Dict[int, int]) -> bool:
        return posterior_overlap_prob(
            c1, c2, tol_bits=tol_bits, n_samples=n_samples,
        ) >= p_min

    def score(c1: Dict[int, int], c2: Dict[int, int]) -> float:
        # Score by JS as a tiebreaker (smaller = more similar)
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
        method="bayesian_merge",
        lever_value=p_min,
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=mp,
        metrics=metrics,
    )
