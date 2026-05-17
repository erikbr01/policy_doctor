"""Uniform API for all simplification methods.

Every method has the signature:
    fn(graph, cluster_labels, metadata, lever, **kwargs) -> SimplificationResult

The lever is a single scalar that controls the size/fidelity tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.behaviors.simplification.metrics import GraphMetrics


@dataclass
class SimplificationResult:
    method: str
    lever_value: float
    graph: BehaviorGraph
    new_labels: np.ndarray
    node_mapping: Dict[int, int]  # old_cluster_id -> new_cluster_id (within-cluster only)
    metrics: GraphMetrics
    extras: Dict[str, float] = field(default_factory=dict)


# Methods register themselves via this dict.
METHODS: Dict[str, Callable] = {}
METHOD_DESCRIPTIONS: Dict[str, str] = {}
METHOD_LEVER_LABELS: Dict[str, str] = {}


def register_method(
    name: str,
    description: str,
    lever_label: str,
) -> Callable:
    """Decorator: register a simplification method under `name`."""

    def _wrap(fn: Callable) -> Callable:
        METHODS[name] = fn
        METHOD_DESCRIPTIONS[name] = description
        METHOD_LEVER_LABELS[name] = lever_label
        return fn

    return _wrap


def run_method(
    name: str,
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float] = None,
    **kwargs,
) -> SimplificationResult:
    """Run a registered simplification method by name."""
    if name not in METHODS:
        raise KeyError(f"Unknown method: {name!r}. Available: {sorted(METHODS)}")
    return METHODS[name](graph, cluster_labels, metadata, lever, **kwargs)


# Trigger registration via import side-effects.
from policy_doctor.behaviors.simplification import (  # noqa: E402,F401
    bayesian,
    mdl,
    merging,
    skeleton,
    spectral,
    splitting,
)


def _reconstruct_mapping(
    original_labels: np.ndarray, new_labels: np.ndarray,
) -> Dict[int, int]:
    """Derive {old_id → new_id} by sampling the first new label seen for each old.

    Works whenever the method is a pure relabeling (every original ID maps
    deterministically to exactly one new ID).
    """
    mapping: Dict[int, int] = {}
    for old, new in zip(original_labels.tolist(), new_labels.tolist()):
        if old == -1 or old == new:
            continue
        if int(old) not in mapping:
            mapping[int(old)] = int(new)
    return mapping


@register_method(
    name="passthrough",
    description="No simplification — the raw graph straight from the clustering.",
    lever_label="(none)",
)
def _passthrough(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    from policy_doctor.behaviors.simplification.metrics import compute_metrics
    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    metrics = compute_metrics(
        graph, cluster_labels, metadata,
        original_labels=original_labels, node_mapping={},
    )
    return SimplificationResult(
        method="passthrough",
        lever_value=0.0,
        graph=graph,
        new_labels=original_labels,
        node_mapping={},
        metrics=metrics,
    )


@register_method(
    name="degree_one_prune",
    description=(
        "Collapse non-branching chains (each pure-pass-through cluster node is "
        "absorbed into its unique neighbor). Always-lossless w.r.t. branching."
    ),
    lever_label="(none — runs to fixed point)",
)
def _degree_one_prune(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    from policy_doctor.behaviors.behavior_graph import degree_one_prune_to_fixed_point
    from policy_doctor.behaviors.simplification.metrics import compute_metrics

    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_graph, new_labels, rounds, n_merged = degree_one_prune_to_fixed_point(
        graph, cluster_labels, metadata,
    )
    mapping = _reconstruct_mapping(original_labels, new_labels)
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=mapping,
    )
    return SimplificationResult(
        method="degree_one_prune",
        lever_value=0.0,
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=mapping,
        metrics=metrics,
        extras={"rounds": float(rounds), "nodes_merged": float(n_merged)},
    )
