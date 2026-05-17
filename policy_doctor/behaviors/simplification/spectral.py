"""Spectral / random-walk based reduction.

Two creative methods inspired by Markov-chain coarse-graining literature:

- **PCCA+** (Perron Cluster Cluster Analysis, Deuflhard-Weber 2005): the
  dominant eigenvectors of the (row-stochastic) transition matrix encode
  *metastable sets* — clusters of states between which the chain rarely
  transitions. Picking the top-k eigenvectors and rounding (here via
  KMeans) gives k meta-states.

- **Markov stability** (Delvenne-Yaliraki-Barahona 2010): at random-walk
  time t, two states are in the same community if the t-step random walk
  starting at each lands in similar distributions. Single lever = t.
  Small t → fine, large t → coarse. Naturally multi-scale.

Both methods produce a node mapping (old_cluster_id → meta_id) and a graph
built from the merged labels.
"""

from __future__ import annotations

from collections import defaultdict
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
from policy_doctor.behaviors.simplification.metrics import compute_metrics


def _transition_matrix(
    graph: BehaviorGraph, node_ids: List[int],
) -> np.ndarray:
    """Row-stochastic transition matrix restricted to `node_ids`.

    Probability mass that escapes to non-listed nodes (terminals, START) is
    redistributed by renormalizing rows. Rows with no outgoing mass become
    self-loops.
    """
    idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    P = np.zeros((n, n))
    for src in node_ids:
        i = idx[src]
        probs = graph.transition_probs.get(src, {})
        for tgt, p in probs.items():
            if tgt in idx:
                P[i, idx[tgt]] += p
        row_sum = P[i].sum()
        if row_sum > 0:
            P[i] /= row_sum
        else:
            P[i, i] = 1.0  # absorbing self-loop
    return P


def _stationary_distribution(P: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """Stationary distribution by power iteration."""
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(n_iter):
        pi_new = pi @ P
        pi_new /= pi_new.sum()
        if np.linalg.norm(pi_new - pi, ord=1) < 1e-10:
            break
        pi = pi_new
    return pi


def _kmeans_on_vectors(
    X: np.ndarray, k: int, n_init: int = 10, rng_seed: int = 0,
) -> np.ndarray:
    """Tiny KMeans (no sklearn dep)."""
    n, d = X.shape
    rng = np.random.RandomState(rng_seed)
    best_inertia = float("inf")
    best_labels = np.zeros(n, dtype=int)
    for _init in range(n_init):
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(100):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                for j in range(k)
            ])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        inertia = float(np.min(d2, axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels


def _apply_meta_mapping(
    cluster_labels: np.ndarray,
    node_ids: List[int],
    meta_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Build a merge map: collapse all node_ids belonging to the same meta-cluster
    onto the smallest cluster_id in that meta-cluster.
    """
    n_meta = int(meta_labels.max()) + 1
    members: Dict[int, List[int]] = defaultdict(list)
    for nid, m in zip(node_ids, meta_labels):
        members[int(m)].append(int(nid))
    merge_map: Dict[int, int] = {}
    for m in range(n_meta):
        ms = sorted(members.get(m, []))
        if len(ms) < 2:
            continue
        rep = ms[0]
        for x in ms[1:]:
            merge_map[x] = rep
    new_labels = _apply_merge_map_to_labels(cluster_labels, merge_map)
    return new_labels, merge_map


@register_method(
    name="pcca_plus",
    description=(
        "**PCCA+ / spectral metastable clustering** (Deuflhard-Weber 2005). "
        "Computes the top-k eigenvectors of the row-stochastic transition "
        "matrix and KMeans-rounds them into k meta-states. Each meta-state is "
        "a *metastable* set — a group of cluster nodes the chain dwells in "
        "before transitioning out."
    ),
    lever_label="k (number of meta-states)",
)
def pcca_plus(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    cluster_ids = sorted(
        nid for nid in graph.nodes if nid not in _SPECIAL_GRAPH_NODE_IDS
    )
    n = len(cluster_ids)
    k = int(round(lever)) if lever is not None else max(2, n // 2)
    k = max(1, min(k, n))
    if k >= n or n <= 1:
        metrics = compute_metrics(graph, cluster_labels, metadata)
        return SimplificationResult(
            method="pcca_plus",
            lever_value=float(k),
            graph=graph,
            new_labels=cluster_labels.copy(),
            node_mapping={},
            metrics=metrics,
        )

    P = _transition_matrix(graph, cluster_ids)
    eigvals, eigvecs = np.linalg.eig(P)
    order = np.argsort(-np.real(eigvals))
    top = np.real(eigvecs[:, order[:k]])
    meta_labels = _kmeans_on_vectors(top, k=k)

    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_labels, merge_map = _apply_meta_mapping(cluster_labels, cluster_ids, meta_labels)
    new_graph = BehaviorGraph.from_cluster_assignments(
        new_labels, metadata, level=graph.level,
    )
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=merge_map,
    )
    return SimplificationResult(
        method="pcca_plus",
        lever_value=float(k),
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=merge_map,
        metrics=metrics,
    )


@register_method(
    name="markov_stability",
    description=(
        "**Markov stability** (Delvenne et al. 2010). Group cluster nodes "
        "whose t-step random-walk distributions are similar (KMeans on the "
        "rows of P^t). Lever t is the random-walk time: t=1 → fine, t→∞ → "
        "stationary clusters. Naturally multi-scale community detection."
    ),
    lever_label="t (random-walk time; 1 = fine, 20 = coarse)",
)
def markov_stability(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    k_meta: Optional[int] = None,
    **kwargs,
) -> SimplificationResult:
    cluster_ids = sorted(
        nid for nid in graph.nodes if nid not in _SPECIAL_GRAPH_NODE_IDS
    )
    n = len(cluster_ids)
    t = 1 if lever is None else max(1, int(round(lever)))
    if n <= 1:
        metrics = compute_metrics(graph, cluster_labels, metadata)
        return SimplificationResult(
            method="markov_stability",
            lever_value=float(t),
            graph=graph,
            new_labels=cluster_labels.copy(),
            node_mapping={},
            metrics=metrics,
        )

    P = _transition_matrix(graph, cluster_ids)
    Pt = np.linalg.matrix_power(P, t)
    # Choose k_meta automatically as the gap-based number of clusters, unless
    # the caller provides one. We use the eigenvalue gap heuristic on P.
    if k_meta is None:
        eigvals = np.sort(np.real(np.linalg.eigvals(P)))[::-1]
        gaps = -np.diff(eigvals)
        # Number of "near-1" eigenvalues
        k_meta = int(np.argmax(gaps[1:]) + 1) if len(gaps) > 1 else 1
        k_meta = max(2, min(k_meta + 1, n))
    k_meta = max(1, min(int(k_meta), n))
    if k_meta >= n:
        metrics = compute_metrics(graph, cluster_labels, metadata)
        return SimplificationResult(
            method="markov_stability",
            lever_value=float(t),
            graph=graph,
            new_labels=cluster_labels.copy(),
            node_mapping={},
            metrics=metrics,
        )

    meta_labels = _kmeans_on_vectors(Pt, k=k_meta)
    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    new_labels, merge_map = _apply_meta_mapping(cluster_labels, cluster_ids, meta_labels)
    new_graph = BehaviorGraph.from_cluster_assignments(
        new_labels, metadata, level=graph.level,
    )
    metrics = compute_metrics(
        new_graph, new_labels, metadata,
        original_labels=original_labels, node_mapping=merge_map,
    )
    return SimplificationResult(
        method="markov_stability",
        lever_value=float(t),
        graph=new_graph,
        new_labels=new_labels,
        node_mapping=merge_map,
        metrics=metrics,
        extras={"k_meta": float(k_meta)},
    )
