"""Data-support measurement for behavior-graph nodes.

For each rollout window (already assigned to a cluster by the upstream
``run_clustering`` step), measure how well it is supported by the training
distribution.  We do this by:

  1. Building a *shared* representation space over aggregated demo + rollout
     windows by re-fitting UMAP on the union of both sets.
  2. For each rollout window in that joint space, computing one or more
     density / proximity metrics against the demo-window cloud.
  3. Aggregating the per-window values into per-cluster distributions.

The whole module is sklearn / umap-learn based and has no Streamlit or
sim-stack dependencies — it can be imported in the ``policy_doctor`` env.

The metric registry pattern keeps things flexible: we fit one BallTree on the
demo windows in the joint UMAP space and run every selected metric off the
same tree, so adding a new metric is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint UMAP fit
# ---------------------------------------------------------------------------


@dataclass
class JointUmapResult:
    """Output of :func:`fit_joint_umap`.

    ``demo_reduced``/``rollout_reduced`` are *views* of the same joint
    embedding: the joint UMAP is fit on ``concat([demo, rollout])`` and split
    back into the two halves so downstream code can keep them separate.
    """

    umap_model: Any
    demo_reduced: np.ndarray   # (N_demo, k)
    rollout_reduced: np.ndarray  # (N_rollout, k)


def fit_joint_umap(
    demo_windows: np.ndarray,
    rollout_windows: np.ndarray,
    *,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
    metric: str = "euclidean",
    normalize: str = "standard",
    n_jobs: int = -1,
) -> JointUmapResult:
    """Fit a single UMAP on ``concat([demo, rollout])`` and split back.

    Args:
        demo_windows:    (N_demo, D) float
        rollout_windows: (N_rollout, D) float
        n_components:    UMAP output dim — keep a few above 2 so density
                         estimates don't collapse to a line.
        normalize:       Pre-UMAP scaling — ``"standard"`` (recommended) /
                         ``"none"`` / ``"l2"``.  Same options as the
                         clustering pipeline's pre-scaler.

    Returns: :class:`JointUmapResult`.
    """
    import umap  # type: ignore[import-not-found]
    from sklearn.preprocessing import StandardScaler, normalize as sk_normalize

    if demo_windows.size == 0:
        raise ValueError("demo_windows is empty — no training-demo windows to compare against.")
    if rollout_windows.size == 0:
        raise ValueError("rollout_windows is empty — nothing to score.")
    if demo_windows.shape[1] != rollout_windows.shape[1]:
        raise ValueError(
            f"demo_windows.shape[1]={demo_windows.shape[1]} must match "
            f"rollout_windows.shape[1]={rollout_windows.shape[1]}; the two "
            "embedding sources must produce the same dimensionality."
        )

    n_demo = demo_windows.shape[0]
    joint = np.concatenate([demo_windows, rollout_windows], axis=0).astype(np.float32)

    if normalize == "standard":
        scaler = StandardScaler().fit(joint)
        joint = scaler.transform(joint).astype(np.float32)
    elif normalize == "l2":
        joint = sk_normalize(joint, norm="l2").astype(np.float32)
    elif normalize not in ("none", None):
        raise ValueError(f"Unknown normalize method: {normalize!r}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric=metric,
        n_jobs=n_jobs,
        low_memory=False,
    )
    reduced = reducer.fit_transform(joint).astype(np.float32)
    return JointUmapResult(
        umap_model=reducer,
        demo_reduced=reduced[:n_demo],
        rollout_reduced=reduced[n_demo:],
    )


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


@dataclass
class MetricContext:
    """Shared inputs for any data-support metric.

    Built once per ``compute_all_metrics`` call so each metric can reuse the
    BallTree / KDE / etc. instead of re-fitting them.
    """

    demo_points: np.ndarray            # (N_demo, k) joint UMAP coords
    rollout_points: np.ndarray         # (N_rollout, k) joint UMAP coords
    tree: Any                          # sklearn.neighbors.BallTree on demo_points
    radius: float
    knn_k: int
    kde_bandwidth: Any                 # float | "scott" | "silverman"
    kde_model: Optional[Any] = None    # cached KernelDensity, fit lazily
    extras: Dict[str, Any] = field(default_factory=dict)


MetricFn = Callable[[MetricContext], np.ndarray]


_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str):
    """Decorator: register a metric function under ``name``.

    The function must accept a :class:`MetricContext` and return a 1-D
    ``np.ndarray`` of shape ``(N_rollout,)`` containing one scalar per
    rollout point.
    """

    def _decorate(fn: MetricFn) -> MetricFn:
        if name in _REGISTRY:
            raise ValueError(f"Data-support metric {name!r} already registered.")
        _REGISTRY[name] = fn
        return fn

    return _decorate


def available_metrics() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_metric(name: str) -> MetricFn:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown data-support metric {name!r}.  Available: {available_metrics()}"
        )
    return _REGISTRY[name]


# --- Concrete metrics -------------------------------------------------------


@register_metric("count_in_radius")
def _count_in_radius(ctx: MetricContext) -> np.ndarray:
    """Number of demo windows within ``ctx.radius`` of each rollout window.

    The user's stated default — interpretable as a local density count.
    """
    counts = ctx.tree.query_radius(ctx.rollout_points, r=ctx.radius, count_only=True)
    return np.asarray(counts, dtype=np.float64)


@register_metric("binary_coverage")
def _binary_coverage(ctx: MetricContext) -> np.ndarray:
    """1 if at least one demo window is within ``ctx.radius`` of the rollout window.

    Per-cluster summary is then naturally a *coverage fraction* — easy to read.
    """
    counts = ctx.tree.query_radius(ctx.rollout_points, r=ctx.radius, count_only=True)
    return (np.asarray(counts) > 0).astype(np.float64)


@register_metric("knn_mean_distance")
def _knn_mean_distance(ctx: MetricContext) -> np.ndarray:
    """Mean Euclidean distance from each rollout window to its k nearest demos.

    Smaller = better-supported.  Less sensitive to radius choice than
    count_in_radius.
    """
    k = max(1, int(ctx.knn_k))
    k = min(k, ctx.demo_points.shape[0])
    dists, _ = ctx.tree.query(ctx.rollout_points, k=k)
    return dists.mean(axis=1).astype(np.float64)


@register_metric("knn_max_distance")
def _knn_max_distance(ctx: MetricContext) -> np.ndarray:
    """Distance from each rollout window to its k-th nearest demo (worst-case support)."""
    k = max(1, int(ctx.knn_k))
    k = min(k, ctx.demo_points.shape[0])
    dists, _ = ctx.tree.query(ctx.rollout_points, k=k)
    return dists[:, -1].astype(np.float64)


@register_metric("kde_log_density")
def _kde_log_density(ctx: MetricContext) -> np.ndarray:
    """Gaussian-KDE log-density of demo cloud evaluated at each rollout window.

    Higher = better-supported.  Smooth, well-behaved across density regimes.
    """
    from sklearn.neighbors import KernelDensity

    if ctx.kde_model is None:
        bw = ctx.kde_bandwidth
        if isinstance(bw, str):
            # sklearn KernelDensity only accepts float bandwidths.  Implement
            # Scott / Silverman manually on the demo cloud.
            n, d = ctx.demo_points.shape
            std = ctx.demo_points.std(axis=0).mean() or 1.0
            if bw == "scott":
                bw_val = std * n ** (-1.0 / (d + 4))
            elif bw == "silverman":
                bw_val = std * (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
            else:
                raise ValueError(f"Unknown bandwidth rule: {bw!r}")
        else:
            bw_val = float(bw)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw_val)
        kde.fit(ctx.demo_points)
        ctx.kde_model = kde
    return ctx.kde_model.score_samples(ctx.rollout_points).astype(np.float64)


# ---------------------------------------------------------------------------
# Driver: run all selected metrics in one pass
# ---------------------------------------------------------------------------


def compute_all_metrics(
    demo_points: np.ndarray,
    rollout_points: np.ndarray,
    *,
    metrics: Iterable[str],
    radius: float,
    knn_k: int = 10,
    kde_bandwidth: Any = "scott",
    leaf_size: int = 40,
) -> Tuple[Dict[str, np.ndarray], MetricContext]:
    """Evaluate every metric in ``metrics`` on the joint UMAP-space points.

    Returns:
        per_metric_values: ``{metric_name: (N_rollout,) np.ndarray}``
        ctx:               the shared :class:`MetricContext` (exposed for
                           debugging / for callers that want to add metrics
                           after the fact).
    """
    from sklearn.neighbors import BallTree

    metrics = list(metrics)
    unknown = [m for m in metrics if m not in _REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}.  Available: {available_metrics()}"
        )

    tree = BallTree(demo_points, leaf_size=leaf_size, metric="euclidean")
    ctx = MetricContext(
        demo_points=demo_points,
        rollout_points=rollout_points,
        tree=tree,
        radius=float(radius),
        knn_k=int(knn_k),
        kde_bandwidth=kde_bandwidth,
    )
    out: Dict[str, np.ndarray] = {}
    for name in metrics:
        out[name] = get_metric(name)(ctx)
    return out, ctx


# ---------------------------------------------------------------------------
# Per-cluster aggregation
# ---------------------------------------------------------------------------


def aggregate_per_cluster(
    per_slice_values: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    exclude_labels: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Bucket the per-slice values by cluster ID and emit summary statistics.

    Returns ``{cluster_id: {"raw": [...], "median": ..., "mean": ...,
    "q10": ..., "q90": ..., "n_slices": ...}}`` for each cluster present in
    ``cluster_labels``.  ``cluster_id`` is a plain Python ``int`` so the
    result round-trips through JSON without surprises.
    """
    if per_slice_values.shape[0] != cluster_labels.shape[0]:
        raise ValueError(
            f"per_slice_values has {per_slice_values.shape[0]} entries but "
            f"cluster_labels has {cluster_labels.shape[0]}; they must align "
            "1-to-1."
        )
    exclude = set(int(x) for x in (exclude_labels or ()))
    out: Dict[int, Dict[str, Any]] = {}
    unique = np.unique(cluster_labels)
    for cid in unique:
        cid_int = int(cid)
        if cid_int in exclude:
            continue
        mask = cluster_labels == cid
        raw = per_slice_values[mask].astype(np.float64)
        if raw.size == 0:
            continue
        out[cid_int] = {
            "raw": raw.tolist(),
            "median": float(np.median(raw)),
            "mean": float(np.mean(raw)),
            "q10": float(np.quantile(raw, 0.10)),
            "q90": float(np.quantile(raw, 0.90)),
            "n_slices": int(raw.size),
        }
    return out
