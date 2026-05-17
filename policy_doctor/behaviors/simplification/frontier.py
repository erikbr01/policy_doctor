"""Pareto-frontier sweep over a method's lever.

For each lever value:
  - Run the method on the full dataset to get the *training* metrics.
  - If `with_heldout=True`, K-fold CV: fit on train fold, apply the resulting
    node mapping to held-out labels, compute NLL of held-out transitions under
    the train graph.
  - Bootstrap CI over episodes for each scalar metric, both train and held-out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph, _apply_merge_map_to_labels
from policy_doctor.behaviors.simplification.api import (
    METHOD_LEVER_LABELS,
    METHODS,
    SimplificationResult,
    run_method,
)
from policy_doctor.behaviors.simplification.metrics import (
    DEFAULT_ALPHA,
    bootstrap_metric,
    bootstrap_mv_ci,
    compute_metrics,
    kfold_episode_splits,
    predictive_nll_bits,
    split_by_episode,
    trajectory_nll_bits,
)


# ---------------------------------------------------------------------------
# Recommended lever grids per method
# ---------------------------------------------------------------------------

LEVER_GRIDS: Dict[str, np.ndarray] = {
    "passthrough": np.array([0.0]),
    "degree_one_prune": np.array([0.0]),
    "js_merge": np.linspace(0.0, 1.0, 21),
    "hoeffding_merge": np.geomspace(1e-4, 0.99, 21),
    "chi2_merge": np.geomspace(1e-4, 0.99, 21),
    "bayesian_merge": np.linspace(0.05, 0.95, 11),
    "vomm_split_merge": np.linspace(0.0, 1.0, 11),
    "mdl_greedy": np.geomspace(0.01, 100.0, 15),
    "pcca_plus": np.arange(2, 12).astype(float),
    "markov_stability": np.array([1, 2, 3, 5, 8, 13, 21, 34], dtype=float),
    "stationary_skeleton": np.linspace(0.0, 0.3, 15),
}


# Methods whose held-out NLL cannot be cleanly evaluated (because they
# introduce IDs unseen on held-out data).
_HELDOUT_INCOMPATIBLE = {"vomm_split_merge"}


@dataclass
class FrontierPoint:
    method: str
    lever: float
    n_nodes: int
    n_edges: int
    n_free_params: int
    train_nll_bits: float
    train_nll_per_trans_bits: float
    train_nll_per_original_bits: float       # FAIR cross-method train metric
    markov_violation_bits: float             # 1st-order MV
    markov_violation_2nd_bits: float         # 2nd-order MV diagnostic
    mdl_score: float
    heldout_nll_per_trans_bits: Optional[float] = None
    heldout_nll_per_original_bits: Optional[float] = None   # FAIR cross-method held-out metric
    heldout_nll_ci_lo: Optional[float] = None
    heldout_nll_ci_hi: Optional[float] = None
    train_nll_ci_lo: Optional[float] = None
    train_nll_ci_hi: Optional[float] = None
    markov_ci_lo: Optional[float] = None
    markov_ci_hi: Optional[float] = None
    markov_2nd_ci_lo: Optional[float] = None
    markov_2nd_ci_hi: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "method": self.method,
            "lever": float(self.lever),
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "n_free_params": int(self.n_free_params),
            "train_nll_bits": float(self.train_nll_bits),
            "train_nll_per_trans_bits": float(self.train_nll_per_trans_bits),
            "train_nll_per_original_bits": float(self.train_nll_per_original_bits),
            "markov_violation_bits": float(self.markov_violation_bits),
            "markov_violation_2nd_bits": float(self.markov_violation_2nd_bits),
            "mdl_score": float(self.mdl_score),
            "heldout_nll_per_trans_bits": self.heldout_nll_per_trans_bits,
            "heldout_nll_per_original_bits": self.heldout_nll_per_original_bits,
            "heldout_nll_ci_lo": self.heldout_nll_ci_lo,
            "heldout_nll_ci_hi": self.heldout_nll_ci_hi,
            "train_nll_ci_lo": self.train_nll_ci_lo,
            "train_nll_ci_hi": self.train_nll_ci_hi,
            "markov_ci_lo": self.markov_ci_lo,
            "markov_ci_hi": self.markov_ci_hi,
            "markov_2nd_ci_lo": self.markov_2nd_ci_lo,
            "markov_2nd_ci_hi": self.markov_2nd_ci_hi,
            "extras": {k: float(v) for k, v in self.extras.items()},
        }


def _heldout_nlls(
    method: str,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    lever: float,
    n_folds: int = 5,
    alpha: float = DEFAULT_ALPHA,
    rng_seed: int = 0,
) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """K-fold CV held-out NLL.

    Returns a dict with two metrics, each as (mean, min, max) across folds:
        "per_trans"    — NLL / # merged transitions  (current-style, biased
                         toward more-merged graphs)
        "per_original" — predictive NLL / # ORIGINAL transitions  (fair
                         cross-method comparison; denominator is fixed
                         across methods for the same fold)
    """
    if method in _HELDOUT_INCOMPATIBLE:
        return None
    folds = kfold_episode_splits(metadata, level=level, n_folds=n_folds, rng_seed=rng_seed)
    if any(len(f) == 0 for f in folds):
        return None
    per_trans: List[float] = []
    per_orig: List[float] = []
    for fold in folds:
        (train_labels, train_meta), (test_labels, test_meta) = split_by_episode(
            cluster_labels, metadata, level=level, test_episodes=fold,
        )
        if len(train_meta) == 0 or len(test_meta) == 0:
            continue
        train_graph = BehaviorGraph.from_cluster_assignments(
            train_labels, train_meta, level=level,
        )
        try:
            result = run_method(method, train_graph, train_labels, train_meta, lever=lever)
        except Exception:
            continue
        # Per-merged-transition NLL (compressive, biased)
        test_labels_mapped = _apply_merge_map_to_labels(test_labels, result.node_mapping)
        nll_t, nt = trajectory_nll_bits(result.graph, test_labels_mapped, test_meta, alpha=alpha)
        if nt > 0:
            per_trans.append(nll_t / nt)
        # Predictive NLL / # original test transitions (fair across methods)
        nll_o, _, n_orig = predictive_nll_bits(
            result.graph, test_labels, test_meta, result.node_mapping, alpha=alpha,
        )
        if n_orig > 0:
            per_orig.append(nll_o / n_orig)
    if not per_trans and not per_orig:
        return None
    out: Dict[str, Tuple[float, float, float]] = {}
    if per_trans:
        a = np.array(per_trans)
        out["per_trans"] = (float(a.mean()), float(a.min()), float(a.max()))
    if per_orig:
        a = np.array(per_orig)
        out["per_original"] = (float(a.mean()), float(a.min()), float(a.max()))
    return out


def _train_metric_ci(
    method: str,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    lever: float,
    metric: str,  # "nll_per_trans" or "markov"
    n_bootstrap: int = 30,
    rng_seed: int = 0,
) -> Optional[Tuple[float, float, float]]:
    """Episode-bootstrap CI on a train-time metric. Re-runs the method per
    bootstrap sample (so it's accurate but slow — use small n_bootstrap)."""

    def fn(labels: np.ndarray, meta: List[Dict]) -> float:
        try:
            graph = BehaviorGraph.from_cluster_assignments(labels, meta, level=level)
            res = run_method(method, graph, labels, meta, lever=lever)
            if metric == "nll_per_trans":
                return float(res.metrics.nll_per_transition_bits)
            elif metric == "markov":
                return float(res.metrics.markov_violation_bits)
            else:
                return 0.0
        except Exception:
            return float("nan")

    try:
        return bootstrap_metric(
            fn, cluster_labels, metadata, level=level,
            n_bootstrap=n_bootstrap, rng_seed=rng_seed,
        )
    except Exception:
        return None


def sweep_method(
    method: str,
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever_grid: Optional[Sequence[float]] = None,
    with_heldout: bool = True,
    n_folds: int = 5,
    with_bootstrap: bool = False,
    n_bootstrap: int = 20,
    rng_seed: int = 0,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> List[FrontierPoint]:
    """Run one method across its lever grid."""
    if method not in METHODS:
        raise KeyError(f"Unknown method: {method!r}")
    if lever_grid is None:
        lever_grid = LEVER_GRIDS.get(method, np.array([0.0]))

    points: List[FrontierPoint] = []
    level = graph.level
    for i, lev in enumerate(lever_grid):
        t0 = time.time()
        result = run_method(method, graph, cluster_labels, metadata, lever=float(lev))
        train_metrics = result.metrics

        heldout_stats = None
        if with_heldout:
            heldout_stats = _heldout_nlls(
                method, cluster_labels, metadata, level, float(lev),
                n_folds=n_folds, rng_seed=rng_seed,
            )

        train_nll_ci = None
        markov_ci = None
        markov_2nd_ci = None
        if with_bootstrap:
            # Fast MV bootstrap: re-evaluate MV on episode-resampled data,
            # holding the simplification (mapping) FIXED. Reflects data noise
            # in the metric estimate; doesn't reflect method instability.
            try:
                _, lo1, hi1 = bootstrap_mv_ci(
                    cluster_labels, metadata, result.node_mapping,
                    level=level, order=1, n_bootstrap=n_bootstrap, rng_seed=rng_seed,
                    current_labels=result.new_labels,
                )
                markov_ci = (float(train_metrics.markov_violation_bits), lo1, hi1)
                _, lo2, hi2 = bootstrap_mv_ci(
                    cluster_labels, metadata, result.node_mapping,
                    level=level, order=2, n_bootstrap=n_bootstrap, rng_seed=rng_seed,
                    current_labels=result.new_labels,
                )
                markov_2nd_ci = (float(train_metrics.markov_violation_2nd_bits), lo2, hi2)
            except Exception:
                markov_ci = None
                markov_2nd_ci = None

        per_trans = heldout_stats.get("per_trans") if heldout_stats else None
        per_orig = heldout_stats.get("per_original") if heldout_stats else None

        pt = FrontierPoint(
            method=method,
            lever=float(lev),
            n_nodes=train_metrics.n_nodes,
            n_edges=train_metrics.n_edges,
            n_free_params=train_metrics.n_free_params,
            train_nll_bits=train_metrics.nll_bits,
            train_nll_per_trans_bits=train_metrics.nll_per_transition_bits,
            train_nll_per_original_bits=train_metrics.nll_per_original_bits,
            markov_violation_bits=train_metrics.markov_violation_bits,
            markov_violation_2nd_bits=train_metrics.markov_violation_2nd_bits,
            mdl_score=train_metrics.mdl_score,
            heldout_nll_per_trans_bits=per_trans[0] if per_trans else None,
            heldout_nll_per_original_bits=per_orig[0] if per_orig else None,
            heldout_nll_ci_lo=per_orig[1] if per_orig else (per_trans[1] if per_trans else None),
            heldout_nll_ci_hi=per_orig[2] if per_orig else (per_trans[2] if per_trans else None),
            train_nll_ci_lo=train_nll_ci[1] if train_nll_ci else None,
            train_nll_ci_hi=train_nll_ci[2] if train_nll_ci else None,
            markov_ci_lo=markov_ci[1] if markov_ci else None,
            markov_ci_hi=markov_ci[2] if markov_ci else None,
            markov_2nd_ci_lo=markov_2nd_ci[1] if markov_2nd_ci else None,
            markov_2nd_ci_hi=markov_2nd_ci[2] if markov_2nd_ci else None,
            extras=dict(result.extras),
        )
        points.append(pt)
        if progress_cb is not None:
            progress_cb(i + 1, len(lever_grid), time.time() - t0)
    return points


def pareto_sweep(
    method: str,
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever_grid: Optional[Sequence[float]] = None,
    **kwargs,
) -> List[FrontierPoint]:
    """Alias for sweep_method (left here for the public API)."""
    return sweep_method(method, graph, cluster_labels, metadata, lever_grid, **kwargs)
