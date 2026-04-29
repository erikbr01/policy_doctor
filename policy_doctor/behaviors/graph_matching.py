"""Behavior graph node matching across flywheel iterations.

Pairs cluster nodes from two BehaviorGraphs using feature descriptors and the
Hungarian algorithm, filtering ambiguous matches with Lowe's ratio test.

Six methods — ``structural``, ``temporal``, ``procrustes``, ``combined``,
``state_action``, ``state_structural`` — share the same match/filter core.

Typical usage::

    result = match_graphs(graph_a, meta_a, labels_a, graph_b, meta_b, labels_b)
    chains = build_tracking_chains([result], ["iter_0", "iter_1"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    TERMINAL_NODE_IDS,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    FAILURE_NODE_ID,
)

_SPECIAL = frozenset({START_NODE_ID}) | TERMINAL_NODE_IDS


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class NodeMatch:
    node_id_a: int
    node_id_b: int
    distance: float
    method: str
    confidence: Optional[float] = None  # fraction of methods that agreed; None for single-method


@dataclass
class GraphMatchResult:
    matches: List[NodeMatch]
    unmatched_a: List[int]
    unmatched_b: List[int]
    method: str


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------

def _ep_key(metadata: List[Dict]) -> str:
    for m in metadata:
        if "rollout_idx" in m:
            return "rollout_idx"
    return "demo_idx"


def _structural_descriptor(
    graph: BehaviorGraph,
    node_id: int,
    values: Dict[int, float],
    total_timesteps: int,
) -> np.ndarray:
    node = graph.nodes[node_id]
    p_to_success = graph.transition_probs.get(node_id, {}).get(SUCCESS_NODE_ID, 0.0)
    p_to_failure = graph.transition_probs.get(node_id, {}).get(FAILURE_NODE_ID, 0.0)
    p_from_start = graph.transition_probs.get(START_NODE_ID, {}).get(node_id, 0.0)
    out_deg = float(sum(
        1 for t, c, _ in graph.get_outgoing_transitions(node_id)
        if c > 0 and t not in TERMINAL_NODE_IDS
    ))
    in_deg = float(sum(
        1 for s, c, _ in graph.get_incoming_transitions(node_id)
        if c > 0 and s != START_NODE_ID
    ))
    ts_frac = node.num_timesteps / max(total_timesteps, 1)
    ep_cov = node.num_episodes / max(graph.num_episodes, 1)
    return np.array([
        values.get(node_id, 0.0),
        p_to_success,
        p_to_failure,
        p_from_start,
        out_deg,
        in_deg,
        ts_frac,
        ep_cov,
    ], dtype=np.float32)


def _temporal_histogram(
    metadata: List[Dict],
    cluster_labels: np.ndarray,
    node_id: int,
    n_bins: int = 10,
) -> np.ndarray:
    ws_key = "window_start" if any("window_start" in m for m in metadata) else "timestep"

    # max window_start per rollout, for normalization
    ep_key = _ep_key(metadata)
    max_ws: Dict[int, int] = {}
    for meta in metadata:
        rid = meta.get(ep_key, 0)
        ws = meta.get(ws_key, 0)
        if rid not in max_ws or ws > max_ws[rid]:
            max_ws[rid] = ws

    positions = []
    for i, meta in enumerate(metadata):
        if int(cluster_labels[i]) != node_id:
            continue
        rid = meta.get(ep_key, 0)
        ws = meta.get(ws_key, 0)
        denom = max(max_ws.get(rid, 1), 1)
        positions.append(ws / denom)

    if not positions:
        return np.zeros(n_bins, dtype=np.float32)
    hist, _ = np.histogram(positions, bins=n_bins, range=(0.0, 1.0))
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence; returns value in [0, log 2]."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _state_action_slice(
    meta: Dict,
    obs: Dict[int, np.ndarray],
    ep_key: str,
) -> Optional[np.ndarray]:
    rid = meta.get(ep_key, 0)
    if rid not in obs:
        return None
    ws = meta.get("window_start", meta.get("timestep", 0))
    we = meta.get("window_end", ws + meta.get("window_width", 1))
    seg = obs[rid][ws:we]
    return seg if len(seg) > 0 else None


def _state_action_descriptor(
    metadata: List[Dict],
    cluster_labels: np.ndarray,
    node_id: int,
    obs: Dict[int, np.ndarray],
    obs_agg: str,
    include_actions: bool,
    action_obs: Optional[Dict[int, np.ndarray]],
) -> Optional[np.ndarray]:
    ep_key = _ep_key(metadata)
    slices = []
    for i, meta in enumerate(metadata):
        if int(cluster_labels[i]) != node_id:
            continue
        seg = _state_action_slice(meta, obs, ep_key)
        if seg is None:
            continue
        if include_actions and action_obs is not None:
            rid = meta.get(ep_key, 0)
            if rid in action_obs:
                ws = meta.get("window_start", meta.get("timestep", 0))
                we = meta.get("window_end", ws + meta.get("window_width", 1))
                act_seg = action_obs[rid][ws:we]
                if len(act_seg) == len(seg):
                    seg = np.concatenate([seg, act_seg], axis=-1)
        slices.append(seg)

    if not slices:
        return None

    if obs_agg == "mean":
        return np.mean([s.mean(axis=0) for s in slices], axis=0).astype(np.float32)

    if obs_agg == "mean_std":
        all_ts = np.concatenate(slices, axis=0)
        return np.concatenate([all_ts.mean(axis=0), all_ts.std(axis=0)]).astype(np.float32)

    if obs_agg == "temporal_profile":
        fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        profiles = []
        for seg in slices:
            n = len(seg)
            pts = [seg[min(int(f * (n - 1)), n - 1)] for f in fracs]
            profiles.append(np.concatenate(pts))
        return np.mean(profiles, axis=0).astype(np.float32)

    raise ValueError(f"Unknown obs_agg: {obs_agg!r}. Expected 'mean', 'mean_std', 'temporal_profile'.")


# ---------------------------------------------------------------------------
# Matching core
# ---------------------------------------------------------------------------

def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-12)
    return m / norms


def _joint_featurewise_normalize(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize each feature dimension jointly across both descriptor matrices."""
    combined = np.concatenate([a, b], axis=0)
    std = combined.std(axis=0).clip(min=1e-12)
    mean = combined.mean(axis=0)
    return (a - mean) / std, (b - mean) / std


def _filter_and_build_result(
    dist_matrix: np.ndarray,
    ids_a: List[int],
    ids_b: List[int],
    row_assign: np.ndarray,
    col_assign: np.ndarray,
    ratio: float,
    max_distance: Optional[float],
    method: str,
) -> GraphMatchResult:
    matched_a: set = set()
    matched_b: set = set()
    matches: List[NodeMatch] = []

    for r, c in zip(row_assign, col_assign):
        d = float(dist_matrix[r, c])

        if max_distance is not None and d > max_distance:
            continue

        if ratio < 1.0 and dist_matrix.shape[1] >= 2:
            row = dist_matrix[r]
            sorted_dists = np.sort(row)
            d_second = sorted_dists[1] if sorted_dists[0] == d else sorted_dists[0]
            if d_second > 1e-12 and d / d_second >= ratio:
                continue

        matches.append(NodeMatch(ids_a[r], ids_b[c], d, method))
        matched_a.add(r)
        matched_b.add(c)

    unmatched_a = [ids_a[i] for i in range(len(ids_a)) if i not in matched_a]
    unmatched_b = [ids_b[j] for j in range(len(ids_b)) if j not in matched_b]
    return GraphMatchResult(matches=matches, unmatched_a=unmatched_a, unmatched_b=unmatched_b, method=method)


def _run_hungarian(
    dist_matrix: np.ndarray,
    ids_a: List[int],
    ids_b: List[int],
    ratio: float,
    max_distance: Optional[float],
    method: str,
) -> GraphMatchResult:
    if dist_matrix.size == 0:
        return GraphMatchResult([], list(ids_a), list(ids_b), method)
    row_assign, col_assign = linear_sum_assignment(dist_matrix)
    return _filter_and_build_result(
        dist_matrix, ids_a, ids_b, row_assign, col_assign, ratio, max_distance, method
    )


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def _match_structural(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    ratio: float,
    max_distance: Optional[float],
) -> GraphMatchResult:
    values_a = graph_a.compute_values()
    values_b = graph_b.compute_values()
    total_ts_a = sum(n.num_timesteps for n in graph_a.cluster_nodes.values())
    total_ts_b = sum(n.num_timesteps for n in graph_b.cluster_nodes.values())
    ids_a = sorted(graph_a.cluster_nodes)
    ids_b = sorted(graph_b.cluster_nodes)
    if not ids_a or not ids_b:
        return GraphMatchResult([], ids_a, ids_b, "structural")
    descs_a = np.array([_structural_descriptor(graph_a, n, values_a, total_ts_a) for n in ids_a])
    descs_b = np.array([_structural_descriptor(graph_b, n, values_b, total_ts_b) for n in ids_b])
    descs_a = _l2_normalize_rows(descs_a)
    descs_b = _l2_normalize_rows(descs_b)
    dist_matrix = cdist(descs_a, descs_b, metric="euclidean")
    return _run_hungarian(dist_matrix, ids_a, ids_b, ratio, max_distance, "structural")


def _match_temporal(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    n_bins: int,
    ratio: float,
    max_distance: Optional[float],
) -> GraphMatchResult:
    ids_a = sorted(graph_a.cluster_nodes)
    ids_b = sorted(graph_b.cluster_nodes)
    if not ids_a or not ids_b:
        return GraphMatchResult([], ids_a, ids_b, "temporal")
    hists_a = [_temporal_histogram(metadata_a, cluster_labels_a, n, n_bins) for n in ids_a]
    hists_b = [_temporal_histogram(metadata_b, cluster_labels_b, n, n_bins) for n in ids_b]
    dist_matrix = np.array(
        [[_js_divergence(ha, hb) for hb in hists_b] for ha in hists_a],
        dtype=np.float32,
    )
    return _run_hungarian(dist_matrix, ids_a, ids_b, ratio, max_distance, "temporal")


def _match_procrustes(
    graph_a: BehaviorGraph,
    clustering_dir_a,
    graph_b: BehaviorGraph,
    clustering_dir_b,
    scale_threshold: float,
    ratio: float,
    max_distance: Optional[float],
) -> GraphMatchResult:
    import pathlib
    from scipy.spatial import procrustes as scipy_procrustes
    from policy_doctor.data.clustering_loader import load_clustering_models

    models_a = load_clustering_models(pathlib.Path(clustering_dir_a))
    models_b = load_clustering_models(pathlib.Path(clustering_dir_b))
    ids_a = sorted(graph_a.cluster_nodes)
    ids_b = sorted(graph_b.cluster_nodes)
    if not ids_a or not ids_b:
        return GraphMatchResult([], ids_a, ids_b, "procrustes")

    ca_full = models_a.kmeans.cluster_centers_
    cb_full = models_b.kmeans.cluster_centers_
    if ca_full.shape != cb_full.shape:
        raise ValueError(
            f"Procrustes requires equal-shaped centroid matrices; "
            f"got {ca_full.shape} vs {cb_full.shape}. Use a different method."
        )
    _, cb_aligned, _ = scipy_procrustes(ca_full, cb_full)

    ca = ca_full[ids_a]
    cb = np.array(cb_aligned)[ids_b]
    dist_matrix = cdist(ca, cb, metric="euclidean")

    effective_max = max_distance
    if effective_max is None:
        nn_dists = dist_matrix.min(axis=1)
        effective_max = float(np.median(nn_dists)) * scale_threshold

    return _run_hungarian(dist_matrix, ids_a, ids_b, ratio, effective_max, "procrustes")


def _match_combined(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    alpha: float,
    n_bins: int,
    ratio: float,
    max_distance: Optional[float],
) -> GraphMatchResult:
    values_a = graph_a.compute_values()
    values_b = graph_b.compute_values()
    total_ts_a = sum(n.num_timesteps for n in graph_a.cluster_nodes.values())
    total_ts_b = sum(n.num_timesteps for n in graph_b.cluster_nodes.values())
    ids_a = sorted(graph_a.cluster_nodes)
    ids_b = sorted(graph_b.cluster_nodes)
    if not ids_a or not ids_b:
        return GraphMatchResult([], ids_a, ids_b, "combined")

    struct_a = _l2_normalize_rows(np.array(
        [_structural_descriptor(graph_a, n, values_a, total_ts_a) for n in ids_a]
    ))
    struct_b = _l2_normalize_rows(np.array(
        [_structural_descriptor(graph_b, n, values_b, total_ts_b) for n in ids_b]
    ))
    hist_a = np.array([_temporal_histogram(metadata_a, cluster_labels_a, n, n_bins) for n in ids_a])
    hist_b = np.array([_temporal_histogram(metadata_b, cluster_labels_b, n, n_bins) for n in ids_b])

    desc_a = np.concatenate([(1.0 - alpha) * struct_a, alpha * hist_a], axis=1)
    desc_b = np.concatenate([(1.0 - alpha) * struct_b, alpha * hist_b], axis=1)
    dist_matrix = cdist(desc_a, desc_b, metric="euclidean")
    return _run_hungarian(dist_matrix, ids_a, ids_b, ratio, max_distance, "combined")


def _build_state_action_descs(
    graph: BehaviorGraph,
    metadata: List[Dict],
    cluster_labels: np.ndarray,
    obs: Dict[int, np.ndarray],
    obs_agg: str,
    include_actions: bool,
    action_obs: Optional[Dict[int, np.ndarray]],
) -> Tuple[List[int], np.ndarray]:
    """Return (valid_ids, descriptors) — nodes without obs data are omitted."""
    ids_all = sorted(graph.cluster_nodes)
    valid_ids, descs = [], []
    for nid in ids_all:
        d = _state_action_descriptor(
            metadata, cluster_labels, nid, obs, obs_agg, include_actions, action_obs
        )
        if d is not None:
            valid_ids.append(nid)
            descs.append(d)
    if not descs:
        return [], np.zeros((0, 1), dtype=np.float32)
    return valid_ids, np.array(descs, dtype=np.float32)


def _match_state_action(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    obs_a: Dict[int, np.ndarray],
    obs_b: Dict[int, np.ndarray],
    obs_agg: str,
    include_actions: bool,
    action_obs_a: Optional[Dict[int, np.ndarray]],
    action_obs_b: Optional[Dict[int, np.ndarray]],
    ratio: float,
    max_distance: Optional[float],
) -> GraphMatchResult:
    ids_a, descs_a = _build_state_action_descs(graph_a, metadata_a, cluster_labels_a, obs_a, obs_agg, include_actions, action_obs_a)
    ids_b, descs_b = _build_state_action_descs(graph_b, metadata_b, cluster_labels_b, obs_b, obs_agg, include_actions, action_obs_b)

    all_a = sorted(graph_a.cluster_nodes)
    all_b = sorted(graph_b.cluster_nodes)
    missing_a = [n for n in all_a if n not in set(ids_a)]
    missing_b = [n for n in all_b if n not in set(ids_b)]

    if not ids_a or not ids_b:
        return GraphMatchResult([], all_a, all_b, "state_action")

    # Normalize jointly so each feature dimension is on the same scale
    descs_a, descs_b = _joint_featurewise_normalize(descs_a, descs_b)
    dist_matrix = cdist(descs_a, descs_b, metric="euclidean")
    result = _run_hungarian(dist_matrix, ids_a, ids_b, ratio, max_distance, "state_action")
    result.unmatched_a.extend(missing_a)
    result.unmatched_b.extend(missing_b)
    return result


def _match_state_structural(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    obs_a: Dict[int, np.ndarray],
    obs_b: Dict[int, np.ndarray],
    obs_agg: str,
    include_actions: bool,
    action_obs_a: Optional[Dict[int, np.ndarray]],
    action_obs_b: Optional[Dict[int, np.ndarray]],
    alpha: float,
    ratio: float,
    max_distance: Optional[float],
) -> GraphMatchResult:
    values_a = graph_a.compute_values()
    values_b = graph_b.compute_values()
    total_ts_a = sum(n.num_timesteps for n in graph_a.cluster_nodes.values())
    total_ts_b = sum(n.num_timesteps for n in graph_b.cluster_nodes.values())

    ids_a, sa_a = _build_state_action_descs(graph_a, metadata_a, cluster_labels_a, obs_a, obs_agg, include_actions, action_obs_a)
    ids_b, sa_b = _build_state_action_descs(graph_b, metadata_b, cluster_labels_b, obs_b, obs_agg, include_actions, action_obs_b)

    all_a = sorted(graph_a.cluster_nodes)
    all_b = sorted(graph_b.cluster_nodes)
    missing_a = [n for n in all_a if n not in set(ids_a)]
    missing_b = [n for n in all_b if n not in set(ids_b)]

    if not ids_a or not ids_b:
        return GraphMatchResult([], all_a, all_b, "state_structural")

    struct_a = _l2_normalize_rows(np.array(
        [_structural_descriptor(graph_a, n, values_a, total_ts_a) for n in ids_a]
    ))
    struct_b = _l2_normalize_rows(np.array(
        [_structural_descriptor(graph_b, n, values_b, total_ts_b) for n in ids_b]
    ))

    # Normalize obs jointly, then L2-normalize rows so both parts are unit-scale
    sa_a, sa_b = _joint_featurewise_normalize(sa_a, sa_b)
    sa_a = _l2_normalize_rows(sa_a)
    sa_b = _l2_normalize_rows(sa_b)

    desc_a = np.concatenate([(1.0 - alpha) * struct_a, alpha * sa_a], axis=1)
    desc_b = np.concatenate([(1.0 - alpha) * struct_b, alpha * sa_b], axis=1)
    dist_matrix = cdist(desc_a, desc_b, metric="euclidean")
    result = _run_hungarian(dist_matrix, ids_a, ids_b, ratio, max_distance, "state_structural")
    result.unmatched_a.extend(missing_a)
    result.unmatched_b.extend(missing_b)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_graphs(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    method: str = "combined",
    ratio: float = 0.8,
    max_distance: Optional[float] = None,
    *,
    # temporal / combined
    n_bins: int = 10,
    # combined / state_structural
    alpha: float = 0.5,
    # procrustes
    clustering_dir_a=None,
    clustering_dir_b=None,
    scale_threshold: float = 2.0,
    # state_action / state_structural
    obs_a: Optional[Dict[int, np.ndarray]] = None,
    obs_b: Optional[Dict[int, np.ndarray]] = None,
    obs_agg: str = "mean",
    include_actions: bool = False,
    action_obs_a: Optional[Dict[int, np.ndarray]] = None,
    action_obs_b: Optional[Dict[int, np.ndarray]] = None,
) -> GraphMatchResult:
    """Match cluster nodes between two BehaviorGraphs.

    Args:
        graph_a, graph_b: Graphs to match.
        metadata_a, metadata_b: Per-slice metadata dicts (same indexing as cluster_labels).
        cluster_labels_a, cluster_labels_b: Per-slice cluster label arrays.
        method: One of:
            ``"structural"``       — Graph topology + Bellman values (no external data needed).
            ``"temporal"``         — Temporal histogram of cluster occupancy over episode.
            ``"combined"``         — Structural + temporal (default; no external data needed).
            ``"procrustes"``       — KMeans centroid alignment via Procrustes (needs clustering_dir).
            ``"state_action"``     — Physical state/action distribution matching (needs obs_a/b).
            ``"state_structural"`` — State-action + structural (recommended when obs available).
        ratio: Lowe ratio test threshold. Match (u→v) is accepted only if
            ``d(u,v) / d(u, v₂) < ratio`` where v₂ is the second-nearest node. 1.0 = disabled.
        max_distance: Absolute distance threshold; matches above this are rejected.
        n_bins: Number of histogram bins for temporal / combined methods.
        alpha: Weight of the second descriptor in combined methods (0 = structural only).
        clustering_dir_a, clustering_dir_b: Paths to clustering result dirs (procrustes method).
        scale_threshold: Procrustes auto-threshold = median_nn_dist × scale_threshold.
        obs_a, obs_b: Pre-loaded rollout observations keyed by rollout_idx
            (shape: T × obs_dim). Required for state_action / state_structural.
        obs_agg: Aggregation over slices — ``"mean"``, ``"mean_std"``, ``"temporal_profile"``.
        include_actions: If True, concatenate action vectors to obs (use with caution;
            actions shift across policies).
        action_obs_a, action_obs_b: Pre-loaded action arrays (same format as obs_a/b).

    Returns:
        GraphMatchResult with matched pairs, unmatched node IDs, and method name.
    """
    if method == "structural":
        return _match_structural(
            graph_a, metadata_a, cluster_labels_a,
            graph_b, metadata_b, cluster_labels_b,
            ratio, max_distance,
        )
    if method == "temporal":
        return _match_temporal(
            graph_a, metadata_a, cluster_labels_a,
            graph_b, metadata_b, cluster_labels_b,
            n_bins, ratio, max_distance,
        )
    if method == "combined":
        return _match_combined(
            graph_a, metadata_a, cluster_labels_a,
            graph_b, metadata_b, cluster_labels_b,
            alpha, n_bins, ratio, max_distance,
        )
    if method == "procrustes":
        if clustering_dir_a is None or clustering_dir_b is None:
            raise ValueError("procrustes method requires clustering_dir_a and clustering_dir_b.")
        return _match_procrustes(
            graph_a, clustering_dir_a,
            graph_b, clustering_dir_b,
            scale_threshold, ratio, max_distance,
        )
    if method in ("state_action", "state_structural"):
        if obs_a is None or obs_b is None:
            raise ValueError(f"{method!r} method requires obs_a and obs_b.")
        if method == "state_action":
            return _match_state_action(
                graph_a, metadata_a, cluster_labels_a,
                graph_b, metadata_b, cluster_labels_b,
                obs_a, obs_b, obs_agg, include_actions, action_obs_a, action_obs_b,
                ratio, max_distance,
            )
        return _match_state_structural(
            graph_a, metadata_a, cluster_labels_a,
            graph_b, metadata_b, cluster_labels_b,
            obs_a, obs_b, obs_agg, include_actions, action_obs_a, action_obs_b,
            alpha, ratio, max_distance,
        )
    raise ValueError(
        f"Unknown method {method!r}. Expected one of: "
        "'structural', 'temporal', 'combined', 'procrustes', 'state_action', 'state_structural'."
    )


def match_graphs_ensemble(
    graph_a: BehaviorGraph,
    metadata_a: List[Dict],
    cluster_labels_a: np.ndarray,
    graph_b: BehaviorGraph,
    metadata_b: List[Dict],
    cluster_labels_b: np.ndarray,
    methods: List[str],
    min_agreement: float = 0.5,
    **kwargs,
) -> GraphMatchResult:
    """Run multiple matching methods and return matches with cross-method agreement scores.

    Each candidate pair (node_id_a, node_id_b) accumulates a vote from every method
    that produces it.  ``NodeMatch.confidence`` is the fraction of methods that agreed
    on that pair.  Only pairs with ``confidence >= min_agreement`` are kept.

    When two accepted pairs conflict (same node_id_a or node_id_b appears in more than
    one), the higher-confidence pair wins; ties are broken by lower mean distance.
    ``NodeMatch.distance`` is the mean distance across all agreeing methods.

    Args:
        methods: Non-empty list of method names (e.g. ``["structural", "temporal"]``).
            Passing a single method is valid; every match gets ``confidence=1.0``.
        min_agreement: Minimum fraction of methods that must agree for a pair to be
            accepted.  Default 0.5 = simple majority.
        **kwargs: Forwarded to each :func:`match_graphs` call (ratio, max_distance, etc.).

    Returns:
        GraphMatchResult with ``method="ensemble"`` and ``NodeMatch.confidence`` set.
    """
    if not methods:
        raise ValueError("methods must be non-empty.")

    # Accumulate distances for every (node_id_a, node_id_b) pair across methods.
    pair_distances: Dict[Tuple[int, int], List[float]] = {}
    for method in methods:
        result = match_graphs(
            graph_a, metadata_a, cluster_labels_a,
            graph_b, metadata_b, cluster_labels_b,
            method=method,
            **kwargs,
        )
        for m in result.matches:
            pair = (m.node_id_a, m.node_id_b)
            pair_distances.setdefault(pair, []).append(m.distance)

    n_methods = len(methods)

    # Build candidates: (confidence, mean_distance, nid_a, nid_b)
    candidates = []
    for (nid_a, nid_b), dists in pair_distances.items():
        conf = len(dists) / n_methods
        if conf >= min_agreement:
            candidates.append((conf, float(np.mean(dists)), nid_a, nid_b))

    # Greedy conflict resolution: highest confidence first, then lowest mean distance.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    matched_a: set = set()
    matched_b: set = set()
    matches: List[NodeMatch] = []
    for conf, mean_dist, nid_a, nid_b in candidates:
        if nid_a in matched_a or nid_b in matched_b:
            continue
        matches.append(NodeMatch(nid_a, nid_b, mean_dist, "ensemble", conf))
        matched_a.add(nid_a)
        matched_b.add(nid_b)

    all_a = sorted(graph_a.cluster_nodes)
    all_b = sorted(graph_b.cluster_nodes)
    unmatched_a = [n for n in all_a if n not in matched_a]
    unmatched_b = [n for n in all_b if n not in matched_b]
    return GraphMatchResult(matches=matches, unmatched_a=unmatched_a, unmatched_b=unmatched_b, method="ensemble")


def match_graph_sequence(
    graphs: List[BehaviorGraph],
    metadata_list: List[List[Dict]],
    cluster_labels_list: List[np.ndarray],
    method: str = "combined",
    **kwargs,
) -> List[GraphMatchResult]:
    """Match each consecutive pair in a sequence of graphs.

    Returns N-1 GraphMatchResults for N graphs.
    """
    if len(graphs) != len(metadata_list) or len(graphs) != len(cluster_labels_list):
        raise ValueError("graphs, metadata_list, and cluster_labels_list must have the same length.")
    results = []
    for i in range(len(graphs) - 1):
        result = match_graphs(
            graphs[i], metadata_list[i], cluster_labels_list[i],
            graphs[i + 1], metadata_list[i + 1], cluster_labels_list[i + 1],
            method=method,
            **kwargs,
        )
        results.append(result)
    return results


def build_tracking_chains(
    match_results: List[GraphMatchResult],
    graph_ids: List[str],
) -> List[List[Optional[int]]]:
    """Build per-node identity chains across multiple graph iterations.

    Given N-1 match results for N graphs, returns one chain per unique node
    identity.  Each chain is a list of length N containing the cluster_id of
    that identity at each iteration, or None if the node was absent.

    Args:
        match_results: Output of match_graph_sequence (length N-1).
        graph_ids: Identifier strings for each graph, e.g. ["iter_0", "iter_1", …].

    Returns:
        List of chains, each of length len(graph_ids).
    """
    n_iters = len(graph_ids)
    if n_iters == 0:
        return []
    if len(match_results) != n_iters - 1:
        raise ValueError(
            f"Expected {n_iters - 1} match results for {n_iters} graphs, "
            f"got {len(match_results)}."
        )

    chains: List[List[Optional[int]]] = []
    # maps (iter_idx, node_id) → chain index
    node_to_chain: Dict[Tuple[int, int], int] = {}

    # Seed chains from all nodes present in iter_0
    if match_results:
        iter0_nodes = (
            {m.node_id_a for m in match_results[0].matches}
            | set(match_results[0].unmatched_a)
        )
    else:
        iter0_nodes = set()

    for nid in sorted(iter0_nodes):
        idx = len(chains)
        chains.append([None] * n_iters)
        chains[-1][0] = nid
        node_to_chain[(0, nid)] = idx

    # Propagate matches forward
    for iter_idx, result in enumerate(match_results):
        next_iter = iter_idx + 1

        for match in result.matches:
            chain_idx = node_to_chain.get((iter_idx, match.node_id_a))
            if chain_idx is None:
                # node_id_a wasn't seeded — create a retroactive chain
                chain_idx = len(chains)
                chains.append([None] * n_iters)
                chains[-1][iter_idx] = match.node_id_a
            chains[chain_idx][next_iter] = match.node_id_b
            node_to_chain[(next_iter, match.node_id_b)] = chain_idx

        # Nodes new in graph_b (unmatched from graph_b)
        for nid in result.unmatched_b:
            chain_idx = len(chains)
            chains.append([None] * n_iters)
            chains[-1][next_iter] = nid
            node_to_chain[(next_iter, nid)] = chain_idx

    return chains


def load_rollout_observations(
    hdf5_path,
    obs_keys: Optional[List[str]] = None,
    include_actions: bool = False,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    """Load per-rollout observation arrays from a robomimic-format HDF5 file.

    Supports the standard ``data/demo_{i}/obs/`` layout.  Each demo is indexed
    by its integer position (0, 1, 2, …), matching the ``rollout_idx`` field in
    clustering metadata.

    Args:
        hdf5_path: Path to the rollout HDF5 file.
        obs_keys: List of observation keys to load and concatenate (e.g.
            ``["robot0_eef_pos", "robot0_gripper_qpos", "object"]``).
            If None, all keys under ``obs/`` are concatenated.
        include_actions: If True, also return a dict of action arrays.

    Returns:
        ``(obs_dict, action_dict)`` where obs_dict maps rollout_idx → array
        of shape ``(T, obs_dim)`` and action_dict is None unless include_actions.
    """
    import h5py

    obs_dict: Dict[int, np.ndarray] = {}
    action_dict: Dict[int, np.ndarray] = {} if include_actions else None  # type: ignore[assignment]

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]))
        for ep_idx, demo_key in enumerate(demo_keys):
            demo = data_grp[demo_key]
            obs_grp = demo["obs"]
            keys = obs_keys if obs_keys is not None else sorted(obs_grp.keys())
            arrays = [np.array(obs_grp[k]) for k in keys if k in obs_grp]
            if arrays:
                obs_dict[ep_idx] = np.concatenate(arrays, axis=-1).astype(np.float32)
            if include_actions and "actions" in demo:
                action_dict[ep_idx] = np.array(demo["actions"], dtype=np.float32)

    return obs_dict, action_dict
