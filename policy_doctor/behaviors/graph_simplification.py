"""Graph simplification methods: smoothing, edge pruning, node merging, layout.

Each function is a pure transformation. Two flavors:
  - **Label transformations** take ``(cluster_labels, metadata, ...)`` and return new labels.
  - **Graph transformations** take a ``BehaviorGraph`` (plus labels for re-derivation) and
    return a new graph (and possibly new labels).

See ``docs/graph_simplification_brainstorm.md`` for the method catalogue and rationale.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    FAILURE_NODE_ID,
    END_NODE_ID,
    TERMINAL_NODE_IDS,
    _SPECIAL_GRAPH_NODE_IDS,
    _apply_merge_map_to_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode_groups(
    labels: np.ndarray, metadata: List[Dict], level: str = "rollout"
) -> Dict[int, List[int]]:
    """Return {episode_idx: [sample_indices_in_time_order]}."""
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    eps: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for i, m in enumerate(metadata):
        sort_key = m.get("timestep", m.get("window_start", 0))
        eps[m[ep_key]].append((sort_key, i))
    out: Dict[int, List[int]] = {}
    for ep, vals in eps.items():
        vals.sort()
        out[ep] = [i for _, i in vals]
    return out


def _cluster_centroids(
    labels: np.ndarray, embeddings: np.ndarray
) -> Dict[int, np.ndarray]:
    """Mean embedding per cluster id (ignores noise -1)."""
    out: Dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        out[int(c)] = embeddings[labels == c].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Smoothing (label-level)
# ---------------------------------------------------------------------------

def median_filter_labels(
    labels: np.ndarray,
    metadata: List[Dict],
    window: int = 5,
    level: str = "rollout",
) -> np.ndarray:
    """Per-episode mode filter on the cluster-label sequence.

    Smooths the A→B→A→B flicker that survives run-length collapse. ``window``
    is the full window size (odd is best); for each timestep replace its label
    with the mode of labels in ``[t - w/2, t + w/2]``.
    """
    if window <= 1:
        return labels.copy()
    half = window // 2
    out = labels.copy()
    eps = _episode_groups(labels, metadata, level)
    for ep, idxs in eps.items():
        ep_labels = labels[idxs]
        for j in range(len(ep_labels)):
            lo = max(0, j - half)
            hi = min(len(ep_labels), j + half + 1)
            window_vals = ep_labels[lo:hi]
            window_vals = window_vals[window_vals != -1]
            if len(window_vals) == 0:
                continue
            vals, counts = np.unique(window_vals, return_counts=True)
            out[idxs[j]] = int(vals[np.argmax(counts)])
    return out


def sticky_decoder(
    labels: np.ndarray,
    embeddings: np.ndarray,
    metadata: List[Dict],
    lambda_stick: float = 2.0,
    level: str = "rollout",
) -> np.ndarray:
    """Sticky DP decoder: minimize Σ d(x_t, μ_{z_t}) + λ · 1[z_t ≠ z_{t-1}].

    Re-assigns each window to a cluster using its embedding's distance to each
    cluster centroid, plus a flat penalty ``lambda_stick`` for any transition.
    Higher λ → fewer transitions, longer runs.

    Centroids are computed once from the original ``labels``; the decoder only
    re-orders / re-assigns assignments — it cannot invent new clusters.
    """
    centroids = _cluster_centroids(labels, embeddings)
    if not centroids:
        return labels.copy()
    cluster_ids = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in cluster_ids])  # (K, D)
    K = len(cluster_ids)
    id_to_idx = {c: i for i, c in enumerate(cluster_ids)}

    out = labels.copy()
    eps = _episode_groups(labels, metadata, level)
    for ep, idxs in eps.items():
        X = embeddings[idxs]  # (T, D)
        # Per-step unary cost: squared distance to each centroid
        # cost: (T, K)
        diffs = X[:, None, :] - C[None, :, :]
        unary = np.sum(diffs * diffs, axis=-1)  # (T, K)
        # Normalize so λ is on a comparable scale
        scale = float(unary.std() + 1e-6)
        unary = unary / scale

        T = X.shape[0]
        if T == 0:
            continue
        # DP
        dp = np.empty((T, K), dtype=np.float64)
        bp = np.empty((T, K), dtype=np.int32)
        dp[0] = unary[0]
        bp[0] = -1
        for t in range(1, T):
            # transition: 0 if same, lambda_stick if different
            prev = dp[t - 1]
            # cost to come if same state: prev[k]
            # cost to come if different state: min_{j != k} prev[j] + λ
            best_same = prev  # shape (K,)
            # best other state cost
            sorted_idx = np.argsort(prev)
            best_other = np.full(K, np.inf)
            # For each k, best j != k = sorted_idx[0] if != k else sorted_idx[1]
            first_min = prev[sorted_idx[0]]
            second_min = prev[sorted_idx[1]] if K > 1 else np.inf
            for k in range(K):
                if sorted_idx[0] != k:
                    best_other[k] = first_min + lambda_stick
                else:
                    best_other[k] = second_min + lambda_stick
            choose_same = best_same <= best_other
            min_prev = np.where(choose_same, best_same, best_other)
            argmin_prev = np.where(
                choose_same,
                np.arange(K),
                np.where(sorted_idx[0] != np.arange(K), sorted_idx[0], sorted_idx[1] if K > 1 else 0),
            )
            dp[t] = unary[t] + min_prev
            bp[t] = argmin_prev

        # Backtrack
        path = np.empty(T, dtype=np.int32)
        path[-1] = int(np.argmin(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        for t in range(T):
            out[idxs[t]] = cluster_ids[path[t]]
    return out


def hmm_smooth(
    embeddings: np.ndarray,
    metadata: List[Dict],
    n_states: int,
    level: str = "rollout",
    covariance_type: str = "diag",
    n_iter: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """Gaussian-HMM smoothing: fit on all windows, Viterbi-decode per-episode.

    Returns new labels (0..n_states-1). The HMM jointly estimates state means
    and a transition matrix; the diagonal of that matrix is the learned
    self-persistence bias.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn required: pip install hmmlearn") from e

    eps = _episode_groups(np.zeros(len(metadata), dtype=np.int32), metadata, level)
    # Build episode-grouped sequences for fit (concatenate, with lengths)
    seqs = []
    lengths = []
    for ep, idxs in eps.items():
        seqs.append(embeddings[idxs])
        lengths.append(len(idxs))
    X = np.concatenate(seqs, axis=0)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X, lengths=lengths)

    out = np.full(len(metadata), -1, dtype=np.int64)
    offset = 0
    for ep, idxs in eps.items():
        T = len(idxs)
        seq_X = X[offset : offset + T]
        offset += T
        states = model.predict(seq_X)
        for i, t in enumerate(idxs):
            out[t] = int(states[i])
    return out


# ---------------------------------------------------------------------------
# Graph-level transformations
# ---------------------------------------------------------------------------

def prune_edges_by_count(
    graph: BehaviorGraph, min_count: int = 2
) -> BehaviorGraph:
    """Drop edges with count < min_count; renormalize probabilities."""
    new_counts: Dict[int, Dict[int, int]] = {}
    for src, targets in graph.transition_counts.items():
        kept = {t: c for t, c in targets.items() if c >= min_count}
        if kept:
            new_counts[src] = kept
    new_probs: Dict[int, Dict[int, float]] = {}
    for src, targets in new_counts.items():
        total = sum(targets.values())
        new_probs[src] = {t: c / total for t, c in targets.items()}
    return BehaviorGraph(
        nodes=dict(graph.nodes),
        transition_counts=new_counts,
        transition_probs=new_probs,
        num_episodes=graph.num_episodes,
        level=graph.level,
    )


def prune_edges_by_prob(
    graph: BehaviorGraph, min_prob: float = 0.1
) -> BehaviorGraph:
    """Drop edges with probability < min_prob; renormalize remaining."""
    new_counts: Dict[int, Dict[int, int]] = {}
    for src, targets in graph.transition_counts.items():
        total = sum(targets.values())
        if total == 0:
            continue
        kept = {t: c for t, c in targets.items() if c / total >= min_prob}
        if kept:
            new_counts[src] = kept
    new_probs: Dict[int, Dict[int, float]] = {}
    for src, targets in new_counts.items():
        total = sum(targets.values())
        new_probs[src] = {t: c / total for t, c in targets.items()}
    return BehaviorGraph(
        nodes=dict(graph.nodes),
        transition_counts=new_counts,
        transition_probs=new_probs,
        num_episodes=graph.num_episodes,
        level=graph.level,
    )


def merge_similar_centroids(
    labels: np.ndarray,
    embeddings: np.ndarray,
    metadata: List[Dict],
    sim_threshold: float = 0.9,
    level: str = "rollout",
) -> np.ndarray:
    """Merge clusters whose centroids are cosine-similar above ``sim_threshold``.

    Iterates: at each step find the most-similar pair, merge (keep smaller id),
    until no pair exceeds threshold. Returns relabeled array.
    """
    cur = labels.copy()
    while True:
        centroids = _cluster_centroids(cur, embeddings)
        if len(centroids) < 2:
            return cur
        ids = sorted(centroids.keys())
        C = np.stack([centroids[c] for c in ids])
        norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        Cn = C / norms
        sims = Cn @ Cn.T
        np.fill_diagonal(sims, -np.inf)
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        max_sim = sims[i, j]
        if max_sim < sim_threshold:
            return cur
        # Merge j into i (smaller id wins)
        a, b = sorted([ids[i], ids[j]])
        cur = np.where(cur == b, a, cur)


def stable_phase_prune(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: List[Dict],
) -> Tuple[BehaviorGraph, np.ndarray]:
    """ENAP-style stable-phase pruning.

    Merge node ``q'`` into ``q`` when:
      1. ``q`` has a self-loop (sees itself in the cluster sequence — actually,
         in run-length-collapsed graphs there are no self-loops; we use a
         stricter definition: ``q'`` is reached *only* from ``q`` and has very
         few outgoing destinations).
      2. The (q → q') transition is much more frequent than (q' → anywhere
         else), i.e., q' is "downstream" of q.

    Concretely: merge q' into q when (count(q→q') / sum(counts out of q'))
    > 0.5 AND q' has a single dominant incoming source q.
    """
    counts = {s: dict(t) for s, t in graph.transition_counts.items()}
    # Build incoming view
    incoming: Dict[int, Dict[int, int]] = defaultdict(dict)
    for s, targets in counts.items():
        for t, c in targets.items():
            incoming[t][s] = c

    merge_map: Dict[int, int] = {}
    for q_prime in list(graph.nodes.keys()):
        if q_prime in _SPECIAL_GRAPH_NODE_IDS or q_prime in TERMINAL_NODE_IDS:
            continue
        inc = incoming.get(q_prime, {})
        if not inc:
            continue
        # Identify dominant incoming source
        q, qc = max(inc.items(), key=lambda kv: kv[1])
        if q in _SPECIAL_GRAPH_NODE_IDS or q in TERMINAL_NODE_IDS:
            continue
        # Mass on dominant incoming edge
        if qc / max(1, sum(inc.values())) < 0.6:
            continue
        # Outgoing from q': mostly back into the system or terminal?
        out_qprime = counts.get(q_prime, {})
        total_out = sum(out_qprime.values())
        if total_out == 0:
            merge_map[q_prime] = q
            continue
        # If most of q's outgoing mass to q' is much larger than q's other
        # outgoing flows, consider q' a "stable continuation" of q
        out_q = counts.get(q, {})
        share_to_qprime = qc / max(1, sum(out_q.values()))
        if share_to_qprime > 0.3:
            merge_map[q_prime] = q

    if not merge_map:
        return graph, labels

    new_labels = _apply_merge_map_to_labels(labels, merge_map)
    new_graph = BehaviorGraph.from_cluster_assignments(
        new_labels, metadata, level=graph.level
    )
    return new_graph, new_labels


# ---------------------------------------------------------------------------
# Alternative clusterings
# ---------------------------------------------------------------------------

def auto_k_kmeans(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (4, 15),
    random_state: int = 42,
) -> Tuple[np.ndarray, int, Dict[int, float]]:
    """Sweep K, return labels for the K with the best silhouette score.

    Returns ``(labels, best_k, scores_by_k)``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: Dict[int, float] = {}
    best_k = k_range[0]
    best_score = -np.inf
    best_labels: Optional[np.ndarray] = None
    # Subsample for silhouette (it's O(n^2))
    n = embeddings.shape[0]
    rng = np.random.default_rng(random_state)
    sub = rng.choice(n, size=min(3000, n), replace=False)
    X_sub = embeddings[sub]
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(embeddings).astype(np.int64)
        if k == 1:
            scores[k] = 0.0
            continue
        sub_labels = labels[sub]
        if len(np.unique(sub_labels)) < 2:
            scores[k] = 0.0
            continue
        sc = float(silhouette_score(X_sub, sub_labels))
        scores[k] = sc
        if sc > best_score:
            best_score = sc
            best_k = k
            best_labels = labels
    if best_labels is None:
        # fallback
        km = KMeans(n_clusters=k_range[0], n_init=10, random_state=random_state)
        best_labels = km.fit_predict(embeddings).astype(np.int64)
    return best_labels, best_k, scores


def spectral_transition_clustering(
    labels: np.ndarray,
    metadata: List[Dict],
    n_macro: int = 8,
    level: str = "rollout",
    random_state: int = 42,
) -> np.ndarray:
    """Build the micro-transition matrix, spectral-cluster it into macro-clusters.

    Idea: even if individual clusters are noisy, the *transition affinity* between
    them carries the true behavioral grouping — clusters that flow into each
    other should be one macro-mode.
    """
    from sklearn.cluster import SpectralClustering

    # Build symmetric affinity = transition counts + transpose
    micro_ids = sorted(set(int(c) for c in labels if c != -1))
    id_to_idx = {c: i for i, c in enumerate(micro_ids)}
    K = len(micro_ids)
    if K <= n_macro:
        # Nothing to do; identity mapping
        return labels.copy()

    M = np.zeros((K, K), dtype=np.float64)
    eps = _episode_groups(labels, metadata, level)
    for ep, idxs in eps.items():
        seq = labels[idxs]
        # collapse run-length
        if len(seq) == 0:
            continue
        collapsed = [int(seq[0])]
        for v in seq[1:]:
            if v == -1:
                continue
            if int(v) != collapsed[-1]:
                collapsed.append(int(v))
        for a, b in zip(collapsed[:-1], collapsed[1:]):
            if a in id_to_idx and b in id_to_idx:
                M[id_to_idx[a], id_to_idx[b]] += 1.0
                M[id_to_idx[b], id_to_idx[a]] += 1.0  # symmetric for spectral
    # Add small self affinity so isolated clusters don't break Laplacian
    np.fill_diagonal(M, M.diagonal() + 1.0)
    # Spectral clustering on the affinity
    sc = SpectralClustering(
        n_clusters=n_macro,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    macro = sc.fit_predict(M)
    micro_to_macro = {micro_ids[i]: int(macro[i]) for i in range(K)}

    out = labels.copy()
    for i, c in enumerate(labels):
        if int(c) in micro_to_macro:
            out[i] = micro_to_macro[int(c)]
    return out


def change_point_segmentation(
    embeddings: np.ndarray,
    metadata: List[Dict],
    n_macro: int,
    penalty: float = 10.0,
    level: str = "rollout",
    random_state: int = 42,
) -> np.ndarray:
    """Detect change-points per episode, then cluster the resulting segments.

    For each episode: run PELT (ruptures library) to find change-points; each
    inter-changepoint segment gets a single label assigned by KMeans on the
    segment's mean embedding.
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("ruptures required: pip install ruptures") from e
    from sklearn.cluster import KMeans

    eps = _episode_groups(np.zeros(len(metadata), dtype=np.int32), metadata, level)
    # Step 1: detect change-points, collect segment means
    segments: List[Tuple[List[int], np.ndarray]] = []  # (indices_in_segment, mean_emb)
    for ep, idxs in eps.items():
        X = embeddings[idxs]
        if len(X) < 4:
            segments.append((idxs, X.mean(axis=0)))
            continue
        algo = rpt.Pelt(model="l2").fit(X)
        bkps = algo.predict(pen=penalty)
        bkps = [0] + bkps  # include start
        for i in range(len(bkps) - 1):
            lo, hi = bkps[i], bkps[i + 1]
            seg_idx = idxs[lo:hi]
            if not seg_idx:
                continue
            segments.append((seg_idx, X[lo:hi].mean(axis=0)))
    # Step 2: KMeans on segment means
    seg_means = np.stack([s[1] for s in segments])
    km = KMeans(n_clusters=n_macro, n_init=10, random_state=random_state)
    seg_labels = km.fit_predict(seg_means).astype(np.int64)

    out = np.full(len(metadata), -1, dtype=np.int64)
    for (seg_idx, _), lab in zip(segments, seg_labels):
        for i in seg_idx:
            out[i] = int(lab)
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def temporal_layout(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    x_min: float = -2.5,
    x_max: float = 2.5,
) -> Dict[int, Tuple[float, float]]:
    """Layout where each cluster node's x = median fraction-of-episode-length.

    START is pinned at x_min, terminals at x_max. Vertical positions are
    spread to avoid overlaps within the same x-band. Timesteps are normalized
    to [0, 1] per episode so episodes of different lengths contribute on the
    same scale.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    # First pass: episode-length per episode
    ep_max_t: Dict[int, float] = defaultdict(float)
    for m in metadata:
        ep = m[ep_key]
        t = float(m.get("timestep", m.get("window_start", 0)))
        if t > ep_max_t[ep]:
            ep_max_t[ep] = t

    cluster_timesteps: Dict[int, List[float]] = defaultdict(list)
    for i, m in enumerate(metadata):
        c = int(labels[i])
        if c == -1:
            continue
        t = float(m.get("timestep", m.get("window_start", 0)))
        ep = m[ep_key]
        ep_len = max(1.0, ep_max_t[ep])
        cluster_timesteps[c].append(t / ep_len)  # fraction 0..1

    # Compute per-cluster median fraction; use rank-based x to guarantee
    # even spread along the horizontal axis (raw medians often cluster around
    # mid-trajectory which produces visual overlap).
    cluster_median: Dict[int, float] = {
        c: float(np.median(ts)) for c, ts in cluster_timesteps.items()
    }
    if not cluster_median:
        return {}
    # Sort clusters by median; assign rank 0..K-1.
    sorted_clusters = sorted(cluster_median.keys(), key=lambda c: cluster_median[c])
    K = len(sorted_clusters)
    cluster_rank: Dict[int, int] = {c: i for i, c in enumerate(sorted_clusters)}

    pos: Dict[int, Tuple[float, float]] = {}
    inner_min = x_min + (x_max - x_min) * 0.1
    inner_max = x_min + (x_max - x_min) * 0.9
    inner_range = inner_max - inner_min

    for c in graph.nodes:
        if c == START_NODE_ID:
            pos[c] = (x_min, 0.0)
        elif c in TERMINAL_NODE_IDS:
            pos[c] = (x_max, 0.0)
        elif c in cluster_rank:
            # Blend rank-position (even) with raw median (preserves "tied" clusters)
            rank_x = inner_min + inner_range * (cluster_rank[c] / max(1, K - 1))
            pos[c] = (rank_x, 0.0)
        else:
            # No data → place mid
            pos[c] = ((x_min + x_max) / 2, 0.0)

    # Resolve y by binning nodes into x-bands and spreading vertically.
    # Tight x-tolerance so we only spread truly-overlapping nodes.
    x_tolerance = (x_max - x_min) * 0.03
    # Sort by x, then by id within
    nodes_by_x = sorted(
        [(nid, x, y) for nid, (x, y) in pos.items() if nid not in TERMINAL_NODE_IDS and nid != START_NODE_ID],
        key=lambda v: (v[1], v[0]),
    )
    bins: List[List[int]] = []
    last_x: Optional[float] = None
    for nid, x, _ in nodes_by_x:
        if last_x is None or abs(x - last_x) > x_tolerance:
            bins.append([nid])
            last_x = x
        else:
            bins[-1].append(nid)
    # Spread y within each bin; spacing scales with bin size
    for bin_nodes in bins:
        n = len(bin_nodes)
        if n == 1:
            continue
        bin_nodes = sorted(bin_nodes, key=lambda nid: nid)
        y_spacing = 0.7 if n <= 4 else 0.5
        for i, nid in enumerate(bin_nodes):
            y = (i - (n - 1) / 2.0) * y_spacing
            pos[nid] = (pos[nid][0], y)

    # Pin success ABOVE failure if both exist
    if SUCCESS_NODE_ID in pos and FAILURE_NODE_ID in pos:
        pos[SUCCESS_NODE_ID] = (pos[SUCCESS_NODE_ID][0], 1.0)
        pos[FAILURE_NODE_ID] = (pos[FAILURE_NODE_ID][0], -1.0)
    if END_NODE_ID in pos and END_NODE_ID != SUCCESS_NODE_ID:
        pos[END_NODE_ID] = (pos[END_NODE_ID][0], 0.0)

    return pos


def sugiyama_layout(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    x_min: float = -2.5,
    x_max: float = 2.5,
    y_scale: float = 1.2,
) -> Dict[int, Tuple[float, float]]:
    """Layered (Sugiyama) layout using NetworkX multipartite_layout.

    Builds a *layered DAG* where each cluster node's layer = its rank by
    median fraction-of-episode-length. Edges within and across layers are
    routed by NetworkX; the resulting y-positions minimise visual crossings
    while keeping x meaningful (= temporal rank).

    START is pinned left, terminals right.
    """
    import networkx as nx

    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    ep_max_t: Dict[int, float] = defaultdict(float)
    for m in metadata:
        ep = m[ep_key]
        t = float(m.get("timestep", m.get("window_start", 0)))
        if t > ep_max_t[ep]:
            ep_max_t[ep] = t

    cluster_med: Dict[int, float] = {}
    by_cluster: Dict[int, List[float]] = defaultdict(list)
    for i, m in enumerate(metadata):
        c = int(labels[i])
        if c == -1:
            continue
        t = float(m.get("timestep", m.get("window_start", 0)))
        ep_len = max(1.0, ep_max_t[m[ep_key]])
        by_cluster[c].append(t / ep_len)
    for c, ts in by_cluster.items():
        cluster_med[c] = float(np.median(ts))

    sorted_clusters = sorted(cluster_med, key=lambda c: cluster_med[c])
    rank = {c: i + 1 for i, c in enumerate(sorted_clusters)}
    K = len(sorted_clusters)

    G = nx.DiGraph()
    layer_of: Dict[int, int] = {}
    for nid in graph.nodes:
        if nid == START_NODE_ID:
            layer_of[nid] = 0
        elif nid in TERMINAL_NODE_IDS:
            layer_of[nid] = K + 1
        else:
            layer_of[nid] = rank.get(nid, K // 2 + 1)
        G.add_node(nid, layer=layer_of[nid])
    for src, targets in graph.transition_probs.items():
        for tgt, p in targets.items():
            if src in G and tgt in G and p > 0:
                G.add_edge(src, tgt, weight=p)

    raw = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    # multipartite_layout returns x in [-0.5, 0.5]-ish — we re-scale.
    xs = [raw[n][0] for n in raw]
    ys = [raw[n][1] for n in raw]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    span_x = max(1e-6, x_hi - x_lo)
    span_y = max(1e-6, y_hi - y_lo)
    pos: Dict[int, Tuple[float, float]] = {}
    for n in raw:
        rx, ry = raw[n]
        nx_norm = (rx - x_lo) / span_x
        ny_norm = (ry - y_lo) / span_y - 0.5
        pos[n] = (x_min + nx_norm * (x_max - x_min),
                  ny_norm * y_scale * 4)

    # Force terminals
    if SUCCESS_NODE_ID in pos and FAILURE_NODE_ID in pos:
        pos[SUCCESS_NODE_ID] = (x_max, max(2.0, pos[SUCCESS_NODE_ID][1]))
        pos[FAILURE_NODE_ID] = (x_max, min(-2.0, pos[FAILURE_NODE_ID][1]))
    return pos


def force_directed_x_pinned_layout(
    graph: BehaviorGraph,
    labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    x_min: float = -2.5,
    x_max: float = 2.5,
    iterations: int = 200,
    seed: int = 42,
) -> Dict[int, Tuple[float, float]]:
    """Spring layout with x fixed to temporal rank.

    Each node's x is set by its median timestep rank; spring_layout then
    only adjusts y to minimise edge length. The result preserves the
    temporal axis while letting y-positions drift to avoid edges passing
    through nodes.
    """
    import networkx as nx

    base = temporal_layout(
        graph, labels, metadata, level=level, x_min=x_min, x_max=x_max,
    )
    # Build a NetworkX graph with weights so spring layout treats high-
    # probability edges as stronger springs.
    G = nx.DiGraph()
    for nid in graph.nodes:
        G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        for tgt, p in targets.items():
            if src in G and tgt in G and p > 0:
                G.add_edge(src, tgt, weight=float(p))

    # spring_layout treats fixed positions in raw (x, y); we want only x
    # fixed. Workaround: run unconstrained spring, then override x.
    initial = {nid: np.array([x, np.random.default_rng(seed + nid).uniform(-0.5, 0.5)])
               for nid, (x, _) in base.items()}
    new_pos = nx.spring_layout(
        G.to_undirected(),
        pos=initial,
        iterations=iterations,
        seed=seed,
        k=0.6,
    )
    # Rescale y to a readable range, restore x from base
    ys = [v[1] for v in new_pos.values()]
    y_lo, y_hi = min(ys), max(ys)
    span = max(1e-6, y_hi - y_lo)
    out: Dict[int, Tuple[float, float]] = {}
    for nid, (x_base, _) in base.items():
        y_raw = new_pos[nid][1]
        y_norm = (y_raw - y_lo) / span - 0.5  # in [-0.5, 0.5]
        out[nid] = (x_base, y_norm * 5.0)
    # Force terminals
    if SUCCESS_NODE_ID in out and FAILURE_NODE_ID in out:
        out[SUCCESS_NODE_ID] = (x_max, 2.0)
        out[FAILURE_NODE_ID] = (x_max, -2.0)
    if START_NODE_ID in out:
        out[START_NODE_ID] = (x_min, 0.0)
    return out


def build_trajectory_tree(
    labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
) -> List[Dict]:
    """Build a prefix-tree (trie) of run-length-collapsed cluster sequences.

    Each tree node is identified by its full path from the root (a tuple of
    cluster IDs). Stats:
      - n_episodes: how many episodes pass through this prefix
      - n_success / n_failure: outcome breakdown
      - parent_path: the prefix-1 path (so we can render it as a tree)

    Returns a flat list of nodes; the root has parent_path=None.

    The path "(c1, c2, c3, SUCCESS)" means "an episode that visited clusters
    c1 → c2 → c3 and succeeded."
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    eps: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    success: Dict[int, Optional[bool]] = {}
    for i, m in enumerate(metadata):
        sort_key = m.get("timestep", m.get("window_start", 0))
        eps[m[ep_key]].append((sort_key, int(labels[i])))
        if "success" in m and m[ep_key] not in success:
            success[m[ep_key]] = m["success"]

    # Per-episode run-length-collapsed sequence, then append terminal token
    sequences: List[Tuple[int, Tuple[int, ...], Optional[bool]]] = []  # (ep_id, seq, success)
    for ep, pairs in eps.items():
        pairs.sort()
        seq: List[int] = []
        for _, c in pairs:
            if c == -1:
                continue
            if not seq or seq[-1] != c:
                seq.append(c)
        # Terminal token
        if success.get(ep) is True:
            seq.append(SUCCESS_NODE_ID)
        elif success.get(ep) is False:
            seq.append(FAILURE_NODE_ID)
        else:
            seq.append(END_NODE_ID)
        sequences.append((int(ep), tuple(seq), success.get(ep)))

    # Build trie. node_counts[path] = (n_episodes, n_success, n_failure).
    # Also track which episode indices pass through each prefix so node
    # panels can list videos for the episodes that took that path.
    counts: Dict[Tuple[int, ...], List[int]] = defaultdict(lambda: [0, 0, 0])
    ep_through: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for ep_id, seq, succ in sequences:
        for depth in range(1, len(seq) + 1):
            prefix = seq[:depth]
            counts[prefix][0] += 1
            ep_through[prefix].append(ep_id)
            if succ is True:
                counts[prefix][1] += 1
            elif succ is False:
                counts[prefix][2] += 1

    # Root node (empty path)
    nodes: List[Dict] = [{
        "path": (),
        "label": "START",
        "cluster_id": START_NODE_ID,
        "depth": 0,
        "n_episodes": len(sequences),
        "n_success": sum(1 for _, _, s in sequences if s is True),
        "n_failure": sum(1 for _, _, s in sequences if s is False),
        "parent_path": None,
        "episode_indices": [ep_id for ep_id, _, _ in sequences],
    }]
    for path, (n_ep, n_succ, n_fail) in counts.items():
        cid = path[-1]
        nodes.append({
            "path": path,
            "label": f"Behavior {cid}" if cid >= 0 else (
                "SUCCESS" if cid == SUCCESS_NODE_ID else
                "FAILURE" if cid == FAILURE_NODE_ID else "END"
            ),
            "cluster_id": cid,
            "depth": len(path),
            "n_episodes": n_ep,
            "n_success": n_succ,
            "n_failure": n_fail,
            "parent_path": path[:-1],
            "episode_indices": list(ep_through[path]),
        })
    return nodes
