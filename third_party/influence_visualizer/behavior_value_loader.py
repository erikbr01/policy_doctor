"""Pure helpers for behavior graph and slice value computation (no Streamlit).

Callers load cluster_labels and metadata from disk (e.g. clustering_results.load_clustering_result)
and pass them in. Used by the Learning tab advantage-based curation and optionally by the
Behavior Graph tab.
"""

from typing import Dict, List, Tuple

import numpy as np

from influence_visualizer.behavior_graph import (
    BehaviorGraph,
    FAILURE_NODE_ID,
    SUCCESS_NODE_ID,
    get_rollout_slices_for_paths,
)
from influence_visualizer.data_loader import InfluenceData


def get_behavior_graph_and_slice_values(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    gamma: float = 0.99,
    reward_success: float = 1.0,
    reward_failure: float = -1.0,
    reward_end: float = 0.0,
) -> Tuple[BehaviorGraph, Dict[int, float], np.ndarray, np.ndarray]:
    """Build behavior graph, compute V(s), then per-slice Q and advantage.

    Level is inferred from metadata: "rollout" if "rollout_idx" in first entry, else "demo".

    Returns:
        (graph, node_values, q_values, advantages)
        - node_values: dict node_id -> V(s)
        - q_values, advantages: arrays of shape (len(cluster_labels),)
    """
    level = "rollout" if (metadata and "rollout_idx" in metadata[0]) else "demo"
    graph = BehaviorGraph.from_cluster_assignments(
        cluster_labels, metadata, level=level
    )
    values = graph.compute_values(
        gamma=gamma,
        reward_success=reward_success,
        reward_failure=reward_failure,
        reward_end=reward_end,
    )
    q_values, advantages, _ = graph.compute_slice_values(
        cluster_labels, metadata, values, gamma=gamma
    )
    return graph, values, q_values, advantages


def slice_indices_to_rollout_slices(
    metadata: List[Dict],
    data: InfluenceData,
    cluster_labels: np.ndarray,
    indices: np.ndarray,
) -> List[Dict]:
    """Map slice indices (e.g. from advantage < threshold) to rollout slice dicts.

    Each returned dict has keys: rollout_idx, rollout_ep, start, end, suitable
    for _run_slice_search. Skips indices where cluster_labels[i] == -1 (noise).

    Uses data.rollout_episodes to resolve rollout_idx to the episode object.
    """
    rollout_episodes = getattr(data, "rollout_episodes", None)
    if not rollout_episodes:
        return []

    ep_by_idx = {ep.index: ep for ep in rollout_episodes}
    # Deduplicate by (rollout_idx, start, end): multiple timesteps within the same
    # cluster segment all share identical metadata, so without this each N-timestep
    # segment would produce N identical rollout slice entries causing N-fold redundant
    # influence searches and a bloated curation config.
    seen: set = set()
    result = []
    for i in np.atleast_1d(indices):
        i = int(i)
        if i < 0 or i >= len(metadata) or cluster_labels[i] == -1:
            continue
        meta = metadata[i]
        rollout_idx = meta.get("rollout_idx")
        if rollout_idx is None:
            continue
        start = meta.get("window_start", 0)
        end = meta.get("window_end")
        if end is None:
            # Fallback: compute inclusive end from window_width
            end = start + meta.get("window_width", 1) - 1
        else:
            # window_end is stored as exclusive (Python slice convention:
            # episode_data[start:end] in clustering). Convert to inclusive so
            # _run_slice_search_one()'s "+ 1" produces the correct exclusive
            # bound and the influence query covers exactly the same rows as
            # the clustering window that generated this sample.
            end = end - 1
        ep = ep_by_idx.get(rollout_idx)
        if ep is None:
            continue
        key = (rollout_idx, start, end)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "rollout_idx": rollout_idx,
            "rollout_ep": ep,
            "start": start,
            "end": end,
        })
    return result


def get_path_based_rollout_slices(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    data: InfluenceData,
    terminal_id: int,
    top_k_paths: int = 10,
    min_path_probability: float = 0.0,
    min_edge_probability: float = 0.0,
) -> List[Dict]:
    """Get rollout slice dicts for path-based curation (filtering or selection).

    Hierarchy: rollout → segments (same-label runs) → slices (fixed-length from clustering) → samples.
    Paths to terminal_id are enumerated and top top_k_paths kept. We return the original
    rollout slices that compose the segments on those paths—one (rollout_idx, start, end)
    per clustering slice (fixed-length window), so attribution runs per slice. Format
    matches slice_indices_to_rollout_slices (rollout_idx, rollout_ep, start, end) for
    _run_slice_search.

    Args:
        graph: Behavior graph built from rollout-level cluster assignments.
        cluster_labels: Cluster labels array (same as used to build graph).
        metadata: Per-sample metadata (same as used to build graph).
        data: InfluenceData for resolving rollout_ep from rollout_idx.
        terminal_id: SUCCESS_NODE_ID for selection (paths to success), FAILURE_NODE_ID for filtering.
        top_k_paths: Number of top-probability paths to include.
        min_path_probability: Minimum path probability threshold.
        min_edge_probability: Minimum edge probability during path enumeration.

    Returns:
        List of rollout slice dicts with keys rollout_idx, rollout_ep, start, end.
    """
    if terminal_id not in (SUCCESS_NODE_ID, FAILURE_NODE_ID):
        return []
    paths_with_probs = graph.enumerate_paths_to_terminal(
        terminal_id=terminal_id,
        max_paths=max(top_k_paths, 100),
        min_probability=min_path_probability,
        min_edge_probability=min_edge_probability,
    )
    top_paths = [path for path, _prob, _loops in paths_with_probs[:top_k_paths]]
    if not top_paths:
        return []

    level = "rollout"
    slice_tuples = get_rollout_slices_for_paths(
        cluster_labels=cluster_labels,
        metadata=metadata,
        level=level,
        paths=top_paths,
    )

    rollout_episodes = getattr(data, "rollout_episodes", None)
    if not rollout_episodes:
        return []
    ep_by_idx = {ep.index: ep for ep in rollout_episodes}

    result = []
    seen = set()
    for (rollout_idx, start, end) in slice_tuples:
        if (rollout_idx, start, end) in seen:
            continue
        seen.add((rollout_idx, start, end))
        ep = ep_by_idx.get(rollout_idx)
        if ep is None:
            continue
        result.append({
            "rollout_idx": rollout_idx,
            "rollout_ep": ep,
            "start": start,
            "end": end,
        })
    return result
