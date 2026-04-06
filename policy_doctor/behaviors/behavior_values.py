"""Behavior graph and slice value helpers (no Streamlit)."""

from typing import Dict, List, Tuple, Any

import numpy as np

from policy_doctor.data.structures import EpisodeInfo
from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    FAILURE_NODE_ID,
    SUCCESS_NODE_ID,
    get_rollout_slices_for_paths,
)


def build_behavior_graph_from_clustering(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
) -> BehaviorGraph:
    """Build the transition graph from cluster assignments (no rewards or values)."""
    level = "rollout" if (metadata and "rollout_idx" in metadata[0]) else "demo"
    return BehaviorGraph.from_cluster_assignments(
        cluster_labels, metadata, level=level
    )


def compute_mrp_slice_values(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    gamma: float = 0.99,
    reward_success: float = 1.0,
    reward_failure: float = -1.0,
    reward_end: float = 0.0,
) -> Tuple[Dict[int, float], np.ndarray, np.ndarray]:
    """Bellman V(s) on the graph plus per-slice transition value and advantage (uses existing graph)."""
    values = graph.compute_values(
        gamma=gamma,
        reward_success=reward_success,
        reward_failure=reward_failure,
        reward_end=reward_end,
    )
    q_values, advantages, _ = graph.compute_slice_values(
        cluster_labels, metadata, values, gamma=gamma
    )
    return values, q_values, advantages


def get_behavior_graph_and_slice_values(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    gamma: float = 0.99,
    reward_success: float = 1.0,
    reward_failure: float = -1.0,
    reward_end: float = 0.0,
) -> Tuple[BehaviorGraph, Dict[int, float], np.ndarray, np.ndarray]:
    """Build behavior graph, compute V(s), then per-slice transition values and advantage."""
    graph = build_behavior_graph_from_clustering(cluster_labels, metadata)
    values, q_values, advantages = compute_mrp_slice_values(
        graph,
        cluster_labels,
        metadata,
        gamma=gamma,
        reward_success=reward_success,
        reward_failure=reward_failure,
        reward_end=reward_end,
    )
    return graph, values, q_values, advantages


def slice_indices_to_rollout_slices(
    metadata: List[Dict],
    rollout_episodes: List[EpisodeInfo],
    cluster_labels: np.ndarray,
    indices: np.ndarray,
) -> List[Dict[str, Any]]:
    """Map slice indices to rollout slice dicts (rollout_idx, rollout_ep, start, end)."""
    ep_by_idx = {ep.index: ep for ep in rollout_episodes}
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
            end = start + meta.get("window_width", 1) - 1
        else:
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
