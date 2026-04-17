"""Helpers to select MimicGen seed trajectories from a BehaviorGraph.

Given cluster labels + per-sample metadata from the clustering pipeline, this
module reconstructs the collapsed per-episode behavior sequences and finds
rollout episodes whose sequence exactly matches a given behavior path from the
graph.

The main entry points are:

* :func:`reconstruct_episode_sequences` — rebuild per-episode collapsed label
  sequences from the flat ``cluster_labels`` + ``metadata`` arrays.
* :func:`path_cluster_sequence` — strip START / terminal node IDs from a
  behavior-graph path, leaving only real cluster IDs.
* :func:`find_rollouts_for_path` — return rollout indices whose collapsed
  sequence exactly matches a path's cluster sequence.
* :func:`top_paths_with_rollouts` — combine the two to rank paths by
  probability and match each against the available rollouts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
)

_SPECIAL_IDS: frozenset[int] = frozenset({START_NODE_ID}) | TERMINAL_NODE_IDS


# ---------------------------------------------------------------------------
# Sequence reconstruction
# ---------------------------------------------------------------------------


def reconstruct_episode_sequences(
    cluster_labels: np.ndarray,
    metadata: list[dict[str, Any]],
    level: str = "rollout",
) -> dict[int, list[int]]:
    """Reconstruct the collapsed cluster-label sequence for every episode.

    Mirrors the logic inside :meth:`BehaviorGraph.from_cluster_assignments` so
    results are consistent with what the graph was built from.

    Args:
        cluster_labels: ``(N,)`` int array; ``-1`` marks noise samples.
        metadata:       List of per-sample dicts.  Each dict must contain
                        ``"rollout_idx"`` (or ``"demo_idx"`` when
                        ``level="demo"``), plus ``"window_start"`` or
                        ``"timestep"`` for ordering.
        level:          ``"rollout"`` (default) or ``"demo"``.

    Returns:
        ``{episode_idx: [c0, c1, ...]}`` — collapsed (run-length-encoded)
        cluster ID sequence per episode, noise samples excluded.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    raw: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        label = int(cluster_labels[i])
        if label == -1:
            continue
        ep_idx = meta[ep_key]
        sort_key = meta.get("window_start", meta.get("timestep", 0))
        raw[ep_idx].append((sort_key, label))

    collapsed: dict[int, list[int]] = {}
    for ep_idx, seq in raw.items():
        seq.sort(key=lambda x: x[0])
        result = [seq[0][1]]
        for _, label in seq[1:]:
            if label != result[-1]:
                result.append(label)
        collapsed[ep_idx] = result
    return collapsed


def episode_success_map(
    metadata: list[dict[str, Any]],
    level: str = "rollout",
) -> dict[int, bool | None]:
    """Return ``{episode_idx: success}`` extracted from metadata.

    Only the first occurrence of each episode index is used.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    result: dict[int, bool | None] = {}
    for meta in metadata:
        ep_idx = meta[ep_key]
        if ep_idx not in result:
            result[ep_idx] = meta.get("success")
    return result


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def path_cluster_sequence(path: list[int]) -> list[int]:
    """Strip START and terminal node IDs, returning only real cluster IDs."""
    return [n for n in path if n not in _SPECIAL_IDS]


def find_rollouts_for_path(
    path: list[int],
    episode_sequences: dict[int, list[int]],
    success_only: bool = True,
    success_map: dict[int, bool | None] | None = None,
) -> list[int]:
    """Return episode indices whose collapsed sequence exactly matches *path*.

    Args:
        path:              Full behavior-graph path (includes START / terminal).
        episode_sequences: ``{ep_idx: [c0, c1, ...]}`` from
                           :func:`reconstruct_episode_sequences`.
        success_only:      If ``True`` (default), only return episodes that
                           are marked successful.
        success_map:       ``{ep_idx: bool|None}`` from
                           :func:`episode_success_map`.  Required when
                           ``success_only=True``.

    Returns:
        Sorted list of matching episode indices.
    """
    target = path_cluster_sequence(path)
    matches = []
    for ep_idx, seq in episode_sequences.items():
        if seq != target:
            continue
        if success_only and success_map is not None:
            if not success_map.get(ep_idx):
                continue
        matches.append(ep_idx)
    return sorted(matches)


# ---------------------------------------------------------------------------
# Top-level helper: rank paths and find rollouts
# ---------------------------------------------------------------------------


def top_paths_with_rollouts(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    top_k: int = 5,
    min_path_probability: float = 0.0,
    min_edge_probability: float = 0.0,
    success_only: bool = True,
    level: str = "rollout",
) -> list[dict[str, Any]]:
    """Find the top-k success paths and, for each, the matching rollouts.

    Returns a list of dicts (one per path, sorted by descending probability):
    ::

        {
            "path":         [-2, 3, 1, 5, -4],
            "path_prob":    0.32,
            "cluster_seq":  [3, 1, 5],
            "rollout_idxs": [12, 47, 83],   # may be empty
            "has_match":    True,
        }
    """
    ep_seqs = reconstruct_episode_sequences(cluster_labels, metadata, level=level)
    s_map = episode_success_map(metadata, level=level) if success_only else None

    paths = graph.enumerate_paths_to_terminal(
        SUCCESS_NODE_ID,
        max_paths=top_k * 4,  # fetch extra so we have top_k after filtering
        min_probability=min_path_probability,
        min_edge_probability=min_edge_probability,
    )

    results = []
    for path, prob, _loops in paths:
        if len(results) >= top_k:
            break
        cluster_seq = path_cluster_sequence(path)
        rollout_idxs = find_rollouts_for_path(
            path, ep_seqs, success_only=success_only, success_map=s_map
        )
        results.append(
            {
                "path": path,
                "path_prob": prob,
                "cluster_seq": cluster_seq,
                "rollout_idxs": rollout_idxs,
                "has_match": len(rollout_idxs) > 0,
            }
        )
    return results
