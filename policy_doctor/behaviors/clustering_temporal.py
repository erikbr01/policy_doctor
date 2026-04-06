"""Build per-episode cluster timelines from clustering labels + slice metadata (no Streamlit)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from policy_doctor.data.structures import EpisodeInfo


def build_episode_cluster_map(
    cluster_labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    representation: str,
    level: str,
) -> Dict[int, List[Dict[str, Any]]]:
    """Map episode id -> list of window or timestep cluster entries (same logic as influence_visualizer)."""
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    episode_cluster_map: Dict[int, List[Dict[str, Any]]] = {}

    if representation == "sliding_window":
        for i, meta in enumerate(metadata):
            ep_idx = int(meta[ep_key])
            cluster_id = int(cluster_labels[i])
            window_start = int(meta["window_start"])
            window_end = int(meta["window_end"])
            episode_cluster_map.setdefault(ep_idx, []).append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "cluster": cluster_id,
                }
            )
    else:
        for i, meta in enumerate(metadata):
            ep_idx = int(meta[ep_key])
            cluster_id = int(cluster_labels[i])
            timestep = int(meta["timestep"])
            episode_cluster_map.setdefault(ep_idx, []).append(
                {"timestep": timestep, "cluster": cluster_id}
            )

    return episode_cluster_map


def build_cluster_timeline(
    num_frames: int,
    ep_id: int,
    episode_cluster_map: Dict[int, List[Dict[str, Any]]],
    representation: str,
) -> List[int]:
    """Per-frame cluster ids; -1 = unassigned / not in clustering sample."""
    cluster_timeline = [-1] * num_frames
    entries = episode_cluster_map.get(ep_id)
    if not entries:
        return cluster_timeline

    if representation == "sliding_window":
        for window_info in entries:
            for t in range(window_info["window_start"], window_info["window_end"]):
                if t < num_frames:
                    cluster_timeline[t] = window_info["cluster"]
    else:
        for ts_info in entries:
            t = ts_info["timestep"]
            if t < num_frames:
                cluster_timeline[t] = ts_info["cluster"]

    return cluster_timeline


def resolve_rollout_episode(
    rollout_episodes: List[EpisodeInfo],
    ep_id: int,
) -> Optional[EpisodeInfo]:
    """Find rollout episode by dataset index, or fall back to list position."""
    for ep in rollout_episodes:
        if ep.index == ep_id:
            return ep
    if 0 <= ep_id < len(rollout_episodes):
        return rollout_episodes[ep_id]
    return None


def resolve_demo_episode(
    demo_episodes: List[EpisodeInfo],
    holdout_episodes: List[EpisodeInfo],
    demo_split: str,
    ep_id: int,
) -> Optional[EpisodeInfo]:
    """Find demo episode by dataset index within train or holdout list."""
    if demo_split == "holdout":
        episodes = holdout_episodes
    else:
        episodes = demo_episodes
    for ep in episodes:
        if ep.index == ep_id:
            return ep
    return None
