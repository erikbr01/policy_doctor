"""Slice search (attribution): find demo slices that influence given rollout slices."""

from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from policy_doctor.data.structures import GlobalInfluenceMatrix, EpisodeInfo
from policy_doctor.computations.slice_influence import rank_demo_indices_by_slice_influence


def resolve_candidates_to_demo_slices(
    candidates: List[Dict[str, Any]],
    demo_sample_infos: List[Any],
    demo_episodes: List[EpisodeInfo],
    window_width: int,
) -> List[Tuple[int, int, int]]:
    """Resolve candidate dicts (with local_sample_idx) to unique (episode_idx, start, end) in raw timestep space.

    Each output slice is clamped to the episode (demo_end = min(demo_start + window_width - 1, ep_len - 1))
    so the final (episode_idx, start, end) never crosses demo episode boundaries.

    demo_sample_infos: list of objects with .episode_idx and .timestep (or .buffer_start_idx) for each
        sample in the split being resolved (e.g. holdout_sample_infos when candidates are from
        slice search over holdout). local_sample_idx in candidates indexes into this list.
    demo_episodes: episodes for this split (e.g. holdout only).
    window_width: used to compute demo_end = min(demo_start + window_width - 1, ep_len - 1).
    Returns: list of unique (episode_idx, start, end).
    """
    ep_len_by_index: Dict[int, int] = {
        ep.index: int(getattr(ep, "raw_length", None) or ep.num_samples)
        for ep in demo_episodes
    }
    lookup: Dict[int, Tuple[int, int, int]] = {}
    for local_idx in range(len(demo_sample_infos)):
        info = demo_sample_infos[local_idx]
        episode_idx = info.episode_idx if hasattr(info, "episode_idx") else info["episode_idx"]
        timestep = info.timestep if hasattr(info, "timestep") else info.get("timestep", getattr(info, "buffer_start_idx", 0))
        demo_start = int(timestep)
        ep_len = ep_len_by_index.get(episode_idx, 0)
        demo_end = min(demo_start + window_width - 1, ep_len - 1) if ep_len else demo_start
        lookup[local_idx] = (episode_idx, demo_start, demo_end)
    seen: set = set()
    result: List[Tuple[int, int, int]] = []
    for c in candidates:
        local_sample_idx = c.get("local_sample_idx")
        if local_sample_idx is None:
            continue
        t = lookup.get(int(local_sample_idx))
        if t is None:
            continue
        key = t
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def per_slice_percentile_selection(
    per_slice_candidates: List[List[Dict[str, Any]]],
    percentile: float,
) -> List[Dict[str, Any]]:
    """Select candidates at or above the given empirical percentile within each rollout slice."""
    selected: List[Dict[str, Any]] = []
    for slice_candidates in per_slice_candidates:
        if not slice_candidates:
            continue
        scores = np.array([c["score"] for c in slice_candidates], dtype=np.float64)
        cutoff = float(np.percentile(scores, percentile))
        for c in slice_candidates:
            if c["score"] >= cutoff:
                selected.append(c)
    return selected


def per_slice_n_sigma_selection(
    per_slice_candidates: List[List[Dict[str, Any]]],
    n_sigma: float,
) -> List[Dict[str, Any]]:
    """Select candidates with z-score >= n_sigma within each rollout slice (union).

    Matches influence_visualizer's _per_slice_n_sigma_selection. For heavy-tailed TRAK
    score distributions this is much more conservative than a fixed percentile threshold
    because the z-score cutoff adapts to the spread of each slice's distribution.
    """
    selected: List[Dict[str, Any]] = []
    for slice_candidates in per_slice_candidates:
        if not slice_candidates:
            continue
        scores = np.array([c["score"] for c in slice_candidates], dtype=np.float64)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std == 0:
            std = 1.0
        for c in slice_candidates:
            if (c["score"] - mean) / std >= n_sigma:
                selected.append(c)
    return selected


def run_slice_search(
    global_matrix: GlobalInfluenceMatrix,
    rollout_slices: List[Dict[str, Any]],
    demo_episodes: List[EpisodeInfo],
    window_width_demo: int = 1,
    per_slice_top_k: int = 10,
    ascending: bool = False,
    demo_start_idx: int = 0,
    demo_end_idx: Optional[int] = None,
    use_all_demos_per_slice: bool = False,
    show_progress: bool = False,
    aggregation_method: str = "sum",
    selection_percentile: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """For each rollout slice, rank demo samples by influence and return top-k candidates.

    Uses left-aligned sliding window over the demo axis to match influence_visualizer:
    score at demo index i = aggregation over (rollout slice x demo window [i, i+window_width)).

    Episode boundaries: Rollout slices are defined per episode (start/end within one
    rollout_ep), so the row range never crosses rollout episode borders. The demo
    columns [demo_start_idx, demo_end_idx) may span multiple demo episodes; the
    sliding window used for scoring can therefore cross demo episode boundaries.
    Resolve step (resolve_candidates_to_demo_slices) clamps each output slice to the
    episode so the final (episode_idx, start, end) never crosses.

    rollout_slices: list of dicts with keys rollout_idx, rollout_ep (EpisodeInfo), start, end.
    demo_start_idx, demo_end_idx: optional column range (default all demos). When set, only these
        demo columns are ranked and local_sample_idx in candidates is 0..(demo_end_idx - demo_start_idx)-1.
    use_all_demos_per_slice: if True, include all demo samples per rollout slice (for per-slice percentile).
    show_progress: if True, show a tqdm progress bar over rollout slices.
    aggregation_method: 'sum' or 'mean' (matches influence_visualizer rank_demos_by_slice_influence).
    selection_percentile: if set, filter each slice's candidates to those at or above this
        percentile of scores (computed on ALL scores per slice). This avoids creating dict objects
        for the vast majority of candidates that will be discarded. When set,
        use_all_demos_per_slice is implicitly True.
    Returns (all_candidates_flat, per_slice_candidates).
    """
    if demo_end_idx is None:
        demo_end_idx = global_matrix.num_demo_samples
    if selection_percentile is not None:
        use_all_demos_per_slice = True
    per_slice_candidates: List[List[Dict[str, Any]]] = []
    iterator = rollout_slices
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(rollout_slices, desc="Slice search", unit="slice")
        except ImportError:
            pass
    for rs in iterator:
        rollout_ep = rs["rollout_ep"]
        start, end = rs["start"], rs["end"]
        rollout_start_idx = rollout_ep.sample_start_idx + start
        rollout_end_idx = rollout_ep.sample_start_idx + end + 1
        block = global_matrix.get_slice(
            rollout_start_idx, rollout_end_idx, demo_start_idx, demo_end_idx
        )
        sorted_indices, sorted_scores, raw_scores = rank_demo_indices_by_slice_influence(
            block,
            window_width_demo=window_width_demo,
            aggregation_method=aggregation_method,
            ascending=ascending,
        )

        if selection_percentile is not None:
            cutoff = float(np.percentile(raw_scores, selection_percentile))
            mask = sorted_scores >= cutoff
            sorted_indices = sorted_indices[mask]
            sorted_scores = sorted_scores[mask]
            take = len(sorted_indices)
        elif use_all_demos_per_slice:
            take = len(sorted_indices)
        else:
            take = min(per_slice_top_k, len(sorted_indices))

        slice_candidates = []
        for i in range(take):
            idx_in_block = int(sorted_indices[i])
            score = float(sorted_scores[i])
            slice_candidates.append({
                "local_sample_idx": idx_in_block,
                "score": score,
                "source_episode_idx": rs["rollout_idx"],
                "source_start": start,
                "source_end": end,
            })
        per_slice_candidates.append(slice_candidates)
    all_candidates = [c for sl in per_slice_candidates for c in sl]
    return all_candidates, per_slice_candidates
