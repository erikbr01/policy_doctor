"""Extract influence embeddings (per timestep or per slice; sum aggregation for slices)."""

from typing import List, Dict, Any, Tuple, Literal

import numpy as np

from policy_doctor.data.structures import GlobalInfluenceMatrix, EpisodeInfo


def extract_slice_embeddings_sum(
    global_matrix: GlobalInfluenceMatrix,
    episodes: List[EpisodeInfo],
    level: Literal["rollout", "demo"],
    window_width: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract embeddings by aggregating over a sliding window with sum.

    level='rollout': each row is one rollout sample (or window of rollout samples);
        we aggregate over demo dimension (sum) -> one vector per rollout (sample or window).
    level='demo': each column is one demo sample; we aggregate over rollout dimension (sum)
        -> one vector per demo (sample or window).

    For window_width=1, stride=1: one embedding per timestep (sum over the other axis).
    For window_width>1: sliding window of width window_width, sum within window.

    Returns:
        embeddings: (n_embeddings, n_other_axis) e.g. (n_rollout_windows, n_demo_samples) or (n_rollout_samples, n_demo_windows).
        metadata: list of dicts with episode index, start, end, etc.
    """
    if level == "rollout":
        # Rows = rollout, cols = demo. Embedding per rollout (row): sum over demos or window over rows then sum demos.
        n_rows, n_cols = global_matrix.shape
        if window_width == 1 and stride == 1:
            embeddings = np.zeros((n_rows, n_cols), dtype=np.float32)
            for r in range(n_rows):
                row = global_matrix.get_slice(r, r + 1, 0, n_cols)
                embeddings[r] = np.nansum(row, axis=0)
            metadata = _metadata_rollout_samples(episodes)
            return embeddings, metadata
        # Sliding window over rows
        n_windows = max(0, (n_rows - window_width) // stride + 1)
        embeddings = np.zeros((n_windows, n_cols), dtype=np.float32)
        metadata = []
        for i in range(n_windows):
            r_lo = i * stride
            r_hi = r_lo + window_width
            block = global_matrix.get_slice(r_lo, r_hi, 0, n_cols)
            embeddings[i] = np.nansum(block, axis=0)
            ep_idx, _ = _global_idx_to_episode(r_lo, episodes)
            metadata.append({
                "rollout_idx": ep_idx,
                "window_start": r_lo,
                "window_end": r_hi,
                "window_width": window_width,
            })
        return embeddings, metadata
    else:
        # level == "demo": embedding per demo column
        n_rows, n_cols = global_matrix.shape
        if window_width == 1 and stride == 1:
            embeddings = np.zeros((n_rows, n_cols), dtype=np.float32)
            for d in range(n_cols):
                col = global_matrix.get_slice(0, n_rows, d, d + 1)
                embeddings[:, d] = np.nansum(col, axis=1).flatten()
            metadata = _metadata_demo_samples(episodes)
            return embeddings, metadata
        n_windows = max(0, (n_cols - window_width) // stride + 1)
        embeddings = np.zeros((n_rows, n_windows), dtype=np.float32)
        metadata = []
        for i in range(n_windows):
            d_lo = i * stride
            d_hi = d_lo + window_width
            block = global_matrix.get_slice(0, n_rows, d_lo, d_hi)
            embeddings[:, i] = np.nansum(block, axis=1).flatten()
            ep_idx, _ = _global_idx_to_episode(d_lo, episodes)
            metadata.append({
                "demo_idx": ep_idx,
                "window_start": d_lo,
                "window_end": d_hi,
                "window_width": window_width,
            })
        return embeddings, metadata


def _global_idx_to_episode(
    global_idx: int, episodes: List[EpisodeInfo]
) -> Tuple[int, int]:
    """Return (episode_index, offset_within_episode) for a global sample index."""
    for ep in episodes:
        if ep.sample_start_idx <= global_idx < ep.sample_end_idx:
            return ep.index, global_idx - ep.sample_start_idx
    return -1, 0


def _metadata_rollout_samples(episodes: List[EpisodeInfo]) -> List[Dict[str, Any]]:
    meta = []
    for ep in sorted(episodes, key=lambda e: e.index):
        for t in range(ep.num_samples):
            meta.append({
                "rollout_idx": ep.index,
                "timestep": t,
                "success": ep.success,
            })
    return meta


def _metadata_demo_samples(episodes: List[EpisodeInfo]) -> List[Dict[str, Any]]:
    meta = []
    for ep in sorted(episodes, key=lambda e: e.index):
        for t in range(ep.num_samples):
            meta.append({
                "demo_idx": ep.index,
                "timestep": t,
            })
    return meta
