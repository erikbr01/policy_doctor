"""Slice influence: rank demo slices by influence from a (rollout slice x demo cols) block.

Single definition: aggregate over rollout dimension, optionally apply left-aligned sliding
window over demo dimension, then rank demo indices by score. Used by both the local-matrix
API (one rollout x one demo trajectory) and the attribution pipeline (one rollout slice x
all demo columns).

Episode boundaries: The block's rows are assumed to be one rollout slice (no episode
crossing on the rollout side). The block's columns may span multiple demo episodes; the
sliding window over the demo axis is applied in global column index and can cross demo
episode borders. Callers that need episode-respecting output (e.g. attribution) clamp
the resolved slice to the episode in a later step (see resolve_candidates_to_demo_slices).
"""

from typing import Tuple

import numpy as np

from policy_doctor.data.structures import LocalInfluenceMatrix
from policy_doctor.data.aggregation import sliding_window_aggregate_left_aligned


def slice_influence_scores_from_block(
    block: np.ndarray,
    window_width_demo: int = 1,
    aggregation_method: str = "sum",
) -> np.ndarray:
    """Compute per-demo slice-influence scores from a (rollout_rows x demo_cols) block.

    Aggregates over the rollout axis (axis=0), then if window_width_demo > 1 applies
    a left-aligned sliding window over the demo axis. Same convention as influence_visualizer.

    Note: The demo-axis window is over consecutive column indices and may cross demo
    episode boundaries; this layer does not enforce per-episode windows.

    Args:
        block: Shape (rollout_height, num_demo_samples).
        window_width_demo: If > 1, left-aligned window over demo dimension.
        aggregation_method: 'sum' or 'mean'.

    Returns:
        Array of shape (num_demo_samples,) of scores per demo index.
    """
    if window_width_demo > 1:
        return sliding_window_aggregate_left_aligned(
            block,
            window_width=window_width_demo,
            kind=aggregation_method,
            pad_mode="edge",
        )
    if aggregation_method == "mean":
        return np.nanmean(block, axis=0).astype(np.float32)
    return np.nansum(block, axis=0).astype(np.float32)


def rank_demo_indices_by_slice_influence(
    block: np.ndarray,
    window_width_demo: int = 1,
    aggregation_method: str = "sum",
    ascending: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank demo (column) indices by slice-influence score. Single implementation for all call sites.

    Args:
        block: Shape (rollout_height, num_demo_samples).
        window_width_demo: If > 1, left-aligned window over demo axis.
        aggregation_method: 'sum' or 'mean'.
        ascending: If True, sort ascending (lowest first).

    Returns:
        sorted_demo_indices: Demo indices sorted by score.
        sorted_scores: Scores at those indices.
        raw_scores: Full 1D array of scores (length num_demo_samples).
    """
    raw_scores = slice_influence_scores_from_block(
        block, window_width_demo=window_width_demo, aggregation_method=aggregation_method
    )
    if ascending:
        sorted_indices = np.argsort(raw_scores)
    else:
        sorted_indices = np.argsort(raw_scores)[::-1]
    sorted_scores = raw_scores[sorted_indices]
    return sorted_indices, sorted_scores, raw_scores


def rank_demo_slices_by_influence(
    local_matrix: LocalInfluenceMatrix,
    rollout_sample_lo: int,
    rollout_sample_hi: int,
    window_width_demo: int = 1,
    aggregation_method: str = "sum",
    ascending: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank demo (column) indices by influence within one local influence matrix.

    Extracts the (rollout slice x demo cols) block from the local matrix and calls
    rank_demo_indices_by_slice_influence. Use this when you have a single rollout
    trajectory x single demo trajectory.

    Args:
        local_matrix: LocalInfluenceMatrix (one rollout trajectory x one demo trajectory).
        rollout_sample_lo: Start of rollout slice (local row index).
        rollout_sample_hi: End of rollout slice (exclusive).
        window_width_demo: If > 1, left-aligned sliding window over demo dimension.
        aggregation_method: 'sum' or 'mean'.
        ascending: If True, sort ascending (lowest influence first).

    Returns:
        sorted_demo_indices: Demo sample indices (local) sorted by influence.
        sorted_scores: Corresponding scores.
        raw_scores: Full 1D array of scores for all demo samples.
    """
    n_demo = local_matrix.shape[1]
    block = local_matrix.get_slice(
        rollout_sample_lo, rollout_sample_hi, 0, n_demo
    )
    return rank_demo_indices_by_slice_influence(
        block,
        window_width_demo=window_width_demo,
        aggregation_method=aggregation_method,
        ascending=ascending,
    )
