"""Trajectory-level influence explanations (aggregate sample scores to trajectory scores)."""

from typing import Callable, List

import numpy as np


def mean_of_mean_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    return float(np.nanmean(scores_ij))


def mean_of_mean_influence_success(scores_ij: np.ndarray, is_success: bool) -> float:
    return float(np.nanmean(scores_ij)) if is_success else 0.0


def sum_of_sum_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    return float(np.nansum(scores_ij))


def sum_of_sum_influence_success(scores_ij: np.ndarray, is_success: bool) -> float:
    return float(np.nansum(scores_ij)) if is_success else 0.0


def min_of_max_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = np.nanmax(scores_ij, axis=1).min()
    return -float(s) if is_success else float(s)


def max_of_min_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = np.nanmin(scores_ij, axis=1).max()
    return float(s) if is_success else -float(s)


AGGREGATION_FUNCTIONS = {
    "mean_of_mean": mean_of_mean_influence,
    "mean_of_mean_success": mean_of_mean_influence_success,
    "sum_of_sum": sum_of_sum_influence,
    "sum_of_sum_success": sum_of_sum_influence_success,
    "min_of_max": min_of_max_influence,
    "max_of_min": max_of_min_influence,
}


def trajectory_scores(
    pairwise_sample_scores: np.ndarray,
    test_ep_lens: np.ndarray,
    train_ep_lens: np.ndarray,
    success_mask: np.ndarray,
    aggr_fn: Callable[[np.ndarray, bool], float] = mean_of_mean_influence,
) -> np.ndarray:
    """Aggregate pairwise sample scores to (num_test_eps, num_train_eps) trajectory scores.

    pairwise_sample_scores shape: (sum(test_ep_lens), sum(train_ep_lens)).
    """
    num_test_eps = len(test_ep_lens)
    num_train_eps = len(train_ep_lens)
    assert pairwise_sample_scores.shape[0] == test_ep_lens.sum()
    assert pairwise_sample_scores.shape[1] == train_ep_lens.sum()
    assert len(success_mask) == num_test_eps

    test_offsets = np.concatenate([[0], np.cumsum(test_ep_lens)[:-1]])
    train_offsets = np.concatenate([[0], np.cumsum(train_ep_lens)[:-1]])

    traj_scores = np.zeros((num_test_eps, num_train_eps), dtype=np.float32)
    for i in range(num_test_eps):
        r_lo = test_offsets[i]
        r_hi = r_lo + test_ep_lens[i]
        for j in range(num_train_eps):
            d_lo = train_offsets[j]
            d_hi = d_lo + train_ep_lens[j]
            block = pairwise_sample_scores[r_lo:r_hi, d_lo:d_hi]
            traj_scores[i, j] = aggr_fn(block, bool(success_mask[i]))
    return traj_scores
