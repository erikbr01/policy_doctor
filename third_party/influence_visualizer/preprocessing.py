"""Preprocessing utilities for influence matrices.

This module provides preprocessing functions for influence matrices,
including soft thresholding as recommended by the TRAK paper.
"""

import numpy as np


def soft_threshold(influence_matrix: np.ndarray, lambda_threshold: float) -> np.ndarray:
    """Apply soft thresholding to influence matrix to encourage sparsity.

    Implements the soft-thresholding operator S(τ; λ) from the TRAK paper:

    S(τᵢ; λ) = {
        τᵢ - λ     if τᵢ > λ
        τᵢ + λ     if τᵢ < -λ
        0          if |τᵢ| ≤ λ
    }

    This operator:
    1. Shrinks positive values by λ (if above threshold)
    2. Shrinks negative values by λ (if below negative threshold)
    3. Zeros out values with magnitude ≤ λ

    The TRAK paper recommends this to encourage sparsity in attribution scores,
    which aligns with the intuition that most training samples have negligible
    influence on most test predictions.

    Args:
        influence_matrix: Raw influence matrix, shape (num_rollouts, num_demos)
        lambda_threshold: Threshold parameter λ. Higher values = more sparsity.
                         Typical values: 0.001 to 0.1 depending on score scale.

    Returns:
        Thresholded influence matrix with the same shape, now sparse.

    Example:
        >>> scores = np.array([0.5, -0.3, 0.05, -0.02, 0.8])
        >>> soft_threshold(scores, lambda_threshold=0.1)
        array([0.4, -0.2, 0.0, 0.0, 0.7])

    References:
        TRAK: Attributing Model Behavior at Scale
        https://arxiv.org/abs/2303.14186
    """
    # Vectorized implementation for efficiency
    result = np.zeros_like(influence_matrix)

    # Positive values above threshold: subtract λ
    positive_mask = influence_matrix > lambda_threshold
    result[positive_mask] = influence_matrix[positive_mask] - lambda_threshold

    # Negative values below threshold: add λ (make less negative)
    negative_mask = influence_matrix < -lambda_threshold
    result[negative_mask] = influence_matrix[negative_mask] + lambda_threshold

    # Values with |τᵢ| ≤ λ are already zero (from initialization)

    return result


def compute_sparsity(
    influence_matrix: np.ndarray, zero_threshold: float = 1e-10
) -> float:
    """Compute the sparsity percentage of an influence matrix.

    Args:
        influence_matrix: Influence matrix
        zero_threshold: Values below this absolute magnitude are considered zero

    Returns:
        Sparsity as a percentage (0-100)
    """
    near_zero = np.abs(influence_matrix) < zero_threshold
    sparsity_pct = 100.0 * near_zero.sum() / influence_matrix.size
    return sparsity_pct


def suggest_threshold(
    influence_matrix: np.ndarray, target_sparsity: float = 0.5
) -> float:
    """Suggest a threshold value to achieve a target sparsity level.

    Args:
        influence_matrix: Influence matrix
        target_sparsity: Target sparsity level (0 to 1), e.g., 0.5 = 50% zeros

    Returns:
        Suggested lambda threshold value
    """
    abs_values = np.abs(influence_matrix.flatten())
    # Find the value at the target percentile
    percentile = target_sparsity * 100
    suggested_lambda = np.percentile(abs_values, percentile)
    return suggested_lambda


def analyze_influence_distribution(influence_matrix: np.ndarray) -> dict:
    """Analyze the distribution of influence scores.

    Args:
        influence_matrix: Influence matrix

    Returns:
        Dictionary with distribution statistics
    """
    flat = influence_matrix.flatten()
    abs_flat = np.abs(flat)

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "abs_median": float(np.median(abs_flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p05": float(np.percentile(abs_flat, 5)),
        "p25": float(np.percentile(abs_flat, 25)),
        "p50": float(np.percentile(abs_flat, 50)),
        "p75": float(np.percentile(abs_flat, 75)),
        "p95": float(np.percentile(abs_flat, 95)),
        "p99": float(np.percentile(abs_flat, 99)),
        "sparsity": compute_sparsity(influence_matrix),
        "n_positive": int((flat > 0).sum()),
        "n_negative": int((flat < 0).sum()),
        "n_zero": int((np.abs(flat) < 1e-10).sum()),
    }
