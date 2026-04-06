"""Aggregation helpers: sum/mean over axis with optional windowing."""

from typing import Optional

import numpy as np


def aggregate_axis(
    data: np.ndarray,
    axis: Optional[int] = None,
    agg_fn: str = "sum",
) -> np.ndarray:
    """Aggregate over axis. agg_fn in ('sum', 'mean')."""
    if agg_fn == "sum":
        out = np.nansum(data, axis=axis)
    elif agg_fn == "mean":
        out = np.nanmean(data, axis=axis)
    else:
        raise ValueError(f"agg_fn must be 'sum' or 'mean', got {agg_fn!r}")
    return np.asarray(out, dtype=np.float32)


def sliding_window_sum(
    data: np.ndarray,
    window_width: int,
    axis: int = 1,
    pad_mode: str = "edge",
) -> np.ndarray:
    """For each position, sum over a window of width window_width along axis. Padding applied at edges."""
    if window_width < 1:
        raise ValueError("window_width must be >= 1")
    if axis == 0:
        data = data.T
    n = data.shape[1]
    out = np.zeros((data.shape[0], n), dtype=np.float32)
    half = window_width // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i - half + window_width)
        if lo >= hi:
            lo, hi = max(0, i - window_width + 1), i + 1
        out[:, i] = np.nansum(data[:, lo:hi], axis=1)
    if axis == 0:
        out = out.T
    return out


def sliding_window_aggregate_left_aligned(
    influence_slice: np.ndarray,
    window_width: int,
    kind: str = "sum",
    pad_mode: str = "edge",
) -> np.ndarray:
    """Left-aligned sliding window over the demo axis (axis=1). Matches influence_visualizer.

    For each demo index i, the score is aggregation over rollout rows x demo samples [i, i+window_width)
    with edge padding when the window extends past the end. Output length = influence_slice.shape[1].

    Args:
        influence_slice: Shape (rollout_height, num_demo_samples).
        window_width: Width of the window along the demo axis.
        kind: 'sum' or 'mean'. For 'mean', divides by rollout_height * window_width.
        pad_mode: 'edge' (repeat boundary) or 'constant' (zero).

    Returns:
        Array of shape (num_demo_samples,) with one score per demo sample index.
    """
    if window_width < 1:
        raise ValueError("window_width must be >= 1")
    rollout_height, num_demo_samples = influence_slice.shape
    pad_width = window_width - 1
    if pad_mode == "edge":
        padded = np.pad(
            influence_slice.astype(np.float64),
            ((0, 0), (0, pad_width)),
            mode="edge",
        )
    else:
        padded = np.pad(
            influence_slice.astype(np.float64),
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )
    kernel = np.ones(window_width, dtype=padded.dtype)
    window_sums = np.zeros(num_demo_samples, dtype=np.float64)
    for r in range(padded.shape[0]):
        window_sums += np.convolve(padded[r, :], kernel, mode="valid")
    if kind == "mean" and (rollout_height * window_width) > 0:
        window_sums /= rollout_height * window_width
    return window_sums.astype(np.float32)
