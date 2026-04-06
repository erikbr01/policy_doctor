"""Load rollout window frames from eval_save_episodes episode pickles."""

from __future__ import annotations

import pathlib
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def list_rollout_episode_pkls(episodes_dir: pathlib.Path) -> List[pathlib.Path]:
    """Sorted episode pickles (excludes metadata)."""
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
    pkls = sorted(episodes_dir.glob("ep*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No ep*.pkl under {episodes_dir}")
    return pkls


def resolve_window_indices(meta: dict) -> Tuple[int, int, int]:
    """Return (rollout_idx, window_start, window_end) from clustering metadata."""
    if "demo_idx" in meta and "rollout_idx" not in meta:
        raise ValueError(
            "Demo-level clustering metadata is not supported for rollout episode images."
        )
    r_idx = int(meta["rollout_idx"])
    if "window_start" in meta and "window_end" in meta:
        w0, w1 = int(meta["window_start"]), int(meta["window_end"])
    elif "start" in meta and "end" in meta:
        w0, w1 = int(meta["start"]), int(meta["end"])
    else:
        raise KeyError(f"metadata missing window bounds: keys={list(meta.keys())}")
    return r_idx, w0, w1


def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return x


def extract_window_frames(
    eval_dir: pathlib.Path,
    rollout_idx: int,
    window_start: int,
    window_end: int,
    *,
    max_frames: Optional[int],
    rng: np.random.Generator,
) -> List[Image.Image]:
    """Load RGB PIL images from ``[window_start, window_end)``.

    If *max_frames* is ``None``, returns every timestep in the window (no subsampling).
    Otherwise returns up to *max_frames* uniformly spaced timesteps.
    """
    ep_dir = eval_dir / "episodes"
    pkls = list_rollout_episode_pkls(ep_dir)
    if rollout_idx < 0 or rollout_idx >= len(pkls):
        raise IndexError(
            f"rollout_idx={rollout_idx} out of range for {len(pkls)} episode files"
        )
    with open(pkls[rollout_idx], "rb") as f:
        df = pickle.load(f)

    n = window_end - window_start
    if n <= 0:
        raise ValueError(f"empty window [{window_start}, {window_end})")

    lo = max(0, window_start)
    hi = min(len(df), window_end)
    if lo >= hi:
        raise ValueError(
            f"window [{window_start}, {window_end}) outside episode length {len(df)}"
        )

    idxs: Sequence[int]
    span = hi - lo
    if max_frames is None:
        _ = rng  # API symmetry with subsampled path; full window is deterministic
        idxs = list(range(lo, hi))
    elif max_frames <= 0:
        raise ValueError("max_frames must be positive or None")
    elif span <= max_frames:
        idxs = list(range(lo, hi))
    else:
        pick = np.linspace(lo, hi - 1, num=max_frames)
        idxs = [int(round(x)) for x in pick]
        idxs = sorted(set(idxs))

    images: List[Image.Image] = []
    for i in idxs:
        row = df.iloc[i]
        if "img" not in row:
            raise KeyError(f"episode row {i} has no 'img' column — need image policy eval save")
        arr = _ensure_rgb_uint8(np.asarray(row["img"]))
        images.append(Image.fromarray(arr, mode="RGB"))
    return images
