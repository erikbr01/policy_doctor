"""Resize all img_* columns in eval rollout .pkl files in-place.

Each .pkl is a pandas DataFrame whose `img*` columns hold uint8 arrays per
timestep. We downscale every frame to (128, 128, 3) using cv2.INTER_AREA,
write to a sibling `*.tmp` and atomically rename over the original. Idempotent:
files whose first image is already <= target size are skipped.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TARGET = 128
IMG_COL_PREFIX = "img"


def _is_image_column(df: pd.DataFrame, col: str) -> bool:
    if not col.startswith(IMG_COL_PREFIX):
        return False
    v = df[col].iloc[0]
    return isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3 and v.dtype == np.uint8


def _resize_frame(frame: np.ndarray, target: int) -> np.ndarray:
    if frame.shape[0] == target and frame.shape[1] == target:
        return frame
    return cv2.resize(frame, (target, target), interpolation=cv2.INTER_AREA)


def resize_pickle(path: Path, target: int = TARGET) -> tuple[bool, int, int]:
    """Returns (changed, bytes_before, bytes_after)."""
    bytes_before = path.stat().st_size
    with open(path, "rb") as f:
        df = pickle.load(f)
    assert isinstance(df, pd.DataFrame), f"unexpected payload type in {path}: {type(df)}"

    img_cols = [c for c in df.columns if _is_image_column(df, c)]
    if not img_cols:
        return (False, bytes_before, bytes_before)

    first = df[img_cols[0]].iloc[0]
    if first.shape[0] <= target and first.shape[1] <= target:
        return (False, bytes_before, bytes_before)

    for col in img_cols:
        df[col] = [_resize_frame(frame, target) for frame in df[col].values]

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    bytes_after = path.stat().st_size
    return (True, bytes_before, bytes_after)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="Directories containing ep*.pkl files")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_before = 0
    total_after = 0
    n_changed = 0
    n_skipped = 0

    for d in args.dirs:
        d = Path(d)
        files = sorted(d.glob("ep*.pkl"))
        print(f"[{d}] {len(files)} files", flush=True)
        for i, p in enumerate(files):
            t0 = time.time()
            if args.dry_run:
                print(f"  would resize {p.name}")
                continue
            changed, b0, b1 = resize_pickle(p, args.target)
            total_before += b0
            total_after += b1
            if changed:
                n_changed += 1
                dt = time.time() - t0
                print(
                    f"  [{i+1:>3}/{len(files)}] {p.name}: "
                    f"{b0/1e6:.1f} -> {b1/1e6:.1f} MB  ({dt:.1f}s)",
                    flush=True,
                )
            else:
                n_skipped += 1
                print(f"  [{i+1:>3}/{len(files)}] {p.name}: skip (already <= {args.target})", flush=True)

    if not args.dry_run:
        print(
            f"\ndone: {n_changed} resized, {n_skipped} skipped. "
            f"total {total_before/1e9:.1f} -> {total_after/1e9:.1f} GB "
            f"(freed {(total_before-total_after)/1e9:.1f} GB)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
