"""Merge multiple DROID robomimic HDF5 files into one.

Concatenates demo groups from each source file, renumbering them sequentially.
Regenerates train/valid/test masks with a configurable split.

Usage:
    python scripts/merge_droid_hdf5.py \
        --inputs ~/data/droid_data/kendama_may13.hdf5 \
                 ~/data/droid_data/kendama_may19_may20.hdf5 \
        --output ~/data/droid_data/kendama_may13_may20.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def merge(inputs: list[str], output: str, train_frac: float, val_frac: float, seed: int) -> None:
    input_paths = [Path(p) for p in inputs]
    output_path = Path(output)

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as out:
        data_grp = out.create_group("data")
        mask_grp = out.create_group("mask")

        demo_idx = 0
        total_steps = 0

        for src_path in input_paths:
            print(f"Reading {src_path} ...")
            with h5py.File(src_path, "r") as src:
                src_data = src["data"]
                src_keys = sorted(src_data.keys(), key=lambda k: int(k.split("_")[1]))
                for key in src_keys:
                    dst_key = f"demo_{demo_idx}"
                    src.copy(src_data[key], data_grp, name=dst_key)
                    total_steps += data_grp[dst_key].attrs.get(
                        "num_samples", len(data_grp[dst_key]["actions"])
                    )
                    demo_idx += 1
                    print(f"  {key} → {dst_key}")

        n = demo_idx
        data_grp.attrs["total"] = total_steps

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        train_keys = [f"demo_{i}" for i in perm[:n_train]]
        val_keys   = [f"demo_{i}" for i in perm[n_train : n_train + n_val]]
        test_keys  = [f"demo_{i}" for i in perm[n_train + n_val :]]

        dt = h5py.special_dtype(vlen=str)
        for split_name, keys in [("train", train_keys), ("valid", val_keys), ("test", test_keys)]:
            if keys:
                ds = mask_grp.create_dataset(split_name, (len(keys),), dtype=dt)
                for j, k in enumerate(keys):
                    ds[j] = k

    print(
        f"\nDone. {n} demos ({total_steps} steps) → {output_path}\n"
        f"  train={len(train_keys)}  valid={len(val_keys)}  test={len(test_keys)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="Source HDF5 files (merged in order)")
    p.add_argument("--output", required=True, help="Destination HDF5 file")
    p.add_argument("--train_frac", type=float, default=0.85)
    p.add_argument("--val_frac",   type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=0)
    args = p.parse_args()
    merge(args.inputs, args.output, args.train_frac, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
