"""Merge multiple DROID robomimic HDF5 files into one (zero-copy by default).

Builds an output HDF5 whose ``data/demo_*`` entries are ``h5py.ExternalLink``
references into the source files, with sequential renumbering and regenerated
``mask/{train,valid,test}`` splits. No bytes are duplicated — the output file
is a thin index, typically a few MB regardless of input size.

The dataset loader (``RobomimicReplayImageDataset``) reads demo groups via
``data_grp[key]``, which transparently resolves external links. The downstream
zarr cache is built once and snapshots the data; after that the merged HDF5
is only re-read if the cache is missing.

Trade-off: the merged file depends on the source paths remaining valid.
Pass ``--copy_data`` for a self-contained output (slower; copies all bytes).

Usage:
    python scripts/merge_droid_hdf5.py \
        --inputs ~/data/droid_data/kendama_may13.hdf5 \
                 ~/data/droid_data/kendama_may19_may20.hdf5 \
        --output ~/data/droid_data/kendama_may13_may20.hdf5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def _sorted_demo_keys(data_grp: h5py.Group) -> list[str]:
    keys = [k for k in data_grp.keys() if k.startswith("demo_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _demo_steps(demo_grp: h5py.Group) -> int:
    n = demo_grp.attrs.get("num_samples")
    return int(n) if n is not None else int(len(demo_grp["actions"]))


def merge(
    inputs: list[str],
    output: str,
    train_frac: float,
    val_frac: float,
    seed: int,
    copy_data: bool = False,
) -> None:
    input_paths = [Path(p).resolve() for p in inputs]
    output_path = Path(output)
    out_parent = output_path.parent.resolve()

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    demo_idx = 0
    total_steps = 0

    with h5py.File(output_path, "w") as out:
        data_grp = out.create_group("data")

        for src_path in input_paths:
            mode = "copy" if copy_data else "link"
            print(f"{mode}: {src_path}")
            # Use a path relative to the output file's directory if both live on
            # the same filesystem branch — keeps the link stable under common
            # rename/move operations on the output directory.
            try:
                link_target = os.path.relpath(src_path, start=out_parent)
            except ValueError:
                link_target = str(src_path)
            with h5py.File(src_path, "r") as src:
                src_data = src["data"]
                for key in _sorted_demo_keys(src_data):
                    dst_key = f"demo_{demo_idx}"
                    total_steps += _demo_steps(src_data[key])
                    if copy_data:
                        src.copy(src_data[key], data_grp, name=dst_key)
                    else:
                        data_grp[dst_key] = h5py.ExternalLink(
                            link_target, f"data/{key}"
                        )
                    demo_idx += 1

        n = demo_idx
        data_grp.attrs["total"] = total_steps

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        train_keys = [f"demo_{i}" for i in perm[:n_train]]
        val_keys   = [f"demo_{i}" for i in perm[n_train : n_train + n_val]]
        test_keys  = [f"demo_{i}" for i in perm[n_train + n_val :]]

        mask_grp = out.create_group("mask")
        dt = h5py.special_dtype(vlen=str)
        for split_name, keys in [("train", train_keys), ("valid", val_keys), ("test", test_keys)]:
            if keys:
                ds = mask_grp.create_dataset(split_name, (len(keys),), dtype=dt)
                for j, k in enumerate(keys):
                    ds[j] = k

    out_size = output_path.stat().st_size
    print(
        f"\nDone. {n} demos ({total_steps} steps) → {output_path} "
        f"({out_size/1e6:.2f} MB)\n"
        f"  train={len(train_keys)}  valid={len(val_keys)}  test={len(test_keys)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="Source HDF5 files (merged in order)")
    p.add_argument("--output", required=True, help="Destination HDF5 file")
    p.add_argument("--train_frac", type=float, default=0.85)
    p.add_argument("--val_frac",   type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument(
        "--copy_data",
        action="store_true",
        help="Copy every demo's bytes into the output (self-contained, slow). "
             "Default: use h5py.ExternalLink (fast, zero-copy; output depends "
             "on the source files staying in place).",
    )
    args = p.parse_args()
    merge(args.inputs, args.output, args.train_frac, args.val_frac, args.seed, args.copy_data)


if __name__ == "__main__":
    main()
