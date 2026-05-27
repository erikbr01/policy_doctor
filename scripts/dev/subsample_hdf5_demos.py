"""Copy the first N demos of a robomimic-format HDF5 into a new file.

Used to materialise per-N training subsets (e.g. ``square_d1_60.hdf5`` =
first 60 demos of the 1000-demo MimicGen source).  Demo groups in HDF5 are
ordered by insertion, *not* lexicographically — so we sort by the numeric
suffix of ``demo_<int>`` to get a stable "first N" interpretation.

Usage::

    python scripts/subsample_hdf5_demos.py \\
        --in  ~/data/mimicgen_data/core_datasets/square/demo_src_square_task_D1/demo.hdf5 \\
        --out data/mimicgen/square_d1_60.hdf5 \\
        --n   60
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py


_DEMO_RE = re.compile(r"^demo_(\d+)$")


def _sorted_demo_keys(group: h5py.Group) -> list[str]:
    keys = [k for k in group.keys() if _DEMO_RE.match(k)]
    keys.sort(key=lambda k: int(_DEMO_RE.match(k).group(1)))
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--in", dest="src", type=Path, required=True, help="Input HDF5.")
    p.add_argument("--out", dest="dst", type=Path, required=True, help="Output HDF5.")
    p.add_argument("--n", type=int, required=True, help="Number of demos to keep.")
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.exists():
        raise FileExistsError(f"Output exists, refusing to overwrite: {args.dst}")

    with h5py.File(args.src, "r") as src, h5py.File(args.dst, "w") as dst:
        if "data" not in src:
            raise KeyError(f"{args.src} has no 'data' group")
        src_data = src["data"]
        all_keys = _sorted_demo_keys(src_data)
        if args.n > len(all_keys):
            raise ValueError(f"Requested {args.n} demos but source has only {len(all_keys)}")
        keep = all_keys[: args.n]

        dst_data = dst.create_group("data")
        for k, v in src_data.attrs.items():
            dst_data.attrs[k] = v

        for key in keep:
            src_data.copy(key, dst_data, name=key)

        if "mask" in src:
            src.copy("mask", dst, name="mask")

        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst_data.attrs["total"] = sum(
            int(src_data[k].attrs.get("num_samples", src_data[k]["actions"].shape[0]))
            for k in keep
        )

    print(f"Wrote {len(keep)} demos -> {args.dst}")


if __name__ == "__main__":
    main()
