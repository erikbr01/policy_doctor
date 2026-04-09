"""Print layout summary for a robomimic- or MimicGen-style HDF5 (for diffusion_policy training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

from diffusion_policy.common.hdf5_robomimic_layout import sorted_robomimic_demo_keys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hdf5", type=Path, help="Path to .hdf5 dataset")
    args = p.parse_args()
    path = args.hdf5.expanduser()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    with h5py.File(path, "r") as f:
        if "data" not in f:
            print("Missing top-level 'data' group.", file=sys.stderr)
            return 1
        data = f["data"]
        demo_keys = sorted_robomimic_demo_keys(data)
        print(f"File: {path}")
        print(f"Demos ({len(demo_keys)}): {demo_keys[:5]}{' ...' if len(demo_keys) > 5 else ''}")
        env_args = data.attrs.get("env_args")
        if env_args is not None:
            try:
                meta = json.loads(env_args if isinstance(env_args, str) else env_args.decode())
                print(f"env_name: {meta.get('env_name', '?')}")
            except Exception as e:
                print(f"env_args present but not JSON: {e}")
        if demo_keys:
            d0 = data[demo_keys[0]]
            obs = d0["obs"]
            obs_keys = sorted(obs.keys())
            print(f"First demo obs keys ({len(obs_keys)}): {obs_keys}")
            act = d0["actions"]
            print(f"First demo actions shape: {act.shape}")
            if "datagen_info" in d0:
                print("Note: MimicGen datagen_info present (ignored by diffusion_policy loader).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
