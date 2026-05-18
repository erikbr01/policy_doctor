"""Rebuild `<eval_dir>/episodes/metadata.yaml` from the .pkl files.

The eval_save_episodes runner skips this write when `n_test_vis=0` (no media
dir produced). compute_policy_embeddings.py needs the file. We infer
`episode_lengths` from the saved pkls and `episode_successes` from the
filename suffix (`_succ` / `_fail`).

Usage:
    python scripts/regen_metadata_yaml.py <eval_dir1> [<eval_dir2> ...]
"""

from __future__ import annotations

import pathlib
import pickle
import sys

import yaml


def _episode_length(pkl_path: pathlib.Path) -> int:
    """Count timesteps in one episode pkl.

    Episode format (robomimic lowdim runner):
        list of dicts with keys 'obs', 'action', 'reward', ...
    The list length IS the episode length.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # DataFrame (canonical robomimic_lowdim_runner format): one row per timestep.
    if hasattr(data, "shape") and hasattr(data, "columns"):
        return int(data.shape[0])
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for k in ("action", "obs", "reward"):
            if k in data:
                arr = data[k]
                if isinstance(arr, dict):
                    arr = next(iter(arr.values()))
                return int(arr.shape[0])
    raise ValueError(f"Unrecognized pkl format: {pkl_path} (type={type(data).__name__})")


def regen(eval_dir: pathlib.Path) -> None:
    ep_dir = eval_dir / "episodes"
    if not ep_dir.is_dir():
        print(f"  skip {eval_dir}: no episodes/")
        return
    pkls = sorted(ep_dir.glob("ep*.pkl"))
    if not pkls:
        print(f"  skip {eval_dir}: no ep*.pkl")
        return
    lengths = []
    successes = []
    for pkl in pkls:
        # Resolve symlinks for length read.
        try:
            T = _episode_length(pkl.resolve())
        except Exception as e:  # noqa: BLE001
            print(f"    error {pkl.name}: {e}")
            continue
        succ = pkl.stem.endswith("_succ")
        lengths.append(T)
        successes.append(succ)
    out = ep_dir / "metadata.yaml"
    out.write_text(
        yaml.safe_dump({
            "length": int(sum(lengths)),
            "episode_lengths": [int(x) for x in lengths],
            "episode_successes": [bool(x) for x in successes],
        })
    )
    print(f"  wrote {out}: {len(lengths)} episodes, total length {sum(lengths)}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        regen(pathlib.Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
