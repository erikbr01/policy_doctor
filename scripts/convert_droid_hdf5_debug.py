"""Convert DROID trajectory.h5 files → robomimic HDF5 for debug training.

Reads state observations directly from HDF5 (no ZED/pyzed dependency).
Image keys are filled with zeros (placeholder) — sufficient for verifying
the training pipeline end-to-end without real camera frames.

Output cameras: hand_camera_image, exterior_image_1_left (two views).
Action space: joint_velocity (7) + gripper_position (1) = 8-dim.

Usage:
    conda activate cupid_torch2
    python scripts/convert_droid_hdf5_debug.py \
        --input_path /mnt/ssdB/erik/droid_data/data \
        --output_path /mnt/ssdB/erik/droid_data/droid_debug_dataset.hdf5 \
        --image_size 84 84
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def _find_trajectories(root: str) -> list[str]:
    folders = []
    for dirpath, _, files in os.walk(root):
        if "trajectory.h5" in files:
            folders.append(dirpath)
    return sorted(folders)


def _load_trajectory(folder: str, image_h: int, image_w: int) -> dict | None:
    traj_file = os.path.join(folder, "trajectory.h5")
    try:
        with h5py.File(traj_file, "r") as f:
            success = bool(f.attrs.get("success", False))

            movement = f["observation/controller_info/movement_enabled"][:]
            valid = movement.astype(bool)

            # joint_velocity actions: (T, 7)
            jv = f["action/joint_velocity"][:][valid].astype(np.float32)
            # gripper position: (T,) → (T, 1)
            gp_action = f["action/gripper_position"][:][valid].astype(np.float32)[:, None]
            actions = np.concatenate([jv, gp_action], axis=-1)  # (T, 8)

            T = len(actions)
            if T == 0:
                return None

            joint_pos = f["observation/robot_state/joint_positions"][:][valid].astype(np.float32)
            cart_pos = f["observation/robot_state/cartesian_position"][:][valid].astype(np.float32)
            gripper_pos = f["observation/robot_state/gripper_position"][:][valid].astype(np.float32)[:, None]

    except Exception as e:
        print(f"  [skip] {folder}: {e}")
        return None

    # placeholder images — zeros as uint8 RGB
    placeholder = np.zeros((T, image_h, image_w, 3), dtype=np.uint8)

    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = float(success)

    return {
        "actions": actions,
        "obs": {
            "joint_positions": joint_pos,
            "cartesian_position": cart_pos,
            "gripper_position": gripper_pos,
            "hand_camera_image": placeholder,
            "exterior_image_1_left": placeholder,
        },
        "dones": dones,
        "rewards": rewards,
        "success": success,
    }


def convert(
    input_path: str,
    output_path: str,
    image_h: int,
    image_w: int,
    train_frac: float,
    val_frac: float,
) -> None:
    folders = _find_trajectories(input_path)
    print(f"Found {len(folders)} trajectory folders under {input_path}")

    demos = []
    for i, folder in enumerate(folders):
        print(f"  [{i+1}/{len(folders)}] {folder}")
        result = _load_trajectory(folder, image_h, image_w)
        if result is not None:
            demos.append(result)

    if not demos:
        raise RuntimeError("No valid trajectories found.")

    rng = np.random.default_rng(seed=0)
    idx = rng.permutation(len(demos))
    n_train = max(1, int(len(demos) * train_frac))
    n_val = max(1, int(len(demos) * val_frac))
    n_test = max(0, len(demos) - n_train - n_val)
    # adjust if rounding over-allocated
    while n_train + n_val + n_test > len(demos):
        n_test = max(0, n_test - 1)

    split_of = {}
    for i in idx[:n_train]:
        split_of[int(i)] = "train"
    for i in idx[n_train : n_train + n_val]:
        split_of[int(i)] = "valid"
    for i in idx[n_train + n_val : n_train + n_val + n_test]:
        split_of[int(i)] = "test"
    # anything unassigned (shouldn't happen) → train
    for j in range(len(demos)):
        split_of.setdefault(j, "train")

    total_steps = sum(len(d["actions"]) for d in demos)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["total"] = total_steps

        mask_grp = f.create_group("mask")
        split_lists: dict[str, list[str]] = {"train": [], "valid": [], "test": []}

        for demo_idx, demo in enumerate(demos):
            key = f"demo_{demo_idx}"
            dgrp = data_grp.create_group(key)
            dgrp.create_dataset("actions", data=demo["actions"])
            dgrp.create_dataset("dones", data=demo["dones"])
            dgrp.create_dataset("rewards", data=demo["rewards"])

            obs_grp = dgrp.create_group("obs")
            for obs_key, arr in demo["obs"].items():
                obs_grp.create_dataset(obs_key, data=arr)

            split_lists[split_of[demo_idx]].append(key)

        dt = h5py.special_dtype(vlen=str)
        for split_name, keys in split_lists.items():
            if keys:
                ds = mask_grp.create_dataset(split_name, (len(keys),), dtype=dt)
                for j, k in enumerate(keys):
                    ds[j] = k

    print(
        f"Done. {len(demos)} demos ({total_steps} steps) → {output_path}\n"
        f"  train={len(split_lists['train'])} "
        f"valid={len(split_lists['valid'])} "
        f"test={len(split_lists['test'])}\n"
        f"  images: placeholder zeros {image_h}x{image_w} (no ZED SDK available)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--image_size", nargs=2, type=int, default=[84, 84], metavar=("H", "W"))
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--val_frac", type=float, default=0.3)
    args = p.parse_args()
    convert(
        input_path=args.input_path,
        output_path=args.output_path,
        image_h=args.image_size[0],
        image_w=args.image_size[1],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
