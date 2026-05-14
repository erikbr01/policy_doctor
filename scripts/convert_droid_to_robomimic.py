"""Convert a folder of DROID trajectory.h5 files into a single robomimic-format HDF5.

The output is compatible with the policy_doctor attribution pipeline (TRAK, InfEmbed,
clustering, curation) and with cupid diffusion_policy image training.

Usage:
    conda activate policy_doctor
    python scripts/convert_droid_to_robomimic.py \
        --input_path /path/to/droid/success \
        --output_path data/source/droid/droid_dataset.hdf5 \
        --wrist_serial 14313307 \
        --ext1_serial 36716034 \
        --ext2_serial 37617599 \
        --image_size 180 320 \
        --action_space joint_velocity \
        --train_frac 0.9 --val_frac 0.05

The resulting HDF5 has the standard robomimic layout:
    data/demo_N/actions   (T, 8)  — joint_velocity (7) + gripper_position (1)
    data/demo_N/obs/joint_positions       (T, 7)
    data/demo_N/obs/cartesian_position    (T, 6)
    data/demo_N/obs/gripper_position      (T, 1)
    data/demo_N/obs/hand_camera_image     (T, H, W, 3)
    data/demo_N/obs/exterior_image_1_left (T, H, W, 3)
    data/demo_N/obs/exterior_image_2_left (T, H, W, 3)
    data/demo_N/dones  (T,)
    data/demo_N/rewards (T,)
    mask/train, mask/valid, mask/test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def _add_droid_to_path():
    src = Path("~/src_droid").expanduser()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[:2] == (height, width):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _convert_one_trajectory(
    folder: str,
    wrist_serial: str,
    ext1_serial: str,
    ext2_serial: str,
    image_h: int,
    image_w: int,
    action_space: str,
) -> dict | None:
    from droid.trajectory_utils.misc import load_trajectory

    traj_file = os.path.join(folder, "trajectory.h5")
    if not os.path.isfile(traj_file):
        return None

    recording_dir = os.path.join(folder, "recordings", "MP4")
    if not os.path.isdir(recording_dir):
        recording_dir = None

    try:
        timesteps = load_trajectory(
            filepath=traj_file,
            read_cameras=(recording_dir is not None),
            recording_folderpath=recording_dir,
            remove_skipped_steps=True,
        )
    except Exception as e:
        print(f"  [skip] {folder}: {e}")
        return None

    if len(timesteps) == 0:
        return None

    # Read trajectory-level success flag from HDF5 attrs.
    success = False
    try:
        with h5py.File(traj_file, "r") as f:
            success = bool(f.attrs.get("success", False))
    except Exception:
        pass

    T = len(timesteps)

    # --- actions ---
    arm_key = action_space  # e.g. "joint_velocity"
    try:
        arm_actions = np.stack([t["action"][arm_key] for t in timesteps], axis=0)  # (T, 7)
        gripper_actions = np.stack(
            [np.atleast_1d(t["action"]["gripper_position"]) for t in timesteps], axis=0
        )  # (T, 1)
        actions = np.concatenate([arm_actions, gripper_actions], axis=-1).astype(np.float32)
    except KeyError as e:
        print(f"  [skip] {folder}: missing action key {e}")
        return None

    # --- low-dim obs ---
    try:
        joint_pos = np.stack(
            [t["observation"]["robot_state"]["joint_positions"] for t in timesteps], axis=0
        ).astype(np.float32)
        cartesian_pos = np.stack(
            [t["observation"]["robot_state"]["cartesian_position"] for t in timesteps], axis=0
        ).astype(np.float32)
        gripper_pos = np.stack(
            [np.atleast_1d(t["observation"]["robot_state"]["gripper_position"]) for t in timesteps],
            axis=0,
        ).astype(np.float32)
    except KeyError as e:
        print(f"  [skip] {folder}: missing obs key {e}")
        return None

    obs: dict[str, np.ndarray] = {
        "joint_positions": joint_pos,
        "cartesian_position": cartesian_pos,
        "gripper_position": gripper_pos,
    }

    # --- images (optional — skipped gracefully if absent) ---
    serial_to_key = {
        f"{wrist_serial}_left": "hand_camera_image",
        f"{ext1_serial}_left": "exterior_image_1_left",
        f"{ext2_serial}_left": "exterior_image_2_left",
    }

    first_obs = timesteps[0]["observation"]
    has_images = "image" in first_obs and len(first_obs["image"]) > 0

    if has_images:
        for serial_key, obs_key in serial_to_key.items():
            frames = []
            for t in timesteps:
                img = t["observation"]["image"].get(serial_key)
                if img is None:
                    break
                # DROID images come as BGRA or BGR from ZED; convert to RGB.
                if img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = img[..., ::-1].copy()  # BGR → RGB
                frames.append(_resize_image(img, image_h, image_w))
            if len(frames) == T:
                obs[obs_key] = np.stack(frames, axis=0).astype(np.uint8)

    # --- terminal flags ---
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = float(success)

    return {
        "actions": actions,
        "obs": obs,
        "dones": dones,
        "rewards": rewards,
        "success": success,
    }


def convert(
    input_path: str,
    output_path: str,
    wrist_serial: str,
    ext1_serial: str,
    ext2_serial: str,
    image_size: tuple[int, int],
    action_space: str,
    train_frac: float,
    val_frac: float,
) -> None:
    _add_droid_to_path()
    from droid.data_loading.trajectory_sampler import crawler

    image_h, image_w = image_size
    folders = crawler(input_path)
    print(f"Found {len(folders)} trajectory folders under {input_path}")

    demos: list[dict] = []
    for i, folder in enumerate(folders):
        print(f"  [{i+1}/{len(folders)}] {folder}")
        result = _convert_one_trajectory(
            folder, wrist_serial, ext1_serial, ext2_serial, image_h, image_w, action_space
        )
        if result is not None:
            demos.append(result)

    if not demos:
        raise RuntimeError("No valid trajectories found — check input_path and camera serials.")

    print(f"Converted {len(demos)} trajectories. Writing {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Shuffle and split.
    rng = np.random.default_rng(seed=0)
    idx = rng.permutation(len(demos))
    n_train = int(len(demos) * train_frac)
    n_val = int(len(demos) * val_frac)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train : n_train + n_val].tolist()
    test_idx = idx[n_train + n_val :].tolist()

    split_of = {}
    for i in train_idx:
        split_of[i] = "train"
    for i in val_idx:
        split_of[i] = "valid"
    for i in test_idx:
        split_of[i] = "test"

    total_steps = sum(len(d["actions"]) for d in demos)

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

            split = split_of.get(demo_idx, "train")
            split_lists[split].append(key)

        for split_name, keys in split_lists.items():
            if keys:
                dt = h5py.special_dtype(vlen=str)
                ds = mask_grp.create_dataset(split_name, (len(keys),), dtype=dt)
                for j, k in enumerate(keys):
                    ds[j] = k

    n_with_images = sum(1 for d in demos if any("image" in k for k in d["obs"]))
    print(
        f"Done. {len(demos)} demos ({total_steps} steps). "
        f"{n_with_images} have image obs. "
        f"train={len(split_lists['train'])} val={len(split_lists['valid'])} "
        f"test={len(split_lists['test'])}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Convert DROID trajectory.h5 files to robomimic HDF5")
    p.add_argument("--input_path", required=True, help="Root folder of DROID trajectories")
    p.add_argument("--output_path", required=True, help="Output HDF5 path")
    p.add_argument("--wrist_serial", default="14313307", help="ZED serial for wrist camera")
    p.add_argument("--ext1_serial", default="36716034", help="ZED serial for exterior camera 1")
    p.add_argument("--ext2_serial", default="37617599", help="ZED serial for exterior camera 2")
    p.add_argument("--image_size", nargs=2, type=int, default=[180, 320], metavar=("H", "W"))
    p.add_argument(
        "--action_space",
        default="joint_velocity",
        choices=["joint_velocity", "cartesian_velocity"],
    )
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--val_frac", type=float, default=0.05)
    args = p.parse_args()

    convert(
        input_path=args.input_path,
        output_path=args.output_path,
        wrist_serial=args.wrist_serial,
        ext1_serial=args.ext1_serial,
        ext2_serial=args.ext2_serial,
        image_size=(args.image_size[0], args.image_size[1]),
        action_space=args.action_space,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
