"""Convert DROID trajectory.h5 + SVO2 recordings → robomimic HDF5 for training.

Reads state observations directly from HDF5 and real images from ZED SVO2 files
using pyzed. Requires the ZED SDK shared libraries on LD_LIBRARY_PATH.

Camera mapping:
    hand_camera_image       ← <wrist_serial>.svo2  (LEFT view)
    exterior_image_1_left   ← <ext1_serial>.svo2   (LEFT view)

SVO2 frames are 1:1 with HDF5 steps (same count, ~constant timestamp offset).
movement_enabled=False steps are filtered out from both state and images.

Demos are streamed directly to HDF5 one at a time to keep RAM flat, regardless
of dataset size. Parallel workers (--num_workers) decode SVO2 frames concurrently;
the main process writes results to HDF5 sequentially.

Usage:
    LD_LIBRARY_PATH=/mnt/ssdB/erik/zed_sdk_extracted/lib \\
    conda run -n cupid_torch2 \\
    python scripts/convert_droid_hdf5_debug.py \\
        --input_path /mnt/ssdB/erik/droid_data/data/success/2026-05-13 \\
        --output_path /mnt/ssdB/erik/droid_data/kendama_may13.hdf5 \\
        --image_size 256 256 \\
        --num_workers 4
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

WRIST_SERIAL = "14313307"
EXT1_SERIAL = "36716034"


def _find_trajectories(root: str) -> list[str]:
    folders = []
    for dirpath, _, files in os.walk(root):
        if "trajectory.h5" in files:
            folders.append(dirpath)
    return sorted(folders)


def _read_svo_images(
    svo_path: str,
    valid_mask: np.ndarray,
    image_h: int,
    image_w: int,
    zed_settings: str,
) -> np.ndarray | None:
    """Read LEFT images from an SVO2 file, filter by valid_mask, resize.

    Returns uint8 array (T_valid, H, W, 3) or None if SVO can't be opened.
    """
    try:
        import pyzed.sl as sl
    except ImportError:
        raise RuntimeError(
            "pyzed not found. Install it and set LD_LIBRARY_PATH to the ZED SDK lib dir."
        )

    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.optional_settings_path = zed_settings

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"    [warn] SVO open failed ({status}): {svo_path}")
        return None

    rt = sl.RuntimeParameters()
    img_mat = sl.Mat()
    frames = []
    frame_idx = 0

    while True:
        grab_status = cam.grab(rt)
        if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if grab_status != sl.ERROR_CODE.SUCCESS:
            break

        if frame_idx < len(valid_mask) and valid_mask[frame_idx]:
            cam.retrieve_image(img_mat, sl.VIEW.LEFT)
            bgra = img_mat.get_data()          # (H, W, 4) uint8 BGRA
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            if rgb.shape[:2] != (image_h, image_w):
                rgb = cv2.resize(rgb, (image_w, image_h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)

        frame_idx += 1

    cam.close()

    if not frames:
        return None
    return np.stack(frames, axis=0).astype(np.uint8)


def _load_trajectory(
    folder: str,
    image_h: int,
    image_w: int,
    zed_settings: str,
) -> dict | None:
    traj_file = os.path.join(folder, "trajectory.h5")
    try:
        with h5py.File(traj_file, "r") as f:
            success = bool(f.attrs.get("success", False))
            movement = f["observation/controller_info/movement_enabled"][:].astype(bool)

            jv = f["action/joint_velocity"][:][movement].astype(np.float32)
            gp_action = f["action/gripper_position"][:][movement].astype(np.float32)[:, None]
            actions = np.concatenate([jv, gp_action], axis=-1)

            T = len(actions)
            if T == 0:
                return None

            joint_pos = f["observation/robot_state/joint_positions"][:][movement].astype(np.float32)
            cart_pos = f["observation/robot_state/cartesian_position"][:][movement].astype(np.float32)
            gripper_pos = f["observation/robot_state/gripper_position"][:][movement].astype(np.float32)[:, None]

    except Exception as e:
        print(f"  [skip] {folder}: {e}")
        return None

    svo_dir = os.path.join(folder, "recordings", "SVO")

    def _read(serial: str) -> np.ndarray | None:
        svo_path = os.path.join(svo_dir, f"{serial}.svo2")
        if not os.path.isfile(svo_path):
            print(f"    [warn] SVO not found: {svo_path}")
            return None
        return _read_svo_images(svo_path, movement, image_h, image_w, zed_settings)

    print(f"    reading wrist SVO ({WRIST_SERIAL})...")
    wrist_imgs = _read(WRIST_SERIAL)
    print(f"    reading ext1 SVO ({EXT1_SERIAL})...")
    ext1_imgs = _read(EXT1_SERIAL)

    if wrist_imgs is None or ext1_imgs is None:
        print(f"  [skip] {folder}: failed to read images")
        return None

    if len(wrist_imgs) != T or len(ext1_imgs) != T:
        print(f"  [skip] {folder}: image count mismatch "
              f"(wrist={len(wrist_imgs)} ext1={len(ext1_imgs)} state={T})")
        return None

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
            "hand_camera_image": wrist_imgs,
            "exterior_image_1_left": ext1_imgs,
        },
        "dones": dones,
        "rewards": rewards,
        "success": success,
    }


def _worker(args: tuple) -> tuple[int, dict | None]:
    """Top-level worker callable for ProcessPoolExecutor (must be picklable)."""
    folder_idx, folder, image_h, image_w, zed_settings = args
    result = _load_trajectory(folder, image_h, image_w, zed_settings)
    return folder_idx, result


def convert(
    input_path: str,
    output_path: str,
    image_h: int,
    image_w: int,
    zed_settings: str,
    train_frac: float,
    val_frac: float,
    num_workers: int,
) -> None:
    folders = _find_trajectories(input_path)
    n_folders = len(folders)
    print(f"Found {n_folders} trajectory folders under {input_path}")

    # Pre-compute split assignments before we start reading SVO files.
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(n_folders).tolist()
    n_train = max(1, int(n_folders * train_frac))
    n_val = max(1, int(n_folders * val_frac))
    folder_split: dict[int, str] = {}
    for j in perm[:n_train]:
        folder_split[j] = "train"
    for j in perm[n_train : n_train + n_val]:
        folder_split[j] = "valid"
    for j in perm[n_train + n_val :]:
        folder_split[j] = "test"
    for j in range(n_folders):
        folder_split.setdefault(j, "train")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    split_lists: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    total_steps = 0
    demo_idx = 0

    worker_args = [
        (folder_idx, folder, image_h, image_w, zed_settings)
        for folder_idx, folder in enumerate(folders)
    ]

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        mask_grp = f.create_group("mask")

        if num_workers > 1:
            # Parallel decode: workers read SVO frames concurrently.
            # Results arrive out of order; buffer and write in folder_idx order.
            pending: dict[int, dict | None] = {}
            next_to_write = 0

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_worker, arg): arg[0] for arg in worker_args}

                for future in as_completed(futures):
                    folder_idx, result = future.result()
                    pending[folder_idx] = result

                    # Flush contiguous completed results in order.
                    while next_to_write in pending:
                        res = pending.pop(next_to_write)
                        folder = folders[next_to_write]
                        print(f"  [{next_to_write+1}/{n_folders}] {folder}")
                        if res is not None:
                            key = f"demo_{demo_idx}"
                            _write_demo(data_grp, key, res)
                            split_lists[folder_split[next_to_write]].append(key)
                            total_steps += len(res["actions"])
                            demo_idx += 1
                            del res
                        next_to_write += 1
        else:
            # Sequential mode — simpler, no pickling overhead.
            for folder_idx, folder in enumerate(folders):
                print(f"  [{folder_idx+1}/{n_folders}] {folder}")
                result = _load_trajectory(folder, image_h, image_w, zed_settings)
                if result is not None:
                    key = f"demo_{demo_idx}"
                    _write_demo(data_grp, key, result)
                    split_lists[folder_split[folder_idx]].append(key)
                    total_steps += len(result["actions"])
                    demo_idx += 1
                    del result

        if demo_idx == 0:
            raise RuntimeError("No valid trajectories found.")

        data_grp.attrs["total"] = total_steps

        dt = h5py.special_dtype(vlen=str)
        for split_name, keys in split_lists.items():
            if keys:
                ds = mask_grp.create_dataset(split_name, (len(keys),), dtype=dt)
                for j, k in enumerate(keys):
                    ds[j] = k

    print(
        f"\nDone. {demo_idx} demos ({total_steps} steps) → {output_path}\n"
        f"  train={len(split_lists['train'])} "
        f"valid={len(split_lists['valid'])} "
        f"test={len(split_lists['test'])}\n"
        f"  images: real RGB from SVO2, resized to {image_h}x{image_w}"
    )


def _write_demo(data_grp: h5py.Group, key: str, demo: dict) -> None:
    dgrp = data_grp.create_group(key)
    dgrp.create_dataset("actions", data=demo["actions"])
    dgrp.create_dataset("dones", data=demo["dones"])
    dgrp.create_dataset("rewards", data=demo["rewards"])
    obs_grp = dgrp.create_group("obs")
    for obs_key, arr in demo["obs"].items():
        obs_grp.create_dataset(obs_key, data=arr)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert DROID trajectory.h5 + SVO2 files to robomimic HDF5"
    )
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument(
        "--zed_settings",
        default="/mnt/ssdB/erik/zed_settings",
        help="Directory containing SN<serial>.conf calibration files",
    )
    p.add_argument("--image_size", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    p.add_argument("--train_frac", type=float, default=0.85)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument(
        "--num_workers", type=int, default=4,
        help="Parallel SVO decode workers (set 1 to disable multiprocessing)",
    )
    args = p.parse_args()
    convert(
        input_path=args.input_path,
        output_path=args.output_path,
        image_h=args.image_size[0],
        image_w=args.image_size[1],
        zed_settings=args.zed_settings,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
