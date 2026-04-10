#!/usr/bin/env python3
"""Convert a LeRobot v2 dataset to robomimic-layout HDF5.

Outputs a file loadable by ``RobomimicReplayImageDataset`` (diffusion policy).
Images are stored as uint8 (T, H, W, C) with per-frame gzip+shuffle compression
(lossless, comparable ratios to PNG). State and action are stored as float32.

Layout written:
  data/
    demo_0/
      obs/
        {camera_key}          uint8  (T, H, W, C)  -- one per camera
        robot0_proprioception float32 (T, state_dim)
      actions                 float32 (T, action_dim)
    demo_1/ ...
  mask/
    train   dataset of episode indices (0 .. N-1)

Usage:
  python scripts/convert_lerobot_to_robomimic_hdf5.py \\
      --input  data/source/robocasa/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \\
      --output data/robocasa/PickPlaceCounterToCabinet.hdf5

  # Select a subset of cameras (default: all)
  python scripts/convert_lerobot_to_robomimic_hdf5.py \\
      --input  .../lerobot \\
      --output .../out.hdf5 \\
      --cameras robot0_agentview_right robot0_eye_in_hand

  # Limit episodes (useful for smoke tests)
  python scripts/convert_lerobot_to_robomimic_hdf5.py \\
      --input .../lerobot --output .../out.hdf5 --max-episodes 10
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional

import av
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _decode_video_pyav(video_path: pathlib.Path) -> np.ndarray:
    """Decode all frames from an MP4 file using PyAV.

    Returns:
        uint8 array of shape (T, H, W, C) in RGB order.
    """
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# LeRobot v2 helpers
# ---------------------------------------------------------------------------

def _episode_chunk(episode_idx: int, chunks_size: int) -> int:
    return episode_idx // chunks_size


def _load_episode_parquet(
    root: pathlib.Path,
    episode_idx: int,
    data_path_template: str,
    chunks_size: int,
) -> pd.DataFrame:
    chunk = _episode_chunk(episode_idx, chunks_size)
    rel = data_path_template.format(
        episode_chunk=chunk, episode_index=episode_idx
    )
    return pd.read_parquet(root / rel)


def _video_path(
    root: pathlib.Path,
    episode_idx: int,
    video_key: str,
    video_path_template: str,
    chunks_size: int,
) -> pathlib.Path:
    chunk = _episode_chunk(episode_idx, chunks_size)
    rel = video_path_template.format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=episode_idx,
    )
    return root / rel


def _lerobot_camera_keys(features: dict) -> List[str]:
    """Return feature keys whose dtype is 'video'."""
    return [k for k, v in features.items() if v.get("dtype") == "video"]


def _camera_hdf5_name(lerobot_key: str) -> str:
    """Strip 'observation.images.' prefix and add '_image' suffix → robomimic convention."""
    prefix = "observation.images."
    if lerobot_key.startswith(prefix):
        return lerobot_key[len(prefix):] + "_image"
    return lerobot_key.replace(".", "_") + "_image"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(
    lerobot_root: pathlib.Path,
    output_path: pathlib.Path,
    cameras: Optional[List[str]] = None,
    state_key: str = "observation.state",
    action_key: str = "action",
    max_episodes: Optional[int] = None,
    gzip_level: int = 4,
) -> None:
    """Convert a LeRobot v2 dataset to robomimic HDF5.

    Args:
        lerobot_root: Path to the ``.../lerobot`` directory (contains meta/, data/, videos/).
        output_path: Destination ``.hdf5`` file.
        cameras: LeRobot video feature keys to include (e.g. ``["observation.images.robot0_agentview_right"]``).
                 ``None`` = all video keys in the dataset.
        state_key: Parquet column to use as low-dim proprioception obs.
        action_key: Parquet column to use as actions.
        max_episodes: If set, convert only the first N episodes.
        gzip_level: HDF5 gzip compression level (1-9). 4 is a good balance.
    """
    lerobot_root = pathlib.Path(lerobot_root).resolve()
    output_path = pathlib.Path(output_path).resolve()

    # ---- load metadata ----
    info_path = lerobot_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes: int = info["total_episodes"]
    chunks_size: int = info["chunks_size"]
    data_path_tmpl: str = info["data_path"]
    video_path_tmpl: str = info["video_path"]
    features: dict = info["features"]
    fps: float = info.get("fps", 20)

    all_camera_keys = _lerobot_camera_keys(features)
    if not all_camera_keys:
        raise ValueError("No video features found in info.json")

    if cameras is None:
        selected_camera_keys = all_camera_keys
    else:
        # Accept either full keys or short names
        selected_camera_keys = []
        for cam in cameras:
            if cam in all_camera_keys:
                selected_camera_keys.append(cam)
            else:
                full = f"observation.images.{cam}"
                if full in all_camera_keys:
                    selected_camera_keys.append(full)
                else:
                    raise ValueError(
                        f"Camera {cam!r} not found. Available: {all_camera_keys}"
                    )

    n_episodes = min(total_episodes, max_episodes) if max_episodes else total_episodes

    print(f"LeRobot root : {lerobot_root}")
    print(f"Output       : {output_path}")
    print(f"Episodes     : {n_episodes} / {total_episodes}")
    print(f"Cameras      : {[_camera_hdf5_name(k) for k in selected_camera_keys]}")
    print(f"FPS          : {fps}")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        mask_grp = f.create_group("mask")

        episode_indices = list(range(n_episodes))
        total_steps = 0

        for ep_idx in tqdm(episode_indices, desc="Converting episodes"):
            df = _load_episode_parquet(
                lerobot_root, ep_idx, data_path_tmpl, chunks_size
            )
            T = len(df)
            total_steps += T

            demo_grp = data_grp.create_group(f"demo_{ep_idx}")
            demo_grp.attrs["num_samples"] = T
            obs_grp = demo_grp.create_group("obs")

            # ---- actions ----
            actions = np.stack(df[action_key].values).astype(np.float32)
            demo_grp.create_dataset(
                "actions",
                data=actions,
                compression="gzip",
                compression_opts=gzip_level,
                shuffle=True,
            )

            # ---- low-dim state ----
            state = np.stack(df[state_key].values).astype(np.float32)
            obs_grp.create_dataset(
                "robot0_proprioception",
                data=state,
                compression="gzip",
                compression_opts=gzip_level,
                shuffle=True,
            )

            # ---- images ----
            for cam_key in selected_camera_keys:
                hdf5_name = _camera_hdf5_name(cam_key)
                vpath = _video_path(
                    lerobot_root, ep_idx, cam_key, video_path_tmpl, chunks_size
                )
                frames = _decode_video_pyav(vpath)  # (T_video, H, W, C)

                # Trim/pad to match parquet length (should be equal, but guard anyway)
                if frames.shape[0] != T:
                    if frames.shape[0] > T:
                        frames = frames[:T]
                    else:
                        pad = np.repeat(frames[-1:], T - frames.shape[0], axis=0)
                        frames = np.concatenate([frames, pad], axis=0)

                _, H, W, C = frames.shape
                obs_grp.create_dataset(
                    hdf5_name,
                    data=frames,
                    shape=(T, H, W, C),
                    dtype=np.uint8,
                    chunks=(1, H, W, C),       # one chunk per frame
                    compression="gzip",
                    compression_opts=gzip_level,
                    shuffle=True,
                )

        # ---- masks ----
        mask_grp.create_dataset(
            "train",
            data=np.arange(n_episodes, dtype=np.int32),
        )

        # ---- top-level attributes ----
        f.attrs["total"] = n_episodes
        f.attrs["total_steps"] = total_steps
        f.attrs["fps"] = fps
        f.attrs["source"] = "lerobot_v2"
        f.attrs["lerobot_root"] = str(lerobot_root)

    # ---- print shape_meta snippet ----
    df0 = _load_episode_parquet(lerobot_root, 0, data_path_tmpl, chunks_size)
    state_dim = np.array(df0[state_key].iloc[0]).shape[0]
    action_dim = np.array(df0[action_key].iloc[0]).shape[0]
    cam_feature = features[selected_camera_keys[0]]
    H, W, C = cam_feature["shape"]

    print(f"\nDone. {n_episodes} episodes → {output_path}")
    print(f"  total_steps : {total_steps}")
    print(f"  state_dim   : {state_dim}")
    print(f"  action_dim  : {action_dim}")
    print(f"  image shape : ({H}, {W}, {C})")
    print()
    print("── shape_meta snippet for your Hydra config ──────────────────────────")
    print("shape_meta:")
    print("  obs:")
    print("    robot0_proprioception:")
    print(f"      shape: [{state_dim}]")
    print("      type: low_dim")
    for cam_key in selected_camera_keys:
        name = _camera_hdf5_name(cam_key)
        feat = features[cam_key]
        h, w, c = feat["shape"]
        print(f"    {name}:")
        print(f"      shape: [{c}, {h}, {w}]")
        print("      type: rgb")
    print("  action:")
    print(f"    shape: [{action_dim}]")
    print("───────────────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True,
                   help="Path to the LeRobot dataset root (.../lerobot)")
    p.add_argument("--output", "-o", default=None,
                   help="Destination HDF5 file path (default: <input>/robomimic.hdf5)")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Camera keys to include (short name or full observation.images.* key). "
                        "Default: all cameras in the dataset.")
    p.add_argument("--state-key", default="observation.state",
                   help="Parquet column for proprioceptive state (default: observation.state)")
    p.add_argument("--action-key", default="action",
                   help="Parquet column for actions (default: action)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Convert only the first N episodes (useful for smoke tests)")
    p.add_argument("--gzip-level", type=int, default=4, choices=range(1, 10),
                   help="HDF5 gzip compression level 1-9 (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = args.output or str(pathlib.Path(args.input) / "robomimic.hdf5")
    convert(
        lerobot_root=args.input,
        output_path=output,
        cameras=args.cameras,
        state_key=args.state_key,
        action_key=args.action_key,
        max_episodes=args.max_episodes,
        gzip_level=args.gzip_level,
    )
