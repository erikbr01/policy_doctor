from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import re
from typing import List, Optional

import imageio
import numpy as np


def _normalize_frame(frame) -> Optional[np.ndarray]:
    frame = np.array(frame, copy=True)

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    if (
        frame.ndim == 3
        and frame.shape[0] in (1, 3, 4)
        and frame.shape[2] not in (1, 3, 4)
    ):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.concatenate([frame] * 3, axis=-1)

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        print(f"Warning: unexpected frame shape {frame.shape}, skipping")
        return None

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    return np.ascontiguousarray(frame)


def _write_video(frames: List[np.ndarray], output_path: pathlib.Path, fps: int):
    if not frames:
        return

    h, w = frames[0].shape[:2]
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    if new_h != h or new_w != w:
        padded = []
        for f in frames:
            p = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            p[:h, :w] = f
            padded.append(p)
        frames = padded

    imageio.mimwrite(str(output_path), frames, fps=fps, format="ffmpeg")


def _episode_number(path: pathlib.Path) -> int:
    # handles ep0000_succ.pkl, ep0000_fail.pkl, ep0.pkl, ep42.pkl
    m = re.search(r"ep(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _success_from_filename(path: pathlib.Path) -> Optional[bool]:
    name = path.stem  # e.g. ep0000_succ, ep0001_fail
    if "_succ" in name:
        return True
    if "_fail" in name:
        return False
    return None


def _load_frames_and_success(
    pkl_path: pathlib.Path, obs_key: str
) -> tuple[List[np.ndarray], Optional[bool]]:
    """Load frames and success flag from a pickle that may be a dict or DataFrame."""
    import pickle as _pickle
    with open(pkl_path, "rb") as f:
        data = _pickle.load(f)

    # DataFrame format (e.g. transport_mh_jan28)
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            frames_raw = list(data[obs_key]) if obs_key in data.columns else []
            success = _success_from_filename(pkl_path)
            if success is None and "success" in data.columns:
                s = data["success"].iloc[-1]
                success = bool(s) if s is not None else None
            return frames_raw, success
    except ImportError:
        pass

    # Dict format
    if isinstance(data, dict):
        raw = data.get(obs_key)
        if raw is None:
            return [], None
        if hasattr(raw, "tolist"):
            frames_raw = list(raw)
        elif hasattr(raw, "__iter__") and not isinstance(raw, np.ndarray):
            frames_raw = list(raw)
        else:
            frames_raw = [raw[i] for i in range(len(raw))]
        success = _success_from_filename(pkl_path)
        if success is None:
            s = data.get("success")
            success = bool(s) if s is not None else None
        return frames_raw, success

    return [], None


def main():
    parser = argparse.ArgumentParser(
        description="Render rollout episode pickles to MP4 files."
    )
    parser.add_argument("--eval_dir", required=True, type=pathlib.Path)
    parser.add_argument("--output_dir", required=True, type=pathlib.Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--obs_key", type=str, default="img")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    episodes_dir = args.eval_dir / "episodes"
    pkl_paths = sorted(episodes_dir.glob("ep*.pkl"), key=_episode_number)

    if args.max_episodes is not None:
        pkl_paths = pkl_paths[: args.max_episodes]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_entries = []

    for pkl_path in pkl_paths:
        ep_num = _episode_number(pkl_path)
        mp4_name = f"ep{ep_num}.mp4"
        mp4_path = args.output_dir / mp4_name

        if mp4_path.exists() and not args.overwrite:
            print(f"Skipping {mp4_name} (already exists)")
            try:
                frames_raw, success = _load_frames_and_success(pkl_path, args.obs_key)
                frame_count = len(frames_raw)
            except Exception as e:
                print(f"  Warning: could not read for index: {e}")
                frame_count = 0
                success = _success_from_filename(pkl_path)
            index_entries.append(
                {"index": ep_num, "path": mp4_name, "frame_count": frame_count, "success": success}
            )
            continue

        print(f"Processing {pkl_path.name} -> {mp4_name}")

        try:
            frames_raw, success = _load_frames_and_success(pkl_path, args.obs_key)
        except Exception as e:
            print(f"  Warning: failed to load: {e}, skipping")
            continue

        if not frames_raw:
            print(f"  Warning: no frames found (key '{args.obs_key}'), skipping")
            continue

        frames: List[np.ndarray] = []
        for frame in frames_raw:
            normalized = _normalize_frame(frame)
            if normalized is not None:
                frames.append(normalized)

        if not frames:
            print(f"  Warning: no valid frames after normalization, skipping")
            continue

        try:
            _write_video(frames, mp4_path, args.fps)
        except Exception as e:
            print(f"  Warning: failed to write video: {e}, skipping")
            continue

        print(f"  Wrote {len(frames)} frames  success={success}")
        index_entries.append(
            {"index": ep_num, "path": mp4_name, "frame_count": len(frames), "success": success}
        )

    index = {
        "episodes": index_entries,
        "fps": args.fps,
        "eval_dir": str(args.eval_dir.resolve()),
    }
    index_path = args.output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote index with {len(index_entries)} episodes -> {index_path}")


if __name__ == "__main__":
    main()
