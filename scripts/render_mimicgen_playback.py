"""Render simulation playback videos from MimicGen-generated demo HDF5 files.

Replays states from demo.hdf5 and demo_failed.hdf5 through the robosuite
simulator and saves MP4 videos — one for successful demos, one for failures.

Run in the mimicgen conda env:
    conda run -n mimicgen --no-capture-output python scripts/render_mimicgen_playback.py

Outputs (written to --out_dir, default /tmp/mimicgen_eef_test):
    playback_success.mp4   — first --n_success successful demos side-by-side
    playback_failed.mp4    — first --n_failed failed demos side-by-side
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
_MG = _REPO / "third_party" / "mimicgen"
if _MG.is_dir():
    sys.path.insert(0, str(_MG))

os.environ.setdefault("MUJOCO_GL", "egl")

import mimicgen  # noqa: F401 — registers MimicGen environments with robosuite
import h5py
import imageio
import numpy as np
import robomimic.utils.obs_utils as ObsUtils

# Initialize obs utils before any env creation (required by robomimic internals)
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(
    obs=dict(low_dim=["robot0_eef_pos"], rgb=[])
))


def _patch_base_env() -> None:
    try:
        import robomimic.envs.env_robosuite as er
        if not hasattr(er.EnvRobosuite, "base_env"):
            er.EnvRobosuite.base_env = property(lambda self: self.env)
    except ImportError:
        pass


def _make_env(env_args: dict, camera_names: list[str], camera_height: int, camera_width: int):
    """Create a robosuite environment with offscreen rendering enabled."""
    import robomimic.utils.env_utils as eu

    kwargs = dict(env_args.get("env_kwargs", {}))
    kwargs["has_offscreen_renderer"] = True
    kwargs["has_renderer"] = False
    kwargs["use_camera_obs"] = True
    kwargs["camera_names"] = camera_names
    kwargs["camera_heights"] = camera_height
    kwargs["camera_widths"] = camera_width
    kwargs["camera_depths"] = False

    patched = dict(env_args)
    patched["env_kwargs"] = kwargs

    env = eu.create_env_from_metadata(
        env_meta=patched,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    return env


def _render_demo(env, states: np.ndarray, camera_names: list[str]) -> list[np.ndarray]:
    """Replay states through the env and return list of RGB frames (H, W*n_cams, 3)."""
    frames: list[np.ndarray] = []
    for t, state in enumerate(states):
        env.reset_to({"states": state})
        obs = env.get_observation()
        # Stitch cameras horizontally
        imgs = []
        for cam in camera_names:
            key = f"{cam}_image"
            if key in obs:
                img = obs[key]
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                imgs.append(img)
        if imgs:
            frame = np.concatenate(imgs, axis=1)  # side-by-side
            frames.append(frame)
    return frames


def _render_hdf5(
    hdf5_path: Path,
    env_args: dict,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    n_demos: int,
    fps: int,
) -> list[list[np.ndarray]]:
    """Render up to n_demos from hdf5_path. Returns list of per-demo frame lists."""
    env = _make_env(env_args, camera_names, camera_height, camera_width)
    results = []
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[:n_demos]
        for i, key in enumerate(demo_keys):
            states = np.array(f[f"data/{key}/states"])
            print(f"  rendering {key} ({len(states)} steps) ...")
            frames = _render_demo(env, states, camera_names)
            results.append(frames)
            print(f"  {key}: {len(frames)} frames")
    try:
        env.close()
    except AttributeError:
        pass
    return results


def _write_video(
    per_demo_frames: list[list[np.ndarray]],
    output_path: Path,
    fps: int,
    separator_width: int = 4,
) -> None:
    """Concatenate per-demo frame sequences with a black separator and write MP4."""
    if not per_demo_frames:
        print(f"  [skip] no frames to write for {output_path.name}")
        return

    # Build timeline: all demos in sequence, separated by a brief pause frame
    all_frames: list[np.ndarray] = []
    for i, demo_frames in enumerate(per_demo_frames):
        if not demo_frames:
            continue
        all_frames.extend(demo_frames)
        # 0.5 s black pause between demos
        h, w = demo_frames[0].shape[:2]
        pause = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(max(1, fps // 2)):
            all_frames.append(pause)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p")
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    print(f"  written: {output_path}  ({len(all_frames)} frames @ {fps} fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_hdf5", default="/tmp/mimicgen_eef_test/demo.hdf5")
    ap.add_argument("--failed_hdf5", default="/tmp/mimicgen_eef_test/demo_failed.hdf5")
    ap.add_argument("--out_dir", default="/tmp/mimicgen_eef_test")
    ap.add_argument("--n_success", type=int, default=8,
                    help="Number of successful demos to render")
    ap.add_argument("--n_failed", type=int, default=8,
                    help="Number of failed demos to render")
    ap.add_argument("--cameras", nargs="+",
                    default=["agentview", "robot0_eye_in_hand"],
                    help="Camera names to render")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    _patch_base_env()

    demo_hdf5 = Path(args.demo_hdf5)
    failed_hdf5 = Path(args.failed_hdf5)
    out_dir = Path(args.out_dir)

    with h5py.File(demo_hdf5, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    # --- Successful demos ---
    print(f"\n[render] Successful demos ({args.n_success}) from {demo_hdf5.name}")
    succ_frames = _render_hdf5(
        demo_hdf5, env_args, args.cameras, args.height, args.width, args.n_success, args.fps
    )
    _write_video(succ_frames, out_dir / "playback_success.mp4", args.fps)

    # --- Failed demos ---
    if failed_hdf5.exists():
        print(f"\n[render] Failed demos ({args.n_failed}) from {failed_hdf5.name}")
        fail_frames = _render_hdf5(
            failed_hdf5, env_args, args.cameras, args.height, args.width, args.n_failed, args.fps
        )
        _write_video(fail_frames, out_dir / "playback_failed.mp4", args.fps)
    else:
        print(f"[render] no demo_failed.hdf5 found at {failed_hdf5}")

    print("\n[render] done.")


if __name__ == "__main__":
    main()
