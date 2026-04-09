"""Render individual demonstration videos from an HDF5 dataset.

This script extends robomimic's playback_dataset.py to generate one video
per demonstration episode, which is required for the influence visualizer.

Usage:
    python influence_visualizer/scripts/render_demo_videos.py \
        --dataset data/robomimic/datasets/lift/mh/low_dim.hdf5 \
        --output_dir data/outputs/demo_videos/lift_mh \
        --render_image_names agentview

This will create:
    data/outputs/demo_videos/lift_mh/
        demo_0000.mp4
        demo_0001.mp4
        ...
        metadata.json
"""

import argparse
import json
import os
import pathlib
import sys

import h5py
import imageio
import numpy as np
import tqdm

# Add robomimic to path
sys.path.insert(
    0, str(pathlib.Path(__file__).parent.parent.parent / "third_party" / "robomimic")
)

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase


def render_single_trajectory(
    env: EnvBase,
    initial_state: dict,
    states: np.ndarray,
    video_path: str,
    camera_names: list,
    video_skip: int = 5,
    fps: int = 20,
    height: int = 256,
    width: int = 256,
) -> dict:
    """Render a single trajectory to a video file.

    Args:
        env: The simulation environment.
        initial_state: Initial state dict to reset to.
        states: Array of states to playback.
        video_path: Output path for the video.
        camera_names: Camera names to render.
        video_skip: Render every N frames.
        fps: Video frames per second.
        height: Render height.
        width: Render width.

    Returns:
        Metadata dict with episode info.
    """
    env.reset_to(initial_state)

    video_writer = imageio.get_writer(video_path, fps=fps)
    frame_count = 0

    for i in range(states.shape[0]):
        env.reset_to({"states": states[i]})

        if frame_count % video_skip == 0:
            video_img = []
            for cam_name in camera_names:
                video_img.append(
                    env.render(
                        mode="rgb_array",
                        height=height,
                        width=width,
                        camera_name=cam_name,
                    )
                )
            video_img = np.concatenate(video_img, axis=1)
            video_writer.append_data(video_img)
        frame_count += 1

    video_writer.close()

    return {
        "num_states": int(states.shape[0]),
        "num_frames": int(
            frame_count // video_skip + (1 if frame_count % video_skip else 0)
        ),
    }


def render_demo_videos(
    dataset_path: str,
    output_dir: str,
    camera_names: list = None,
    video_skip: int = 5,
    fps: int = 20,
    height: int = 256,
    width: int = 256,
    n_demos: int = None,
    start_idx: int = 0,
) -> None:
    """Render all demonstrations from a dataset to individual video files.

    Args:
        dataset_path: Path to the HDF5 dataset.
        output_dir: Directory to save videos.
        camera_names: Camera names to render (default: agentview).
        video_skip: Render every N frames.
        fps: Video frames per second.
        height: Render height.
        width: Render width.
        n_demos: Number of demos to render (None = all).
        start_idx: Starting demonstration index.
    """
    if camera_names is None:
        camera_names = ["agentview"]

    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize observation utils with dummy spec
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    # Create environment
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
    )
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # Open dataset
    f = h5py.File(dataset_path, "r")

    # Get sorted list of demonstrations
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # Limit demos if specified
    end_idx = len(demos) if n_demos is None else min(start_idx + n_demos, len(demos))
    demos = demos[start_idx:end_idx]

    print(f"Rendering {len(demos)} demonstrations from {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Camera names: {camera_names}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Skip: {video_skip}")

    metadata = {
        "dataset_path": str(dataset_path),
        "camera_names": camera_names,
        "video_skip": video_skip,
        "fps": fps,
        "height": height,
        "width": width,
        "episodes": {},
    }

    for demo in tqdm.tqdm(demos, desc="Rendering demos"):
        demo_idx = int(demo[5:])  # Extract index from "demo_X"
        video_path = output_dir / f"demo_{demo_idx:04d}.mp4"

        # Get states and initial state
        states = f[f"data/{demo}/states"][()]
        initial_state = {"states": states[0]}
        if is_robosuite_env:
            initial_state["model"] = f[f"data/{demo}"].attrs["model_file"]
            initial_state["ep_meta"] = f[f"data/{demo}"].attrs.get("ep_meta", None)

        # Render the trajectory
        episode_meta = render_single_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            video_path=str(video_path),
            camera_names=camera_names,
            video_skip=video_skip,
            fps=fps,
            height=height,
            width=width,
        )

        metadata["episodes"][demo_idx] = {
            "video_file": f"demo_{demo_idx:04d}.mp4",
            **episode_meta,
        }

    f.close()

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")
    print(f"Rendered {len(demos)} demonstration videos")


def main():
    parser = argparse.ArgumentParser(
        description="Render individual demonstration videos from an HDF5 dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to HDF5 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save videos",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="Camera name(s) to render",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="Render every N frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frames per second",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Render height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Render width",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of demonstrations to render (default: all)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting demonstration index",
    )

    args = parser.parse_args()

    render_demo_videos(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        camera_names=args.render_image_names,
        video_skip=args.video_skip,
        fps=args.fps,
        height=args.height,
        width=args.width,
        n_demos=args.n,
        start_idx=args.start_idx,
    )


if __name__ == "__main__":
    main()
