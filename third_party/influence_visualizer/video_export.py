"""Video export utilities for the influence visualizer.

This module provides functions to export video clips of rollout and demo slices
for offline analysis.
"""

import pathlib
from typing import List, Optional, Tuple

import imageio
import numpy as np

from influence_visualizer.data_loader import InfluenceData


def _normalize_frame(frame) -> Optional[np.ndarray]:
    """Normalize a frame to a uint8 (H, W, 3) contiguous numpy array.

    Handles CHW->HWC transpose, dtype conversion, and memory layout.
    Returns None if the frame can't be normalized.
    """
    frame = np.array(frame, copy=True)  # force a concrete numpy copy

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    # Handle (C, H, W) -> (H, W, C)
    if (
        frame.ndim == 3
        and frame.shape[0] in (1, 3, 4)
        and frame.shape[2] not in (1, 3, 4)
    ):
        frame = np.transpose(frame, (1, 2, 0))

    # Ensure (H, W, 3) RGB
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.concatenate([frame] * 3, axis=-1)

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        print(f"Warning: unexpected frame shape {frame.shape}, skipping")
        return None

    # Ensure RGBA -> RGB
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    return np.ascontiguousarray(frame)


def _write_video(frames: List[np.ndarray], output_path: pathlib.Path, fps: int):
    """Write frames to an mp4 video file.

    Pads frames to even dimensions (h264 requirement) and uses
    imageio.mimwrite for reliable single-shot encoding.
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]
    # h264 requires even dimensions
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    if new_h != h or new_w != w:
        padded = []
        for f in frames:
            p = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            p[:h, :w] = f
            padded.append(p)
        frames = padded

    imageio.mimwrite(str(output_path), frames, fps=fps, codec="h264")


def export_slice_videos(
    data: InfluenceData,
    rollout_episode_idx: int,
    rollout_start_offset: int,
    rollout_end_offset: int,
    demo_influences: List[dict],
    output_dir: pathlib.Path,
    task_config_name: str,
    obs_key: str = "agentview_image",
    fps: int = 10,
    demo_window_width: Optional[int] = None,
) -> Tuple[List[pathlib.Path], Optional[str]]:
    """Export videos of a rollout slice and its top-k influential demo slices.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode
        rollout_start_offset: Start offset within the episode (inclusive)
        rollout_end_offset: End offset within the episode (exclusive)
        demo_influences: List of influence dicts (from render_influences functions)
        output_dir: Base output directory (e.g., outputs/slice_exports)
        task_config_name: Name of the task config for file naming
        obs_key: Observation key for images (for demos)
        fps: Frames per second for the output videos
        demo_window_width: Window width for demo slices (if None, uses full buffer length)

    Returns:
        Tuple of (list of exported video paths, error message if any)
    """
    # Create output directory structure
    rollout_ep = data.rollout_episodes[rollout_episode_idx]
    slice_name = (
        f"{task_config_name}_rollout{rollout_ep.index}_"
        f"slice{rollout_start_offset}_{rollout_end_offset}_"
        f"top{len(demo_influences)}"
    )
    slice_dir = output_dir / slice_name
    slice_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    error_msg = None

    try:
        # Export full rollout video
        full_rollout_path = (
            slice_dir
            / f"rollout_ep{rollout_ep.index}_full_0_{rollout_ep.num_samples}.mp4"
        )
        success = _export_rollout_slice_video(
            data=data,
            rollout_episode_idx=rollout_episode_idx,
            start_offset=0,
            end_offset=rollout_ep.num_samples,
            output_path=full_rollout_path,
            fps=fps,
        )
        if success:
            exported_paths.append(full_rollout_path)

        # Export rollout slice video
        rollout_path = (
            slice_dir
            / f"rollout_ep{rollout_ep.index}_{rollout_start_offset}_{rollout_end_offset}.mp4"
        )
        success = _export_rollout_slice_video(
            data=data,
            rollout_episode_idx=rollout_episode_idx,
            start_offset=rollout_start_offset,
            end_offset=rollout_end_offset,
            output_path=rollout_path,
            fps=fps,
        )
        if success:
            exported_paths.append(rollout_path)
        else:
            error_msg = "Failed to export rollout slice video"

        # Export demo slice videos
        quality_labels = data.demo_quality_labels
        for rank, influence in enumerate(demo_influences, start=1):
            demo_ep_idx = influence["demo_episode_idx"]
            demo_sample_idx = influence["global_demo_sample_idx"]
            sample_info = influence["sample_info"]

            # Calculate demo slice bounds
            demo_start = sample_info.timestep
            demo_end = demo_start + (
                sample_info.buffer_end_idx - sample_info.buffer_start_idx
            )

            # Include quality label in filename if available
            quality_suffix = ""
            if quality_labels is not None and demo_ep_idx in quality_labels:
                quality_suffix = f"_{quality_labels[demo_ep_idx]}"

            demo_path = (
                slice_dir
                / f"demo_ep{demo_ep_idx}_{demo_start}_{demo_end}_rank{rank}{quality_suffix}.mp4"
            )

            success = _export_demo_slice_video(
                data=data,
                demo_sample_idx=demo_sample_idx,
                output_path=demo_path,
                obs_key=obs_key,
                fps=fps,
                window_width=demo_window_width,
            )
            if success:
                exported_paths.append(demo_path)

    except Exception as e:
        error_msg = f"Error during export: {str(e)}"

    return exported_paths, error_msg


def _export_rollout_slice_video(
    data: InfluenceData,
    rollout_episode_idx: int,
    start_offset: int,
    end_offset: int,
    output_path: pathlib.Path,
    fps: int = 10,
) -> bool:
    """Export a rollout slice to video.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode
        start_offset: Start offset within the episode (inclusive)
        end_offset: End offset within the episode (exclusive)
        output_path: Output video path
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    try:
        rollout_episode = data.rollout_episodes[rollout_episode_idx]

        # Collect frames for the slice
        frames = []
        for offset in range(start_offset, end_offset):
            sample_idx = rollout_episode.sample_start_idx + offset

            # Get the first observation from each sample (n_obs_steps can be > 1)
            # Rollout frames use 'img' key by default
            frame = data.get_rollout_frame(sample_idx, obs_key="img")

            if frame is not None:
                frame = _normalize_frame(frame)
                if frame is not None:
                    frames.append(frame)

        if len(frames) == 0:
            return False

        _write_video(frames, output_path, fps)

        return True

    except Exception as e:
        print(f"Error exporting rollout slice: {e}")
        import traceback

        traceback.print_exc()
        return False


def _export_demo_slice_video(
    data: InfluenceData,
    demo_sample_idx: int,
    output_path: pathlib.Path,
    obs_key: str = "agentview_image",
    fps: int = 10,
    window_width: Optional[int] = None,
) -> bool:
    """Export a demo slice to video.

    Args:
        data: InfluenceData object
        demo_sample_idx: Global demo sample index (start of the window)
        output_path: Output video path
        obs_key: Observation key for images
        fps: Frames per second
        window_width: Number of demo samples in the window (if None, exports single sample buffer)

    Returns:
        True if successful, False otherwise
    """
    try:
        frames = []

        if window_width is None:
            # Export single sample's buffer
            sample_info = data.get_demo_sample_info(demo_sample_idx)
            num_timesteps = sample_info.buffer_end_idx - sample_info.buffer_start_idx

            for timestep_in_horizon in range(num_timesteps):
                frame = data.get_demo_frame(
                    demo_sample_idx,
                    obs_key=obs_key,
                    timestep_in_horizon=timestep_in_horizon,
                )
                if frame is not None:
                    frame = _normalize_frame(frame)
                    if frame is not None:
                        frames.append(frame)
        else:
            # Export frames from all samples in the window [demo_sample_idx : demo_sample_idx + window_width]
            num_demo_samples = len(data.all_demo_sample_infos)
            end_sample_idx = min(demo_sample_idx + window_width, num_demo_samples)

            for sample_idx in range(demo_sample_idx, end_sample_idx):
                # Get the first frame from each sample (the observation at that timestep)
                frame = data.get_demo_frame(
                    sample_idx,
                    obs_key=obs_key,
                    timestep_in_horizon=0,
                )

                if frame is not None:
                    frame = _normalize_frame(frame)
                    if frame is not None:
                        frames.append(frame)

        if len(frames) == 0:
            return False

        _write_video(frames, output_path, fps)

        return True

    except Exception as e:
        print(f"Error exporting demo slice: {e}")
        import traceback

        traceback.print_exc()
        return False
