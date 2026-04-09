"""Video export tab for exporting individual episode slices."""

import pathlib
from typing import Optional

import streamlit as st

from influence_visualizer.data_loader import InfluenceData


def render_video_export_tab(
    data: InfluenceData, task_config: str, obs_key: str = "agentview_image"
):
    """Render the video export tab.

    Args:
        data: InfluenceData object
        task_config: Name of the task config
        obs_key: Observation key for demo images (from sidebar)
    """
    st.header("Video Export")
    st.write("Export individual episode slices as videos.")

    # Split selector
    col1, col2 = st.columns(2)
    with col1:
        split = st.selectbox(
            "Split",
            options=["train", "holdout", "rollout"],
            key="video_export_split",
            help="Select which dataset split to export from",
        )

    # Get episodes for the selected split
    if split == "train":
        episodes = data.demo_episodes
        sample_infos = data.demo_sample_infos
        is_rollout = False
    elif split == "holdout":
        episodes = data.holdout_episodes if data.holdout_episodes else []
        sample_infos = data.holdout_sample_infos if data.holdout_sample_infos else []
        is_rollout = False
    else:  # rollout
        episodes = data.rollout_episodes
        sample_infos = data.rollout_sample_infos
        is_rollout = True

    if not episodes:
        st.warning(f"No episodes available in {split} split.")
        return

    # Episode selector with labels
    with col2:
        if is_rollout:
            # Add success labels for rollouts
            episode_options = []
            for ep in episodes:
                success_label = ""
                if hasattr(ep, "success") and ep.success is not None:
                    success_label = " ✓" if ep.success else " ✗"
                episode_options.append(f"Episode {ep.index}{success_label}")
        else:
            # Add quality labels for demos
            episode_options = []
            quality_labels = data.demo_quality_labels
            for ep in episodes:
                quality_label = ""
                if quality_labels is not None and ep.index in quality_labels:
                    quality_label = f" ({quality_labels[ep.index]})"
                episode_options.append(f"Episode {ep.index}{quality_label}")

        episode_idx = st.selectbox(
            "Episode",
            options=range(len(episodes)),
            format_func=lambda i: episode_options[i],
            key="video_export_episode_idx",
            help="Select which episode to export",
        )

    episode = episodes[episode_idx]

    # Display episode info
    st.divider()
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Episode Index", episode.index)
    with info_col2:
        st.metric("Total Samples", episode.num_samples)
    with info_col3:
        if is_rollout:
            if hasattr(episode, "success") and episode.success is not None:
                st.metric("Success", "✓" if episode.success else "✗")
        else:
            # Show quality label for demos
            quality_labels = data.demo_quality_labels
            if quality_labels is not None and episode.index in quality_labels:
                st.metric("Quality", quality_labels[episode.index])

    # Slice range selectors
    st.subheader("Slice Selection")
    range_col1, range_col2 = st.columns(2)
    with range_col1:
        start_idx = st.number_input(
            "Start Index",
            min_value=0,
            max_value=max(0, episode.num_samples - 1),
            value=0,
            step=1,
            key="video_export_start_idx",
            help="Start index within the episode (inclusive)",
        )
    with range_col2:
        end_idx = st.number_input(
            "End Index",
            min_value=start_idx + 1,
            max_value=episode.num_samples,
            value=min(episode.num_samples, start_idx + 50),
            step=1,
            key="video_export_end_idx",
            help="End index within the episode (exclusive)",
        )

    slice_length = end_idx - start_idx
    st.caption(f"Slice length: {slice_length} samples")

    # Render preview and export in fragments
    _render_preview_fragment(
        data, episode, split, is_rollout, start_idx, end_idx, obs_key
    )
    _render_export_fragment(
        data,
        episode,
        episode_idx,
        split,
        is_rollout,
        start_idx,
        end_idx,
        task_config,
        obs_key,
    )


@st.fragment
def _render_preview_fragment(
    data: InfluenceData,
    episode,
    split: str,
    is_rollout: bool,
    start_idx: int,
    end_idx: int,
    obs_key: str,
):
    """Fragment for frame preview."""
    st.subheader("Frame Preview")

    # Play controls
    play_col1, play_col2, play_col3 = st.columns([1, 1, 2])
    with play_col1:
        play_button = st.button(
            "▶ Play", key="video_export_play", width='stretch'
        )
    with play_col2:
        stop_button = st.button(
            "⏸ Pause", key="video_export_stop", width='stretch'
        )
    with play_col3:
        playback_fps = st.slider(
            "Playback FPS",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="video_export_playback_fps",
            help="Frames per second for playback",
        )

    # Initialize or update playback state
    if "video_export_playing" not in st.session_state:
        st.session_state.video_export_playing = False
    if "video_export_current_frame" not in st.session_state:
        st.session_state.video_export_current_frame = int(start_idx)

    if play_button:
        st.session_state.video_export_playing = True
    if stop_button:
        st.session_state.video_export_playing = False

    # Auto-advance frame if playing
    if st.session_state.video_export_playing:
        import time

        st.session_state.video_export_current_frame = (
            st.session_state.video_export_current_frame + 1
        ) % episode.num_samples
        time.sleep(1.0 / playback_fps)
        st.rerun()

    # Video player for entire episode
    preview_col1, preview_col2 = st.columns([3, 1])
    with preview_col1:
        # Slider through entire episode, not limited to selected slice
        preview_idx = st.slider(
            "Preview Frame",
            min_value=0,
            max_value=episode.num_samples - 1,
            value=int(st.session_state.video_export_current_frame),
            step=1,
            key="video_export_preview_idx",
            help="Scroll through all frames in the episode (selected slice is highlighted)",
        )
        # Update current frame when slider is manually moved
        st.session_state.video_export_current_frame = preview_idx

    with preview_col2:
        # Show if current frame is in the selected slice
        in_slice = start_idx <= preview_idx < end_idx
        if in_slice:
            st.metric(
                "Position",
                f"In slice ({preview_idx - start_idx + 1}/{end_idx - start_idx})",
            )
        else:
            st.metric("Position", f"{preview_idx + 1}/{episode.num_samples}")

    # Display preview frame
    try:
        if is_rollout:
            # Rollout frame
            sample_idx = episode.sample_start_idx + preview_idx
            frame = data.get_rollout_frame(sample_idx, obs_key="img")
        else:
            # Demo frame - use obs_key from sidebar
            sample_idx = episode.sample_start_idx + preview_idx
            frame = data.get_demo_frame(
                sample_idx, obs_key=obs_key, timestep_in_horizon=0
            )

        if frame is not None:
            # Highlight if frame is in the selected slice
            if in_slice:
                caption = f"{split.capitalize()} Episode {episode.index} - Frame {preview_idx} ✓ IN SLICE"
            else:
                caption = f"{split.capitalize()} Episode {episode.index} - Frame {preview_idx}"

            st.image(
                frame,
                caption=caption,
                width='stretch',
            )
        else:
            st.warning("Frame not available for preview.")
    except Exception as e:
        st.error(f"Error loading preview frame: {e}")

    # Show slice range indicator
    st.caption(
        f"Selected slice: frames {start_idx} to {end_idx - 1} ({end_idx - start_idx} frames total)"
    )


@st.fragment
def _render_export_fragment(
    data: InfluenceData,
    episode,
    episode_idx: int,
    split: str,
    is_rollout: bool,
    start_idx: int,
    end_idx: int,
    task_config: str,
    obs_key: str,
):
    """Fragment for export settings and button."""
    st.divider()
    st.subheader("Export Settings")

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        fps = st.number_input(
            "FPS",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            key="video_export_fps",
            help="Frames per second for the exported video",
        )

    with export_col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        export_button = st.button(
            "Export Video",
            key="video_export_button",
            type="primary",
            width='stretch',
        )

    # Export logic
    if export_button:
        with st.spinner("Exporting video..."):
            try:
                # Create output directory
                output_dir = (
                    pathlib.Path("outputs/individual_exports") / task_config / split
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename with labels
                filename_parts = [split, str(episode.index)]

                # Add success/quality label to filename
                if (
                    is_rollout
                    and hasattr(episode, "success")
                    and episode.success is not None
                ):
                    success_str = "success" if episode.success else "fail"
                    filename_parts.append(success_str)
                elif not is_rollout:
                    quality_labels = data.demo_quality_labels
                    if quality_labels is not None and episode.index in quality_labels:
                        filename_parts.append(quality_labels[episode.index])

                filename_parts.extend([f"slice_{start_idx}_{end_idx}"])
                output_filename = "_".join(filename_parts) + ".mp4"
                output_path = output_dir / output_filename

                # Export the video
                if is_rollout:
                    success = _export_rollout_video(
                        data=data,
                        episode_idx=episode_idx,
                        start_offset=start_idx,
                        end_offset=end_idx,
                        output_path=output_path,
                        fps=fps,
                    )
                else:
                    success = _export_demo_video(
                        data=data,
                        episode=episode,
                        start_offset=start_idx,
                        end_offset=end_idx,
                        output_path=output_path,
                        fps=fps,
                        obs_key=obs_key,  # Use obs_key from sidebar
                    )

                if success:
                    with export_col3:
                        st.success(f"✓ Exported to:\n`{output_path}`")
                else:
                    st.error("Failed to export video. Check console for errors.")

            except Exception as e:
                st.error(f"Export failed: {e}")
                import traceback

                traceback.print_exc()


def _export_rollout_video(
    data: InfluenceData,
    episode_idx: int,
    start_offset: int,
    end_offset: int,
    output_path: pathlib.Path,
    fps: int,
) -> bool:
    """Export a rollout episode slice to video."""
    from influence_visualizer.video_export import _export_rollout_slice_video

    return _export_rollout_slice_video(
        data=data,
        rollout_episode_idx=episode_idx,
        start_offset=start_offset,
        end_offset=end_offset,
        output_path=output_path,
        fps=fps,
    )


def _export_demo_video(
    data: InfluenceData,
    episode,
    start_offset: int,
    end_offset: int,
    output_path: pathlib.Path,
    fps: int,
    obs_key: str,
) -> bool:
    """Export a demo episode slice to video."""
    import numpy as np

    from influence_visualizer.video_export import _normalize_frame, _write_video

    try:
        frames = []

        # Collect frames for each sample in the slice
        for offset in range(start_offset, end_offset):
            sample_idx = episode.sample_start_idx + offset

            # Get the first frame from each sample
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
        print(f"Error exporting demo video: {e}")
        import traceback

        traceback.print_exc()
        return False
