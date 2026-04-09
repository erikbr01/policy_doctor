"""Influence rendering functions for top influences and slice opponents."""

from typing import Callable, List, Literal, Tuple

import numpy as np
import streamlit as st

from influence_visualizer.data_loader import EpisodeInfo, InfluenceData
from influence_visualizer.profiling import profile
from influence_visualizer.render_annotation import (
    get_episode_annotations,
    get_label_for_frame,
    load_annotations,
)
from influence_visualizer.render_frames import (
    frame_player,
    render_action_chunk,
    render_annotated_frame,
    render_label_timeline,
)
from influence_visualizer.render_heatmaps import (
    SplitType,
    get_split_data,
    render_trajectory_influence_heatmap,
)


def _render_influence_detail(
    data: InfluenceData,
    influence: dict,
    rank: int,
    rollout_episode_idx: int,
    rollout_sample_idx: int,
    obs_key: str,
    split: SplitType = "train",
    demo_episodes: List[EpisodeInfo] = None,
    key_prefix: str = "",
    annotation_file: str = "",
    task_config: str = "",
    rollout_frame_key: str = "",
    show_trajectory_heatmap: bool = True,
):
    """Render detailed view for a single influence entry.

    This is shared logic between render_top_influences and render_slice_opponents.

    Args:
        data: InfluenceData object
        influence: Dict with influence details
        rank: Rank of this influence
        rollout_episode_idx: Index of the rollout episode
        rollout_sample_idx: Index of the rollout sample (used for influence lookup)
        obs_key: Observation key for images
        split: Which demo split is being used
        demo_episodes: List of demo episodes for this split
        key_prefix: Prefix for unique Streamlit keys
        annotation_file: Path to the annotation file
        rollout_frame_key: Session state key for the rollout frame player (to get current frame for action chunk display)
        show_trajectory_heatmap: Whether to show the trajectory influence heatmap (set False for behavior-aggregated views)
    """
    score = influence["influence_score"]
    demo_ep_idx = influence["demo_episode_idx"]
    demo_timestep = influence["demo_timestep"]
    demo_sample_idx = influence["global_demo_sample_idx"]  # Global index in full matrix
    sample_info = influence["sample_info"]
    episode = influence["episode"]

    # Determine if this is a holdout sample (only relevant when showing both)
    is_holdout = split == "holdout" or (
        split == "both" and demo_sample_idx >= len(data.demo_sample_infos)
    )
    holdout_marker = " [HOLDOUT]" if is_holdout else ""

    # Get quality label if available
    quality_labels = data.demo_quality_labels
    quality = None
    quality_marker = ""
    if quality_labels is not None and demo_ep_idx in quality_labels:
        quality = quality_labels[demo_ep_idx]
        quality_marker = f" [{quality.upper()}]"

    # Calculate episode-relative timesteps
    ep_start_in_buffer = sample_info.buffer_start_idx - sample_info.timestep
    relative_start = sample_info.timestep
    relative_end = relative_start + (
        sample_info.buffer_end_idx - sample_info.buffer_start_idx
    )

    # Get rollout timestep for reference in the title (only used when showing trajectory heatmap)
    if show_trajectory_heatmap:
        rollout_sample_info = data.get_rollout_sample_info(rollout_sample_idx)
        rollout_timestep = rollout_sample_info.timestep
        expander_title = (
            f"#{rank}: Demo ep{demo_ep_idx}{holdout_marker}{quality_marker} t={relative_start} → "
            f"Rollout t={rollout_timestep} (Score: {score:.4f})"
        )
    else:
        # For behavior-aggregated views, don't reference a specific rollout sample
        expander_title = (
            f"#{rank}: Demo ep{demo_ep_idx}{holdout_marker}{quality_marker} t={relative_start} "
            f"(Mean Influence: {score:.4f})"
        )

    # Wrap the entire card in a fragment so interactions don't trigger parent reloads
    @st.fragment
    def _render_card():
        with st.expander(
            expander_title,
            expanded=(rank <= 3),
        ):
            # Show sample details
            st.markdown("**Sample Details**")
            num_cols = 5 if quality is not None else 4
            cols = st.columns(num_cols)
            with cols[0]:
                st.metric("Influence Score", f"{score:.4f}")
            with cols[1]:
                st.metric("Demo Episode", demo_ep_idx)
            with cols[2]:
                st.metric("Episode Start", relative_start)
            with cols[3]:
                st.metric(
                    "Horizon Coverage",
                    f"{relative_start}-{relative_end}",
                )
            if quality is not None:
                with cols[4]:
                    st.metric("Quality Tier", quality.capitalize())

            st.divider()

            # Full episode length calculation
            if episode is not None and episode.raw_length is not None:
                full_episode_length = episode.raw_length
            else:
                full_episode_length = relative_end + 10

            # Load annotations for this demo episode
            is_holdout_ep = split == "holdout" or (
                split == "both" and demo_sample_idx >= len(data.demo_sample_infos)
            )
            demo_data_type = "holdout" if is_holdout_ep else "demo"
            demo_episode_id = f"{demo_data_type}_ep{demo_ep_idx}"
            demo_annotations = []
            if annotation_file:
                with profile("load_annotations_in_influence_detail"):
                    all_annotations = load_annotations(
                        annotation_file, task_config=task_config
                    )
                demo_annotations = all_annotations.get(demo_episode_id, [])

            # Full episode sequence viewer
            st.markdown("**Full Demo Episode Sequence**")

            def _render_full_episode_frame(full_ep_timestep):
                # Render label timeline for the full episode
                render_label_timeline(
                    demo_annotations,
                    num_frames=full_episode_length,
                    current_frame=full_ep_timestep,
                    unique_key=f"{key_prefix}timeline_demo_ep{demo_ep_idx}_rank{rank}",
                )

                in_hz = relative_start <= full_ep_timestep < relative_end
                hz_indicator = "in" if in_hz else "out"
                col_full_frame, col_full_info = st.columns([1, 1])
                with col_full_frame:
                    full_frame = _load_full_episode_frame(
                        data,
                        demo_sample_idx,
                        ep_start_in_buffer,
                        full_ep_timestep,
                        obs_key,
                    )
                    if full_frame is not None:
                        lbl = f"ep{demo_ep_idx}, t={full_ep_timestep} ({hz_indicator})"

                        # Get label for this frame
                        current_label = get_label_for_frame(
                            full_ep_timestep, demo_annotations
                        )
                        label_str = (
                            current_label
                            if current_label != "no label"
                            else "no label yet"
                        )

                        render_annotated_frame(
                            full_frame,
                            lbl,
                            caption=f"Label: {label_str}",
                            font_size=10,
                        )
                    else:
                        st.info("Frame not available")
                with col_full_info:
                    st.markdown("**Full Episode Info**")
                    st.text(f"Episode length: {full_episode_length}")
                    st.text(f"Current timestep: {full_ep_timestep}")
                    st.text(f"Horizon window: {relative_start}-{relative_end - 1}")
                    if in_hz:
                        st.success("This frame is in the horizon window")
                    else:
                        st.info("This frame is outside the horizon window")

            full_episode_timestep = frame_player(
                label="Select timestep in full episode:",
                min_value=0,
                max_value=max(0, full_episode_length - 1),
                key=f"{key_prefix}demo_full_ep_player_{rank}",
                default_value=relative_start,
                default_fps=3.0,
                help=f"Full episode has {full_episode_length} timesteps. Horizon window covers {relative_start}-{relative_end - 1}",
                render_fn=_render_full_episode_frame,
                fragment_scope=True,
            )

            # Skip action chunks and heatmap for behavior-aggregated views
            if show_trajectory_heatmap:
                st.divider()

                # Action chunks comparison
                st.markdown("**Action Chunks**")

                col_rollout_action, col_demo_action = st.columns(2)

                with col_rollout_action:
                    st.markdown("*Rollout Action Chunk*")
                    # Get current rollout sample index from session state if available
                    if rollout_frame_key:
                        rollout_episode = data.rollout_episodes[rollout_episode_idx]
                        current_offset = st.session_state.get(
                            f"{rollout_frame_key}_value", 0
                        )
                        current_rollout_sample_idx = (
                            rollout_episode.sample_start_idx + current_offset
                        )
                        current_rollout_timestep = current_offset
                    else:
                        current_rollout_sample_idx = rollout_sample_idx
                        current_rollout_timestep = rollout_timestep

                    rollout_actions = data.get_rollout_action_chunk(
                        current_rollout_sample_idx
                    )
                    render_action_chunk(
                        rollout_actions,
                        title=f"Rollout Sample {current_rollout_sample_idx}, t={current_rollout_timestep}",
                        unique_key=f"{key_prefix}rollout_action_{rank}_{current_rollout_sample_idx}",
                    )

                with col_demo_action:
                    st.markdown("*Demo Action Chunk*")
                    demo_actions = data.get_demo_action_chunk(demo_sample_idx)
                    render_action_chunk(
                        demo_actions,
                        title=f"Demo Sample {demo_sample_idx}",
                        unique_key=f"{key_prefix}demo_action_{rank}_{demo_sample_idx}",
                    )

                st.divider()

                # Trajectory influence heatmap (lazy-loaded)
                st.markdown("**Trajectory Influence Heatmap** (click to expand)")

                if st.toggle(
                    "Show detailed influence heatmap",
                    value=False,
                    key=f"{key_prefix}show_heatmap_{rank}",
                    help="Click to load and display the detailed timestep-by-timestep influence heatmap",
                ):
                    st.caption(
                        "Shows how each timestep of the rollout is influenced by each timestep "
                        "of this demonstration."
                    )

                    # Find the demo episode index within the split's episode list
                    demo_episode_list_idx = None
                    if demo_episodes is not None:
                        for idx, ep in enumerate(demo_episodes):
                            if ep.index == demo_ep_idx:
                                demo_episode_list_idx = idx
                                break

                    if demo_episode_list_idx is not None:
                        with profile(f"render_trajectory_heatmap_rank{rank}"):
                            render_trajectory_influence_heatmap(
                                data=data,
                                rollout_episode_idx=rollout_episode_idx,
                                demo_episode_idx=demo_episode_list_idx,
                                split=split,
                                unique_key_suffix=f"{key_prefix}t{demo_timestep}",
                                annotation_file=annotation_file,
                                task_config=task_config,
                            )
                    else:
                        st.error(
                            f"Could not find demo episode {demo_ep_idx} in {split} split"
                        )
            else:
                # For behavior-aggregated views, only show the demo action chunk
                st.divider()
                st.markdown("**Demo Action Chunk**")
                demo_actions = data.get_demo_action_chunk(demo_sample_idx)
                render_action_chunk(
                    demo_actions,
                    title=f"Demo Sample {demo_sample_idx}",
                    unique_key=f"{key_prefix}demo_action_{rank}_{demo_sample_idx}",
                )

    _render_card()


def _load_full_episode_frame(
    data: InfluenceData,
    demo_sample_idx: int,
    ep_start_in_buffer: int,
    full_episode_timestep: int,
    obs_key: str,
):
    """Load a frame from the full episode at a specific timestep."""
    full_frame = None
    try:
        if demo_sample_idx < len(data.demo_sample_infos):
            dataset = data.demo_dataset
        else:
            dataset = data.holdout_dataset

        if dataset is not None:
            replay_buffer = dataset.replay_buffer
            abs_buffer_idx = ep_start_in_buffer + full_episode_timestep

            img_dataset = (
                data.image_dataset if data.image_dataset is not None else dataset
            )
            img_replay_buffer = (
                img_dataset.replay_buffer if img_dataset is not None else replay_buffer
            )

            if obs_key in img_replay_buffer.keys():
                full_frame = img_replay_buffer[obs_key][abs_buffer_idx]
            else:
                image_keys = [
                    k for k in img_replay_buffer.keys() if "image" in k.lower()
                ]
                if image_keys:
                    full_frame = img_replay_buffer[image_keys[0]][abs_buffer_idx]

            if full_frame is not None:
                full_frame = np.array(full_frame)
                if full_frame.dtype != np.uint8:
                    if full_frame.max() <= 1.0:
                        full_frame = (full_frame * 255).astype(np.uint8)
                    else:
                        full_frame = full_frame.astype(np.uint8)
    except Exception as e:
        st.warning(f"Could not load full episode frame: {e}")

    return full_frame


def render_top_influences(
    data: InfluenceData,
    rollout_sample_idx: int,
    rollout_episode_idx: int,
    top_k: int = 10,
    obs_key: str = "agentview_image",
    split: SplitType = "train",
    annotation_file: str = "",
    rollout_frame_key: str = "",
):
    """Render the top-k influential demonstrations for a sample.

    Args:
        data: InfluenceData object
        rollout_sample_idx: Index of the rollout sample
        rollout_episode_idx: Index of the rollout episode
        top_k: Number of top influences to show
        obs_key: Observation key for images
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Path to the annotation file
        rollout_frame_key: Session state key for the rollout frame player
    """
    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get influences for this rollout sample
    influence_scores = influence_matrix[rollout_sample_idx, :]

    # Get top-k indices (sorted by influence, descending)
    sorted_indices = np.argsort(influence_scores)[::-1][:top_k]

    # Build influence details
    top_influences = []
    num_train_samples = len(data.demo_sample_infos)

    for local_sample_idx in sorted_indices:
        score = float(influence_scores[local_sample_idx])

        # Find which episode this sample belongs to
        episode = None
        episode_list_idx = None
        for ep_idx, sample_idxs in enumerate(ep_idxs):
            if local_sample_idx in sample_idxs:
                episode = demo_episodes[ep_idx]
                episode_list_idx = ep_idx
                break

        if episode is None:
            continue

        # Calculate global sample index for data access
        if split == "train":
            global_sample_idx = local_sample_idx
        elif split == "holdout":
            global_sample_idx = local_sample_idx + num_train_samples
        else:  # both
            global_sample_idx = local_sample_idx

        # Get sample info using global index
        if global_sample_idx < len(data.all_demo_sample_infos):
            sample_info = data.all_demo_sample_infos[global_sample_idx]
        else:
            continue

        top_influences.append(
            {
                "influence_score": score,
                "demo_episode_idx": episode.index,
                "demo_timestep": sample_info.timestep,
                "local_demo_sample_idx": local_sample_idx,
                "global_demo_sample_idx": global_sample_idx,
                "sample_info": sample_info,
                "episode": episode,
            }
        )

    split_label = split.title() if split != "both" else "Train + Holdout"
    st.subheader(f"Top {top_k} Influential Demonstrations ({split_label})")

    if len(top_influences) == 0:
        st.warning("No influences found for this sample")
        return

    for i, influence in enumerate(top_influences):
        _render_influence_detail(
            data=data,
            influence=influence,
            rank=i + 1,
            rollout_episode_idx=rollout_episode_idx,
            rollout_sample_idx=rollout_sample_idx,
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix="",
            annotation_file=annotation_file,
            rollout_frame_key=rollout_frame_key,
        )


# =============================================================================
# Sliding Window Aggregation for Rollout Slice Analysis
# =============================================================================


def _pad_influence_slice(
    influence_slice: np.ndarray,
    window_width: int,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Pad the influence slice for sliding window computation.

    Args:
        influence_slice: Shape (rollout_slice_height, num_demo_samples)
        window_width: Width of the sliding window
        pad_mode: How to pad - "edge" repeats boundary values (a-b-c-c-c)

    Returns:
        Padded array with shape (rollout_slice_height, num_demo_samples + window_width - 1)
    """
    # Pad only on the right side of the demo axis (axis=1)
    # This allows the window to slide to the end of the demo samples
    pad_width = window_width - 1
    if pad_mode == "edge":
        # Repeat boundary values: a-b-c becomes a-b-c-c-c
        padded = np.pad(influence_slice, ((0, 0), (0, pad_width)), mode="edge")
    else:
        # Zero padding
        padded = np.pad(influence_slice, ((0, 0), (0, pad_width)), mode="constant")
    return padded


def _sliding_window_aggregate_vectorized(
    influence_slice: np.ndarray,
    window_width: int,
    kind: str,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Vectorized sliding window for 'sum' or 'mean' (much faster for large num_demo_samples)."""
    rollout_height, num_demo_samples = influence_slice.shape
    padded = _pad_influence_slice(influence_slice, window_width, pad_mode)
    # Per-row 1D convolution with ones(window_width) gives sum over window; sum over rows
    kernel = np.ones(window_width, dtype=padded.dtype)
    window_sums = np.zeros(num_demo_samples, dtype=np.float64)
    for r in range(padded.shape[0]):
        window_sums += np.convolve(padded[r, :], kernel, mode="valid")
    if kind == "mean":
        window_sums /= rollout_height * window_width
    return window_sums.astype(np.float32)


def sliding_window_aggregate(
    influence_slice: np.ndarray,
    window_width: int,
    aggregation_fn: Callable[[np.ndarray], float],
    pad_mode: str = "edge",
) -> np.ndarray:
    """Compute sliding window aggregation over demonstration timesteps.

    The window slides along the demonstration sample axis. At each position,
    the aggregation function is applied to the window of shape
    (rollout_slice_height, window_width).

    Args:
        influence_slice: Shape (rollout_slice_height, num_demo_samples)
            The influence submatrix for the selected rollout slice.
        window_width: Width of the sliding window along demo axis.
        aggregation_fn: Function that takes a 2D window and returns a scalar.
        pad_mode: "edge" for repeating boundary values, "zero" for zero padding.

    Returns:
        Array of shape (num_demo_samples,) containing aggregated influence
        scores for each demo sample position (window centered/starting at that position).
    """
    rollout_height, num_demo_samples = influence_slice.shape

    # Pad the influence slice
    padded = _pad_influence_slice(influence_slice, window_width, pad_mode)

    # Compute aggregation at each demo timestep position
    aggregated_scores = np.zeros(num_demo_samples)
    for demo_idx in range(num_demo_samples):
        window = padded[:, demo_idx : demo_idx + window_width]
        aggregated_scores[demo_idx] = aggregation_fn(window)

    return aggregated_scores


# Available aggregation methods for sliding window
AGGREGATION_METHODS = {
    "sum": lambda w: np.sum(w),
    "mean": lambda w: np.mean(w),
    "max": lambda w: np.max(w),
    "min": lambda w: np.min(w),
    "sum_of_means": lambda w: np.sum(np.mean(w, axis=1)),  # Sum of row means
    "mean_of_sums": lambda w: np.mean(np.sum(w, axis=1)),  # Mean of row sums
    "abs_sum": lambda w: np.sum(np.abs(w)),  # Sum of absolute values
    "abs_mean": lambda w: np.mean(np.abs(w)),  # Mean of absolute values
}


def rank_demos_by_slice_influence(
    data: InfluenceData,
    rollout_start_idx: int,
    rollout_end_idx: int,
    window_width: int,
    aggregation_method: str = "sum",
    split: SplitType = "train",
    ascending: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank demonstration samples by their influence on a rollout slice.

    Args:
        data: InfluenceData object
        rollout_start_idx: Global start index of the rollout slice (inclusive)
        rollout_end_idx: Global end index of the rollout slice (exclusive)
        window_width: Width of the sliding window for demo aggregation
        aggregation_method: One of the keys in AGGREGATION_METHODS
        split: Which demo split to use
        ascending: If True, sort ascending (for finding lowest influence)

    Returns:
        Tuple of:
        - sorted_demo_indices: Demo sample indices sorted by influence
        - sorted_scores: Corresponding aggregated influence scores
        - raw_scores: Full array of scores for all demo samples
    """
    with profile("rank_demos_by_slice_influence"):
        # Get split-specific data
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

        # Extract the influence slice for the rollout range
        influence_slice = influence_matrix[rollout_start_idx:rollout_end_idx, :]

        # Use vectorized path for sum/mean (much faster for large num_demo_samples)
        if aggregation_method in ("sum", "mean"):
            with profile("sliding_window_vectorized"):
                aggregated_scores = _sliding_window_aggregate_vectorized(
                    influence_slice,
                    window_width=window_width,
                    kind=aggregation_method,
                    pad_mode="edge",
                )
        else:
            with profile("sliding_window_aggregate"):
                agg_fn = AGGREGATION_METHODS.get(aggregation_method, AGGREGATION_METHODS["sum"])
                aggregated_scores = sliding_window_aggregate(
                    influence_slice,
                    window_width=window_width,
                    aggregation_fn=agg_fn,
                    pad_mode="edge",
                )

        # Sort indices
        if ascending:
            sorted_indices = np.argsort(aggregated_scores)
        else:
            sorted_indices = np.argsort(aggregated_scores)[::-1]

        sorted_scores = aggregated_scores[sorted_indices]

    return sorted_indices, sorted_scores, aggregated_scores


def _render_slice_influence_detail(
    data: InfluenceData,
    influence: dict,
    rank: int,
    rollout_episode_idx: int,
    rollout_start_offset: int,
    rollout_end_offset: int,
    window_width: int,
    obs_key: str,
    split: SplitType = "train",
    demo_episodes: List[EpisodeInfo] = None,
    key_prefix: str = "",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render detailed view for a single influence entry from slice analysis.

    Similar to _render_influence_detail but adapted for slice-based analysis.
    """
    score = influence["influence_score"]
    demo_ep_idx = influence["demo_episode_idx"]
    demo_timestep = influence["demo_timestep"]
    demo_sample_idx = influence["global_demo_sample_idx"]
    sample_info = influence["sample_info"]
    episode = influence["episode"]

    # Determine if this is a holdout sample
    is_holdout = split == "holdout" or (
        split == "both" and demo_sample_idx >= len(data.demo_sample_infos)
    )
    holdout_marker = " [HOLDOUT]" if is_holdout else ""

    # Get quality label if available
    quality_labels = data.demo_quality_labels
    quality = None
    quality_marker = ""
    if quality_labels is not None and demo_ep_idx in quality_labels:
        quality = quality_labels[demo_ep_idx]
        quality_marker = f" [{quality.upper()}]"

    # Calculate episode-relative timesteps
    ep_start_in_buffer = sample_info.buffer_start_idx - sample_info.timestep
    relative_start = sample_info.timestep
    relative_end = relative_start + (
        sample_info.buffer_end_idx - sample_info.buffer_start_idx
    )

    # Calculate demo window range (the sliding window covers demo_timestep to demo_timestep + window_width)
    demo_window_end = relative_start + window_width

    expander_title = (
        f"#{rank}: Demo ep{demo_ep_idx}{holdout_marker}{quality_marker} t=[{relative_start}:{demo_window_end}] "
        f"→ Rollout [{rollout_start_offset}:{rollout_end_offset}] "
        f"(Score: {score:.4f})"
    )

    @st.fragment
    def _render_card():
        with st.expander(
            expander_title,
            expanded=(rank <= 3),
        ):
            # Show sample details
            st.markdown("**Sample Details**")
            num_cols = 5 if quality is not None else 4
            cols = st.columns(num_cols)
            with cols[0]:
                st.metric("Influence Score", f"{score:.4f}")
            with cols[1]:
                st.metric("Demo Episode", demo_ep_idx)
            with cols[2]:
                st.metric("Demo Window", f"[{relative_start}:{demo_window_end}]")
            with cols[3]:
                st.metric(
                    "Horizon Coverage",
                    f"{relative_start}-{relative_end}",
                )
            if quality is not None:
                with cols[4]:
                    st.metric("Quality Tier", quality.capitalize())

            st.divider()

            # Full episode length calculation
            if episode is not None and episode.raw_length is not None:
                full_episode_length = episode.raw_length
            else:
                full_episode_length = relative_end + 10

            # Load annotations for this demo episode
            is_holdout_ep = split == "holdout" or (
                split == "both" and demo_sample_idx >= len(data.demo_sample_infos)
            )
            demo_data_type = "holdout" if is_holdout_ep else "demo"
            demo_episode_id = f"{demo_data_type}_ep{demo_ep_idx}"
            demo_annotations = []
            if annotation_file:
                with profile("load_annotations_in_slice_detail"):
                    all_annotations = load_annotations(
                        annotation_file, task_config=task_config
                    )
                demo_annotations = all_annotations.get(demo_episode_id, [])

            # Full episode sequence viewer
            st.markdown("**Demo Episode Sequence**")

            def _render_full_episode_frame(full_ep_timestep):
                render_label_timeline(
                    demo_annotations,
                    num_frames=full_episode_length,
                    current_frame=full_ep_timestep,
                    unique_key=f"{key_prefix}slice_timeline_demo_ep{demo_ep_idx}_rank{rank}",
                )

                in_hz = relative_start <= full_ep_timestep < relative_end
                hz_indicator = "in" if in_hz else "out"
                col_full_frame, col_full_info = st.columns([1, 1])
                with col_full_frame:
                    full_frame = _load_full_episode_frame(
                        data,
                        demo_sample_idx,
                        ep_start_in_buffer,
                        full_ep_timestep,
                        obs_key,
                    )
                    if full_frame is not None:
                        lbl = f"ep{demo_ep_idx}, t={full_ep_timestep} ({hz_indicator})"
                        current_label = get_label_for_frame(
                            full_ep_timestep, demo_annotations
                        )
                        label_str = (
                            current_label
                            if current_label != "no label"
                            else "no label yet"
                        )
                        render_annotated_frame(
                            full_frame,
                            lbl,
                            caption=f"Label: {label_str}",
                            font_size=10,
                        )
                    else:
                        st.info("Frame not available")
                with col_full_info:
                    st.markdown("**Full Episode Info**")
                    st.text(f"Episode length: {full_episode_length}")
                    st.text(f"Current timestep: {full_ep_timestep}")
                    st.text(f"Horizon window: {relative_start}-{relative_end - 1}")
                    if in_hz:
                        st.success("This frame is in the horizon window")
                    else:
                        st.info("This frame is outside the horizon window")

            full_episode_timestep = frame_player(
                label="Select timestep in full episode:",
                min_value=0,
                max_value=max(0, full_episode_length - 1),
                key=f"{key_prefix}slice_demo_full_ep_player_{rank}",
                default_value=relative_start,
                default_fps=3.0,
                help=f"Full episode has {full_episode_length} timesteps.",
                render_fn=_render_full_episode_frame,
                fragment_scope=True,
            )

            st.divider()

            # Demo action chunk
            st.markdown("**Demo Action Chunk**")
            demo_actions = data.get_demo_action_chunk(demo_sample_idx)
            render_action_chunk(
                demo_actions,
                title=f"Demo Sample {demo_sample_idx}",
                unique_key=f"{key_prefix}slice_demo_action_{rank}_{demo_sample_idx}",
            )

            st.divider()

            # Trajectory influence heatmap (lazy-loaded)
            st.markdown("**Trajectory Influence Heatmap** (click to expand)")

            if st.toggle(
                "Show detailed influence heatmap",
                value=False,
                key=f"{key_prefix}slice_show_heatmap_{rank}",
                help="Click to load and display the detailed timestep-by-timestep influence heatmap",
            ):
                st.caption(
                    "Shows how each timestep of the rollout is influenced by each timestep "
                    "of this demonstration."
                )

                # Find the demo episode index within the split's episode list
                demo_episode_list_idx = None
                if demo_episodes is not None:
                    for idx, ep in enumerate(demo_episodes):
                        if ep.index == demo_ep_idx:
                            demo_episode_list_idx = idx
                            break

                if demo_episode_list_idx is not None:
                    with profile(f"render_trajectory_heatmap_slice_rank{rank}"):
                        render_trajectory_influence_heatmap(
                            data=data,
                            rollout_episode_idx=rollout_episode_idx,
                            demo_episode_idx=demo_episode_list_idx,
                            split=split,
                            unique_key_suffix=f"{key_prefix}slice_t{demo_timestep}",
                            annotation_file=annotation_file,
                            task_config=task_config,
                            default_rollout_start=rollout_start_offset,
                            default_rollout_end=rollout_end_offset,
                        )
                else:
                    st.error(
                        f"Could not find demo episode {demo_ep_idx} in {split} split"
                    )

    _render_card()


def render_top_influences_slice(
    data: InfluenceData,
    rollout_episode_idx: int,
    rollout_start_offset: int,
    rollout_end_offset: int,
    window_width: int,
    aggregation_method: str = "sum",
    top_k: int = 10,
    obs_key: str = "agentview_image",
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render top-k influential demo samples for a rollout slice.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode
        rollout_start_offset: Start offset within the episode (inclusive)
        rollout_end_offset: End offset within the episode (exclusive)
        window_width: Sliding window width for demo aggregation
        aggregation_method: Aggregation method name
        top_k: Number of top influences to show
        obs_key: Observation key for images
        split: Which demo split to use
        annotation_file: Path to annotation file
    """
    # Get episode info
    rollout_episode = data.rollout_episodes[rollout_episode_idx]

    # Convert episode offsets to global indices
    rollout_start_idx = rollout_episode.sample_start_idx + rollout_start_offset
    rollout_end_idx = rollout_episode.sample_start_idx + rollout_end_offset

    # Rank demos by slice influence
    sorted_indices, sorted_scores, raw_scores = rank_demos_by_slice_influence(
        data,
        rollout_start_idx,
        rollout_end_idx,
        window_width=window_width,
        aggregation_method=aggregation_method,
        split=split,
        ascending=False,
    )

    # Get split data for episode lookup
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    num_train_samples = len(data.demo_sample_infos)

    # Build top influences
    top_influences = []
    for i, local_sample_idx in enumerate(sorted_indices[:top_k]):
        score = float(sorted_scores[i])

        # Find which episode this sample belongs to
        episode = None
        for ep_idx, sample_idxs in enumerate(ep_idxs):
            if local_sample_idx in sample_idxs:
                episode = demo_episodes[ep_idx]
                break

        if episode is None:
            continue

        # Calculate global sample index
        if split == "train":
            global_sample_idx = local_sample_idx
        elif split == "holdout":
            global_sample_idx = local_sample_idx + num_train_samples
        else:
            global_sample_idx = local_sample_idx

        if global_sample_idx < len(data.all_demo_sample_infos):
            sample_info = data.all_demo_sample_infos[global_sample_idx]
        else:
            continue

        top_influences.append(
            {
                "influence_score": score,
                "demo_episode_idx": episode.index,
                "demo_timestep": sample_info.timestep,
                "local_demo_sample_idx": local_sample_idx,
                "global_demo_sample_idx": global_sample_idx,
                "sample_info": sample_info,
                "episode": episode,
            }
        )

    split_label = split.title() if split != "both" else "Train + Holdout"
    slice_desc = f"[{rollout_start_offset}:{rollout_end_offset}]"
    st.subheader(
        f"Top {top_k} Influential Demos for Rollout Slice {slice_desc} ({split_label})"
    )
    st.caption(f"Aggregation: {aggregation_method}, Window width: {window_width}")

    if len(top_influences) == 0:
        st.warning("No influences found for this slice")
        return

    for i, influence in enumerate(top_influences):
        _render_slice_influence_detail(
            data=data,
            influence=influence,
            rank=i + 1,
            rollout_episode_idx=rollout_episode_idx,
            rollout_start_offset=rollout_start_offset,
            rollout_end_offset=rollout_end_offset,
            window_width=window_width,
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix="slice_top_",
            annotation_file=annotation_file,
            task_config=task_config,
        )


def render_slice_opponents_slice(
    data: InfluenceData,
    rollout_episode_idx: int,
    rollout_start_offset: int,
    rollout_end_offset: int,
    window_width: int,
    aggregation_method: str = "sum",
    top_k: int = 10,
    obs_key: str = "agentview_image",
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render bottom-k (lowest influence) demo samples for a rollout slice.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode
        rollout_start_offset: Start offset within the episode (inclusive)
        rollout_end_offset: End offset within the episode (exclusive)
        window_width: Sliding window width for demo aggregation
        aggregation_method: Aggregation method name
        top_k: Number of bottom influences to show
        obs_key: Observation key for images
        split: Which demo split to use
        annotation_file: Path to annotation file
    """
    # Get episode info
    rollout_episode = data.rollout_episodes[rollout_episode_idx]

    # Convert episode offsets to global indices
    rollout_start_idx = rollout_episode.sample_start_idx + rollout_start_offset
    rollout_end_idx = rollout_episode.sample_start_idx + rollout_end_offset

    # Rank demos by slice influence (ascending for lowest)
    sorted_indices, sorted_scores, raw_scores = rank_demos_by_slice_influence(
        data,
        rollout_start_idx,
        rollout_end_idx,
        window_width=window_width,
        aggregation_method=aggregation_method,
        split=split,
        ascending=True,
    )

    # Get split data for episode lookup
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    num_train_samples = len(data.demo_sample_infos)

    # Build bottom influences
    bottom_influences = []
    for i, local_sample_idx in enumerate(sorted_indices[:top_k]):
        score = float(sorted_scores[i])

        # Find which episode this sample belongs to
        episode = None
        for ep_idx, sample_idxs in enumerate(ep_idxs):
            if local_sample_idx in sample_idxs:
                episode = demo_episodes[ep_idx]
                break

        if episode is None:
            continue

        # Calculate global sample index
        if split == "train":
            global_sample_idx = local_sample_idx
        elif split == "holdout":
            global_sample_idx = local_sample_idx + num_train_samples
        else:
            global_sample_idx = local_sample_idx

        if global_sample_idx < len(data.all_demo_sample_infos):
            sample_info = data.all_demo_sample_infos[global_sample_idx]
        else:
            continue

        bottom_influences.append(
            {
                "influence_score": score,
                "demo_episode_idx": episode.index,
                "demo_timestep": sample_info.timestep,
                "local_demo_sample_idx": local_sample_idx,
                "global_demo_sample_idx": global_sample_idx,
                "sample_info": sample_info,
                "episode": episode,
            }
        )

    split_label = split.title() if split != "both" else "Train + Holdout"
    slice_desc = f"[{rollout_start_offset}:{rollout_end_offset}]"
    st.subheader(
        f"Slice Opponents: {top_k} Lowest Influence for Slice {slice_desc} ({split_label})"
    )
    st.caption(f"Aggregation: {aggregation_method}, Window width: {window_width}")

    if len(bottom_influences) == 0:
        st.warning("No influences found for this slice")
        return

    for i, influence in enumerate(bottom_influences):
        _render_slice_influence_detail(
            data=data,
            influence=influence,
            rank=i + 1,
            rollout_episode_idx=rollout_episode_idx,
            rollout_start_offset=rollout_start_offset,
            rollout_end_offset=rollout_end_offset,
            window_width=window_width,
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix="slice_opp_",
            annotation_file=annotation_file,
            task_config=task_config,
        )


def render_slice_opponents(
    data: InfluenceData,
    rollout_sample_idx: int,
    rollout_episode_idx: int,
    top_k: int = 10,
    obs_key: str = "agentview_image",
    split: SplitType = "train",
    annotation_file: str = "",
    rollout_frame_key: str = "",
):
    """Render the demonstrations with lowest influence (slice opponents) for a sample.

    Args:
        data: InfluenceData object
        rollout_sample_idx: Index of the rollout sample
        rollout_episode_idx: Index of the rollout episode
        top_k: Number of bottom influences to show
        obs_key: Observation key for images
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Path to the annotation file for behavior labels
        rollout_frame_key: Session state key for the rollout frame player
    """
    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get influences for this rollout sample
    influence_scores = influence_matrix[rollout_sample_idx, :]

    # Get bottom-k indices (sorted by influence, ascending)
    sorted_indices = np.argsort(influence_scores)[:top_k]

    # Build influence details
    bottom_influences = []
    num_train_samples = len(data.demo_sample_infos)

    for local_sample_idx in sorted_indices:
        score = float(influence_scores[local_sample_idx])

        # Find which episode this sample belongs to
        episode = None
        for ep_idx, sample_idxs in enumerate(ep_idxs):
            if local_sample_idx in sample_idxs:
                episode = demo_episodes[ep_idx]
                break

        if episode is None:
            continue

        # Calculate global sample index for data access
        if split == "train":
            global_sample_idx = local_sample_idx
        elif split == "holdout":
            global_sample_idx = local_sample_idx + num_train_samples
        else:  # both
            global_sample_idx = local_sample_idx

        # Get sample info using global index
        if global_sample_idx < len(data.all_demo_sample_infos):
            sample_info = data.all_demo_sample_infos[global_sample_idx]
        else:
            continue

        bottom_influences.append(
            {
                "influence_score": score,
                "demo_episode_idx": episode.index,
                "demo_timestep": sample_info.timestep,
                "local_demo_sample_idx": local_sample_idx,
                "global_demo_sample_idx": global_sample_idx,
                "sample_info": sample_info,
                "episode": episode,
            }
        )

    split_label = split.title() if split != "both" else "Train + Holdout"
    st.subheader(
        f"Slice Opponents: {top_k} Lowest Influence Demonstrations ({split_label})"
    )

    if len(bottom_influences) == 0:
        st.warning("No influences found for this sample")
        return

    for i, influence in enumerate(bottom_influences):
        _render_influence_detail(
            data=data,
            influence=influence,
            rank=i + 1,
            rollout_episode_idx=rollout_episode_idx,
            rollout_sample_idx=rollout_sample_idx,
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix="opponent_",
            annotation_file=annotation_file,
            rollout_frame_key=rollout_frame_key,
        )
