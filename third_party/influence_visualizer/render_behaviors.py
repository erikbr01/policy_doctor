"""Behavior analysis and visualization for the Behaviors tab.

This module provides functions to:
1. Aggregate behavior labels from rollout annotations
2. Compute mean influence per training sample across rollout samples for each behavior
3. Render pie chart showing label distribution
4. Render top-k influential demos per behavior
"""

from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from influence_visualizer import plotting
from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.render_annotation import (
    get_episode_annotations,
    load_annotations,
)
from influence_visualizer.render_heatmaps import SplitType, get_split_data
from influence_visualizer.render_influences import _render_influence_detail


def get_behavior_statistics(
    annotations: Dict, rollout_episodes: List
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Get statistics about behavior labels from rollout annotations.

    Args:
        annotations: Dictionary of annotations from annotation file
        rollout_episodes: List of EpisodeInfo for rollout episodes

    Returns:
        Dict mapping label -> list of (episode_idx, start_frame, end_frame) tuples
    """
    behavior_slices = {}

    for episode in rollout_episodes:
        episode_id_str = str(episode.index)
        slices = get_episode_annotations(annotations, episode_id_str, split="rollout")

        for slice_info in slices:
            label = slice_info["label"]
            start = slice_info["start"]
            end = slice_info["end"]

            if label not in behavior_slices:
                behavior_slices[label] = []
            behavior_slices[label].append((episode.index, start, end))

    return behavior_slices


def get_rollout_samples_for_behavior(
    data: InfluenceData, behavior_slices: List[Tuple[int, int, int]]
) -> np.ndarray:
    """Get all rollout sample indices that belong to a behavior.

    Args:
        data: InfluenceData object
        behavior_slices: List of (episode_idx, start_frame, end_frame) tuples

    Returns:
        Array of rollout sample indices
    """
    sample_indices = []

    for episode_idx, start_frame, end_frame in behavior_slices:
        # Find the episode
        episode = None
        for ep in data.rollout_episodes:
            if ep.index == episode_idx:
                episode = ep
                break

        if episode is None:
            continue

        # Get sample indices for frames in range [start_frame, end_frame]
        for sample_offset in range(episode.num_samples):
            # The sample's timestep is its offset within the episode
            if start_frame <= sample_offset <= end_frame:
                abs_idx = episode.sample_start_idx + sample_offset
                sample_indices.append(abs_idx)

    return np.array(sample_indices, dtype=int)


def compute_behavior_influence_per_slice(
    data: InfluenceData,
    behavior_slices: List[Tuple[int, int, int]],
    split: SplitType = "train",
    aggregation_method: str = "mean",
) -> List[Tuple[np.ndarray, str]]:
    """Compute influence for each behavior slice separately (no cross-episode aggregation).

    Returns per-slice aggregated influences to preserve temporal coherence within each slice.

    Args:
        data: InfluenceData object
        behavior_slices: List of (episode_idx, start_frame, end_frame) tuples
        split: Which demo split to use
        aggregation_method: How to aggregate across timesteps within a slice.
            One of: "mean", "sum", "max", "min", "sum_of_means", "mean_of_sums",
            "abs_sum", "abs_mean"

    Returns:
        List of (aggregated_influences, slice_label) tuples where:
        - aggregated_influences: Array of shape (num_demo_samples,) with aggregated influence for this slice
        - slice_label: String describing the slice (e.g., "Rollout ep1 t[10:20]")
    """
    from influence_visualizer.render_influences import AGGREGATION_METHODS

    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get aggregation function
    agg_fn = AGGREGATION_METHODS.get(aggregation_method, AGGREGATION_METHODS["mean"])

    results = []

    for episode_idx, start_frame, end_frame in behavior_slices:
        # Find the episode
        episode = None
        for ep in data.rollout_episodes:
            if ep.index == episode_idx:
                episode = ep
                break

        if episode is None:
            continue

        # Get contiguous block for this slice
        rollout_start_idx = episode.sample_start_idx + start_frame
        rollout_end_idx = episode.sample_start_idx + end_frame + 1

        # Extract influence slice
        influence_slice = influence_matrix[rollout_start_idx:rollout_end_idx, :]

        # Compute aggregation across this slice
        # For each demo sample (column), apply aggregation function
        aggregated_influences = np.array(
            [
                agg_fn(influence_slice[:, demo_idx : demo_idx + 1])
                for demo_idx in range(influence_slice.shape[1])
            ]
        )

        # Create label for this slice
        slice_label = f"Rollout ep{episode_idx} t[{start_frame}:{end_frame}]"

        results.append((aggregated_influences, slice_label))

    return results


def compute_behavior_influence(
    data: InfluenceData,
    behavior_slices: List[Tuple[int, int, int]],
    split: SplitType = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute aggregated influence for a behavior.

    For each training sample, compute the mean influence across all rollout
    samples that have this behavior label.

    Args:
        data: InfluenceData object
        behavior_slices: List of (episode_idx, start_frame, end_frame) tuples
        split: Which demo split to use

    Returns:
        Tuple of (mean_influences, rollout_sample_indices)
        - mean_influences: Array of shape (num_demo_samples,) with mean influence
        - rollout_sample_indices: Array of rollout sample indices used
    """
    # Get rollout sample indices for this behavior
    rollout_indices = get_rollout_samples_for_behavior(data, behavior_slices)

    if len(rollout_indices) == 0:
        return np.array([]), np.array([])

    # Get the split-specific influence matrix
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Extract influences for the relevant rollout samples
    behavior_influences = influence_matrix[rollout_indices, :]

    # Compute mean across rollout samples
    mean_influences = np.mean(behavior_influences, axis=0)

    return mean_influences, rollout_indices


def render_behavior_pie_chart(
    behavior_stats: Dict[str, List[Tuple[int, int, int]]],
    unique_key: str = "behavior_pie",
):
    """Render a pie chart showing distribution of rollout samples by behavior.

    Args:
        behavior_stats: Dict mapping label -> list of (episode_idx, start, end)
        unique_key: Unique key for the Streamlit chart
    """
    # Count total samples per label
    label_counts = {}
    for label, slices in behavior_stats.items():
        count = sum(end - start + 1 for _, start, end in slices)
        label_counts[label] = count

    if not label_counts:
        st.info("No behavior annotations found for rollout episodes.")
        return

    # Labels and values for the pie chart
    labels = list(label_counts.keys())
    values = list(label_counts.values())

    # Create pie chart using pure plotting function
    fig = plotting.create_behavior_pie_chart(
        labels=labels,
        values=values,
        title="Rollout Samples by Behavior Label",
    )

    st.plotly_chart(fig, width="stretch", key=unique_key)


def render_behavior_influence_bar_chart(
    data: InfluenceData,
    label: str,
    mean_influences: np.ndarray,
    split: SplitType,
    unique_key: str = "behavior_influence_bar",
):
    """Render a bar chart showing aggregated influence per demonstration sample.

    Args:
        data: InfluenceData object
        label: Behavior label
        mean_influences: Array of mean influences per demo sample
        split: Which demo split is being used
        unique_key: Unique key for the chart
    """
    # Sort for better visualization
    sort_indices = np.argsort(mean_influences)[::-1]
    sorted_influences = mean_influences[sort_indices]
    x_labels = [f"s{i}" for i in sort_indices]

    # Create bar chart using pure plotting function
    fig = plotting.create_behavior_influence_bar_chart(
        sample_labels=x_labels,
        influences=sorted_influences,
        title=f"Mean Influence per Demo Sample for '{label}' (sorted)",
    )

    st.plotly_chart(fig, width="stretch", key=unique_key)


@st.fragment
def render_behavior_top_influences_per_slice(
    data: InfluenceData,
    label: str,
    behavior_slices: List[Tuple[int, int, int]],
    top_k: int = 10,
    obs_key: str = "agentview_image",
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
    aggregation_method: str = "mean",
):
    """Render top-k influential demos across all behavior slices (globally ranked).

    Collects top-k from each slice independently, then re-ranks them globally.

    Args:
        data: InfluenceData object
        label: Behavior label
        behavior_slices: List of (episode_idx, start_frame, end_frame) tuples
        top_k: Number of top influences to show globally
        obs_key: Observation key for images
        split: Which demo split to use
        annotation_file: Path to annotation file
        task_config: Task config name for annotations
        aggregation_method: How to aggregate influence within each slice
    """
    from influence_visualizer.render_influences import rank_demos_by_slice_influence

    if len(behavior_slices) == 0:
        st.warning(f"No rollout samples found for behavior '{label}'")
        return

    st.info(
        f"Analyzing influence on **{len(behavior_slices)}** behavior slices labeled as **'{label}'**. "
        f"Results ranked globally across all slices using **{aggregation_method}** aggregation."
    )

    # Add video export button
    export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
    with export_col1:
        do_export = st.button(
            "Export Videos",
            key=f"export_behavior_videos_{label}_{split}",
            help="Export videos of all behavior slices and their top-k demo slices to outputs/behavior_exports/",
        )
    with export_col2:
        export_fps = st.number_input(
            "FPS",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            key=f"export_behavior_fps_{label}_{split}",
            help="Frames per second for exported videos",
        )

    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    num_train_samples = len(data.demo_sample_infos)

    # Collect top-k results from each slice
    all_candidates = []

    for episode_idx, start_frame, end_frame in behavior_slices:
        # Find the episode
        episode = None
        for ep in data.rollout_episodes:
            if ep.index == episode_idx:
                episode = ep
                break

        if episode is None:
            continue

        # Convert to global indices
        rollout_start_idx = episode.sample_start_idx + start_frame
        rollout_end_idx = episode.sample_start_idx + end_frame + 1

        # Use the existing ranking function (with window_width=1 for no windowing)
        sorted_indices, sorted_scores, _ = rank_demos_by_slice_influence(
            data,
            rollout_start_idx,
            rollout_end_idx,
            window_width=1,
            aggregation_method=aggregation_method,
            split=split,
            ascending=False,
        )

        # Collect top-k from this slice
        for i in range(min(top_k, len(sorted_indices))):
            local_sample_idx = sorted_indices[i]
            score = float(sorted_scores[i])

            all_candidates.append(
                {
                    "local_sample_idx": local_sample_idx,
                    "score": score,
                    "source_episode_idx": episode_idx,
                    "source_start": start_frame,
                    "source_end": end_frame,
                }
            )

    # Re-rank globally
    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Take global top-k
    global_top_k = all_candidates[:top_k]

    # Handle video export
    if do_export:
        _export_behavior_videos(
            data=data,
            label=label,
            behavior_slices=behavior_slices,
            global_top_k=global_top_k,
            split=split,
            task_config=task_config,
            obs_key=obs_key,
            fps=int(export_fps),
            export_col=export_col3,
        )

    st.divider()

    # Top-k highest influence
    st.subheader(f"Top {top_k} Most Influential Demos for '{label}'")

    for rank, candidate in enumerate(global_top_k):
        local_sample_idx = candidate["local_sample_idx"]
        score = candidate["score"]
        source_ep = candidate["source_episode_idx"]
        source_start = candidate["source_start"]
        source_end = candidate["source_end"]

        # Find episode
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

        if global_sample_idx >= len(data.all_demo_sample_infos):
            continue

        sample_info = data.all_demo_sample_infos[global_sample_idx]

        influence = {
            "influence_score": score,
            "demo_episode_idx": episode.index,
            "demo_timestep": sample_info.timestep,
            "local_demo_sample_idx": local_sample_idx,
            "global_demo_sample_idx": global_sample_idx,
            "sample_info": sample_info,
            "episode": episode,
        }

        # Add source slice info header
        st.markdown(
            f"**📍 Source:** Rollout ep{source_ep} t[{source_start}:{source_end}]"
        )

        _render_influence_detail(
            data=data,
            influence=influence,
            rank=rank + 1,
            rollout_episode_idx=0,
            rollout_sample_idx=0,
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix=f"behavior_{label}_top_",
            annotation_file=annotation_file,
            task_config=task_config,
            show_trajectory_heatmap=False,
        )

    st.divider()

    # Bottom-k (collect and rank from all slices)
    st.subheader(f"Bottom {top_k} Least Influential Demos for '{label}'")

    # Collect bottom-k from each slice
    all_bottom_candidates = []
    for episode_idx, start_frame, end_frame in behavior_slices:
        episode = None
        for ep in data.rollout_episodes:
            if ep.index == episode_idx:
                episode = ep
                break

        if episode is None:
            continue

        rollout_start_idx = episode.sample_start_idx + start_frame
        rollout_end_idx = episode.sample_start_idx + end_frame + 1

        sorted_indices, sorted_scores, _ = rank_demos_by_slice_influence(
            data,
            rollout_start_idx,
            rollout_end_idx,
            window_width=1,
            aggregation_method=aggregation_method,
            split=split,
            ascending=True,  # Get lowest influence
        )

        for i in range(min(top_k, len(sorted_indices))):
            local_sample_idx = sorted_indices[i]
            score = float(sorted_scores[i])

            all_bottom_candidates.append(
                {
                    "local_sample_idx": local_sample_idx,
                    "score": score,
                    "source_episode_idx": episode_idx,
                    "source_start": start_frame,
                    "source_end": end_frame,
                }
            )

    # Re-rank globally (ascending for bottom-k)
    all_bottom_candidates.sort(key=lambda x: x["score"])
    global_bottom_k = all_bottom_candidates[:top_k]

    for rank, candidate in enumerate(global_bottom_k):
        local_sample_idx = candidate["local_sample_idx"]
        score = candidate["score"]
        source_ep = candidate["source_episode_idx"]
        source_start = candidate["source_start"]
        source_end = candidate["source_end"]

        # Find episode
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

        if global_sample_idx >= len(data.all_demo_sample_infos):
            continue

        sample_info = data.all_demo_sample_infos[global_sample_idx]

        influence = {
            "influence_score": score,
            "demo_episode_idx": episode.index,
            "demo_timestep": sample_info.timestep,
            "local_demo_sample_idx": local_sample_idx,
            "global_demo_sample_idx": global_sample_idx,
            "sample_info": sample_info,
            "episode": episode,
        }

        # Add source slice info header
        st.markdown(
            f"**📍 Source:** Rollout ep{source_ep} t[{source_start}:{source_end}]"
        )

        _render_influence_detail(
            data=data,
            influence=influence,
            rank=rank + 1,
            rollout_episode_idx=0,
            rollout_sample_idx=0,
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix=f"behavior_{label}_bottom_",
            annotation_file=annotation_file,
            task_config=task_config,
            show_trajectory_heatmap=False,
        )


@st.fragment
def render_behavior_top_influences(
    data: InfluenceData,
    label: str,
    behavior_slices: List[Tuple[int, int, int]],
    top_k: int = 10,
    obs_key: str = "agentview_image",
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render top-k and bottom-k influential demos for a behavior.

    Args:
        data: InfluenceData object
        label: Behavior label
        behavior_slices: List of (episode_idx, start_frame, end_frame) tuples
        top_k: Number of top/bottom influences to show
        obs_key: Observation key for images
        split: Which demo split to use
        annotation_file: Path to annotation file
    """
    mean_influences, rollout_indices = compute_behavior_influence(
        data, behavior_slices, split
    )

    if len(mean_influences) == 0:
        st.warning(f"No rollout samples found for behavior '{label}'")
        return

    st.info(
        f"Aggregated influence computed over **{len(rollout_indices)}** rollout samples "
        f"labeled as **'{label}'**"
    )

    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    num_train_samples = len(data.demo_sample_infos)

    # Bar chart showing all demo sample influences
    render_behavior_influence_bar_chart(
        data=data,
        label=label,
        mean_influences=mean_influences,
        split=split,
        unique_key=f"behavior_bar_{label}",
    )

    st.divider()

    # Top-k highest influence
    st.subheader(f"Top {top_k} Most Influential Demos for '{label}'")
    top_indices = np.argsort(mean_influences)[::-1][:top_k]

    for rank, local_sample_idx in enumerate(top_indices):
        score = float(mean_influences[local_sample_idx])

        # Find episode
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

        if global_sample_idx >= len(data.all_demo_sample_infos):
            continue

        sample_info = data.all_demo_sample_infos[global_sample_idx]

        influence = {
            "influence_score": score,
            "demo_episode_idx": episode.index,
            "demo_timestep": sample_info.timestep,
            "local_demo_sample_idx": local_sample_idx,
            "global_demo_sample_idx": global_sample_idx,
            "sample_info": sample_info,
            "episode": episode,
        }

        _render_influence_detail(
            data=data,
            influence=influence,
            rank=rank + 1,
            rollout_episode_idx=0,  # Not used when show_trajectory_heatmap=False
            rollout_sample_idx=0,  # Not used when show_trajectory_heatmap=False
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix=f"behavior_{label}_top_",
            annotation_file=annotation_file,
            task_config=task_config,
            show_trajectory_heatmap=False,
        )

    st.divider()

    # Bottom-k (lowest influence / slice opponents)
    st.subheader(f"Bottom {top_k} Least Influential Demos for '{label}'")
    bottom_indices = np.argsort(mean_influences)[:top_k]

    for rank, local_sample_idx in enumerate(bottom_indices):
        score = float(mean_influences[local_sample_idx])

        # Find episode
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

        if global_sample_idx >= len(data.all_demo_sample_infos):
            continue

        sample_info = data.all_demo_sample_infos[global_sample_idx]

        influence = {
            "influence_score": score,
            "demo_episode_idx": episode.index,
            "demo_timestep": sample_info.timestep,
            "local_demo_sample_idx": local_sample_idx,
            "global_demo_sample_idx": global_sample_idx,
            "sample_info": sample_info,
            "episode": episode,
        }

        _render_influence_detail(
            data=data,
            influence=influence,
            rank=rank + 1,
            rollout_episode_idx=0,  # Not used when show_trajectory_heatmap=False
            rollout_sample_idx=0,  # Not used when show_trajectory_heatmap=False
            obs_key=obs_key,
            split=split,
            demo_episodes=demo_episodes,
            key_prefix=f"behavior_{label}_bottom_",
            annotation_file=annotation_file,
            task_config=task_config,
            show_trajectory_heatmap=False,
        )


@st.fragment
def _render_label_dist_fragment(behavior_stats, demo_split):
    with st.expander("Load Label Distribution", expanded=False):
        col_chart, col_stats = st.columns([2, 1])
        with col_chart:
            show_pie_key = f"show_behavior_pie_{demo_split}"
            if st.button("Generate Pie Chart", key=f"gen_behavior_pie_{demo_split}"):
                st.session_state[show_pie_key] = True
            if st.session_state.get(show_pie_key, False):
                render_behavior_pie_chart(
                    behavior_stats, unique_key="main_behavior_pie"
                )
        with col_stats:
            st.markdown("**Label Summary**")
            for label, slices in sorted(behavior_stats.items()):
                num_samples = sum(end - start + 1 for _, start, end in slices)
                st.text(f"• {label}: {len(slices)} slices, {num_samples} samples")


@st.fragment
def _render_behavior_analysis_fragment(
    data,
    behavior_stats,
    selected_label,
    demo_split,
    top_k,
    obs_key,
    annotation_file,
    task_config,
):
    with st.expander(f"Load Influence Analysis for '{selected_label}'", expanded=False):
        # Add aggregation method selector
        from influence_visualizer.render_influences import AGGREGATION_METHODS

        aggregation_method = st.selectbox(
            "Aggregation method",
            options=list(AGGREGATION_METHODS.keys()),
            index=0,
            key=f"behavior_analysis_agg_{selected_label}_{demo_split}",
            help="How to aggregate influence values within each behavior slice",
        )

        show_analysis_key = f"show_behavior_analysis_{selected_label}_{demo_split}"
        if st.button(
            f"Generate Influence Analysis",
            key=f"gen_behavior_influence_{selected_label}_{demo_split}",
        ):
            st.session_state[show_analysis_key] = True
        if st.session_state.get(show_analysis_key, False):
            render_behavior_top_influences_per_slice(
                data=data,
                label=selected_label,
                behavior_slices=behavior_stats[selected_label],
                top_k=top_k,
                obs_key=obs_key,
                split=demo_split,
                annotation_file=annotation_file,
                task_config=task_config,
                aggregation_method=aggregation_method,
            )


def _export_behavior_videos(
    data: InfluenceData,
    label: str,
    behavior_slices: List[Tuple[int, int, int]],
    global_top_k: List[dict],
    split: SplitType,
    task_config: str,
    obs_key: str,
    fps: int,
    export_col,
):
    """Export videos for all behavior slices and their top-k demos."""
    import pathlib

    from influence_visualizer.video_export import export_slice_videos

    with st.spinner("Exporting behavior videos..."):
        try:
            all_exported_paths = []

            # Export videos for each behavior slice
            for slice_idx, (episode_idx, start_frame, end_frame) in enumerate(
                behavior_slices
            ):
                # Find the episode
                episode = None
                for ep in data.rollout_episodes:
                    if ep.index == episode_idx:
                        episode = ep
                        break

                if episode is None:
                    continue

                # Get top-k demos for this slice
                slice_demos = [
                    candidate
                    for candidate in global_top_k
                    if candidate["source_episode_idx"] == episode_idx
                    and candidate["source_start"] == start_frame
                    and candidate["source_end"] == end_frame
                ][:10]  # Limit to top 10 per slice

                if not slice_demos:
                    continue

                # Build demo influences list
                num_train_samples = len(data.demo_sample_infos)
                demo_influences = []

                for candidate in slice_demos:
                    local_sample_idx = candidate["local_sample_idx"]
                    score = candidate["score"]

                    # Calculate global sample index
                    if split == "train":
                        global_sample_idx = local_sample_idx
                    elif split == "holdout":
                        global_sample_idx = local_sample_idx + num_train_samples
                    else:
                        global_sample_idx = local_sample_idx

                    if global_sample_idx >= len(data.all_demo_sample_infos):
                        continue

                    sample_info = data.all_demo_sample_infos[global_sample_idx]

                    # Find demo episode
                    demo_episode = None
                    for ep in data.all_demo_episodes:
                        if ep.sample_start_idx <= global_sample_idx < ep.sample_end_idx:
                            demo_episode = ep
                            break

                    if demo_episode is None:
                        continue

                    demo_influences.append(
                        {
                            "influence_score": score,
                            "demo_episode_idx": demo_episode.index,
                            "demo_timestep": sample_info.timestep,
                            "local_demo_sample_idx": local_sample_idx,
                            "global_demo_sample_idx": global_sample_idx,
                            "sample_info": sample_info,
                            "episode": demo_episode,
                        }
                    )

                if not demo_influences:
                    continue

                # Export videos for this slice
                output_dir = pathlib.Path("outputs/behavior_exports")
                exported_paths, error_msg = export_slice_videos(
                    data=data,
                    rollout_episode_idx=episode_idx,
                    rollout_start_offset=start_frame,
                    rollout_end_offset=end_frame + 1,
                    demo_influences=demo_influences,
                    output_dir=output_dir,
                    task_config_name=f"{task_config}_{label}_slice{slice_idx}",
                    obs_key=obs_key,
                    fps=fps,
                    demo_window_width=1,  # No windowing for behavior slices
                )

                if error_msg:
                    st.error(f"Error exporting slice {slice_idx}: {error_msg}")
                else:
                    all_exported_paths.extend(exported_paths)

            # Show success message
            if all_exported_paths:
                with export_col:
                    st.success(
                        f"✓ Exported {len(all_exported_paths)} videos to outputs/behavior_exports/"
                    )
            else:
                st.warning("No videos were exported.")

        except Exception as e:
            st.error(f"Export failed: {e}")
            import traceback

            traceback.print_exc()


def render_behaviors_tab(
    data: InfluenceData,
    demo_split: SplitType,
    top_k: int,
    obs_key: str,
    annotation_file: str,
    task_config: str = "",
):
    """Render the Behaviors tab using localized fragments."""
    st.header("Behavior-wise Influence Analysis")
    st.markdown("""
    This tab shows how training demonstrations influence specific **behavioral segments**
    of the rollouts. Behaviors are defined by annotations created in the Annotation tab.
    """)

    annotations = load_annotations(annotation_file, task_config=task_config)
    behavior_stats = get_behavior_statistics(annotations, data.rollout_episodes)

    if not behavior_stats:
        st.warning(
            "No behavior annotations found. Please annotate some episodes first."
        )
        return

    st.subheader("Behavior Label Distribution")
    _render_label_dist_fragment(behavior_stats, demo_split)

    st.divider()
    st.subheader("Behavior Influence Analysis")
    available_labels = sorted(behavior_stats.keys())
    selected_label = st.selectbox(
        "Select behavior to analyze:",
        options=available_labels,
        key="behavior_label_select",
    )

    if selected_label:
        _render_behavior_analysis_fragment(
            data,
            behavior_stats,
            selected_label,
            demo_split,
            top_k,
            obs_key,
            annotation_file,
            task_config,
        )

    st.divider()

    # Add behavior slice search section
    from influence_visualizer.render_local_behaviors import render_behavior_slice_search

    render_behavior_slice_search(
        data=data,
        demo_split=demo_split,
        annotation_file=annotation_file,
        task_config=task_config,
        obs_key=obs_key,
    )
