"""Heatmap rendering functions for the influence visualizer."""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter

from diffusion_policy.common.error_util import (
    compute_demo_quality_scores,
    max_of_min_influence,
    mean_of_mean_influence,
    min_of_max_influence,
    pairwise_sample_to_trajectory_scores,
    sum_of_sum_influence,
)
from influence_visualizer import plotting
from influence_visualizer.data_loader import EpisodeInfo, InfluenceData
from influence_visualizer.profiling import profile
from influence_visualizer.render_annotation import (
    get_episode_annotations,
    get_label_for_frame,
    load_annotations,
)
from influence_visualizer.render_frames import LABEL_COLORS, _get_label_color

# Type alias for split options
SplitType = Literal["train", "holdout", "both"]


def get_episode_label_summary(
    annotations: Dict,
    episode_id: str,
    num_frames: int,
    split: str = "rollout",
) -> Tuple[str, List[str], Dict[str, int]]:
    """Get a summary of labels for an episode.

    Args:
        annotations: Dictionary of all annotations
        episode_id: Episode ID (can be old format like "demo_ep0" or new format like "0")
        num_frames: Total number of frames in the episode
        split: Split name for new format ("train", "holdout", "rollout")

    Returns:
        Tuple of:
        - dominant_label: Most common label (or "no label" if no annotations)
        - unique_labels: List of unique labels in the episode
        - label_counts: Dict mapping label to frame count
    """
    # Try to get annotations - support both old and new formats
    if episode_id in annotations:
        # Could be old format (direct key) or new format (episode number)
        ep_data = annotations[episode_id]
        if isinstance(ep_data, list):
            # Old format - direct list of slices
            episode_annotations = ep_data
        elif isinstance(ep_data, dict) and split in ep_data:
            # New format - nested dict with splits
            episode_annotations = ep_data[split]
        else:
            episode_annotations = []
    else:
        # Not found in either format
        episode_annotations = []

    if not episode_annotations:
        return "no label", [], {}

    # Count frames for each label
    label_counts: Dict[str, int] = {}
    for ann in episode_annotations:
        num_ann_frames = ann["end"] - ann["start"] + 1
        label = ann["label"]
        label_counts[label] = label_counts.get(label, 0) + num_ann_frames

    # Get unique labels in order of appearance
    unique_labels = []
    for ann in sorted(episode_annotations, key=lambda x: x["start"]):
        if ann["label"] not in unique_labels:
            unique_labels.append(ann["label"])

    # Find dominant label
    dominant_label = (
        max(label_counts, key=label_counts.get) if label_counts else "no label"
    )

    return dominant_label, unique_labels, label_counts


def sum_of_normalized_scores(
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return sum of normalized scores.

    This mirrors the notebook's sum_of_normalized_scores function.

    Args:
        scores: (N, M) array where N is the number of score types and M is the number of demos
        weights: Optional weights for each score type. Defaults to uniform weights.

    Returns:
        (M,) array of weighted normalized scores
    """
    # Weights for weighted average.
    if weights is None:
        weights = np.ones(len(scores)) / len(scores)
    assert len(weights) == len(scores)

    # Remove invalid scores (e.g., all zeros)
    mask = scores.sum(axis=1) != 0
    scores = scores[mask]
    weights = weights[mask] / weights[mask].sum()

    # Normalize scores between [0, 1], return weighted average.
    def norm(scores: np.ndarray) -> np.ndarray:
        score_min = scores.min()
        score_max = scores.max()
        if score_max == score_min:
            return np.zeros_like(scores)
        return (scores - score_min) / (score_max - score_min)

    return np.array([norm(s) * weights[i] for i, s in enumerate(scores)]).sum(axis=0)


def get_split_data(
    data: InfluenceData,
    split: SplitType,
) -> Tuple[np.ndarray, List[EpisodeInfo], List[np.ndarray], np.ndarray]:
    """Get influence matrix and episode metadata for a specific split.

    This mirrors the slicing logic in eval_demonstration_scores.py:online_trak_influence_routine.

    Args:
        data: InfluenceData object
        split: "train", "holdout", or "both"

    Returns:
        Tuple of:
        - influence_matrix: Sliced influence matrix for the split
        - demo_episodes: List of EpisodeInfo for the split
        - ep_idxs: List of sample index arrays (relative to sliced matrix, starting from 0)
        - ep_lens: Array of episode lengths
    """
    # Get the number of train samples (for slicing)
    num_train_samples = len(data.demo_sample_infos)

    if split == "train":
        # Slice to train samples only (columns 0 to num_train_samples)
        influence_matrix = data.influence_matrix[:, :num_train_samples]
        demo_episodes = data.demo_episodes

        # Build ep_idxs relative to the sliced matrix (starting from 0)
        ep_lens = np.array([ep.num_samples for ep in demo_episodes], dtype=np.int64)
        ep_ends = ep_lens.cumsum()
        ep_idxs = []
        for i, ep_end in enumerate(ep_ends):
            start_idx = 0 if i == 0 else ep_ends[i - 1]
            ep_idxs.append(np.arange(start_idx, ep_end))

    elif split == "holdout":
        # Slice to holdout samples only (columns num_train_samples onwards)
        influence_matrix = data.influence_matrix[:, num_train_samples:]
        demo_episodes = data.holdout_episodes

        # Build ep_idxs relative to the sliced matrix (starting from 0)
        ep_lens = np.array([ep.num_samples for ep in demo_episodes], dtype=np.int64)
        ep_ends = ep_lens.cumsum()
        ep_idxs = []
        for i, ep_end in enumerate(ep_ends):
            start_idx = 0 if i == 0 else ep_ends[i - 1]
            ep_idxs.append(np.arange(start_idx, ep_end))

    elif split == "both":
        # Use full matrix
        influence_matrix = data.influence_matrix
        demo_episodes = data.all_demo_episodes

        # Build ep_idxs for the full matrix
        # Train episodes first, then holdout
        ep_idxs = []
        ep_lens_list = []

        # Train episodes (indices 0 to num_train_samples)
        train_ep_lens = np.array(
            [ep.num_samples for ep in data.demo_episodes], dtype=np.int64
        )
        train_ep_ends = train_ep_lens.cumsum()
        for i, ep_end in enumerate(train_ep_ends):
            start_idx = 0 if i == 0 else train_ep_ends[i - 1]
            ep_idxs.append(np.arange(start_idx, ep_end))
        ep_lens_list.extend(train_ep_lens.tolist())

        # Holdout episodes (indices num_train_samples onwards)
        holdout_ep_lens = np.array(
            [ep.num_samples for ep in data.holdout_episodes], dtype=np.int64
        )
        holdout_ep_ends = holdout_ep_lens.cumsum() + num_train_samples
        for i, ep_end in enumerate(holdout_ep_ends):
            start_idx = num_train_samples if i == 0 else holdout_ep_ends[i - 1]
            ep_idxs.append(np.arange(start_idx, ep_end))
        ep_lens_list.extend(holdout_ep_lens.tolist())

        ep_lens = np.array(ep_lens_list, dtype=np.int64)

    else:
        raise ValueError(
            f"Invalid split: {split}. Must be 'train', 'holdout', or 'both'"
        )

    return influence_matrix, demo_episodes, ep_idxs, ep_lens


def get_rollout_data(
    data: InfluenceData,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Get rollout episode metadata.

    Returns:
        Tuple of:
        - ep_idxs: List of sample index arrays for each rollout episode
        - ep_lens: Array of episode lengths
        - success_mask: Boolean array indicating success/failure
    """
    ep_lens = np.array([ep.num_samples for ep in data.rollout_episodes], dtype=np.int64)
    ep_ends = ep_lens.cumsum()
    ep_idxs = []
    for i, ep_end in enumerate(ep_ends):
        start_idx = 0 if i == 0 else ep_ends[i - 1]
        ep_idxs.append(np.arange(start_idx, ep_end))

    success_mask = np.array(
        [
            ep.success if ep.success is not None else False
            for ep in data.rollout_episodes
        ],
        dtype=bool,
    )

    return ep_idxs, ep_lens, success_mask


def compute_trajectory_influence_matrix(
    data: InfluenceData,
    split: SplitType = "train",
) -> Tuple[np.ndarray, List[EpisodeInfo]]:
    """Compute trajectory-wise influence matrix from action-level influences.

    This mirrors the computation in eval_demonstration_scores.py:online_trak_influence_routine.

    Args:
        data: InfluenceData object containing influence matrix and episode info
        split: Which demo split to use ("train", "holdout", or "both")

    Returns:
        Tuple of:
        - Trajectory influence matrix of shape (num_rollout_episodes, num_demo_episodes)
        - List of demo episodes used
    """
    # Get split-specific data (mirrors eval_demonstration_scores.py slicing)
    influence_matrix, demo_episodes, train_ep_idxs, train_ep_lens = get_split_data(
        data, split
    )

    # Get rollout data
    test_ep_idxs, test_ep_lens, success_mask = get_rollout_data(data)

    num_test_eps = len(data.rollout_episodes)
    num_train_eps = len(demo_episodes)

    # Compute trajectory-level influence matrix
    traj_influence = pairwise_sample_to_trajectory_scores(
        pairwise_sample_scores=influence_matrix,
        num_test_eps=num_test_eps,
        num_train_eps=num_train_eps,
        test_ep_idxs=test_ep_idxs,
        train_ep_idxs=train_ep_idxs,
        test_ep_lens=test_ep_lens,
        train_ep_lens=train_ep_lens,
        success_mask=success_mask,
        aggr_fn=mean_of_mean_influence,
        return_dtype=np.float32,
    )

    return traj_influence, demo_episodes


def compute_performance_influence(
    data: InfluenceData,
    split: SplitType = "train",
    metric: str = "net",
) -> Tuple[np.ndarray, List[EpisodeInfo]]:
    """Compute performance influence scores for each demonstration.

    This mirrors the computation in eval_demonstration_scores.py:online_trak_influence_routine.
    Uses sum_of_sum_influence aggregation, matching the notebook's default behavior.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        metric: Quality metric ("net", "succ", or "fail")

    Returns:
        Tuple of:
        - Performance influence array of shape (num_demo_episodes,) with raw (un-normalized) scores
        - List of demo episodes used
    """
    # Get split-specific data
    influence_matrix, demo_episodes, train_ep_idxs, train_ep_lens = get_split_data(
        data, split
    )

    # Get rollout data
    test_ep_idxs, test_ep_lens, success_mask = get_rollout_data(data)

    num_test_eps = len(data.rollout_episodes)
    num_train_eps = len(demo_episodes)

    # Compute trajectory-level influence matrix using sum_of_sum_influence
    traj_influence = pairwise_sample_to_trajectory_scores(
        pairwise_sample_scores=influence_matrix,
        num_test_eps=num_test_eps,
        num_train_eps=num_train_eps,
        test_ep_idxs=test_ep_idxs,
        train_ep_idxs=train_ep_idxs,
        test_ep_lens=test_ep_lens,
        train_ep_lens=train_ep_lens,
        success_mask=success_mask,
        aggr_fn=sum_of_sum_influence,
        return_dtype=np.float32,
    )

    # Compute demo quality scores (performance influence)
    performance_influence = compute_demo_quality_scores(
        traj_scores=traj_influence,
        success_mask=success_mask,
        metric=metric,
    )

    if performance_influence is None:
        performance_influence = np.zeros(num_train_eps, dtype=np.float32)

    return performance_influence, demo_episodes


def render_influence_heatmap(
    data: InfluenceData,
    rollout_episode_idx: int,
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render a heatmap of influences for a rollout episode.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode to visualize
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Path to the annotation file for behavior labels
        task_config: Task config name for loading annotations from JSON
    """
    rollout_ep = data.rollout_episodes[rollout_episode_idx]
    sample_indices = np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx)

    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    split_label = split.title() if split != "both" else "Train + Holdout"

    # Get influence submatrix for this rollout episode
    influence_submatrix = influence_matrix[sample_indices, :]

    # Aggregate by demo episode for visualization
    num_demo_eps = len(demo_episodes)
    num_rollout_samples = len(sample_indices)
    episode_influences = np.zeros((num_rollout_samples, num_demo_eps))

    for demo_ep_idx, demo_sample_idxs in enumerate(ep_idxs):
        if len(demo_sample_idxs) > 0:
            episode_influences[:, demo_ep_idx] = influence_submatrix[
                :, demo_sample_idxs
            ].mean(axis=1)

    # Smoothing controls (smooth along rollout timesteps only)
    col1, col2 = st.columns([1, 3])
    with col1:
        apply_gaussian = st.checkbox(
            "Apply Gaussian smoothing (vertical)",
            value=False,
            key=f"influence_heatmap_gaussian_{rollout_episode_idx}_{split}",
            help="Denoise by smoothing along rollout timesteps (vertical axis only)",
        )
    with col2:
        if apply_gaussian:
            gaussian_sigma = st.slider(
                "Gaussian sigma",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key=f"influence_heatmap_sigma_{rollout_episode_idx}_{split}",
                help="Standard deviation of Gaussian kernel (larger = more smoothing)",
            )
        else:
            gaussian_sigma = 1.0

    # Apply Gaussian smoothing if requested (only along axis 0 = rollout timesteps)
    if apply_gaussian:
        episode_influences = gaussian_filter(
            episode_influences, sigma=(gaussian_sigma, 0)
        )

    # Load annotations if available
    with profile("load_annotations_in_influence_heatmap"):
        annotations = (
            load_annotations(annotation_file, task_config=task_config)
            if annotation_file
            else {}
        )

    # Get rollout annotations
    episode_id_str = str(rollout_ep.index)
    rollout_annotations = get_episode_annotations(
        annotations, episode_id_str, split="rollout"
    )

    # Build rollout labels for each timestep
    rollout_labels = []
    for t in range(num_rollout_samples):
        label = get_label_for_frame(t, rollout_annotations)
        rollout_labels.append(label)

    # Build demo label summaries
    demo_label_summaries = []
    for i, demo_ep in enumerate(demo_episodes):
        is_holdout = split == "holdout" or (
            split == "both" and i >= len(data.demo_episodes)
        )
        demo_data_type = "holdout" if is_holdout else "demo"
        episode_id = f"{demo_data_type}_ep{demo_ep.index}"
        dominant_label, unique_labels, label_counts = get_episode_label_summary(
            annotations, episode_id, demo_ep.num_samples
        )
        demo_label_summaries.append(
            {
                "dominant": dominant_label,
                "unique": unique_labels,
                "counts": label_counts,
            }
        )

    # Debug: show annotation status
    num_demo_with_labels = sum(1 for s in demo_label_summaries if s["unique"])
    has_rollout_labels = any(lbl != "no label" for lbl in rollout_labels)
    if num_demo_with_labels == 0 and not has_rollout_labels:
        if annotation_file:
            st.caption(
                f"No annotations for rollout '{rollout_ep.index}' or demos (file has {len(annotations)} keys)"
            )
    else:
        st.caption(
            f"Annotations: {num_demo_with_labels}/{num_demo_eps} demos, rollout has {'labels' if has_rollout_labels else 'no labels'}"
        )

    # Build x-axis labels
    x_labels = []
    for demo_ep in demo_episodes:
        label = f"Demo {demo_ep.index}"
        x_labels.append(label)

    # Custom color map for labels
    custom_color_map: Dict[str, str] = {}

    # Create demo label colors - use dominant label color
    demo_label_colors = [
        _get_label_color(s["dominant"], custom_color_map)
        if s["dominant"] != "no label"
        else "#DDDDDD"
        for s in demo_label_summaries
    ]

    # Create rollout label colors
    rollout_label_colors = [
        _get_label_color(lbl, custom_color_map) if lbl != "no label" else "#DDDDDD"
        for lbl in rollout_labels
    ]

    # Label bar dimensions
    label_bar_width = 3.0  # Width of label bar on left
    label_bar_height = 1.5  # Height of label bar on top

    # Build custom hover data with quality and behavior labels
    quality_labels = data.demo_quality_labels
    has_quality = quality_labels is not None

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Rollout t=%{customdata[1]}<br>"
        "Influence: %{z:.4f}<br>"
    )
    if has_quality:
        hovertemplate += "Quality: %{customdata[2]}<br>"
    hovertemplate += "Demo Labels: %{customdata[3]}<br>Rollout Label: %{customdata[4]}<extra></extra>"

    customdata_array = []
    for t_idx in range(num_rollout_samples):
        row = []
        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            quality = quality_labels.get(demo_ep.index, "N/A") if has_quality else "N/A"
            demo_labels_str = (
                ", ".join(demo_label_summaries[demo_ep_idx]["unique"]) or "no label"
            )
            row.append(
                [
                    x_labels[demo_ep_idx],
                    t_idx,
                    quality,
                    demo_labels_str,
                    rollout_labels[t_idx],
                ]
            )
        customdata_array.append(row)

    # Create the figure using pure plotting function
    fig = plotting.create_influence_heatmap(
        influence_matrix=episode_influences,
        x_labels=x_labels,
        y_labels=rollout_labels,
        title=f"Influence Heatmap: Rollout Episode {rollout_episode_idx} ({split_label} Demos)",
        x_title="Demo Episode",
        y_title="Rollout Timestep",
        show_label_bars=True,
        x_label_colors=demo_label_colors,
        y_label_colors=rollout_label_colors,
        customdata=customdata_array,
        hovertemplate=hovertemplate,
        height=max(400, num_rollout_samples * 8 + 100),
        x_tickvals=[
            i + label_bar_width
            for i in range(0, num_demo_eps, max(1, num_demo_eps // 10))
        ],
        x_ticktext=[
            x_labels[i] for i in range(0, num_demo_eps, max(1, num_demo_eps // 10))
        ],
        y_tickvals=list(
            range(0, num_rollout_samples, max(1, num_rollout_samples // 10))
        ),
        y_ticktext=[
            f"t={i}"
            for i in range(0, num_rollout_samples, max(1, num_rollout_samples // 10))
        ],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"influence_heatmap_{rollout_episode_idx}_{split}",
    )


def render_influence_magnitude_over_time(
    data: InfluenceData,
    rollout_episode_idx: int,
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render a plot showing positive/negative influence magnitude over rollout timesteps.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode to visualize
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Path to the annotation file for behavior labels
        task_config: Task config name for loading annotations from JSON
    """
    rollout_ep = data.rollout_episodes[rollout_episode_idx]
    sample_indices = np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx)

    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    split_label = split.title() if split != "both" else "Train + Holdout"

    # Get influence submatrix for this rollout episode
    # Shape: (num_rollout_timesteps, num_demo_samples)
    influence_submatrix = influence_matrix[sample_indices, :]

    num_rollout_samples = len(sample_indices)

    # For each rollout timestep, aggregate positive and negative influences
    positive_influence = np.zeros(num_rollout_samples)
    negative_influence = np.zeros(num_rollout_samples)

    for t_idx in range(num_rollout_samples):
        timestep_influences = influence_submatrix[t_idx, :]
        positive_influence[t_idx] = np.sum(timestep_influences[timestep_influences > 0])
        negative_influence[t_idx] = np.sum(timestep_influences[timestep_influences < 0])

    # Total magnitude (absolute value of negative + positive)
    total_magnitude = np.abs(negative_influence) + positive_influence

    # Load annotations if available
    with profile("load_annotations_in_magnitude_over_time"):
        annotations = (
            load_annotations(annotation_file, task_config=task_config)
            if annotation_file
            else {}
        )
    episode_id_str = str(rollout_ep.index)
    rollout_annotations = get_episode_annotations(
        annotations, episode_id_str, split="rollout"
    )

    # Build rollout labels for each timestep
    rollout_labels = []
    for t in range(num_rollout_samples):
        label = get_label_for_frame(t, rollout_annotations)
        rollout_labels.append(label)

    # Create the plot
    # Create the figure using pure plotting function
    fig = plotting.create_magnitude_over_time_plot(
        positive_influence=positive_influence,
        negative_influence=negative_influence,
        title=f"Influence Magnitude Over Time: Rollout Episode {rollout_episode_idx} ({split_label} Demos)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"influence_magnitude_{rollout_episode_idx}_{split}",
    )

    # Add statistics
    st.caption("Influence Magnitude Statistics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Positive", f"{np.mean(positive_influence):.4f}")
    with col2:
        st.metric("Avg Negative", f"{np.mean(negative_influence):.4f}")
    with col3:
        st.metric("Avg Total Magnitude", f"{np.mean(total_magnitude):.4f}")


def render_trajectory_influence_heatmap(
    data: InfluenceData,
    rollout_episode_idx: int,
    demo_episode_idx: int,
    split: SplitType = "train",
    unique_key_suffix: str = "",
    annotation_file: str = "",
    task_config: str = "",
    default_rollout_start: int = None,
    default_rollout_end: int = None,
):
    """Render trajectory-level influence heatmap for a rollout-demo pair.

    Args:
        data: InfluenceData object
        rollout_episode_idx: Index of the rollout episode
        demo_episode_idx: Index of the demo episode within the split
        split: Which demo split to use
        unique_key_suffix: Suffix for unique Streamlit key
        annotation_file: Path to the annotation file for behavior labels
        default_rollout_start: Default start index for rollout slice (None = 0)
        default_rollout_end: Default end index for rollout slice (None = full length)
    """
    rollout_ep = data.rollout_episodes[rollout_episode_idx]

    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    if demo_episode_idx >= len(demo_episodes):
        st.warning(
            f"Demo episode index {demo_episode_idx} out of range for split '{split}'"
        )
        return

    demo_ep = demo_episodes[demo_episode_idx]
    demo_sample_idxs = ep_idxs[demo_episode_idx]

    # Extract the submatrix for this rollout-demo pair
    rollout_sample_indices = np.arange(
        rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
    )

    if len(demo_sample_idxs) == 0:
        st.warning("No valid demo samples for this episode")
        return

    # Index into the sliced influence matrix
    traj_influence_matrix = influence_matrix[
        np.ix_(rollout_sample_indices, demo_sample_idxs)
    ]

    # Smoothing controls
    col1, col2 = st.columns([1, 3])
    with col1:
        apply_gaussian = st.checkbox(
            "Apply Gaussian smoothing",
            value=False,
            key=f"trajectory_heatmap_gaussian_{rollout_episode_idx}_{demo_episode_idx}_{split}_{unique_key_suffix}",
            help="Denoise matrix by convolving with a Gaussian kernel",
        )
    with col2:
        if apply_gaussian:
            gaussian_sigma = st.slider(
                "Gaussian sigma",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key=f"trajectory_heatmap_sigma_{rollout_episode_idx}_{demo_episode_idx}_{split}_{unique_key_suffix}",
                help="Standard deviation of Gaussian kernel (larger = more smoothing)",
            )
        else:
            gaussian_sigma = 1.0

    # Rollout slice selection
    max_rollout_len = rollout_ep.num_samples

    # Set default values
    if default_rollout_start is None:
        default_rollout_start = 0
    if default_rollout_end is None:
        default_rollout_end = max_rollout_len

    # Clamp defaults to valid range
    default_rollout_start = max(0, min(default_rollout_start, max_rollout_len - 1))
    default_rollout_end = max(1, min(default_rollout_end, max_rollout_len))

    # Determine if slice is non-trivial (not full range)
    has_custom_slice = (
        default_rollout_start > 0 or default_rollout_end < max_rollout_len
    )

    # Checkbox to enable/disable slice selection
    enable_slice = st.checkbox(
        "Enable rollout slice",
        value=has_custom_slice,
        key=f"trajectory_heatmap_enable_slice_{rollout_episode_idx}_{demo_episode_idx}_{split}_{unique_key_suffix}",
        help="When enabled, only show a slice of the rollout timesteps",
    )

    if enable_slice:
        col_slice1, col_slice2 = st.columns(2)
        with col_slice1:
            rollout_slice_start = st.number_input(
                "Rollout start (inclusive)",
                min_value=0,
                max_value=max_rollout_len - 1,
                value=default_rollout_start,
                key=f"trajectory_heatmap_slice_start_{rollout_episode_idx}_{demo_episode_idx}_{split}_{unique_key_suffix}",
                help="Start index of rollout slice to display",
            )
        with col_slice2:
            rollout_slice_end = st.number_input(
                "Rollout end (exclusive)",
                min_value=1,
                max_value=max_rollout_len,
                value=default_rollout_end,
                key=f"trajectory_heatmap_slice_end_{rollout_episode_idx}_{demo_episode_idx}_{split}_{unique_key_suffix}",
                help="End index of rollout slice to display",
            )

        # Validate slice
        if rollout_slice_start >= rollout_slice_end:
            st.error("Start index must be less than end index")
            return
    else:
        # Full rollout when slice is disabled
        rollout_slice_start = 0
        rollout_slice_end = max_rollout_len

    # Apply rollout slice
    traj_influence_matrix = traj_influence_matrix[
        rollout_slice_start:rollout_slice_end, :
    ]
    rollout_sample_indices = rollout_sample_indices[
        rollout_slice_start:rollout_slice_end
    ]

    # Apply Gaussian smoothing if requested
    if apply_gaussian:
        traj_influence_matrix = gaussian_filter(
            traj_influence_matrix, sigma=gaussian_sigma
        )

    # Load annotations if available
    with profile("load_annotations_in_trajectory_heatmap"):
        annotations = (
            load_annotations(annotation_file, task_config=task_config)
            if annotation_file
            else {}
        )

    # Get rollout annotations - use rollout_ep.index (the actual episode index)
    episode_id_str = str(rollout_ep.index)
    rollout_episode_id = f"rollout_ep{rollout_ep.index}"
    rollout_annotations = get_episode_annotations(
        annotations, episode_id_str, split="rollout"
    )

    # Get demo annotations (determine if holdout or train)
    is_holdout = split == "holdout" or (
        split == "both" and demo_episode_idx >= len(data.demo_episodes)
    )
    demo_split = "holdout" if is_holdout else "train"
    episode_id_str = str(demo_ep.index)
    demo_episode_id = f"{demo_split}_ep{demo_ep.index}"
    demo_annotations = get_episode_annotations(
        annotations, episode_id_str, split=demo_split
    )

    # Debug: show annotation status
    if not annotations:
        st.caption(f"No annotations loaded (file: '{annotation_file}')")
    elif not rollout_annotations and not demo_annotations:
        st.caption(
            f"No annotations for '{rollout_episode_id}' or '{demo_episode_id}' (file has {len(annotations)} keys: {list(annotations.keys())[:3]}...)"
        )
    elif not rollout_annotations:
        st.caption(
            f"No annotations for '{rollout_episode_id}' (demo has {len(demo_annotations)} slices)"
        )
    elif not demo_annotations:
        st.caption(
            f"No annotations for '{demo_episode_id}' (rollout has {len(rollout_annotations)} slices)"
        )

    # Determine actual number of rollout samples (after slice)
    num_rollout_samples_display = len(rollout_sample_indices)

    # Build label arrays for rollout timesteps and demo samples
    # Use the slice offset for correct annotation lookup
    rollout_labels = []
    for t in range(num_rollout_samples_display):
        actual_t = rollout_slice_start + t  # Map back to original timestep
        label = get_label_for_frame(actual_t, rollout_annotations)
        rollout_labels.append(label)

    demo_labels = []
    for t in range(len(demo_sample_idxs)):
        label = get_label_for_frame(t, demo_annotations)
        demo_labels.append(label)

    # Compute symmetric colorbar limits
    abs_max = max(
        abs(np.nanmin(traj_influence_matrix)), abs(np.nanmax(traj_influence_matrix))
    )
    vmin, vmax = -abs_max, abs_max

    # Custom color map for labels
    custom_color_map: Dict[str, str] = {}

    # Create demo label colors (top row)
    demo_label_colors = [
        _get_label_color(lbl, custom_color_map) if lbl != "no label" else "#DDDDDD"
        for lbl in demo_labels
    ]

    # Create rollout label colors (left column)
    rollout_label_colors = [
        _get_label_color(lbl, custom_color_map) if lbl != "no label" else "#DDDDDD"
        for lbl in rollout_labels
    ]

    # Create the main heatmap with offset to leave room for label bars
    # Label bar dimensions (in data coordinates)
    label_bar_width = 3.0  # Width of label bar on left
    label_bar_height = 1.5  # Height of label bar on top

    num_demo_samples = len(demo_sample_idxs)
    num_rollout_samples = num_rollout_samples_display

    # Build custom hover data with labels (optimized for large matrices)
    # Check user preference from session state
    enable_detailed_hover = st.session_state.get("enable_detailed_hover", True)

    # For large matrices (>50k cells) or if disabled by user, skip custom hover
    total_cells = num_rollout_samples * num_demo_samples
    if total_cells > 50000 or not enable_detailed_hover:
        customdata_list = None
        hovertemplate_to_use = (
            "<b>Demo t=%{x}, Rollout t=%{y}</b><br>Influence: %{z:.4f}<extra></extra>"
        )
    else:
        customdata_list = []
        for r_idx in range(num_rollout_samples):
            actual_rollout_t = rollout_slice_start + r_idx  # Map to actual timestep
            row = []
            for d_idx in range(num_demo_samples):
                row.append(
                    [
                        d_idx,
                        actual_rollout_t,
                        f"Rollout: {rollout_labels[r_idx]}<br>Demo: {demo_labels[d_idx]}",
                    ]
                )
            customdata_list.append(row)
        hovertemplate_to_use = (
            "<b>Demo t=%{customdata[0]}, Rollout t=%{customdata[1]}</b><br>"
            "Influence: %{z:.4f}<br>"
            "%{customdata[2]}<extra></extra>"
        )

    # Create title with slice info
    if rollout_slice_start > 0 or rollout_slice_end < max_rollout_len:
        slice_info = f" (rollout t=[{rollout_slice_start}:{rollout_slice_end}])"
    else:
        slice_info = ""
    heatmap_title = f"Trajectory Influence: Rollout {rollout_ep.index} vs Demo {demo_ep.index}{slice_info}"

    # Create the figure using pure plotting function
    fig = plotting.create_influence_heatmap(
        influence_matrix=traj_influence_matrix,
        x_labels=demo_labels,
        y_labels=rollout_labels,
        title=heatmap_title,
        x_title="Demo Episode Sample Index",
        y_title="Rollout Timestep",
        show_label_bars=True,
        x_label_colors=demo_label_colors,
        y_label_colors=rollout_label_colors,
        customdata=customdata_list,
        hovertemplate=hovertemplate_to_use,
        height=max(400, num_rollout_samples * 8 + 100),
        zmin=vmin,
        zmax=vmax,
        x_tickvals=[
            i + label_bar_width
            for i in range(0, num_demo_samples, max(1, num_demo_samples // 10))
        ],
        x_ticktext=[
            str(i) for i in range(0, num_demo_samples, max(1, num_demo_samples // 10))
        ],
        y_tickvals=list(
            range(0, num_rollout_samples, max(1, num_rollout_samples // 10))
        ),
        y_ticktext=[
            str(rollout_slice_start + i)
            for i in range(0, num_rollout_samples, max(1, num_rollout_samples // 10))
        ],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"traj_heatmap_{rollout_episode_idx}_{demo_episode_idx}_{split}_{unique_key_suffix}",
    )

    # Add statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{np.nanmin(traj_influence_matrix):.4f}")
    with col2:
        st.metric("Max", f"{np.nanmax(traj_influence_matrix):.4f}")
    with col3:
        st.metric("Mean", f"{np.nanmean(traj_influence_matrix):.4f}")
    with col4:
        st.metric("Std", f"{np.nanstd(traj_influence_matrix):.4f}")


def render_full_trajectory_influence_heatmap(
    data: InfluenceData,
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render the full trajectory-wise influence matrix.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Path to the annotation file for behavior labels
        task_config: Task config name for loading annotations from JSON
    """
    # Compute trajectory influence matrix
    traj_influence, demo_episodes = compute_trajectory_influence_matrix(data, split)

    # Load annotations if available
    with profile("load_annotations_in_full_trajectory_heatmap"):
        annotations = (
            load_annotations(annotation_file, task_config=task_config)
            if annotation_file
            else {}
        )

    # Build x-axis labels (demo episodes) and get label summaries
    x_labels = []
    demo_label_summaries = []
    quality_labels = data.demo_quality_labels
    has_quality = quality_labels is not None

    for i, demo_ep in enumerate(demo_episodes):
        label = f"Demo {demo_ep.index}"
        x_labels.append(label)

        # Determine if this is holdout
        is_holdout = split == "holdout" or (
            split == "both" and i >= len(data.demo_episodes)
        )
        demo_split = "holdout" if is_holdout else "train"
        episode_id_str = str(demo_ep.index)
        dominant_label, unique_labels, label_counts = get_episode_label_summary(
            annotations, episode_id_str, demo_ep.num_samples, split=demo_split
        )
        demo_label_summaries.append(
            {
                "dominant": dominant_label,
                "unique": unique_labels,
                "counts": label_counts,
            }
        )

    # Build y-axis labels (rollout episodes) and get label summaries
    y_labels = []
    rollout_label_summaries = []
    for rollout_ep in data.rollout_episodes:
        status = (
            "✓"
            if rollout_ep.success
            else "✗"
            if rollout_ep.success is not None
            else "?"
        )
        y_labels.append(f"Rollout {rollout_ep.index} [{status}]")

        episode_id = f"rollout_ep{rollout_ep.index}"
        dominant_label, unique_labels, label_counts = get_episode_label_summary(
            annotations, episode_id, rollout_ep.num_samples
        )
        rollout_label_summaries.append(
            {
                "dominant": dominant_label,
                "unique": unique_labels,
                "counts": label_counts,
            }
        )

    # Debug: show annotation status
    num_demo_with_labels = sum(1 for s in demo_label_summaries if s["unique"])
    num_rollout_with_labels = sum(1 for s in rollout_label_summaries if s["unique"])
    if num_demo_with_labels == 0 and num_rollout_with_labels == 0:
        st.caption(
            f"No annotations found. Annotation file: '{annotation_file}', keys in file: {list(annotations.keys())[:5]}..."
        )
    else:
        st.caption(
            f"Annotations: {num_demo_with_labels}/{len(demo_episodes)} demos, {num_rollout_with_labels}/{len(data.rollout_episodes)} rollouts"
        )

    # Custom color map for labels
    custom_color_map: Dict[str, str] = {}

    # Create demo label colors - use dominant label color
    demo_label_colors = [
        _get_label_color(s["dominant"], custom_color_map)
        if s["dominant"] != "no label"
        else "#DDDDDD"
        for s in demo_label_summaries
    ]

    # Create rollout label colors - use dominant label color
    rollout_label_colors = [
        _get_label_color(s["dominant"], custom_color_map)
        if s["dominant"] != "no label"
        else "#DDDDDD"
        for s in rollout_label_summaries
    ]

    # Dimensions
    num_demos = len(demo_episodes)
    num_rollouts = len(data.rollout_episodes)
    label_bar_width = 2.0  # Width of label bar on left
    label_bar_height = 1.0  # Height of label bar on top

    # Build customdata for hover
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "Trajectory Influence: %{z:.4f}<br>"
    )
    if has_quality:
        hovertemplate += "Demo Quality: %{customdata[2]}<br>"
    hovertemplate += "Demo Labels: %{customdata[3]}<br>Rollout Labels: %{customdata[4]}<extra></extra>"

    customdata_array = []
    for i, rollout_ep in enumerate(data.rollout_episodes):
        row = []
        for j, demo_ep in enumerate(demo_episodes):
            quality = quality_labels.get(demo_ep.index, "N/A") if has_quality else "N/A"
            demo_labels_str = ", ".join(demo_label_summaries[j]["unique"]) or "no label"
            rollout_labels_str = (
                ", ".join(rollout_label_summaries[i]["unique"]) or "no label"
            )
            row.append(
                [x_labels[j], y_labels[i], quality, demo_labels_str, rollout_labels_str]
            )
        customdata_array.append(row)

    split_label = split.title() if split != "both" else "Train + Holdout"

    # Create the figure using pure plotting function
    fig = plotting.create_trajectory_heatmap(
        traj_influence=traj_influence,
        rollout_labels=y_labels,
        demo_labels=x_labels,
        title=f"Trajectory-wise Influence Matrix ({split_label} Demos)",
        rollout_label_colors=rollout_label_colors,
        demo_label_colors=demo_label_colors,
        customdata=customdata_array,
        hovertemplate=hovertemplate,
        x_tickvals=[i + label_bar_width for i in range(num_demos)],
        x_ticktext=x_labels,
        y_tickvals=list(range(num_rollouts)),
        y_ticktext=y_labels,
    )

    # Additional layout tweaks that might be Streamlit-specific or for better density
    fig.update_layout(
        height=max(400, num_rollouts * 25 + 100),
        margin=dict(l=120, r=20, t=40, b=100),
    )
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"full_trajectory_influence_heatmap_{split}",
    )

    # Add statistics
    st.caption(f"Statistics for the trajectory-wise influence matrix ({split_label}):")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{np.nanmin(traj_influence):.4f}")
    with col2:
        st.metric("Max", f"{np.nanmax(traj_influence):.4f}")
    with col3:
        st.metric("Mean", f"{np.nanmean(traj_influence):.4f}")
    with col4:
        st.metric("Std", f"{np.nanstd(traj_influence):.4f}")


def compute_transition_level_statistics(
    data: InfluenceData,
    split: SplitType = "train",
) -> Tuple[np.ndarray, List[Tuple[int, int, str, bool]]]:
    """Compute statistics for all transition-level influence matrices.

    For each rollout-demo pair, computes the mean, std, min, and max of the
    transition-level influence matrix (rollout timesteps x demo timesteps).

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")

    Returns:
        Tuple of:
        - stats: Array of shape (num_rollout_episodes * num_demo_episodes, 4) where each row
          contains [mean, std, min, max] for one transition-level influence matrix.
        - metadata: List of tuples (rollout_idx, demo_idx, quality_label, success) for each row
    """
    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    num_rollouts = len(data.rollout_episodes)
    num_demos = len(demo_episodes)

    # Preallocate statistics array
    stats = np.zeros((num_rollouts * num_demos, 4), dtype=np.float32)
    metadata = []

    # Get quality labels if available
    quality_labels = data.demo_quality_labels

    # Iterate through all rollout-demo pairs
    for rollout_idx, rollout_ep in enumerate(data.rollout_episodes):
        rollout_sample_indices = np.arange(
            rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
        )

        # Get rollout success (default to False if None)
        rollout_success = (
            rollout_ep.success if rollout_ep.success is not None else False
        )

        for demo_idx in range(num_demos):
            demo_sample_idxs = ep_idxs[demo_idx]
            demo_ep = demo_episodes[demo_idx]

            # Get quality label for this demo
            if quality_labels is not None and demo_ep.index in quality_labels:
                quality_label = quality_labels[demo_ep.index]
            else:
                quality_label = "N/A"

            if len(demo_sample_idxs) == 0:
                metadata.append(
                    (rollout_ep.index, demo_ep.index, quality_label, rollout_success)
                )
                continue

            # Extract the transition-level influence matrix for this pair
            traj_influence_matrix = influence_matrix[
                np.ix_(rollout_sample_indices, demo_sample_idxs)
            ]

            # Compute statistics
            stat_idx = rollout_idx * num_demos + demo_idx
            stats[stat_idx, 0] = np.mean(traj_influence_matrix)  # mean
            stats[stat_idx, 1] = np.std(traj_influence_matrix)  # std
            stats[stat_idx, 2] = np.min(traj_influence_matrix)  # min
            stats[stat_idx, 3] = np.max(traj_influence_matrix)  # max

            metadata.append(
                (rollout_ep.index, demo_ep.index, quality_label, rollout_success)
            )

    return stats, metadata


def render_transition_statistics_scatter(
    data: InfluenceData,
    split: SplitType = "train",
    annotation_file: str = "",
    task_config: str = "",
):
    """Render scatter plot of transition-level influence statistics.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Path to the annotation file for behavior labels
        task_config: Task config name for loading annotations from JSON
    """
    # Get demo episodes for later use
    _, demo_episodes_for_stats, _, _ = get_split_data(data, split)

    # Compute statistics
    with st.spinner(
        "Computing transition-level statistics for all rollout-demo pairs..."
    ):
        stats, metadata = compute_transition_level_statistics(data, split)

    # Create dropdowns for axis selection
    col_x, col_y = st.columns(2)

    stat_options = ["mean", "std", "min", "max", "success"]

    with col_x:
        x_stat = st.selectbox(
            "X-axis statistic",
            options=stat_options,
            index=0,
            key=f"transition_stats_x_{split}",
        )

    with col_y:
        y_stat = st.selectbox(
            "Y-axis statistic",
            options=stat_options,
            index=1,
            key=f"transition_stats_y_{split}",
        )

    # Map stat names to column indices or extract success values
    stat_to_idx = {"mean": 0, "std": 1, "min": 2, "max": 3}

    # Extract x and y values
    if x_stat == "success":
        x_values = np.array([1 if success else 0 for _, _, _, success in metadata])
    else:
        x_idx = stat_to_idx[x_stat]
        x_values = stats[:, x_idx]

    if y_stat == "success":
        y_values = np.array([1 if success else 0 for _, _, _, success in metadata])
    else:
        y_idx = stat_to_idx[y_stat]
        y_values = stats[:, y_idx]

    # Build customdata array with stats and metadata
    customdata = []
    for i, (rollout_idx, demo_idx, quality_label, success) in enumerate(metadata):
        success_str = "Success" if success else "Failure"
        customdata.append(
            [
                stats[i, 0],  # mean
                stats[i, 1],  # std
                stats[i, 2],  # min
                stats[i, 3],  # max
                rollout_idx,
                demo_idx,
                quality_label,
                success_str,
            ]
        )

    split_label = split.title() if split != "both" else "Train + Holdout"

    # Create the figure using pure plotting function
    fig = plotting.create_transition_statistics_scatter(
        x_values=x_values,
        y_values=y_values,
        x_stat_name=x_stat,
        y_stat_name=y_stat,
        metadata=[
            {
                "rollout_idx": m[0],
                "demo_idx": m[1],
                "quality": m[2],
                "success": m[3],
                "success_str": "Success" if m[3] else "Failure",
                "mean": stats[i, 0],
                "std": stats[i, 1],
                "min": stats[i, 2],
                "max": stats[i, 3],
            }
            for i, m in enumerate(metadata)
        ],
        title=f"Transition-Level Influence Statistics ({split_label} Demos)",
    )

    split_label = split.title() if split != "both" else "Train + Holdout"

    # Format axis titles
    x_title = (
        "Rollout Success (0=Failure, 1=Success)"
        if x_stat == "success"
        else f"{x_stat.title()} Influence"
    )
    y_title = (
        "Rollout Success (0=Failure, 1=Success)"
        if y_stat == "success"
        else f"{y_stat.title()} Influence"
    )

    fig.update_layout(
        title=f"Transition-Level Influence Statistics ({split_label} Demos)",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=600,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            title="Rollout Status",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    # Display the scatter plot
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"transition_stats_scatter_{split}",
    )

    # Add selector for viewing specific influence matrices
    st.caption("💡 Select a rollout-demo pair below to view its influence matrix")

    col_select_rollout, col_select_demo = st.columns(2)

    with col_select_rollout:
        rollout_options = sorted(list(set([m[0] for m in metadata])))
        selected_rollout = st.selectbox(
            "Select Rollout",
            options=rollout_options,
            key=f"select_rollout_{split}",
        )

    with col_select_demo:
        demo_options = sorted(list(set([m[1] for m in metadata])))
        selected_demo = st.selectbox(
            "Select Demo",
            options=demo_options,
            key=f"select_demo_{split}",
        )

    # Display selected influence matrix
    if selected_rollout is not None and selected_demo is not None:
        # Find quality label
        quality_label = next(
            (
                m[2]
                for m in metadata
                if m[0] == selected_rollout and m[1] == selected_demo
            ),
            "N/A",
        )

        st.subheader(
            f"Influence Matrix: Rollout {selected_rollout} × Demo {selected_demo}"
        )
        st.caption(f"Demo Quality: {quality_label}")

        # Find the demo_idx in the demo_episodes list
        demo_idx_in_list = next(
            (
                i
                for i, ep in enumerate(demo_episodes_for_stats)
                if ep.index == selected_demo
            ),
            None,
        )

        if demo_idx_in_list is not None:
            # Render the trajectory influence heatmap for this pair
            rollout_ep_idx = next(
                (
                    i
                    for i, ep in enumerate(data.rollout_episodes)
                    if ep.index == selected_rollout
                ),
                None,
            )

            if rollout_ep_idx is not None:
                render_trajectory_influence_heatmap(
                    data,
                    rollout_ep_idx,
                    demo_idx_in_list,
                    split=split,
                    unique_key_suffix=f"from_scatter_{selected_rollout}_{selected_demo}",
                    annotation_file=annotation_file,
                )

    # Add overall statistics
    st.caption("Overall Statistics Across All Transition Matrices:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Mean of Means",
            f"{np.mean(stats[:, 0]):.4f}",
            help="Average of mean influences across all transition matrices",
        )
    with col2:
        st.metric(
            "Mean of Stds",
            f"{np.mean(stats[:, 1]):.4f}",
            help="Average of standard deviations across all transition matrices",
        )
    with col3:
        st.metric(
            "Mean of Mins",
            f"{np.mean(stats[:, 2]):.4f}",
            help="Average of minimum influences across all transition matrices",
        )
    with col4:
        st.metric(
            "Mean of Maxs",
            f"{np.mean(stats[:, 3]):.4f}",
            help="Average of maximum influences across all transition matrices",
        )

    st.divider()

    # Compute performance influence for correlation analysis
    performance_influence, demo_episodes = compute_performance_influence(
        data, split=split, metric="net"
    )

    # Build aggregated statistics per demonstration (aggregate across all rollouts)
    demo_stats_dict = {}
    for i, (rollout_idx, demo_ep_idx, quality_label, success) in enumerate(metadata):
        # demo_ep_idx is already the actual episode index from the data

        if demo_ep_idx not in demo_stats_dict:
            demo_stats_dict[demo_ep_idx] = {
                "demo_idx": demo_ep_idx,
                "quality_label": quality_label,
                "mean_vals": [],
                "std_vals": [],
                "min_vals": [],
                "max_vals": [],
                "success_vals": [],  # Track which rollouts succeeded
            }

        demo_stats_dict[demo_ep_idx]["mean_vals"].append(stats[i, 0])
        demo_stats_dict[demo_ep_idx]["std_vals"].append(stats[i, 1])
        demo_stats_dict[demo_ep_idx]["min_vals"].append(stats[i, 2])
        demo_stats_dict[demo_ep_idx]["max_vals"].append(stats[i, 3])
        demo_stats_dict[demo_ep_idx]["success_vals"].append(1 if success else 0)

    # Build data table with aggregated demo-level statistics
    table_data = []
    for demo_ep_index, demo_data in demo_stats_dict.items():
        # Find performance influence for this demo
        demo_idx_in_list = next(
            (i for i, ep in enumerate(demo_episodes) if ep.index == demo_ep_index), None
        )
        perf_infl = (
            performance_influence[demo_idx_in_list]
            if demo_idx_in_list is not None
            else 0.0
        )

        table_data.append(
            {
                "Demo ID": demo_ep_index,
                "Quality": demo_data["quality_label"],
                "Performance Influence": f"{perf_infl:.4f}",
                "Avg Mean": f"{np.mean(demo_data['mean_vals']):.6f}",
                "Avg Std": f"{np.mean(demo_data['std_vals']):.6f}",
                "Avg Min": f"{np.mean(demo_data['min_vals']):.6f}",
                "Avg Max": f"{np.mean(demo_data['max_vals']):.6f}",
                "Std of Means": f"{np.std(demo_data['mean_vals']):.6f}",
                "Std of Stds": f"{np.std(demo_data['std_vals']):.6f}",
                "Count": len(demo_data["mean_vals"]),
            }
        )

    # Sort by performance influence
    table_data = sorted(
        table_data, key=lambda x: float(x["Performance Influence"]), reverse=True
    )

    st.header("Statistics Table")

    # Toggle between aggregated and per-pair views
    view_mode = st.radio(
        "Table View",
        options=["Aggregated by Demo", "Individual Rollout-Demo Pairs"],
        horizontal=True,
        key=f"table_view_mode_{split}",
    )

    import pandas as pd

    if view_mode == "Aggregated by Demo":
        st.markdown("""
        **Column Definitions:**
        - **Demo ID**: Demonstration episode index
        - **Quality**: Human-annotated quality label (better/okay/worse)
        - **Performance Influence**: Net influence on successful vs. failed rollouts (from performance ranking above)
        - **Avg Mean**: Average of mean influence values across all rollout pairs for this demo
        - **Avg Std**: Average of std influence values across all rollout pairs
        - **Avg Min**: Average of minimum influence values across all rollout pairs
        - **Avg Max**: Average of maximum influence values across all rollout pairs
        - **Std of Means**: How much the mean influence varies across different rollouts for the same demo (cross-rollout variability)
        - **Std of Stds**: How much the std influence varies across different rollouts for the same demo (cross-rollout consistency)
        - **Count**: Number of rollout episodes paired with this demonstration

        All statistics are computed from transition-level influence matrices (rollout timesteps × demo timesteps),
        then aggregated per demonstration across all rollouts.
        """)

        st.caption("Sortable by clicking column headers.")
        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            width="stretch",
            height=400,
            key=f"transition_stats_table_agg_{split}",
        )
    else:
        # Individual rollout-demo pairs
        st.markdown("""
        **Column Definitions:**
        - **Rollout ID**: Rollout episode index
        - **Success**: Rollout success (1 = success, 0 = failure)
        - **Demo ID**: Demonstration episode index
        - **Quality**: Demo quality label
        - **Mean**: Mean influence of this specific rollout-demo pair
        - **Std**: Standard deviation of influence
        - **Min**: Minimum influence value
        - **Max**: Maximum influence value

        Each row represents one transition-level influence matrix (rollout timesteps × demo timesteps).
        """)

        # Build per-pair table
        pair_table_data = []
        for i, (rollout_idx, demo_ep_idx, quality_label, success) in enumerate(
            metadata
        ):
            pair_table_data.append(
                {
                    "Rollout ID": rollout_idx,
                    "Success": 1 if success else 0,
                    "Demo ID": demo_ep_idx,
                    "Quality": quality_label,
                    "Mean": f"{stats[i, 0]:.6f}",
                    "Std": f"{stats[i, 1]:.6f}",
                    "Min": f"{stats[i, 2]:.6f}",
                    "Max": f"{stats[i, 3]:.6f}",
                }
            )

        st.caption("Sortable by clicking column headers.")
        df_pairs = pd.DataFrame(pair_table_data)
        st.dataframe(
            df_pairs,
            width="stretch",
            height=400,
            key=f"transition_stats_table_pairs_{split}",
        )

    st.divider()

    # Correlation matrix
    st.header("Correlation Matrix")

    # Toggle between aggregated and per-pair correlation
    corr_view_mode = st.radio(
        "Correlation View",
        options=["Aggregated by Demo", "Individual Rollout-Demo Pairs"],
        horizontal=True,
        key=f"corr_view_mode_{split}",
    )

    if corr_view_mode == "Individual Rollout-Demo Pairs":
        st.caption(
            "Pearson correlation coefficients for individual transition matrices. "
            "Shows relationships between rollout/demo IDs and influence statistics."
        )

        # Build correlation data for individual pairs
        corr_data_pairs = {
            "Rollout ID": [],
            "Demo ID": [],
            "Rollout Success": [],
            "Mean": [],
            "Std": [],
            "Min": [],
            "Max": [],
        }

        # One-hot encoding for quality labels
        quality_better = []
        quality_okay = []
        quality_worse = []

        for i, (rollout_idx, demo_ep_idx, quality_label, success) in enumerate(
            metadata
        ):
            corr_data_pairs["Rollout ID"].append(rollout_idx)
            corr_data_pairs["Demo ID"].append(demo_ep_idx)
            corr_data_pairs["Rollout Success"].append(1 if success else 0)
            corr_data_pairs["Mean"].append(stats[i, 0])
            corr_data_pairs["Std"].append(stats[i, 1])
            corr_data_pairs["Min"].append(stats[i, 2])
            corr_data_pairs["Max"].append(stats[i, 3])

            # One-hot encode quality
            quality_better.append(1 if quality_label == "better" else 0)
            quality_okay.append(1 if quality_label == "okay" else 0)
            quality_worse.append(1 if quality_label == "worse" else 0)

        # Add one-hot encoded quality labels
        corr_data_pairs["Q: Better"] = quality_better
        corr_data_pairs["Q: Okay"] = quality_okay
        corr_data_pairs["Q: Worse"] = quality_worse

        # Create correlation matrix
        corr_df = pd.DataFrame(corr_data_pairs)

        # Drop columns with no variance
        variances = corr_df.std()
        zero_var_cols = variances[variances <= 1e-10].index.tolist()

        if zero_var_cols:
            st.caption(
                f"⚠️ Excluded from correlation (no variance): {', '.join(zero_var_cols)}"
            )
            corr_df = corr_df.loc[:, variances > 1e-10]

        if len(corr_df.columns) < 2:
            st.warning(
                "Not enough variables with variance to compute meaningful correlations."
            )
        else:
            correlation_matrix = corr_df.corr()

            # Create heatmap using pure plotting function
            fig_corr = plotting.create_correlation_matrix_heatmap(
                correlation_matrix=correlation_matrix,
                title="Correlation Matrix (Individual Pairs)",
            )

            st.plotly_chart(
                fig_corr,
                width="stretch",
                key=f"transition_stats_corr_pairs_{split}",
            )

            # Add key insights
            st.caption("**Key Insights:**")

            # Find strongest correlations (excluding diagonal)
            corr_values = []
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    corr_values.append(
                        (
                            correlation_matrix.index[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j],
                        )
                    )

            # Sort by absolute correlation
            corr_values.sort(key=lambda x: abs(x[2]), reverse=True)

            # Display top 5 correlations
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Strongest Positive Correlations:**")
                positive_corrs = [c for c in corr_values if c[2] > 0][:5]
                for var1, var2, corr in positive_corrs:
                    st.text(f"{var1} ↔ {var2}: {corr:.3f}")

            with col_b:
                st.markdown("**Strongest Negative Correlations:**")
                negative_corrs = [c for c in corr_values if c[2] < 0][:5]
                for var1, var2, corr in negative_corrs:
                    st.text(f"{var1} ↔ {var2}: {corr:.3f}")

    else:
        st.caption(
            "Pearson correlation coefficients between demonstration-level attributes. "
            "Shows relationships between quality metrics and influence statistics."
        )

        # Build correlation data (numerical columns only)
        corr_data = {
            "Perf. Influence": [],
            "Avg Mean": [],
            "Avg Std": [],
            "Avg Min": [],
            "Avg Max": [],
            "Std of Means": [],
            "Std of Stds": [],
            "Avg Success": [],  # Average success rate for rollouts paired with this demo
        }

        # One-hot encoding for quality labels
        quality_better = []
        quality_okay = []
        quality_worse = []

        for demo_ep_index, demo_data in demo_stats_dict.items():
            demo_idx_in_list = next(
                (i for i, ep in enumerate(demo_episodes) if ep.index == demo_ep_index),
                None,
            )
            perf_infl = (
                performance_influence[demo_idx_in_list]
                if demo_idx_in_list is not None
                else 0.0
            )

            corr_data["Perf. Influence"].append(perf_infl)
            corr_data["Avg Mean"].append(np.mean(demo_data["mean_vals"]))
            corr_data["Avg Std"].append(np.mean(demo_data["std_vals"]))
            corr_data["Avg Min"].append(np.mean(demo_data["min_vals"]))
            corr_data["Avg Max"].append(np.mean(demo_data["max_vals"]))
            corr_data["Std of Means"].append(np.std(demo_data["mean_vals"]))
            corr_data["Std of Stds"].append(np.std(demo_data["std_vals"]))
            corr_data["Avg Success"].append(np.mean(demo_data["success_vals"]))

            # One-hot encode quality
            quality_label = demo_data["quality_label"]
            quality_better.append(1 if quality_label == "better" else 0)
            quality_okay.append(1 if quality_label == "okay" else 0)
            quality_worse.append(1 if quality_label == "worse" else 0)

        # Add one-hot encoded quality labels
        corr_data["Q: Better"] = quality_better
        corr_data["Q: Okay"] = quality_okay
        corr_data["Q: Worse"] = quality_worse

        # Debug: Show quality distribution
        unique_quality_labels = [
            demo_data["quality_label"] for demo_data in demo_stats_dict.values()
        ]
        quality_dist = pd.Series(unique_quality_labels).value_counts().to_dict()
        st.caption(f"📊 Quality label distribution: {quality_dist}")

        # Create correlation matrix
        corr_df = pd.DataFrame(corr_data)

        # Drop columns with no variance (they would produce NaN correlations)
        # Keep track of which columns we're dropping
        variances = corr_df.std()
        zero_var_cols = variances[variances <= 1e-10].index.tolist()

        if zero_var_cols:
            st.caption(
                f"⚠️ Excluded from correlation (no variance): {', '.join(zero_var_cols)}"
            )
            corr_df = corr_df.loc[:, variances > 1e-10]

        if len(corr_df.columns) < 2:
            st.warning(
                "Not enough variables with variance to compute meaningful correlations."
            )
            return

        correlation_matrix = corr_df.corr()

        # Create heatmap using pure plotting function
        fig_corr = plotting.create_correlation_matrix_heatmap(
            correlation_matrix=correlation_matrix,
            title="Correlation Matrix of Demonstration Statistics",
        )

        st.plotly_chart(
            fig_corr,
            width="stretch",
            key=f"transition_stats_corr_{split}",
        )

        # Add key insights
        st.caption("**Key Insights:**")

        # Find strongest correlations (excluding diagonal)
        corr_values = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr_values.append(
                    (
                        correlation_matrix.index[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j],
                    )
                )

        # Sort by absolute correlation
        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)

        # Display top 5 correlations
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Strongest Positive Correlations:**")
            positive_corrs = [c for c in corr_values if c[2] > 0][:5]
            for var1, var2, corr in positive_corrs:
                st.text(f"{var1} ↔ {var2}: {corr:.3f}")

        with col_b:
            st.markdown("**Strongest Negative Correlations:**")
            negative_corrs = [c for c in corr_values if c[2] < 0][:5]
            for var1, var2, corr in negative_corrs:
                st.text(f"{var1} ↔ {var2}: {corr:.3f}")


def render_transition_statistics_density(
    data: InfluenceData,
    split: SplitType = "train",
):
    """Render overlapping density histograms for a selected statistic comparing success vs. failure.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
    """
    # Compute transition statistics
    with st.spinner("Computing transition-level statistics..."):
        stats, metadata = compute_transition_level_statistics(data, split)

    # Selector for which statistic to plot
    stat_to_plot = st.selectbox(
        "Select statistic to visualize",
        options=["mean", "std", "min", "max"],
        index=1,  # Default to std
        key=f"density_stat_{split}",
        help="Choose which influence statistic to compare between successful and failed rollouts",
    )

    stat_to_idx = {"mean": 0, "std": 1, "min": 2, "max": 3}
    stat_idx = stat_to_idx[stat_to_plot]

    # Separate data by success/failure
    success_values = []
    failure_values = []

    for i, (rollout_idx, demo_idx, quality_label, success) in enumerate(metadata):
        if success:
            success_values.append(stats[i, stat_idx])
        else:
            failure_values.append(stats[i, stat_idx])

    success_values = np.array(success_values)
    failure_values = np.array(failure_values)

    split_label = split.title() if split != "both" else "Train + Holdout"

    # Create the figure using pure plotting function
    fig = plotting.create_density_plots(
        success_values=success_values,
        failure_values=failure_values,
        stat_name=stat_to_plot.title(),
        plot_type="histogram",
        title=f"Histograms: {stat_to_plot.title()} Influence ({split_label} Demos)",
    )

    st.plotly_chart(fig, width="stretch", key=f"density_hist_{split}")

    st.divider()

    # Create normalized probability plot
    st.subheader("Normalized Probability Distribution")

    fig_norm = plotting.create_density_plots(
        success_values=success_values,
        failure_values=failure_values,
        stat_name=stat_to_plot.title(),
        plot_type="normalized",
        title=f"Normalized Probability: {stat_to_plot.title()} Influence ({split_label} Demos)",
    )

    st.plotly_chart(fig_norm, width="stretch", key=f"density_norm_{split}")

    st.divider()

    # Create CDF plot
    st.subheader("Cumulative Distribution Function (CDF)")

    fig_cdf = plotting.create_density_plots(
        success_values=success_values,
        failure_values=failure_values,
        stat_name=stat_to_plot.title(),
        plot_type="cdf",
        title=f"Cumulative Distribution: {stat_to_plot.title()} Influence ({split_label} Demos)",
    )

    # Add grid for easier reading
    fig_cdf.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig_cdf.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    st.plotly_chart(fig_cdf, width="stretch", key=f"density_cdf_{split}")

    # Add statistics comparison
    st.caption("**Distribution Statistics:**")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Successful Rollouts**")
        st.text(f"Mean: {np.mean(success_values):.6f}")
        st.text(f"Std: {np.std(success_values):.6f}")
        st.text(f"Median: {np.median(success_values):.6f}")
        st.text(f"Count: {len(success_values):,}")

    with col2:
        st.markdown("**Failed Rollouts**")
        st.text(f"Mean: {np.mean(failure_values):.6f}")
        st.text(f"Std: {np.std(failure_values):.6f}")
        st.text(f"Median: {np.median(failure_values):.6f}")
        st.text(f"Count: {len(failure_values):,}")


def render_influence_distribution_by_success(
    data: InfluenceData,
    split: SplitType = "train",
):
    """Render histograms comparing influence value distributions for successful vs. failed rollouts.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
    """
    # Get split-specific data
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get rollout data
    rollout_ep_idxs, rollout_ep_lens, success_mask = get_rollout_data(data)

    # Collect all influence values for successful and failed rollouts
    success_influences = []
    failure_influences = []

    for rollout_idx, rollout_ep in enumerate(data.rollout_episodes):
        rollout_sample_indices = np.arange(
            rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
        )

        # Get all influence values for this rollout (flatten the matrix)
        rollout_influences = influence_matrix[rollout_sample_indices, :].flatten()

        # Categorize by success/failure
        if rollout_ep.success:
            success_influences.append(rollout_influences)
        elif rollout_ep.success is False:  # Explicitly check for False (not None)
            failure_influences.append(rollout_influences)

    # Convert to numpy arrays - concatenate all rollouts
    success_influences = (
        np.concatenate(success_influences) if success_influences else np.array([])
    )
    failure_influences = (
        np.concatenate(failure_influences) if failure_influences else np.array([])
    )

    # Determine common bin range for fair comparison
    all_influences = np.concatenate([success_influences, failure_influences])
    bin_min, bin_max = np.percentile(
        all_influences, [1, 99]
    )  # Use 1-99 percentile to avoid outliers

    # Pre-compute histogram bins server-side (much more efficient than sending raw data)
    num_bins = 50
    bins = np.linspace(bin_min, bin_max, num_bins + 1)

    # Compute histogram counts
    success_counts, _ = np.histogram(success_influences, bins=bins)
    failure_counts, _ = np.histogram(failure_influences, bins=bins)

    # Bin centers for x-axis
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    split_label = split.title() if split != "both" else "Train + Holdout"

    # Create the figure using pure plotting function
    fig = plotting.create_distribution_comparison_plot(
        success_counts=success_counts,
        failure_counts=failure_counts,
        bin_centers=bin_centers,
        bin_width=bin_width,
        title=f"Influence Distribution Comparison: Success vs. Failure ({split_label} Demos)",
    )

    st.plotly_chart(fig, width="stretch", key=f"influence_dist_comparison_{split}")

    # Add statistics comparison
    st.caption("**Distribution Statistics (computed on all influence values):**")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Successful Rollouts**")
        st.text(f"Mean: {np.mean(success_influences):.6f}")
        st.text(f"Std: {np.std(success_influences):.6f}")
        st.text(f"Median: {np.median(success_influences):.6f}")
        st.text(f"Min: {np.min(success_influences):.6f}")
        st.text(f"Max: {np.max(success_influences):.6f}")
        st.text(f"Total values: {len(success_influences):,}")

    with col2:
        st.markdown("**Failed Rollouts**")
        st.text(f"Mean: {np.mean(failure_influences):.6f}")
        st.text(f"Std: {np.std(failure_influences):.6f}")
        st.text(f"Median: {np.median(failure_influences):.6f}")
        st.text(f"Min: {np.min(failure_influences):.6f}")
        st.text(f"Max: {np.max(failure_influences):.6f}")
        st.text(f"Total values: {len(failure_influences):,}")


def render_performance_influence(
    data: InfluenceData,
    split: SplitType = "train",
    metric: str = "net",
    top_k: int = 20,
):
    """Render performance influence scores for each demonstration trajectory.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        metric: Quality metric ("net", "succ", or "fail")
        top_k: Number of top/bottom demonstrations to display
    """
    # Compute performance influence
    performance_influence, demo_episodes = compute_performance_influence(
        data, split=split, metric=metric
    )

    # Build labels with episode indices and quality tiers
    demo_labels = []
    quality_labels = data.demo_quality_labels

    # Track which episodes are holdout (only relevant when split="both")
    holdout_start_idx = len(data.demo_episodes) if split == "both" else float("inf")

    for i, demo_ep in enumerate(demo_episodes):
        label = f"Demo {demo_ep.index}"

        # Add quality tier if available
        if quality_labels is not None and demo_ep.index in quality_labels:
            quality = quality_labels[demo_ep.index]
            label += f" [{quality}]"

        # Mark holdout demos when showing both
        if split == "both" and i >= holdout_start_idx:
            label += " [HOLDOUT]"

        demo_labels.append(label)

    # Sort by performance influence
    sorted_indices = np.argsort(performance_influence)[::-1]  # Descending order

    # Display statistics
    split_label = split.title() if split != "both" else "Train + Holdout"
    st.markdown(f"**Performance Influence Statistics ({split_label})**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{np.min(performance_influence):.4f}")
    with col2:
        st.metric("Max", f"{np.max(performance_influence):.4f}")
    with col3:
        st.metric("Mean", f"{np.mean(performance_influence):.4f}")
    with col4:
        st.metric("Std", f"{np.std(performance_influence):.4f}")

    st.divider()

    # Determine color based on sign
    colors = [
        "green" if val > 0 else "red" for val in performance_influence[sorted_indices]
    ]

    # Create the figure using pure plotting function
    fig = plotting.create_performance_influence_bar_plot(
        scores=performance_influence[sorted_indices],
        labels=[demo_labels[idx] for idx in sorted_indices],
        title=f"Performance Influence by Demonstration - Sorted ({split_label})",
    )

    st.plotly_chart(fig, width="stretch", key=f"perf_influence_bar_{split}")

    st.divider()

    # Display top and bottom demonstrations
    col_top, col_bottom = st.columns(2)

    with col_top:
        st.markdown(f"**Top {top_k} Demonstrations (Highest Performance Influence)**")
        top_indices = sorted_indices[:top_k]

        top_data = []
        for rank, idx in enumerate(top_indices, 1):
            demo_ep = demo_episodes[idx]
            score = performance_influence[idx]
            top_data.append(
                {
                    "Rank": rank,
                    "Demo ID": demo_ep.index,
                    "Score": f"{score:.4f}",
                    "Label": demo_labels[idx],
                }
            )

        st.table(top_data)

    with col_bottom:
        st.markdown(f"**Bottom {top_k} Demonstrations (Lowest Performance Influence)**")
        bottom_indices = sorted_indices[-top_k:][::-1]  # Reverse to show worst first

        bottom_data = []
        for rank, idx in enumerate(bottom_indices, 1):
            demo_ep = demo_episodes[idx]
            score = performance_influence[idx]
            bottom_data.append(
                {
                    "Rank": rank,
                    "Demo ID": demo_ep.index,
                    "Score": f"{score:.4f}",
                    "Label": demo_labels[idx],
                }
            )

        st.table(bottom_data)
