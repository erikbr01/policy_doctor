"""Plotly plotting functions for influence visualizations (no Streamlit dependency).

This module provides pure Plotly functions that return figures instead of rendering
to Streamlit. Use these for notebooks, scripts, or other non-Streamlit contexts.
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from influence_visualizer.data_loader import EpisodeInfo, InfluenceData
from influence_visualizer.render_annotation import get_episode_annotations, get_label_for_frame, load_annotations
from influence_visualizer.render_frames import LABEL_COLORS, _get_label_color
from influence_visualizer.render_heatmaps import (
    compute_performance_influence,
    compute_trajectory_influence_matrix,
    get_rollout_data,
    get_split_data,
)

# Type alias for split options
SplitType = Literal["train", "holdout", "both"]


def get_episode_label_summary(
    annotations: Dict,
    episode_id: str,
    num_frames: int,
) -> Tuple[str, List[str], Dict[str, int]]:
    """Get a summary of labels for an episode.

    Args:
        annotations: Dictionary of all annotations
        episode_id: ID of the episode (e.g., "demo_ep0", "rollout_ep1")
        num_frames: Total number of frames in the episode

    Returns:
        Tuple of:
        - dominant_label: Most common label (or "no label" if no annotations)
        - unique_labels: List of unique labels in the episode
        - label_counts: Dict mapping label to frame count
    """
    episode_annotations = annotations.get(episode_id, [])

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


def plot_trajectory_influence_heatmap(
    data: InfluenceData,
    split: SplitType = "train",
    annotation_file: Optional[str] = None,
) -> go.Figure:
    """Create trajectory-wise influence matrix heatmap.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        annotation_file: Optional path to annotation file for behavior labels

    Returns:
        Plotly Figure object
    """
    # Compute trajectory influence matrix
    traj_influence, demo_episodes = compute_trajectory_influence_matrix(data, split)

    # Load annotations if available
    annotations = load_annotations(annotation_file, task_config=task_config) if annotation_file else {}

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

    fig = go.Figure()

    # Add main influence heatmap
    fig.add_trace(
        go.Heatmap(
            z=traj_influence,
            x=x_labels,
            y=y_labels,
            colorscale=[(0, "red"), (0.5, "white"), (1, "green")],
            zmid=0,
            customdata=customdata_array,
            hovertemplate=hovertemplate,
            colorbar=dict(title="Influence"),
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Trajectory-wise Influence Matrix ({split} demos)",
        xaxis_title="Demo Episode",
        yaxis_title="Rollout Episode",
        height=max(400, len(data.rollout_episodes) * 20),
        width=max(600, len(demo_episodes) * 20),
    )

    return fig


def plot_performance_influence(
    data: InfluenceData,
    split: SplitType = "train",
    metric: str = "net",
    top_k: int = 20,
) -> go.Figure:
    """Create performance influence visualization.

    Args:
        data: InfluenceData object
        split: Which demo split to use
        metric: "net", "succ", or "fail"
        top_k: Number of top/bottom demos to show

    Returns:
        Plotly Figure with subplots
    """
    perf_influence, demo_episodes = compute_performance_influence(data, split, metric)

    # Get top and bottom demonstrations
    sorted_indices = np.argsort(perf_influence)[::-1]
    top_indices = sorted_indices[:top_k]
    bottom_indices = sorted_indices[-top_k:][::-1]

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Distribution of Performance Influence ({metric})",
            f"Top {top_k} and Bottom {top_k} Demonstrations",
        ),
        horizontal_spacing=0.15,
    )

    # Plot 1: Distribution histogram
    fig.add_trace(
        go.Histogram(
            x=perf_influence,
            nbinsx=50,
            name="Distribution",
            marker_color="steelblue",
            opacity=0.7,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add zero line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Zero",
        row=1,
        col=1,
    )

    # Plot 2: Top and bottom bars
    demo_labels = [f"Demo {i}" for i in range(len(perf_influence))]
    y_pos = list(range(2 * top_k))
    values = np.concatenate(
        [perf_influence[top_indices], perf_influence[bottom_indices]]
    )
    colors = ["green" if v > 0 else "red" for v in values]
    labels = [demo_labels[i] for i in top_indices] + [
        demo_labels[i] for i in bottom_indices
    ]

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            opacity=0.7,
            showlegend=False,
            text=[f"{v:.4f}" for v in values],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    # Add zero line and separator
    fig.add_vline(x=0, line_color="black", line_width=1, row=1, col=2)
    fig.add_hline(
        y=top_k - 0.5,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        row=1,
        col=2,
    )

    # Update axes
    fig.update_xaxes(title_text="Performance Influence Score", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Performance Influence Score", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    # Update layout
    fig.update_layout(
        height=600,
        width=1400,
        showlegend=False,
        title_text=f"Performance Influence Analysis (metric={metric}, split={split})",
    )

    return fig


def plot_influence_distribution_by_success(
    data: InfluenceData,
    split: SplitType = "train",
) -> go.Figure:
    """Create overlapping histogram comparing success vs failure influence distributions.

    Args:
        data: InfluenceData object
        split: Which demo split to use

    Returns:
        Plotly Figure object
    """
    # Get split-specific influence matrix
    influence_matrix, demo_episodes, _, _ = get_split_data(data, split)

    # Separate by success/failure
    success_influences = []
    failure_influences = []

    for ep_idx, ep in enumerate(data.rollout_episodes):
        sample_indices = np.arange(ep.sample_start_idx, ep.sample_end_idx)
        influences = influence_matrix[sample_indices, :].flatten()

        if ep.success:
            success_influences.extend(influences)
        else:
            failure_influences.extend(influences)

    success_influences = np.array(success_influences)
    failure_influences = np.array(failure_influences)

    # Create figure
    fig = go.Figure()

    # Add histograms
    fig.add_trace(
        go.Histogram(
            x=success_influences,
            nbinsx=100,
            name="Success",
            marker_color="green",
            opacity=0.6,
            histnorm="probability density",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=failure_influences,
            nbinsx=100,
            name="Failure",
            marker_color="red",
            opacity=0.6,
            histnorm="probability density",
        )
    )

    # Update layout
    fig.update_layout(
        title="Influence Distribution: Success vs. Failure",
        xaxis_title="Influence Score",
        yaxis_title="Density",
        barmode="overlay",
        height=600,
        width=1000,
        hovermode="x unified",
    )

    return fig


def plot_transition_statistics_density(
    data: InfluenceData,
    split: SplitType = "train",
) -> go.Figure:
    """Create density plots for transition-level influence statistics.

    Args:
        data: InfluenceData object
        split: Which demo split to use

    Returns:
        Plotly Figure with subplots
    """
    # Return placeholder - full implementation requires expensive computation
    fig = go.Figure()
    fig.add_annotation(
        text="Transition statistics visualization not yet implemented in notebook mode.<br>"
        "This requires compute_transition_level_statistics which is computationally expensive.<br>"
        "Use the Streamlit app for this visualization.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
        align="center",
    )

    fig.update_layout(
        title="Transition Statistics Density (Not Available)",
        height=400,
        width=800,
    )

    return fig

    # Original implementation below (commented out - requires expensive computation)
    """
    from influence_visualizer.render_heatmaps import compute_transition_level_statistics

    # Compute statistics
    stats, metadata = compute_transition_level_statistics(data, split)

    if stats is None:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for transition statistics",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    """

    # Create subplots for each statistic
    stat_names = ["mean", "std", "min", "max"]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{s.capitalize()} Influence" for s in stat_names],
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for stat_name, (row, col) in zip(stat_names, positions):
        success_vals = stats_by_success.get("success", {}).get(stat_name, [])
        failure_vals = stats_by_success.get("failure", {}).get(stat_name, [])

        if len(success_vals) > 0:
            fig.add_trace(
                go.Histogram(
                    x=success_vals,
                    nbinsx=50,
                    name="Success",
                    marker_color="green",
                    opacity=0.6,
                    histnorm="probability density",
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )

        if len(failure_vals) > 0:
            fig.add_trace(
                go.Histogram(
                    x=failure_vals,
                    nbinsx=50,
                    name="Failure",
                    marker_color="red",
                    opacity=0.6,
                    histnorm="probability density",
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=f"{stat_name.capitalize()} Value", row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)

    fig.update_layout(
        title="Transition Statistics Density: Success vs. Failure",
        height=800,
        width=1200,
        barmode="overlay",
    )

    return fig
