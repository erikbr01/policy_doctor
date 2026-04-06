"""Pure frame and action plotting functions.

This module contains functions for visualizing frames, actions, and timelines
without any Streamlit dependencies. Functions return PIL Images or Plotly Figures.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from policy_doctor.plotting.common import get_label_color


def create_annotated_frame(
    img: np.ndarray,
    label: str,
    font_size: int = 12,
) -> Image.Image:
    """Create an annotated PIL Image with text overlay.

    Args:
        img: RGB numpy array (H, W, 3) or (H, W) for grayscale
        label: Text to overlay on the image
        font_size: Font size for the overlay text

    Returns:
        PIL Image with annotation
    """
    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

    # Get text dimensions
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position in top-left corner with padding
    padding = 3
    x, y = padding, padding

    # Draw background rectangle for visibility
    draw.rectangle(
        [x - 2, y - 2, x + text_width + 4, y + text_height + 4],
        fill=(0, 0, 0, 180),
    )

    # Draw text
    draw.text((x, y), label, fill=(255, 255, 255), font=font)

    return pil_img


def create_action_plot(
    actions: np.ndarray,
    action_labels: List[str],
    title: str = "Action Chunk",
    max_dims: int = 10,
) -> go.Figure:
    """Create action chunk visualization.

    Args:
        actions: Action array (T, D) for sequence or (D,) for single action
        action_labels: Labels for each action dimension
        title: Plot title
        max_dims: Maximum number of dimensions to display

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Handle 1D actions
    if len(actions.shape) == 1:
        actions = actions.reshape(1, -1)

    timesteps = np.arange(len(actions))
    action_dim = actions.shape[-1]

    # Plot each dimension
    for dim in range(min(action_dim, max_dims)):
        label = action_labels[dim] if dim < len(action_labels) else f"dim_{dim}"
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=actions[:, dim],
                mode="lines+markers",
                name=label,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Timestep in Horizon",
        yaxis_title="Action Value",
        height=250,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_label_timeline(
    annotations: List[Dict],
    num_frames: int,
    current_frame: Optional[int] = None,
    unlabeled_name: str = "unlabeled",
) -> go.Figure:
    """Create a color-coded timeline bar showing labels over time.

    Displays horizontal bars like a linear video editor, where each annotation
    slice is shown as a colored segment. Unlabeled regions are shown in gray.
    A vertical marker indicates the current frame position.

    Args:
        annotations: List of annotation dicts with 'start', 'end', 'label' keys
        num_frames: Total number of frames in the episode
        current_frame: Optional current frame index for position marker
        unlabeled_name: Legend and hover label for unannotated segments (e.g. "Unassigned")

    Returns:
        Plotly Figure object
    """
    if num_frames <= 0:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(height=80)
        return fig

    fig = go.Figure()
    custom_color_map: Dict[str, str] = {}
    legend_labels: Set[str] = set()

    # Sort annotations by start time
    sorted_annotations = sorted(annotations, key=lambda x: x["start"])

    # Collect unlabeled gaps
    gaps = []
    prev_end = -1
    for ann in sorted_annotations:
        if ann["start"] > prev_end + 1:
            gaps.append({"start": prev_end + 1, "end": ann["start"] - 1})
        prev_end = max(prev_end, ann["end"])
    if prev_end < num_frames - 1:
        gaps.append({"start": prev_end + 1, "end": num_frames - 1})

    # Draw unlabeled gaps
    for gap in gaps:
        show_legend = unlabeled_name not in legend_labels
        legend_labels.add(unlabeled_name)
        fig.add_trace(
            go.Bar(
                x=[gap["end"] - gap["start"] + 1],
                y=["Labels"],
                base=[gap["start"]],
                orientation="h",
                marker=dict(color="rgba(220, 220, 220, 0.5)"),
                name=unlabeled_name,
                showlegend=show_legend,
                legendgroup=unlabeled_name,
                hovertemplate=(
                    f"{unlabeled_name}<br>frames {gap['start']}-{gap['end']}<extra></extra>"
                ),
            )
        )

    # Draw annotation bars
    for ann in sorted_annotations:
        color = get_label_color(ann["label"], custom_color_map)
        show_legend = ann["label"] not in legend_labels
        legend_labels.add(ann["label"])
        fig.add_trace(
            go.Bar(
                x=[ann["end"] - ann["start"] + 1],
                y=["Labels"],
                base=[ann["start"]],
                orientation="h",
                marker=dict(color=color),
                name=ann["label"],
                showlegend=show_legend,
                legendgroup=ann["label"],
                hovertemplate=(
                    f"{ann['label']}<br>"
                    f"frames {ann['start']}-{ann['end']}<extra></extra>"
                ),
            )
        )

    # Draw current frame marker
    if current_frame is not None:
        fig.add_vline(
            x=current_frame,
            line_width=2,
            line_dash="solid",
            line_color="black",
        )

    fig.update_layout(
        barmode="stack",
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[0, num_frames],
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
        ),
        showlegend=True,
    )

    return fig


def create_cluster_timeline(
    cluster_assignments: List[int],
    num_frames: int,
    current_frame: Optional[int] = None,
    has_true_noise: bool = False,
    all_cluster_ids: Optional[List[int]] = None,
) -> go.Figure:
    """Create a color-coded timeline bar showing cluster assignments over time.

    Similar to label timeline but for cluster IDs. Each cluster gets a distinct color.

    Args:
        cluster_assignments: List of cluster IDs (length = num_frames)
            -1 can mean either "noise from clustering algorithm" or "frame not in clustering sample"
        num_frames: Total number of frames in the episode
        current_frame: Optional current frame index for position marker
        has_true_noise: If True, -1 represents noise from algorithm (DBSCAN/OPTICS).
                       If False, -1 represents unassigned frames (not included in clustering)
        all_cluster_ids: Optional global list of all cluster IDs for consistent coloring.
                        If provided, colors are assigned based on global cluster ID order.
                        If None, colors are assigned based on clusters present in this timeline.

    Returns:
        Plotly Figure object
    """
    if num_frames <= 0 or len(cluster_assignments) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(height=80)
        return fig

    fig = go.Figure()

    # Color map for clusters - use distinctive colors with high contrast
    import plotly.express as px

    # Combine multiple qualitative palettes for more distinctive colors
    colors = (
        px.colors.qualitative.Bold  # Vivid colors
        + px.colors.qualitative.Vivid  # More vivid colors
        + px.colors.qualitative.Dark24  # Additional distinct colors
    )

    # Use global cluster IDs if provided for consistent coloring across episodes
    if all_cluster_ids is not None:
        # Build color map based on global cluster ID order
        cluster_color_map = {}
        for i, cluster_id in enumerate(all_cluster_ids):
            if cluster_id == -1:
                # Gray for -1, but label depends on context
                cluster_color_map[cluster_id] = "rgba(128, 128, 128, 0.5)"
            else:
                cluster_color_map[cluster_id] = colors[i % len(colors)]
    else:
        # Fall back to local unique clusters
        unique_clusters = sorted(set(cluster_assignments))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            unique_clusters.append(-1)  # Put noise at the end

        cluster_color_map = {}
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:
                # Gray for -1, but label depends on context
                cluster_color_map[cluster_id] = "rgba(128, 128, 128, 0.5)"
            else:
                cluster_color_map[cluster_id] = colors[i % len(colors)]

    legend_added = set()

    # Group consecutive frames with the same cluster
    segments = []
    start_idx = 0
    current_cluster = cluster_assignments[0]

    for i in range(1, len(cluster_assignments)):
        if cluster_assignments[i] != current_cluster:
            segments.append(
                {
                    "start": start_idx,
                    "end": i - 1,
                    "cluster": current_cluster,
                }
            )
            start_idx = i
            current_cluster = cluster_assignments[i]

    # Add final segment
    segments.append(
        {
            "start": start_idx,
            "end": len(cluster_assignments) - 1,
            "cluster": current_cluster,
        }
    )

    # Draw segments
    for seg in segments:
        cluster_id = seg["cluster"]

        # Handle -1 (unassigned/noise) even if not in color map
        if cluster_id == -1 and cluster_id not in cluster_color_map:
            cluster_color_map[cluster_id] = "rgba(128, 128, 128, 0.5)"

        color = cluster_color_map[cluster_id]

        # Label -1 based on context
        if cluster_id == -1:
            cluster_name = "Noise" if has_true_noise else "Unassigned"
        else:
            cluster_name = f"Cluster {cluster_id}"

        show_legend = cluster_id not in legend_added
        legend_added.add(cluster_id)

        fig.add_trace(
            go.Bar(
                x=[seg["end"] - seg["start"] + 1],
                y=["Clusters"],
                base=[seg["start"]],
                orientation="h",
                marker=dict(color=color),
                name=cluster_name,
                showlegend=show_legend,
                legendgroup=f"cluster_{cluster_id}",
                hovertemplate=(
                    f"{cluster_name}<br>"
                    f"frames {seg['start']}-{seg['end']}<extra></extra>"
                ),
            )
        )

    # Draw current frame marker
    if current_frame is not None:
        fig.add_vline(
            x=current_frame,
            line_width=2,
            line_dash="solid",
            line_color="black",
        )

    fig.update_layout(
        barmode="stack",
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[0, num_frames],
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
        ),
        showlegend=True,
    )

    return fig


def create_state_action_scatter(
    embeddings: np.ndarray,
    labels: List[str],
    colors: Optional[List[str]] = None,
    title: str = "State-Action Embeddings",
    marker_size: int = 6,
) -> go.Figure:
    """Create a 2D scatter plot of state-action embeddings.

    Args:
        embeddings: 2D array of shape (N, 2) with x, y coordinates
        labels: Labels for each point (for hover)
        colors: Optional colors for each point
        title: Plot title
        marker_size: Size of scatter markers

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if colors is None:
        colors = ["blue"] * len(embeddings)

    fig.add_trace(
        go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=colors,
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=labels,
            hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=500,
        showlegend=False,
    )

    return fig
