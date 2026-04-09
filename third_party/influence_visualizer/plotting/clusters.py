"""Pure plotting functions for clustering algorithm results.

This module provides Streamlit-independent plotting functions for visualizing
clustering results, including cluster scatter plots, label coherency charts,
silhouette plots, and evaluation metric displays.

No Streamlit imports allowed in this module.
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from influence_visualizer.plotting.common import get_label_color

ColorByOption = Literal[
    "cluster", "success", "timestep", "quality", "rollout_idx", "demo_idx"
]

# Qualitative color palette for clusters
CLUSTER_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#f7b6d2",  # light pink
    "#c7c7c7",  # light gray
    "#dbdb8d",  # light olive
    "#9edae5",  # light cyan
]

NOISE_COLOR = "#cccccc"


def _get_cluster_color(cluster_id: int) -> str:
    """Get color for a cluster ID. Noise (-1) gets gray."""
    if cluster_id == -1:
        return NOISE_COLOR
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def _build_hover_parts(meta: Dict, cluster_label: int) -> List[str]:
    """Build hover text parts for a single point."""
    parts = [f"Cluster: {cluster_label}"]
    if "rollout_idx" in meta:
        parts.append(f"Rollout: {meta['rollout_idx']}")
        parts.append(f"Success: {'Yes' if meta.get('success') else 'No'}")
    elif "demo_idx" in meta:
        parts.append(f"Demo: {meta['demo_idx']}")
        parts.append(f"Quality: {meta.get('quality_label', 'unknown')}")
    if "timestep" in meta:
        parts.append(f"Timestep: {meta['timestep']}")
    if "window_start" in meta:
        parts.append(f"Window: {meta['window_start']}-{meta['window_end']}")
    if "annotation_label" in meta:
        parts.append(f"Label: {meta['annotation_label']}")
    if "mean_influence" in meta:
        parts.append(f"Mean influence: {meta['mean_influence']:.2f}")
    return parts


def create_cluster_scatter_2d(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    title: str = "Cluster Assignments",
    height: int = 600,
    color_by: ColorByOption = "cluster",
) -> go.Figure:
    """Create a 2D scatter plot colored by cluster or metadata attribute.

    Args:
        embeddings_2d: 2D array of shape (N, 2) from t-SNE/UMAP projection
        cluster_labels: Array of cluster assignments (shape N). -1 = noise.
        metadata: List of metadata dicts for hover text
        title: Plot title
        height: Plot height
        color_by: "cluster" | "success" | "timestep" | "quality" | "rollout_idx" | "demo_idx"

    Returns:
        Plotly Figure object
    """
    n = len(embeddings_2d)
    hover_texts = [
        "<br>".join(_build_hover_parts(metadata[i], int(cluster_labels[i])))
        for i in range(n)
    ]

    fig = go.Figure()

    if color_by == "cluster":
        unique_labels = sorted(set(cluster_labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
            unique_labels.append(-1)
        for label in unique_labels:
            mask = cluster_labels == label
            indices = np.where(mask)[0]
            if label == -1:
                name = f"Noise ({len(indices)})"
                color = NOISE_COLOR
                opacity = 0.3
            else:
                name = f"Cluster {label} ({len(indices)})"
                color = _get_cluster_color(int(label))
                opacity = 0.7
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[indices, 0],
                    y=embeddings_2d[indices, 1],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=0.5, color="white"),
                        opacity=opacity,
                    ),
                    text=[hover_texts[i] for i in indices],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    elif color_by == "success":
        # Categorical: Success / Failure / Unknown (by rollout success)
        success_colors = {"Success": "#2ca02c", "Failure": "#d62728", "Unknown": "#7f7f7f"}
        has_success = np.array(["success" in metadata[i] for i in range(n)])
        success_true = np.array([metadata[i].get("success") is True for i in range(n)])
        success_false = has_success & np.array([metadata[i].get("success") is False for i in range(n)])
        unknown = ~has_success
        for name_key, mask in (
            ("Success", success_true),
            ("Failure", success_false),
            ("Unknown", unknown),
        ):
            color = success_colors[name_key]
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[indices, 0],
                    y=embeddings_2d[indices, 1],
                    mode="markers",
                    name=f"{name_key} ({len(indices)})",
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=0.5, color="white"),
                        opacity=0.7,
                    ),
                    text=[hover_texts[i] for i in indices],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    elif color_by == "quality":
        # Categorical: unique quality_label values
        quality_to_indices: Dict[str, List[int]] = {}
        for i in range(n):
            q = str(metadata[i].get("quality_label", "unknown"))
            quality_to_indices.setdefault(q, []).append(i)
        for idx, (q, indices) in enumerate(
            sorted(quality_to_indices.items(), key=lambda x: -len(x[1]))
        ):
            color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[indices, 0],
                    y=embeddings_2d[indices, 1],
                    mode="markers",
                    name=f"{q} ({len(indices)})",
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=0.5, color="white"),
                        opacity=0.7,
                    ),
                    text=[hover_texts[i] for i in indices],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    elif color_by in ("timestep", "rollout_idx", "demo_idx"):
        # Continuous: one trace with colorbar
        key = "timestep" if color_by == "timestep" else (
            "rollout_idx" if color_by == "rollout_idx" else "demo_idx"
        )

        def _get_value(meta: Dict, k: str) -> float:
            if k == "timestep":
                # For slices: use window middle when timestep is absent
                if "timestep" in meta:
                    return float(meta["timestep"])
                if "window_start" in meta and "window_end" in meta:
                    return float((meta["window_start"] + meta["window_end"]) / 2)
            return float(meta.get(k, 0))

        values = np.array([_get_value(metadata[i], key) for i in range(n)], dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=key.replace("_", " ").title()),
                    line=dict(width=0.5, color="white"),
                    opacity=0.7,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                name="",
            )
        )
    else:
        # Fallback to cluster
        unique_labels = sorted(set(cluster_labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
            unique_labels.append(-1)
        for label in unique_labels:
            mask = cluster_labels == label
            indices = np.where(mask)[0]
            name = f"Noise ({len(indices)})" if label == -1 else f"Cluster {label} ({len(indices)})"
            color = NOISE_COLOR if label == -1 else _get_cluster_color(int(label))
            opacity = 0.3 if label == -1 else 0.7
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[indices, 0],
                    y=embeddings_2d[indices, 1],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=0.5, color="white"),
                        opacity=opacity,
                    ),
                    text=[hover_texts[i] for i in indices],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=height,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_label_coherency_chart(
    cluster_stats: List[Dict],
    title: str = "Label Coherency per Cluster",
    height: int = 500,
) -> go.Figure:
    """Create a stacked horizontal bar chart showing annotation label distribution per cluster.

    Args:
        cluster_stats: List of dicts, each with keys:
            - 'cluster_id': int
            - 'size': int (total samples)
            - 'label_counts': dict mapping label -> count
            - 'purity': float (fraction of dominant label)
            - 'dominant_label': str
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    if not cluster_stats:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    # Sort clusters by index (noise last)
    sorted_stats = sorted(
        cluster_stats,
        key=lambda s: (s["cluster_id"] == -1, s["cluster_id"]),
    )

    # Collect all labels across clusters
    all_labels = set()
    for stat in sorted_stats:
        all_labels.update(stat["label_counts"].keys())
    all_labels = sorted(all_labels)

    # Build y-axis labels
    y_labels = [
        f"Cluster {s['cluster_id']} (n={s['size']}, purity={s['purity']:.0%})"
        for s in sorted_stats
    ]

    custom_color_map: Dict[str, str] = {}
    fig = go.Figure()

    for label in all_labels:
        proportions = []
        for stat in sorted_stats:
            count = stat["label_counts"].get(label, 0)
            proportions.append(count / stat["size"] if stat["size"] > 0 else 0)

        color = get_label_color(label, custom_color_map)
        fig_trace = go.Bar(
            name=label,
            y=y_labels,
            x=proportions,
            orientation="h",
            marker_color=color,
            hovertemplate=(
                f"Label: {label}<br>Proportion: %{{x:.1%}}<br><extra></extra>"
            ),
        )
        if label == all_labels[0]:
            fig = go.Figure(data=[fig_trace])
        else:
            fig.add_trace(fig_trace)

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Proportion",
        yaxis_title="Cluster",
        height=max(height, len(sorted_stats) * 40 + 150),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis=dict(range=[0, 1]),
    )

    return fig


def create_cluster_size_chart(
    cluster_stats: List[Dict],
    title: str = "Cluster Sizes",
    height: int = 400,
) -> go.Figure:
    """Create a bar chart showing the number of samples per cluster.

    Args:
        cluster_stats: List of dicts with 'cluster_id' and 'size' keys
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    # Order by cluster index (noise last)
    sorted_stats = sorted(
        cluster_stats,
        key=lambda s: (s["cluster_id"] == -1, s["cluster_id"]),
    )

    cluster_names = [
        f"Cluster {s['cluster_id']}" if s["cluster_id"] != -1 else "Noise"
        for s in sorted_stats
    ]
    sizes = [s["size"] for s in sorted_stats]
    colors = [_get_cluster_color(s["cluster_id"]) for s in sorted_stats]

    fig = go.Figure(
        data=[
            go.Bar(
                x=cluster_names,
                y=sizes,
                marker_color=colors,
                text=sizes,
                textposition="auto",
                hovertemplate="Cluster: %{x}<br>Size: %{y}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Number of Samples",
        height=height,
    )

    return fig


def create_silhouette_plot(
    silhouette_values: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Silhouette Plot",
    height: int = 500,
) -> go.Figure:
    """Create a per-sample silhouette coefficient plot grouped by cluster.

    Each cluster forms a horizontal band of sorted silhouette values,
    creating the classic "knife" shape visualization.

    Args:
        silhouette_values: Per-sample silhouette coefficients (shape N)
        cluster_labels: Cluster assignments (shape N). -1 excluded.
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Exclude noise points
    valid_mask = cluster_labels != -1
    valid_sil = silhouette_values[valid_mask]
    valid_labels = cluster_labels[valid_mask]

    if len(valid_sil) == 0:
        fig.update_layout(title=title, height=height)
        return fig

    unique_labels = sorted(set(valid_labels))
    y_lower = 0

    for label in unique_labels:
        cluster_mask = valid_labels == label
        cluster_sil = valid_sil[cluster_mask]
        cluster_sil_sorted = np.sort(cluster_sil)

        size = len(cluster_sil_sorted)
        y_upper = y_lower + size
        y_range = np.arange(y_lower, y_upper)

        color = _get_cluster_color(label)
        avg_sil = float(np.mean(cluster_sil))

        fig.add_trace(
            go.Bar(
                x=cluster_sil_sorted,
                y=y_range,
                orientation="h",
                marker_color=color,
                name=f"Cluster {label} (avg={avg_sil:.3f})",
                showlegend=True,
                hovertemplate=(
                    f"Cluster {label}<br>Silhouette: %{{x:.3f}}<extra></extra>"
                ),
            )
        )

        y_lower = y_upper + 2  # gap between clusters

    # Add vertical line at mean silhouette
    mean_sil = float(np.mean(valid_sil))
    fig.add_vline(
        x=mean_sil,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_sil:.3f}",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Samples (sorted within cluster)",
        height=height,
        bargap=0,
        showlegend=True,
        yaxis=dict(showticklabels=False),
    )

    return fig


def create_cluster_influence_box_plot(
    metadata: List[Dict],
    cluster_labels: np.ndarray,
    title: str = "Influence Distribution per Cluster",
    height: int = 400,
) -> go.Figure:
    """Create box plots of mean influence distribution per cluster.

    Args:
        metadata: List of metadata dicts, each with 'mean_influence' key
        cluster_labels: Cluster assignments (shape N)
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    unique_labels = sorted(set(cluster_labels))
    for label in unique_labels:
        mask = cluster_labels == label
        influences = [
            metadata[i].get("mean_influence", 0)
            for i in range(len(metadata))
            if mask[i]
        ]

        name = f"Cluster {label}" if label != -1 else "Noise"
        color = _get_cluster_color(label)

        fig.add_trace(
            go.Box(
                y=influences,
                name=name,
                marker_color=color,
                boxmean=True,
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="Mean Influence",
        height=height,
    )

    return fig


def _timesteps_per_cluster(
    metadata: List[Dict],
    cluster_labels: np.ndarray,
    representation: Literal["timestep", "sliding_window"],
) -> Dict[int, List[int]]:
    """Build list of timesteps per cluster.

    For timestep representation: one timestep per point (metadata['timestep']).
    For sliding_window: all timesteps in [window_start, window_end) per point.
    """
    timesteps_by_cluster: Dict[int, List[int]] = {}
    for i, meta in enumerate(metadata):
        c = int(cluster_labels[i])
        timesteps_by_cluster.setdefault(c, [])

        if representation == "timestep":
            t = meta.get("timestep", 0)
            timesteps_by_cluster[c].append(t)
        else:
            # sliding_window: add every timestep in the slice
            start = meta.get("window_start", 0)
            end = meta.get("window_end", start + 1)
            for t in range(start, end):
                timesteps_by_cluster[c].append(t)

    return timesteps_by_cluster


def create_cluster_timestep_distribution(
    metadata: List[Dict],
    cluster_labels: np.ndarray,
    representation: Literal["timestep", "sliding_window"],
    title: str = "Timestep distribution per cluster",
    height: int = 400,
) -> go.Figure:
    """Create a figure showing timestep distribution for each cluster.

    For individual timesteps: each point contributes one timestep.
    For sliding windows: each slice contributes all timesteps in [window_start, window_end).

    One subplot per cluster (excluding noise if present), shared x-axis = timestep.
    """
    timesteps_by_cluster = _timesteps_per_cluster(
        metadata, cluster_labels, representation
    )

    # Sort clusters: valid first (by id), noise last
    cluster_ids = sorted(
        timesteps_by_cluster.keys(),
        key=(lambda c: (c == -1, c)),
    )

    # Exclude clusters with no timesteps for display
    cluster_ids = [c for c in cluster_ids if timesteps_by_cluster[c]]

    if not cluster_ids:
        fig = go.Figure()
        fig.update_layout(title=title, height=height)
        return fig

    n_rows = len(cluster_ids)
    max_t = 0
    for c in cluster_ids:
        ts = timesteps_by_cluster[c]
        if ts:
            max_t = max(max_t, max(ts))

    # One subplot per cluster
    max_vspacing = (1.0 / (n_rows - 1)) if n_rows > 1 else 0.1
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.06, max_vspacing * 0.9),
        subplot_titles=[
            f"Cluster {c}" if c != -1 else "Noise"
            for c in cluster_ids
        ],
    )

    for row, c in enumerate(cluster_ids, start=1):
        ts = timesteps_by_cluster[c]
        if not ts:
            continue
        counts = np.bincount(ts, minlength=max_t + 1)
        x = np.arange(len(counts))
        color = _get_cluster_color(c)
        fig.add_trace(
            go.Bar(
                x=x,
                y=counts,
                marker_color=color,
                name=f"Cluster {c}" if c != -1 else "Noise",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=max(height, n_rows * 120 + 120),
        barmode="overlay",
    )
    fig.update_xaxes(title_text="Timestep", row=n_rows, col=1)
    fig.update_yaxes(title_text="Count")

    return fig


def create_confusion_matrix_plot(
    cluster_labels: np.ndarray,
    annotation_labels: List[str],
    title: str = "Cluster vs Annotation Label Co-occurrence",
    height: int = 500,
) -> go.Figure:
    """Create a heatmap showing co-occurrence of cluster assignments and annotation labels.

    Args:
        cluster_labels: Cluster assignments (shape N)
        annotation_labels: Annotation label strings (shape N)
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    unique_clusters = sorted(set(cluster_labels))
    unique_annotations = sorted(set(annotation_labels))

    # Build co-occurrence matrix
    matrix = np.zeros((len(unique_clusters), len(unique_annotations)))
    for i, (cl, al) in enumerate(zip(cluster_labels, annotation_labels)):
        row = unique_clusters.index(cl)
        col = unique_annotations.index(al)
        matrix[row, col] += 1

    # Normalize each row (cluster) to show proportions
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    cluster_names = [f"Cluster {c}" if c != -1 else "Noise" for c in unique_clusters]

    # Create text annotations showing count and proportion
    text = []
    for i in range(len(unique_clusters)):
        row_text = []
        for j in range(len(unique_annotations)):
            row_text.append(f"{int(matrix[i, j])}<br>({matrix_norm[i, j]:.0%})")
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_norm,
            x=unique_annotations,
            y=cluster_names,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar_title="Proportion",
            hovertemplate=(
                "Cluster: %{y}<br>Label: %{x}<br>Proportion: %{z:.1%}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Annotation Label",
        yaxis_title="Cluster",
        height=max(height, len(unique_clusters) * 40 + 150),
    )

    return fig
