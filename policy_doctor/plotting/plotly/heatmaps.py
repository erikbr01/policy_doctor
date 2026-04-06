"""Pure heatmap plotting functions.

This module contains functions for creating influence heatmap visualizations
without any Streamlit dependencies. All functions return Plotly Figure objects.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from policy_doctor.plotting.common import (
    EXTRA_COLORS,
    get_influence_colorscale,
    get_label_color,
)


def create_influence_heatmap(
    influence_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str = "",
    x_title: str = "Demo",
    y_title: str = "Rollout",
    show_label_bars: bool = True,
    x_label_colors: Optional[List[str]] = None,
    y_label_colors: Optional[List[str]] = None,
    customdata: Optional[List[List[List]]] = None,
    hovertemplate: Optional[str] = None,
    height: Optional[int] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    x_tickvals: Optional[List[float]] = None,
    x_ticktext: Optional[List[str]] = None,
    y_tickvals: Optional[List[float]] = None,
    y_ticktext: Optional[List[str]] = None,
) -> go.Figure:
    """Create an influence heatmap figure.

    Args:
        influence_matrix: 2D array of influence values (rows x cols)
        x_labels: Labels for x-axis (columns)
        y_labels: Labels for y-axis (rows)
        title: Plot title
        x_title: X-axis title
        y_title: Y-axis title
        show_label_bars: Whether to show colored label bars on edges
        x_label_colors: Colors for x-axis label bar (top)
        y_label_colors: Colors for y-axis label bar (left)
        customdata: Custom hover data array
        hovertemplate: Custom hover template string
        height: Plot height in pixels (auto-calculated if None)
        zmin: Minimum value for color scale (default: symmetric around 0)
        zmax: Maximum value for color scale (default: symmetric around 0)
        x_tickvals: Custom tick values for x-axis
        x_ticktext: Custom tick labels for x-axis
        y_tickvals: Custom tick values for y-axis
        y_ticktext: Custom tick labels for y-axis

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    num_rows, num_cols = influence_matrix.shape

    # Compute symmetric color limits centered at 0 if not provided
    if zmin is None or zmax is None:
        abs_max = max(
            abs(np.nanmin(influence_matrix)),
            abs(np.nanmax(influence_matrix)),
        )
        if abs_max == 0:
            abs_max = 1.0  # Avoid division issues
        zmin = -abs_max
        zmax = abs_max

    # Label bar dimensions (in data coordinates)
    label_bar_width = 3.0 if show_label_bars else 0
    label_bar_height = 1.5 if show_label_bars else 0

    # Default hover template
    if hovertemplate is None:
        hovertemplate = "<b>%{x}, %{y}</b><br>Influence: %{z:.4f}<extra></extra>"

    # Main heatmap (offset by label bar dimensions)
    fig.add_trace(
        go.Heatmap(
            z=influence_matrix,
            x=np.arange(num_cols) + label_bar_width,
            y=np.arange(num_rows),
            colorscale=get_influence_colorscale(),
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorbar=dict(title="Influence"),
        )
    )

    # Add label bars if requested
    if show_label_bars:
        _add_label_bars(
            fig,
            num_rows,
            num_cols,
            label_bar_width,
            label_bar_height,
            x_labels,
            y_labels,
            x_label_colors,
            y_label_colors,
        )

    # Calculate height
    if height is None:
        height = max(400, num_rows * 8 + 100)

    # Configure axes
    xaxis_config = dict(
        title=x_title,
        range=[-0.5, num_cols + label_bar_width + 0.5],
    )
    if x_tickvals is not None:
        xaxis_config["tickmode"] = "array"
        xaxis_config["tickvals"] = x_tickvals
        if x_ticktext is not None:
            xaxis_config["ticktext"] = x_ticktext

    yaxis_config = dict(
        title=y_title,
        range=[-0.5, num_rows + label_bar_height + 0.5],
    )
    if y_tickvals is not None:
        yaxis_config["tickmode"] = "array"
        yaxis_config["tickvals"] = y_tickvals
        if y_ticktext is not None:
            yaxis_config["ticktext"] = y_ticktext

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        margin=dict(l=60, r=20, t=40, b=100),
    )

    return fig


def create_variance_ood_plot(
    timesteps: np.ndarray,
    variances: np.ndarray,
    reference_line: Optional[np.ndarray] = None,
    reference_label: str = "Reference",
    title: str = "Variance Across Demos",
    per_demo_variances: Optional[List[np.ndarray]] = None,
    demo_labels: Optional[List[str]] = None,
) -> go.Figure:
    """Create a variance-based OOD detection plot.

    Shows variance of influence across demonstration timesteps for each
    rollout timestep, with optional reference line and per-demo variances.

    Args:
        timesteps: Array of rollout timestep indices
        variances: Array of variance values (aggregated across all demos)
        reference_line: Optional reference line values (e.g., mean across all rollouts)
        reference_label: Label for reference line
        title: Plot title
        per_demo_variances: Optional list of per-demo variance arrays
        demo_labels: Optional list of demo labels (required if per_demo_variances given)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add per-demo variance traces (light gray, in background)
    if per_demo_variances is not None and demo_labels is not None:
        for demo_var, demo_label in zip(per_demo_variances, demo_labels):
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=demo_var,
                    mode="lines",
                    name=demo_label,
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=1),
                    hovertemplate=f"<b>{demo_label}</b><br>Timestep: %{{x}}<br>Variance: %{{y:.6f}}<extra></extra>",
                    showlegend=False,
                )
            )

    # Add reference line (red)
    if reference_line is not None:
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=reference_line,
                mode="lines",
                name=reference_label,
                line=dict(color="red", width=2, dash="dash"),
                hovertemplate=f"<b>{reference_label}</b><br>Timestep: %{{x}}<br>Variance: %{{y:.6f}}<extra></extra>",
            )
        )

    # Add main variance trace (blue, thick)
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=variances,
            mode="lines+markers",
            name="This Rollout",
            line=dict(color="blue", width=3),
            marker=dict(size=5, color="blue"),
            hovertemplate="<b>This Rollout</b><br>Timestep: %{x}<br>Variance: %{y:.6f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Rollout Timestep",
        yaxis_title="Variance",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    return fig


def create_diagonal_ranking_chart(
    all_diagonals: List[List[dict]],
    demo_labels: List[str],
    title: str = "Diagonal Ranking",
) -> go.Figure:
    """Create a bar chart ranking all detected diagonals across demos.

    Args:
        all_diagonals: List of diagonal lists (one per demo)
        demo_labels: Labels for each demo
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Flatten all diagonals and add demo info
    flattened = []
    for demo_idx, diagonals in enumerate(all_diagonals):
        for diag_idx, diag in enumerate(diagonals):
            flattened.append(
                {
                    "demo_idx": demo_idx,
                    "demo_label": demo_labels[demo_idx],
                    "diag_idx": diag_idx,
                    "score": diag["score"],
                    "length": diag["length"],
                    "mean_corr": diag["mean_corr"],
                }
            )

    # Sort by score
    flattened.sort(key=lambda x: x["score"], reverse=True)

    if not flattened:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No diagonals detected",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create bar chart
    labels = [f"{d['demo_label']} (Diag #{d['diag_idx'] + 1})" for d in flattened]
    scores = [d["score"] for d in flattened]
    hover_text = [
        f"Demo: {d['demo_label']}<br>"
        f"Diagonal #{d['diag_idx'] + 1}<br>"
        f"Score: {d['score']:.2f}<br>"
        f"Length: {d['length']}<br>"
        f"Mean Corr: {d['mean_corr']:.4f}"
        for d in flattened
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=scores,
                orientation="v",
                hovertext=hover_text,
                hoverinfo="text",
                marker=dict(
                    color=scores,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Score"),
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Diagonal Score (Length × Mean Correlation)",
        height=500,
        xaxis=dict(tickangle=45),  # Rotate labels for readability
    )

    return fig


def create_diagonal_detection_dual_grid(
    matrices: List[np.ndarray],
    correlation_maps: List[np.ndarray],
    titles: List[str],
    main_title: str = "Diagonal Detection",
    cols: int = 4,
    diagonals_list: Optional[List[List[dict]]] = None,
    show_diagonals: bool = False,
) -> go.Figure:
    """Create a grid showing influence matrices and their correlation maps side-by-side.

    Each row shows pairs of: [influence matrix | correlation map]

    Args:
        matrices: List of influence matrices to display
        correlation_maps: List of sliding window correlation maps
        titles: List of titles for each matrix pair
        main_title: Main title for the entire grid
        cols: Number of matrix pairs per row

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_matrices = len(matrices)
    rows = (num_matrices + cols - 1) // cols

    # Create subplots with 2 columns per matrix pair
    subplot_cols = cols * 2
    column_widths = [0.5, 0.5] * cols  # 50% for matrix, 50% for correlation map

    # Create subplot titles: "Demo X" for influence, "Correlation" for map
    subplot_titles = []
    for title in titles:
        subplot_titles.append(f"{title}<br>(Influence)")
        subplot_titles.append(f"{title}<br>(Correlation)")

    fig = make_subplots(
        rows=rows,
        cols=subplot_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
        column_widths=column_widths,
    )

    # Compute global color scale limits for matrices
    all_values = np.concatenate([m.flatten() for m in matrices])
    abs_max_influence = max(abs(np.nanmin(all_values)), abs(np.nanmax(all_values)))
    if abs_max_influence == 0:
        abs_max_influence = 1.0

    # Compute global limits for correlation maps
    all_corr_values = np.concatenate([cm.flatten() for cm in correlation_maps])
    min_corr = np.nanmin(all_corr_values)
    max_corr = np.nanmax(all_corr_values)

    # Add each matrix and correlation map to the grid
    for idx, (matrix, corr_map, title) in enumerate(
        zip(matrices, correlation_maps, titles)
    ):
        row = idx // cols + 1
        influence_col = (idx % cols) * 2 + 1
        corr_col = influence_col + 1

        # Add influence matrix
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=get_influence_colorscale(),
                zmid=0,
                zmin=-abs_max_influence,
                zmax=abs_max_influence,
                showscale=False,  # Disable individual colorbars
                hovertemplate="Rollout t: %{y}<br>Demo t: %{x}<br>Influence: %{z:.4f}<extra></extra>",
                name="Influence",
            ),
            row=row,
            col=influence_col,
        )

        # Add correlation map
        fig.add_trace(
            go.Heatmap(
                z=corr_map,
                colorscale="RdYlGn",  # Red (low) to Green (high) for correlation
                zmin=min_corr,
                zmax=max_corr,
                showscale=False,  # Disable individual colorbars
                hovertemplate="Window pos: (%{x}, %{y})<br>Correlation: %{z:.4f}<extra></extra>",
                name="Correlation",
            ),
            row=row,
            col=corr_col,
        )

        # Overlay detected diagonals if requested
        if show_diagonals and diagonals_list is not None and idx < len(diagonals_list):
            diagonals = diagonals_list[idx]
            for diag_idx, diag in enumerate(diagonals):
                # Draw diagonal line
                fig.add_trace(
                    go.Scatter(
                        x=[diag["start_x"], diag["end_x"]],
                        y=[diag["start_y"], diag["end_y"]],
                        mode="lines",
                        line=dict(color="yellow", width=2),
                        showlegend=(
                            idx == 0 and diag_idx == 0
                        ),  # Only show legend once
                        name="Detected Diagonal",
                        hovertemplate=f"Diagonal #{diag_idx + 1}<br>Score: {diag['score']:.2f}<br>Length: {diag['length']}<br>Mean Corr: {diag['mean_corr']:.4f}<extra></extra>",
                    ),
                    row=row,
                    col=corr_col,
                )

        # Update axes for influence matrix
        fig.update_xaxes(
            title_text="Demo t", row=row, col=influence_col, showticklabels=False
        )
        fig.update_yaxes(
            title_text="Rollout t", row=row, col=influence_col, showticklabels=False
        )

        # Update axes for correlation map
        fig.update_xaxes(
            title_text="Window X", row=row, col=corr_col, showticklabels=False
        )
        fig.update_yaxes(
            title_text="Window Y", row=row, col=corr_col, showticklabels=False
        )

    # Update layout
    height = max(500, rows * 300)
    fig.update_layout(
        title=main_title,
        height=height,
        showlegend=False,
    )

    return fig


def create_diagonal_detection_grid(
    matrices: List[np.ndarray],
    titles: List[str],
    highlights: List[bool],
    main_title: str = "Diagonal Detection",
    cols: int = 4,
) -> go.Figure:
    """Create a grid of influence matrices with diagonal correlation scores.

    Displays multiple influence matrices in a grid layout with correlation
    scores in titles. Matrices above threshold are highlighted.

    Args:
        matrices: List of influence matrices to display
        titles: List of titles for each matrix (should include correlation score)
        highlights: List of booleans indicating which matrices exceed threshold
        main_title: Main title for the entire grid
        cols: Number of columns in the grid

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_matrices = len(matrices)
    rows = (num_matrices + cols - 1) // cols

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Compute global color scale limits (symmetric around 0)
    all_values = np.concatenate([m.flatten() for m in matrices])
    abs_max = max(abs(np.nanmin(all_values)), abs(np.nanmax(all_values)))
    if abs_max == 0:
        abs_max = 1.0

    # Add each matrix to the grid
    for idx, (matrix, highlight) in enumerate(zip(matrices, highlights)):
        row = idx // cols + 1
        col = idx % cols + 1

        # Use different colorscale for highlighted matrices
        if highlight:
            colorscale = "Viridis"
            showscale = False  # Don't show individual colorbars
        else:
            colorscale = get_influence_colorscale()
            showscale = False

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=colorscale,
                zmid=0,
                zmin=-abs_max,
                zmax=abs_max,
                showscale=showscale,
                hovertemplate="Rollout t: %{y}<br>Demo t: %{x}<br>Influence: %{z:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Update axes for this subplot
        fig.update_xaxes(title_text="Demo t", row=row, col=col, showticklabels=False)
        fig.update_yaxes(title_text="Rollout t", row=row, col=col, showticklabels=False)

    # Add green border to highlighted subplot titles
    for idx, (title, highlight) in enumerate(zip(titles, highlights)):
        if highlight:
            row = idx // cols
            # Update annotation color (subplot titles are annotations)
            annotation_idx = row * cols + (idx % cols)
            if annotation_idx < len(fig.layout.annotations):
                fig.layout.annotations[annotation_idx].font.color = "green"
                fig.layout.annotations[annotation_idx].font.size = 12

    # Update layout
    height = max(400, rows * 200)
    fig.update_layout(
        title=main_title,
        height=height,
        showlegend=False,
    )

    return fig


def create_influence_matrices_with_variance(
    matrices: List[np.ndarray],
    variances: List[np.ndarray],
    titles: List[str],
    reference_variance: Optional[np.ndarray] = None,
    main_title: str = "Influence Matrices with Variance",
    cols: int = 4,
) -> go.Figure:
    """Create a grid of influence matrices with variance plots on the side.

    Each subplot shows an influence matrix on the left and its corresponding
    variance plot (across demo timesteps) on the right.

    Args:
        matrices: List of influence matrices to display
        variances: List of variance arrays (one per matrix)
        titles: List of titles for each matrix
        reference_variance: Optional reference variance line to show in plots
        main_title: Main title for the entire grid
        cols: Number of matrix-variance pairs per row

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_matrices = len(matrices)
    rows = (num_matrices + cols - 1) // cols

    # Create subplots with 2 columns per matrix-variance pair
    # Each row has: matrix1, variance1, matrix2, variance2, ...
    subplot_cols = cols * 2
    column_widths = [0.7, 0.3] * cols  # 70% for matrix, 30% for variance

    fig = make_subplots(
        rows=rows,
        cols=subplot_cols,
        subplot_titles=titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.02,
        column_widths=column_widths,
        specs=[
            [
                {"type": "heatmap"} if i % 2 == 0 else {"type": "scatter"}
                for i in range(subplot_cols)
            ]
            for _ in range(rows)
        ],
    )

    # Compute global color scale limits for matrices
    all_values = np.concatenate([m.flatten() for m in matrices])
    abs_max = max(abs(np.nanmin(all_values)), abs(np.nanmax(all_values)))
    if abs_max == 0:
        abs_max = 1.0

    # Add each matrix and variance plot to the grid
    for idx, (matrix, variance, title) in enumerate(zip(matrices, variances, titles)):
        row = idx // cols + 1
        matrix_col = (idx % cols) * 2 + 1
        variance_col = matrix_col + 1

        # Add influence matrix
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=get_influence_colorscale(),
                zmid=0,
                zmin=-abs_max,
                zmax=abs_max,
                showscale=False,  # Disable colorbars
                hovertemplate="Rollout t: %{y}<br>Demo t: %{x}<br>Influence: %{z:.4f}<extra></extra>",
                name="Influence",
            ),
            row=row,
            col=matrix_col,
        )

        # Add variance plot
        timesteps = np.arange(len(variance))

        # Add reference variance if provided
        if reference_variance is not None:
            ref_var = reference_variance[: len(variance)]
            fig.add_trace(
                go.Scatter(
                    x=ref_var,
                    y=timesteps,
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name="Reference",
                    showlegend=(idx == 0),  # Only show legend for first plot
                    hovertemplate="Ref Var: %{x:.6f}<extra></extra>",
                ),
                row=row,
                col=variance_col,
            )

        # Add this demo's variance
        fig.add_trace(
            go.Scatter(
                x=variance,
                y=timesteps,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Variance",
                showlegend=(idx == 0),  # Only show legend for first plot
                hovertemplate="Variance: %{x:.6f}<br>Timestep: %{y}<extra></extra>",
            ),
            row=row,
            col=variance_col,
        )

        # Update axes for matrix subplot
        fig.update_xaxes(
            title_text="Demo t", row=row, col=matrix_col, showticklabels=False
        )
        fig.update_yaxes(
            title_text="Rollout t", row=row, col=matrix_col, showticklabels=False
        )

        # Update axes for variance subplot
        fig.update_xaxes(title_text="Var", row=row, col=variance_col, side="top")
        fig.update_yaxes(showticklabels=False, row=row, col=variance_col)

    # Update layout
    height = max(400, rows * 300)
    fig.update_layout(
        title=main_title,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_ranking_scores_plot(
    scores: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Ranking Scores",
    highlight_top_k: Optional[int] = None,
    show_cumulative: bool = True,
    color_by_source: Optional[List[int]] = None,
    color_by_in_selection: Optional[List[bool]] = None,
    color_by_category: Optional[List[str]] = None,
    in_global_mask: Optional[List[bool]] = None,
    in_local_mask: Optional[List[bool]] = None,
    show_global_selection: bool = True,
    show_local_selection: bool = True,
    yaxis_title: str = "Score",
    raw_mean: Optional[float] = None,
    raw_std: Optional[float] = None,
    n_sigma_reference: Optional[float] = None,
    sigma_lines_in_z_space: bool = False,
    show_histogram: bool = False,
) -> go.Figure:
    """Create a plot showing ranking scores to help determine optimal top-k.

    This visualization helps users determine an appropriate top-k cutoff by showing:
    1. The distribution of scores across all ranked items
    2. Optional cumulative score to see diminishing returns
    3. Visual highlighting of the top-k items
    4. Optional coloring by source rollout (one color per source rollout episode)

    Args:
        scores: 1D array of scores (should be sorted in descending order for ranking)
        labels: Optional labels for each score (e.g., "Demo 5, t=10-15")
        title: Plot title
        highlight_top_k: If provided, highlight the top k scores with a different color
        show_cumulative: Whether to show cumulative score as a secondary trace
        color_by_source: If provided, same length as scores; each element is the
            source rollout episode index for that candidate. Bars are colored by
            source rollout (one color per unique source).
        color_by_in_selection: If provided, same length as scores; True = in current
            curation selection, False = not. Bars are colored accordingly (e.g. green
            vs gray). Used for per-slice ranking charts.
        color_by_category: If provided, same length as scores; each element is "global",
            "local", or "none". Ignored when in_global_mask and in_local_mask are set.
        in_global_mask: If provided with in_local_mask, same length as scores; True = in
            global selection. Used to show green for all in global when only global is on.
        in_local_mask: If provided with in_global_mask, same length as scores; True = in
            this slice's local selection. Display: blue when show_local and in_local; green
            when show_global and in_global (and not blue); gray else.
        show_global_selection: When using color_by_category or masks, if False then "global" is
            drawn as gray and the Global selection legend entry is omitted.
        show_local_selection: When using color_by_category, if False then "local" is
            drawn as gray and the Local selection legend entry is omitted.
        yaxis_title: Label for the primary y-axis (e.g. "Score" or "Normalized score").
        raw_mean: If provided (with raw_std), show these in the y-axis label as the
            mean/std of the unnormalized values used for normalization.
        raw_std: If provided (with raw_mean), used for the y-axis label stats.
        n_sigma_reference: If provided, add horizontal reference lines at 1σ, 2σ, and
            this value (selected sigma threshold).
        sigma_lines_in_z_space: If True (and n_sigma_reference set), line y-positions
            are 1, 2, n_sigma (z-score space). If False, use raw_mean + k*raw_std when
            raw_mean/raw_std are provided.
        show_histogram: If True, add a second row with a histogram of the score distribution.

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    n_scores = len(scores)
    indices = np.arange(n_scores)

    # Y-axis title: show raw (unnormalized) stats when provided, else stats of displayed scores
    if raw_mean is not None and raw_std is not None:
        mu, sigma = raw_mean, raw_std
    else:
        mu = float(np.mean(scores))
        sigma = float(np.std(scores)) if n_scores > 1 else 0.0
    yaxis_title_with_stats = f"{yaxis_title} (μ={mu:.2f}, σ={sigma:.2f})"

    # Sort scores in descending order if not already sorted
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]

    if labels is not None:
        sorted_labels = [labels[i] for i in sorted_indices]
    else:
        sorted_labels = [f"Rank {i + 1}" for i in range(n_scores)]

    # Create figure with optional secondary y-axis for cumulative and/or histogram row
    if show_histogram:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.12,
            subplot_titles=("", "Score distribution"),
        )
        _histogram_row = 2
        _ranking_row = 1
    elif show_cumulative:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        _histogram_row = None
        _ranking_row = 1
    else:
        fig = go.Figure()
        _histogram_row = None
        _ranking_row = 1

    _rk = {"row": 1, "col": 1} if _histogram_row else {}

    # Determine colors for bars
    if color_by_source is not None and len(color_by_source) == n_scores:
        # Color by source rollout: same order as scores, then apply sort
        sorted_source_ids = [color_by_source[i] for i in sorted_indices]
        unique_ids = sorted(set(sorted_source_ids))
        id_to_color = {
            uid: EXTRA_COLORS[i % len(EXTRA_COLORS)] for i, uid in enumerate(unique_ids)
        }
        colors = [id_to_color[sid] for sid in sorted_source_ids]
    elif in_global_mask is not None and in_local_mask is not None and len(in_global_mask) == n_scores and len(in_local_mask) == n_scores:
        # Per-bar global/local: green for in global, blue for in local (local takes priority when both), gray else
        sorted_in_global = np.array([in_global_mask[i] for i in sorted_indices])
        sorted_in_local = np.array([in_local_mask[i] for i in sorted_indices])
        display_local = show_local_selection & sorted_in_local
        display_global = show_global_selection & sorted_in_global & ~display_local
        display_none = ~display_local & ~display_global
        sorted_cat = np.where(display_local, "local", np.where(display_global, "global", "none"))
        _cat_color = {"global": "rgb(44, 160, 44)", "local": "rgb(31, 119, 180)", "none": "rgb(174, 199, 232)"}
        colors = [_cat_color.get(c, "rgb(174, 199, 232)") for c in sorted_cat]
    elif color_by_category is not None and len(color_by_category) == n_scores:
        # Resolve categories (optionally hide global/local by treating as none)
        sorted_cat = [color_by_category[i] for i in sorted_indices]
        if not show_global_selection:
            sorted_cat = ["none" if c == "global" else c for c in sorted_cat]
        if not show_local_selection:
            sorted_cat = ["none" if c == "local" else c for c in sorted_cat]
        _cat_color = {"global": "rgb(44, 160, 44)", "local": "rgb(31, 119, 180)", "none": "rgb(174, 199, 232)"}
        colors = [_cat_color.get(c, "rgb(174, 199, 232)") for c in sorted_cat]
    elif color_by_in_selection is not None and len(color_by_in_selection) == n_scores:
        # Color by in-selection: True = in current curation, False = not
        sorted_in_sel = [color_by_in_selection[i] for i in sorted_indices]
        colors = [
            "rgb(44, 160, 44)" if sel else "rgb(174, 199, 232)"
            for sel in sorted_in_sel
        ]
    elif highlight_top_k is not None and highlight_top_k > 0:
        colors = [
            "rgb(31, 119, 180)" if i < highlight_top_k else "rgb(174, 199, 232)"
            for i in range(n_scores)
        ]
    else:
        colors = "rgb(31, 119, 180)"

    # Add bar chart of scores
    use_global_local_traces = (
        (color_by_category is not None and len(color_by_category) == n_scores)
        or (in_global_mask is not None and in_local_mask is not None and len(in_global_mask) == n_scores and len(in_local_mask) == n_scores)
    )
    if use_global_local_traces:
        # One trace per category so legend clicks toggle visibility
        if not isinstance(sorted_cat, np.ndarray):
            sorted_cat = np.asarray(sorted_cat)
        _cat_names = {"global": "Global selection", "local": "Local selection", "none": "Not in selection"}
        for cat in ("global", "local", "none"):
            if not show_global_selection and cat == "global":
                continue
            if not show_local_selection and cat == "local":
                continue
            mask = sorted_cat == cat
            if np.any(mask):
                idx = np.where(mask)[0]
                bar = go.Bar(
                    x=idx,
                    y=sorted_scores[mask],
                    marker_color=_cat_color[cat],
                    name=_cat_names[cat],
                    hovertemplate=f"<b>%{{text}}</b><br>{yaxis_title}: %{{y:.4f}}<extra></extra>",
                    text=[sorted_labels[i] for i in idx],
                    showlegend=True,
                    legendgroup=_cat_names[cat],
                )
                if show_cumulative:
                    fig.add_trace(bar, secondary_y=False, **_rk)
                else:
                    fig.add_trace(bar, **_rk)
            else:
                # No bars in this category (e.g. no "local only" in random subsample); add legend-only entry so user sees the color
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=12, color=_cat_color[cat], symbol="square", line=dict(width=0)),
                        name=_cat_names[cat],
                        showlegend=True,
                        legendgroup=_cat_names[cat],
                    ),
                    **_rk,
                )
    else:
        bar_trace = go.Bar(
            x=indices,
            y=sorted_scores,
            marker_color=colors,
            name=yaxis_title,
            hovertemplate=f"<b>%{{text}}</b><br>{yaxis_title}: %{{y:.4f}}<extra></extra>",
            text=sorted_labels,
            showlegend=color_by_source is None and color_by_in_selection is None,
        )
        if show_cumulative:
            fig.add_trace(bar_trace, secondary_y=False, **_rk)
        else:
            fig.add_trace(bar_trace, **_rk)

    # When coloring by source rollout, add legend entries (one per unique source)
    if color_by_source is not None and len(color_by_source) == n_scores:
        for uid in unique_ids:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=id_to_color[uid], symbol="square", line=dict(width=0)),
                    name=f"Rollout ep{uid}",
                    showlegend=True,
                    legendgroup=f"rollout_{uid}",
                ),
                **_rk,
            )
    # When coloring by in-selection (binary), add legend for In selection / Not in selection
    elif color_by_in_selection is not None and len(color_by_in_selection) == n_scores:
        for label, color, sel_val in [
            ("In selection", "rgb(44, 160, 44)", True),
            ("Not in selection", "rgb(174, 199, 232)", False),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="square", line=dict(width=0)),
                    name=label,
                    showlegend=True,
                    legendgroup=label,
                ),
                **_rk,
            )

    # Add cumulative score line if requested
    if show_cumulative:
        cumulative_scores = np.cumsum(sorted_scores)
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=cumulative_scores,
                mode="lines+markers",
                name="Cumulative",
                line=dict(color="rgb(255, 127, 14)", width=2),
                marker=dict(size=4),
                hovertemplate="<b>%{text}</b><br>Cumulative: %{y:.4f}<extra></extra>",
                text=[f"Top {i + 1}" for i in range(n_scores)],
            ),
            secondary_y=True,
            **_rk,
        )

    # Add vertical line at top-k cutoff if specified
    if (
        highlight_top_k is not None
        and highlight_top_k > 0
        and highlight_top_k < n_scores
    ):
        fig.add_vline(
            x=highlight_top_k - 0.5,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Top-{highlight_top_k}",
            annotation_position="top",
            **_rk,
        )

    # Add horizontal reference lines at 1σ, 2σ, and selected σ when requested
    if n_sigma_reference is not None:
        if sigma_lines_in_z_space or (raw_mean is None or raw_std is None):
            def _y(sigma_k: float) -> float:
                return float(sigma_k)
        else:
            def _y(sigma_k: float) -> float:
                return raw_mean + sigma_k * raw_std
        seen = set()
        for sigma_val, label, color in [
            (1.0, "1σ", "rgba(100,100,100,0.7)"),
            (2.0, "2σ", "rgba(100,100,100,0.5)"),
            (n_sigma_reference, f"{n_sigma_reference}σ (selected)", "rgb(200, 80, 80)"),
        ]:
            y_val = _y(sigma_val)
            if y_val in seen:
                continue
            seen.add(y_val)
            fig.add_hline(
                y=y_val,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
                **_rk,
            )

    # Add histogram of score distribution when requested
    if _histogram_row:
        hist_scores = np.asarray(scores, dtype=np.float64)
        fig.add_trace(
            go.Histogram(
                x=hist_scores,
                name="Score distribution",
                nbinsx=min(50, max(10, len(hist_scores) // 5)),
                marker_color="rgb(31, 119, 180)",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=yaxis_title, row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Rank",
        showlegend=True,
        hovermode="closest",
        height=700 if _histogram_row else 500,
    )

    if show_cumulative:
        fig.update_yaxes(title_text=yaxis_title_with_stats, secondary_y=False)
        fig.update_yaxes(title_text=f"Cumulative {yaxis_title}", secondary_y=True)
    else:
        fig.update_yaxes(title_text=yaxis_title_with_stats)

    return fig


def _add_label_bars(
    fig: go.Figure,
    num_rows: int,
    num_cols: int,
    bar_width: float,
    bar_height: float,
    x_labels: List[str],
    y_labels: List[str],
    x_colors: Optional[List[str]],
    y_colors: Optional[List[str]],
) -> None:
    """Add colored label bars to a figure.

    Args:
        fig: Plotly Figure to modify
        num_rows: Number of rows in the heatmap
        num_cols: Number of columns in the heatmap
        bar_width: Width of the left label bar
        bar_height: Height of the top label bar
        x_labels: Labels for x-axis (for hover)
        y_labels: Labels for y-axis (for hover)
        x_colors: Colors for x-axis label bar
        y_colors: Colors for y-axis label bar
    """
    # X-axis label bar (top)
    if x_colors:
        for i, color in enumerate(x_colors):
            fig.add_shape(
                type="rect",
                x0=i + bar_width - 0.5,
                x1=i + bar_width + 0.5,
                y0=num_rows - 0.5,
                y1=num_rows + bar_height - 0.5,
                fillcolor=color,
                line=dict(width=0.5, color="#888888"),
            )
            # Add invisible scatter for hover
            if i < len(x_labels):
                fig.add_trace(
                    go.Scatter(
                        x=[i + bar_width],
                        y=[num_rows + bar_height / 2 - 0.5],
                        mode="markers",
                        marker=dict(size=10, opacity=0),
                        hovertemplate=f"<b>{x_labels[i]}</b><extra></extra>",
                        showlegend=False,
                    )
                )

    # Y-axis label bar (left)
    if y_colors:
        for i, color in enumerate(y_colors):
            fig.add_shape(
                type="rect",
                x0=0,
                x1=bar_width - 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                fillcolor=color,
                line=dict(width=0.5, color="#888888"),
            )
            # Add invisible scatter for hover
            if i < len(y_labels):
                fig.add_trace(
                    go.Scatter(
                        x=[bar_width / 2 - 0.25],
                        y=[i],
                        mode="markers",
                        marker=dict(size=10, opacity=0),
                        hovertemplate=f"<b>{y_labels[i]}</b><extra></extra>",
                        showlegend=False,
                    )
                )


def create_trajectory_heatmap(
    traj_influence: np.ndarray,
    rollout_labels: List[str],
    demo_labels: List[str],
    title: str = "",
    rollout_label_colors: Optional[List[str]] = None,
    demo_label_colors: Optional[List[str]] = None,
    customdata: Optional[List[List[List]]] = None,
    hovertemplate: Optional[str] = None,
    x_tickvals: Optional[List[float]] = None,
    x_ticktext: Optional[List[str]] = None,
    y_tickvals: Optional[List[float]] = None,
    y_ticktext: Optional[List[str]] = None,
) -> go.Figure:
    """Create a trajectory-level influence heatmap.

    Args:
        traj_influence: 2D array (num_rollouts, num_demos)
        rollout_labels: Labels for rollout episodes (y-axis)
        demo_labels: Labels for demo episodes (x-axis)
        title: Plot title
        rollout_label_colors: Colors for rollout label bar
        demo_label_colors: Colors for demo label bar
        customdata: Custom hover data
        hovertemplate: Custom hover template
        x_tickvals: Custom tick values for x-axis
        x_ticktext: Custom tick labels for x-axis
        y_tickvals: Custom tick values for y-axis
        y_ticktext: Custom tick labels for y-axis

    Returns:
        Plotly Figure object
    """
    return create_influence_heatmap(
        influence_matrix=traj_influence,
        x_labels=demo_labels,
        y_labels=rollout_labels,
        title=title,
        x_title="Demo Episodes",
        y_title="Rollout Episodes",
        show_label_bars=True,
        x_label_colors=demo_label_colors,
        y_label_colors=rollout_label_colors,
        customdata=customdata,
        hovertemplate=hovertemplate,
        height=max(400, len(rollout_labels) * 25 + 100),
        x_tickvals=x_tickvals,
        x_ticktext=x_ticktext,
        y_tickvals=y_tickvals,
        y_ticktext=y_ticktext,
    )


def create_magnitude_over_time_plot(
    positive_influence: np.ndarray,
    negative_influence: np.ndarray,
    title: str = "",
    x_labels: Optional[List[str]] = None,
) -> go.Figure:
    """Create a plot showing positive/negative influence magnitude over time.

    Args:
        positive_influence: Array of positive influence sums per timestep
        negative_influence: Array of negative influence sums per timestep
        title: Plot title
        x_labels: Optional labels for x-axis

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    num_timesteps = len(positive_influence)
    timesteps = list(range(num_timesteps))
    total_magnitude = np.abs(negative_influence) + positive_influence

    # Positive influence (green)
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=positive_influence,
            mode="lines",
            name="Positive Influence",
            line=dict(color="green", width=2),
            hovertemplate="<b>Timestep %{x}</b><br>Positive: %{y:.4f}<extra></extra>",
        )
    )

    # Negative influence (red)
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=negative_influence,
            mode="lines",
            name="Negative Influence",
            line=dict(color="red", width=2),
            hovertemplate="<b>Timestep %{x}</b><br>Negative: %{y:.4f}<extra></extra>",
        )
    )

    # Total magnitude (dashed)
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=total_magnitude,
            mode="lines",
            name="Total Magnitude",
            line=dict(color="white", width=2, dash="dash"),
            hovertemplate="<b>Timestep %{x}</b><br>Total: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Rollout Timestep",
        yaxis_title="Influence Magnitude",
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_transition_statistics_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_stat_name: str,
    y_stat_name: str,
    metadata: List[Dict],
    title: str = "Transition-Level Influence Statistics",
) -> go.Figure:
    """Create a scatter plot of transition-level influence statistics.

    Args:
        x_values: Values for x-axis
        y_values: Values for y-axis
        x_stat_name: Name of x-axis statistic
        y_stat_name: Name of y-axis statistic
        metadata: List of dictionaries containing point metadata for hover
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Split indices by success/failure for legend
    success_indices = [i for i, m in enumerate(metadata) if m.get("success", False)]
    failure_indices = [i for i, m in enumerate(metadata) if not m.get("success", False)]

    def add_trace(label, indices, color):
        if not indices:
            return

        custom_data = [metadata[i] for i in indices]

        # Build hover template from available metadata
        hovertemplate = (
            "<b>Rollout %{customdata.rollout_idx} × Demo %{customdata.demo_idx}</b><br>"
            "Success: %{customdata.success_str}<br>"
            "Demo Quality: %{customdata.quality}<br><br>"
            f"{x_stat_name.title()}: %{{x:.4f}}<br>"
            f"{y_stat_name.title()}: %{{y:.4f}}<br><br>"
            "Mean: %{customdata.mean:.4f}<br>"
            "Std: %{customdata.std:.4f}<br>"
            "Min: %{customdata.min:.4f}<br>"
            "Max: %{customdata.max:.4f}"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=x_values[indices],
                y=y_values[indices],
                mode="markers",
                name=label,
                marker=dict(
                    size=6,
                    color=color,
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                ),
                customdata=custom_data,
                hovertemplate=hovertemplate,
            )
        )

    add_trace("Success", success_indices, "green")
    add_trace("Failure", failure_indices, "red")

    return fig


def create_correlation_matrix_heatmap(
    correlation_matrix: Any,  # pandas DataFrame
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Create a heatmap for a correlation matrix.

    Args:
        correlation_matrix: pandas DataFrame containing correlation coefficients
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
    )

    return fig


def create_density_plots(
    success_values: np.ndarray,
    failure_values: np.ndarray,
    stat_name: str,
    plot_type: str = "histogram",  # "histogram", "normalized", "cdf"
    title: Optional[str] = None,
) -> go.Figure:
    """Create density plots comparing success vs. failure distributions.

    Args:
        success_values: Array of values for successful rollouts
        failure_values: Array of values for failed rollouts
        stat_name: Name of the statistic being plotted
        plot_type: Type of plot ("histogram", "normalized", or "cdf")
        title: Optional custom title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if plot_type == "cdf":
        success_sorted = np.sort(success_values)
        failure_sorted = np.sort(failure_values)

        if len(success_sorted) > 0:
            success_cdf = np.arange(1, len(success_sorted) + 1) / len(success_sorted)
            fig.add_trace(
                go.Scatter(
                    x=success_sorted,
                    y=success_cdf,
                    name="Success",
                    line=dict(color="green", width=2),
                    mode="lines",
                    hovertemplate=f"{stat_name}: %{{x:.6f}}<br>Cumulative Prob: %{{y:.4f}}<extra></extra>",
                )
            )

        if len(failure_sorted) > 0:
            failure_cdf = np.arange(1, len(failure_sorted) + 1) / len(failure_sorted)
            fig.add_trace(
                go.Scatter(
                    x=failure_sorted,
                    y=failure_cdf,
                    name="Failure",
                    line=dict(color="red", width=2),
                    mode="lines",
                    hovertemplate=f"{stat_name}: %{{x:.6f}}<br>Cumulative Prob: %{{y:.4f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title or f"Cumulative Distribution: {stat_name}",
            xaxis_title=stat_name,
            yaxis_title="Cumulative Probability",
            hovermode="x unified",
        )
    else:
        histnorm = "probability" if plot_type == "normalized" else None

        fig.add_trace(
            go.Histogram(
                x=success_values,
                name="Success",
                opacity=0.6,
                marker=dict(color="green"),
                nbinsx=50,
                histnorm=histnorm,
            )
        )

        fig.add_trace(
            go.Histogram(
                x=failure_values,
                name="Failure",
                opacity=0.6,
                marker=dict(color="red"),
                nbinsx=50,
                histnorm=histnorm,
            )
        )

        fig.update_layout(
            title=title
            or f"{'Normalized ' if histnorm else ''}Distribution: {stat_name}",
            xaxis_title=stat_name,
            yaxis_title="Probability" if histnorm else "Count",
            barmode="overlay",
        )

    fig.update_layout(height=500, showlegend=True)
    return fig


def create_distribution_comparison_plot(
    success_counts: np.ndarray,
    failure_counts: np.ndarray,
    bin_centers: np.ndarray,
    bin_width: float,
    title: str = "Influence Distribution Comparison",
) -> go.Figure:
    """Create a side-by-side bar plot comparing distributions.

    Args:
        success_counts: Counts for successful rollouts
        failure_counts: Counts for failed rollouts
        bin_centers: X-axis positions for bins
        bin_width: Width of each bin
        title: Plot title

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Successful Rollouts", "Failed Rollouts"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=success_counts,
            name="Success",
            marker=dict(color="green", opacity=0.7),
            width=bin_width * 0.9,
            hovertemplate="Influence: %{x:.4f}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=failure_counts,
            name="Failure",
            marker=dict(color="red", opacity=0.7),
            width=bin_width * 0.9,
            hovertemplate="Influence: %{x:.4f}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Influence Value", row=1, col=1)
    fig.update_xaxes(title_text="Influence Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    return fig


def create_embedding_plot(
    embeddings_2d: np.ndarray,
    groups: List[Dict],
    title: str = "t-SNE Embeddings",
    height: int = 600,
) -> go.Figure:
    """Create an interactive scatter plot for 2D embeddings.

    Args:
        embeddings_2d: 2D array of shape (N, 2)
        groups: List of dictionaries, each containing:
            - 'name': Group name for legend
            - 'indices': Indices of points in this group
            - 'color': Color for this group
            - 'hover_texts': List of hover text strings for points in this group
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for group in groups:
        indices = group["indices"]
        name = group["name"]
        color = group["color"]
        hover_texts = group["hover_texts"]

        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[indices, 0],
                y=embeddings_2d[indices, 1],
                mode="markers",
                name=name,
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.7,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=height,
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )

    return fig


def create_performance_influence_bar_plot(
    scores: np.ndarray,
    labels: List[str],
    title: str = "Performance Influence",
) -> go.Figure:
    """Create a bar plot of performance influence per demonstration.

    Args:
        scores: Performance influence scores
        labels: Labels for each bar (e.g., "Demo 5 [okay]")
        title: Plot title

    Returns:
        Plotly Figure object
    """
    colors = ["green" if s > 0 else "red" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=list(range(len(scores))),
            y=scores,
            text=labels,
            marker_color=colors,
            hovertemplate="<b>%{text}</b><br>Score: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Demonstration Rank",
        yaxis_title="Influence Score",
        height=500,
        showlegend=False,
    )

    return fig


def create_behavior_pie_chart(
    labels: List[str],
    values: List[int],
    title: str = "Behavior Distribution",
) -> go.Figure:
    """Create a pie chart for behavior distribution.

    Args:
        labels: Group labels
        values: Group values
        title: Plot title

    Returns:
        Plotly Figure object
    """
    import plotly.express as px

    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        hole=0.3,  # Donut chart
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True)

    return fig


def create_curation_episode_histogram(
    episode_labels: List[str],
    counts: List[int],
    episode_totals: List[int],
    title: str = "Samples marked for curation by episode",
    episode_quality_labels: Optional[List[str]] = None,
    episode_slice_label_breakdown: Optional[List[Dict[str, int]]] = None,
) -> go.Figure:
    """Create a bar chart of curated sample count per episode, with fraction in hover.

    Bars are colored by episode quality label when episode_quality_labels is provided.

    Args:
        episode_labels: Label per episode (e.g. ["Episode 0", "Episode 3", ...])
        counts: Number of samples marked for curation in that episode
        episode_totals: Total samples in that episode (for hover fraction)
        title: Plot title
        episode_quality_labels: Optional quality label per episode (colors bars, shown in hover)
        episode_slice_label_breakdown: Optional list of dicts: per episode, slice_label -> sample count

    Returns:
        Plotly Figure object
    """

    def _hover_slice_breakdown(breakdown: Optional[Dict[str, int]]) -> str:
        if not breakdown:
            return ""
        parts = [f"{label}: {n}" for label, n in sorted(breakdown.items())]
        return "<br>Slice labels: " + ", ".join(parts)

    hover_texts = []
    for i, (c, total) in enumerate(zip(counts, episode_totals)):
        pct = (100.0 * c / total) if total > 0 else 0
        line = f"{c} of {total} samples ({pct:.1f}%)"
        if episode_quality_labels is not None and i < len(episode_quality_labels):
            line += f"<br>Quality: {episode_quality_labels[i]}"
        if episode_slice_label_breakdown is not None and i < len(episode_slice_label_breakdown):
            line += _hover_slice_breakdown(episode_slice_label_breakdown[i])
        hover_texts.append(line)

    fig = go.Figure()

    if episode_quality_labels is not None and len(episode_quality_labels) == len(
        episode_labels
    ):
        # One trace per quality so bars are colored and legend appears
        quality_color_map: Dict[str, str] = {}
        by_quality: Dict[str, Tuple[List[str], List[int], List[str]]] = {}
        for i, (ep_lbl, cnt, total) in enumerate(zip(episode_labels, counts, episode_totals)):
            q = episode_quality_labels[i]
            if q not in by_quality:
                by_quality[q] = ([], [], [])
            by_quality[q][0].append(ep_lbl)
            by_quality[q][1].append(cnt)
            pct = (100.0 * cnt / total) if total > 0 else 0
            hover = f"{cnt} of {total} samples ({pct:.1f}%)<br>Quality: {q}"
            if episode_slice_label_breakdown is not None and i < len(episode_slice_label_breakdown):
                hover += _hover_slice_breakdown(episode_slice_label_breakdown[i])
            by_quality[q][2].append(hover)
        for q in sorted(by_quality.keys()):
            ep_lbls, cnts, hovers = by_quality[q]
            color = get_label_color(q, quality_color_map)
            fig.add_trace(
                go.Bar(
                    x=ep_lbls,
                    y=cnts,
                    name=q,
                    text=cnts,
                    textposition="outside",
                    customdata=hovers,
                    hovertemplate="%{x}<br>%{customdata}<extra></extra>",
                    marker_color=color,
                    legendgroup=q,
                )
            )
        fig.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": episode_labels},
            showlegend=True,
        )
    else:
        fig.add_trace(
            go.Bar(
                x=episode_labels,
                y=counts,
                text=counts,
                textposition="outside",
                customdata=hover_texts,
                hovertemplate="%{x}<br>%{customdata}<extra></extra>",
            )
        )
        fig.update_layout(showlegend=False)

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Samples marked for curation",
        height=350,
    )
    return fig


def create_selection_source_distribution_plot(
    rollout_labels: List[str],
    counts: List[int],
    title: str = "Source: linked demo slices per rollout episode",
    height: int = 300,
) -> go.Figure:
    """Bar chart: for each rollout episode, how many selected demo slices are linked to it."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rollout_labels,
            y=counts,
            text=counts,
            textposition="outside",
            hovertemplate="%{x}<br>Linked demo slices: %{y}<extra></extra>",
        )
    )
    n_total = len(rollout_labels)
    fig.update_layout(
        title=title,
        xaxis_title=f"Rollout episode ({n_total} total)",
        yaxis_title="Linked demo slices",
        height=height,
    )
    return fig


def create_selection_contribution_stacked_plot(
    per_selection_rollout: List[Dict[str, Any]],
    rollout_eps: List[int],
    title: str = "Contributions by selection per rollout episode",
    height: int = 350,
) -> Optional[go.Figure]:
    """Stacked bar chart: for each rollout episode, show how many linked slices came from each selection.

    per_selection_rollout: list of dicts with selection_id, label, rollout_ep_counts (rollout_ep -> count).
    rollout_eps: ordered list of rollout episode indices to show on x-axis.
    """
    if not per_selection_rollout or not rollout_eps:
        return None
    x_labels = [f"Rollout ep{ep}" for ep in rollout_eps]
    # One trace per selection (stacked)
    fig = go.Figure()
    for i, sel in enumerate(per_selection_rollout):
        counts = [sel["rollout_ep_counts"].get(ep, 0) for ep in rollout_eps]
        if sum(counts) == 0:
            continue
        label = sel.get("label") or f"Sel #{sel['selection_id']}"
        fig.add_trace(
            go.Bar(
                name=label,
                x=x_labels,
                y=counts,
                hovertemplate="%{x}<br>%{fullData.name}: %{y} slices<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Rollout episode",
        yaxis_title="Linked demo slices",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_selection_target_distribution_plot(
    episode_labels: List[str],
    counts: List[int],
    title: str = "Target: selected samples per demo episode",
    height: int = 300,
) -> go.Figure:
    """Bar chart: for each demo episode with selections, number of selected samples."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=episode_labels,
            y=counts,
            text=counts,
            textposition="outside",
            hovertemplate="%{x}<br>Samples: %{y}<extra></extra>",
        )
    )
    n_total = len(episode_labels)
    fig.update_layout(
        title=title,
        xaxis_title=f"Demo episode ({n_total} total)",
        yaxis_title="Selected samples",
        height=height,
    )
    return fig


def create_selection_overlap_heatmap(
    overlap_matrix: np.ndarray,
    labels: List[str],
    title: str = "Selection overlap (IoU)",
    show_iou: bool = True,
) -> go.Figure:
    """Heatmap of sample overlap between each pair of selections.

    Args:
        overlap_matrix: 2D array; entry (i, j) = number of samples (timesteps) in both selection i and j.
        labels: Label for each selection (e.g. ["Sel #0", "Sel #1", ...]).
        title: Plot title.
        show_iou: If True, display cell (i, j) as intersection over union (symmetric, 0–100%).

    Returns:
        Plotly Figure object
    """
    n = overlap_matrix.shape[0]
    if show_iou:
        # IoU(i,j) = |A ∩ B| / |A ∪ B| = overlap[i,j] / (|A| + |B| - overlap[i,j])
        iou = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                intersection = overlap_matrix[i, j]
                union = (
                    overlap_matrix[i, i]
                    + overlap_matrix[j, j]
                    - intersection
                )
                if union > 0:
                    iou[i, j] = 100.0 * intersection / union
        z_show = iou
        text = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                text[i, j] = f"{z_show[i, j]:.0f}%"
        customdata = overlap_matrix
        hovertemplate = (
            "<b>%{y}</b> vs <b>%{x}</b><br>"
            "IoU: %{z:.1f}%<br>Intersection: %{customdata} samples<extra></extra>"
        )
    else:
        z_show = overlap_matrix.astype(np.float64)
        text = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                text[i, j] = str(int(overlap_matrix[i, j]))
        customdata = None
        hovertemplate = "<b>%{y}</b> vs <b>%{x}</b><br>Overlap: %{z} samples<extra></extra>"
    fig = go.Figure(
        data=go.Heatmap(
            z=z_show,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            customdata=customdata,
            colorscale="Blues",
            zmin=0,
            zmax=100 if show_iou else None,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Selection",
        yaxis_title="Selection",
        height=max(300, 120 + 40 * n),
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_selection_quality_distributions_plot(
    selection_labels: List[str],
    per_selection_quality_counts: List[Dict[str, int]],
    title: str = "Demo quality distribution per selection",
    max_cols: int = 2,
    total_samples_by_quality: Optional[Dict[str, int]] = None,
    height_per_row: int = 380,
) -> go.Figure:
    """Create a grid of pie charts: for each selection, show sample count by demo quality label.

    Args:
        selection_labels: Display label per selection (e.g. ["reaching", "pick", ...]).
        per_selection_quality_counts: For each selection, dict mapping quality_label -> sample count.
        title: Overall figure title.
        max_cols: Maximum number of columns in the subplot grid (default 2).
        total_samples_by_quality: If provided, total training samples per quality (for hover "% of all X samples").
        height_per_row: Figure height in pixels per row (total height = height_per_row * n_rows).

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    n = len(selection_labels)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    # Use small spacing so each subplot gets most of the row height (Plotly: spacing <= 1/(n_rows-1))
    vertical_spacing = min(0.06, 1.0 / (n_rows - 1)) if n_rows > 1 else 0.0
    horizontal_spacing = 0.06
    specs = [[{"type": "domain"} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=selection_labels,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    # Collect all quality labels to keep order/colors consistent
    all_qualities: List[str] = []
    for counts in per_selection_quality_counts:
        for q in counts:
            if q not in all_qualities:
                all_qualities.append(q)
    all_qualities = sorted(all_qualities)

    for idx, (sel_label, counts) in enumerate(
        zip(selection_labels, per_selection_quality_counts)
    ):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        labels = [q for q in all_qualities if counts.get(q, 0) > 0]
        values = [counts.get(q, 0) for q in labels]
        if not labels:
            labels = ["(no data)"]
            values = [0]
        # Build hover: "Quality: X samples (Y% of all [quality] samples)" when total_samples_by_quality given
        if total_samples_by_quality and values != [0]:
            total_for_quality = [total_samples_by_quality.get(q, 0) for q in labels]
            pct_of_quality = [
                (100.0 * v / t) if t > 0 else 0.0
                for v, t in zip(values, total_for_quality)
            ]
            hover_text = [
                f"{q}: {v} samples ({pct:.1f}% of all {q} training samples)"
                for q, v, pct in zip(labels, values, pct_of_quality)
            ]
        else:
            hover_text = [f"{q}: {v} samples" for q, v in zip(labels, values)]
        # Show label + absolute count on each slice (not percentage)
        text = [str(v) for v in values]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                text=text,
                textinfo="label+text",
                textposition="inside",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_text,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        height=max(400, height_per_row * n_rows),
    )
    return fig


def create_behavior_influence_bar_chart(
    sample_labels: List[str],
    influences: np.ndarray,
    title: str = "Mean Influence per Demo Sample",
) -> go.Figure:
    """Create a bar chart showing aggregated influence per demo sample.

    Args:
        sample_labels: Labels for each demo sample (e.g., ["s5", "s10", ...])
        influences: Influence values per sample
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=sample_labels,
            y=influences,
            marker=dict(
                color=influences,
                colorscale="RdBu",
                cmid=0,
                colorbar=dict(title="Influence"),
            ),
            hovertemplate="Sample: %{x}<br>Mean Influence: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Demo Sample",
        yaxis_title="Mean Influence",
        height=350,
        showlegend=False,
    )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig

    return fig


def create_histogram(
    values: np.ndarray,
    title: str = "Histogram",
    xaxis_title: str = "Value",
    yaxis_title: str = "Count",
    color: str = "blue",
    nbins: int = 30,
    vline_at: Optional[float] = None,
    vline_label: Optional[str] = None,
) -> go.Figure:
    """Create a simple histogram with optional vertical line.

    Args:
        values: Array of values
        title: Plot title
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        color: Bar color
        nbins: Number of bins
        vline_at: Optional x-position for vertical line
        vline_label: Optional label for vertical line

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=nbins,
            marker=dict(color=color, opacity=0.7),
        )
    )

    if vline_at is not None:
        fig.add_vline(
            x=vline_at,
            line_dash="dash",
            line_color="white",
            annotation_text=vline_label or f"Line at {vline_at:.4f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=400,
    )

    return fig


def create_temporal_patterns_subplots(
    data_list: List[Dict],
    title: str = "Temporal Patterns",
    subplot_titles: Tuple[str, str] = ("Left", "Right"),
) -> go.Figure:
    """Create side-by-side temporal pattern plots.

    Args:
        data_list: List of dicts, each with 'x', 'y', 'name', 'col' (1 or 2)
        title: Main plot title
        subplot_titles: Titles for the two subplots

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
    )

    for entry in data_list:
        fig.add_trace(
            go.Scatter(
                x=entry["x"],
                y=entry["y"],
                mode="lines",
                name=entry["name"],
                opacity=0.6,
            ),
            row=1,
            col=entry["col"],
        )

    fig.update_layout(
        height=500,
        showlegend=True,
        title_text=title,
    )

    return fig


def create_demo_variance_plot(
    demo_indices: List[int],
    avg_stds: List[float],
    counts: List[int],
    qualities: List[str],
    title: str = "Demo Variance on Failures",
) -> go.Figure:
    """Create a bar chart for demo-specific variance.

    Args:
        demo_indices: List of demo indices
        avg_stds: Average standard deviation per demo
        counts: Number of times demo was paired with failure
        qualities: Quality labels for demos
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=demo_indices,
            y=avg_stds,
            marker=dict(color=avg_stds, colorscale="Reds", showscale=True),
            customdata=np.stack([counts, qualities], axis=1),
            hovertemplate="Demo %{x}<br>Avg Std: %{y:.6f}<br>Count: %{customdata[0]}<br>Quality: %{customdata[1]}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Demo Episode Index",
        yaxis_title="Average Std",
        height=400,
    )

    return fig


def create_autocorrelation_plot(
    rollout_lags: List[int],
    rollout_autocorr: List[float],
    demo_lags: List[int],
    demo_autocorr: List[float],
    title: str = "Temporal Autocorrelation",
) -> go.Figure:
    """Create an autocorrelation plot.

    Args:
        rollout_lags: Lags for rollout axis
        rollout_autocorr: Autocorrelation values for rollout axis
        demo_lags: Lags for demo axis
        demo_autocorr: Autocorrelation values for demo axis
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rollout_lags,
            y=rollout_autocorr,
            mode="lines+markers",
            name="Rollout Axis",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=demo_lags,
            y=demo_autocorr,
            mode="lines+markers",
            name="Demo Axis",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (Timesteps)",
        yaxis_title="Autocorrelation",
        height=400,
    )

    return fig


def create_diagonal_analysis_plot(
    diagonal_vals: np.ndarray,
    off_diagonal_vals: np.ndarray,
    title: str = "Diagonal vs Off-Diagonal Distribution",
) -> go.Figure:
    """Create histograms comparing diagonal vs off-diagonal values.

    Args:
        diagonal_vals: Values on the diagonal
        off_diagonal_vals: Values off the diagonal
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=diagonal_vals,
            name="Diagonal",
            opacity=0.7,
            nbinsx=30,
            marker_color="blue",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=off_diagonal_vals,
            name="Off-Diagonal",
            opacity=0.7,
            nbinsx=30,
            marker_color="red",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Influence Value",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )

    return fig


def create_gradient_magnitude_heatmap(
    grad_magnitude: np.ndarray,
    title: str = "Gradient Magnitude",
) -> go.Figure:
    """Create a heatmap for gradient magnitude.

    Args:
        grad_magnitude: 2D array of gradient magnitudes
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=grad_magnitude,
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Demo Timestep",
        yaxis_title="Rollout Timestep",
        height=500,
    )

    return fig


def create_peak_detection_plot(
    transition_matrix: np.ndarray,
    max_coords: np.ndarray,
    min_coords: np.ndarray,
    title: str = "Local Peaks in Influence Matrix",
) -> go.Figure:
    """Create a peak detection visualization.

    Args:
        transition_matrix: 2D influence matrix
        max_coords: Array of (row, col) coordinates for local maxima
        min_coords: Array of (row, col) coordinates for local minima
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Heatmap background
    fig.add_trace(
        go.Heatmap(
            z=transition_matrix,
            colorscale=[(0, "red"), (0.5, "white"), (1, "green")],
            zmid=0,
            showscale=True,
        )
    )

    # Maxima markers
    if len(max_coords) > 0:
        fig.add_trace(
            go.Scatter(
                x=max_coords[:, 1],
                y=max_coords[:, 0],
                mode="markers",
                marker=dict(symbol="circle", size=8, color="blue"),
                name="Local Maxima",
            )
        )

    # Minima markers
    if len(min_coords) > 0:
        fig.add_trace(
            go.Scatter(
                x=min_coords[:, 1],
                y=min_coords[:, 0],
                mode="markers",
                marker=dict(symbol="x", size=10, color="yellow"),
                name="Local Minima",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Demo Timestep",
        yaxis_title="Rollout Timestep",
        height=500,
    )

    return fig


def create_asymmetry_variance_plot(
    data_rollout: np.ndarray,
    data_demo: np.ndarray,
    title: str = "Asymmetry in Variance",
) -> go.Figure:
    """Create side-by-side variance plots for asymmetry analysis.

    Args:
        data_rollout: Variance values along rollout axis
        data_demo: Variance values along demo axis
        title: Plot title

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Var Along Demo Axis", "Var Along Rollout Axis"),
    )

    fig.add_trace(
        go.Scatter(y=data_rollout, mode="lines", name="Demo Axis"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(y=data_demo, mode="lines", name="Rollout Axis"),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=400,
        title_text=title,
    )

    return fig
    return fig


def create_generic_tsne_plot(
    embeddings_2d: np.ndarray,
    metadata: List[Dict],
    color_by: str = "mean_influence",
    title: str = "t-SNE Visualization",
    categorical_color: bool = False,
) -> go.Figure:
    """Create a generic t-SNE scatter plot.

    Args:
        embeddings_2d: Array of shape (n_samples, 2)
        metadata: List of metadata dicts
        color_by: Key in metadata for coloring
        title: Plot title
        categorical_color: Whether color_by is a categorical variable

    Returns:
        Plotly Figure object
    """
    if categorical_color:
        unique_labels = sorted(set(m[color_by] for m in metadata))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        color_values = [label_to_idx[m[color_by]] for m in metadata]
        colorscale = "Viridis"
        showscale = False
    else:
        color_values = [m.get(color_by, 0) for m in metadata]
        colorscale = [[0, "red"], [0.5, "white"], [1, "green"]]
        showscale = True

    hover_texts = []
    for m in metadata:
        lines = [f"{k}: {v}" for k, v in m.items()]
        hover_texts.append("<br>".join(lines))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=color_values,
                colorscale=colorscale,
                showscale=showscale,
                colorbar=dict(title=color_by) if showscale else None,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        hovermode="closest",
        height=600,
    )

    return fig


def create_influence_grid_plot(
    matrices: List[np.ndarray],
    titles: List[str],
    main_title: str = "Influence Matrices Grid",
    rows: Optional[int] = None,
    cols: int = 4,
    colorscale: str = "Red-White-Green",
) -> go.Figure:
    """Create a grid of influence heatmaps.

    Args:
        matrices: List of 2D arrays
        titles: Titles for each subplot
        main_title: Overall plot title
        rows: Number of rows (calculated if None)
        cols: Number of columns
        colorscale: Name of colorscale or list of (pos, color) tuples

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_plots = len(matrices)
    if rows is None:
        rows = (num_plots + cols - 1) // cols

    if colorscale == "Red-White-Green":
        cs = [(0, "red"), (0.5, "white"), (1, "green")]
    elif colorscale == "Red-White-Blue":
        cs = [(0, "red"), (0.5, "white"), (1, "blue")]
    else:
        cs = colorscale

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for i, matrix in enumerate(matrices):
        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=cs,
                showscale=(i == 0),
                zmid=0,
                hovertemplate="Rollout t: %{y}<br>Demo t: %{x}<br>Influence: %{z:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=main_title,
        height=300 * rows,
        showlegend=False,
    )

    return fig


def create_comparison_histograms(
    train_data: np.ndarray,
    rollout_data: np.ndarray,
    labels: List[str],
    title: str = "Distribution Comparison",
) -> go.Figure:
    """Create a grid of histograms comparing two datasets.

    Args:
        train_data: Array of shape (n_samples, n_dims)
        rollout_data: Array of shape (n_samples, n_dims)
        labels: Labels for each dimension
        title: Overall plot title

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_dims = train_data.shape[1]
    cols = 3
    rows = (num_dims + cols - 1) // cols

    max_vertical_spacing = 1.0 / (rows - 1) if rows > 1 else 0.1
    vertical_spacing = min(0.15, max_vertical_spacing * 0.9)
    horizontal_spacing = 0.1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=labels,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    for i in range(num_dims):
        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Histogram(
                x=train_data[:, i],
                name="Training",
                marker_color="blue",
                opacity=0.6,
                nbinsx=50,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Histogram(
                x=rollout_data[:, i],
                name="Rollout",
                marker_color="red",
                opacity=0.6,
                nbinsx=50,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        height=300 * rows,
        barmode="overlay",
        showlegend=True,
    )

    return fig


def create_aggregated_influence_grid(
    matrices: List[np.ndarray],
    titles: List[str],
    std_matrices: Optional[List[np.ndarray]] = None,
    main_title: str = "Aggregated Influence Heatmaps",
    rollouts_per_row: int = 2,
) -> go.Figure:
    """Create a grid of aggregated influence heatmaps, optionally with std dev bars.

    Args:
        matrices: List of aggregated (summed) 2D arrays
        std_matrices: Optional list of std dev 2D arrays (1 column each)
        titles: Titles for each main plot
        main_title: Overall plot title
        rollouts_per_row: Number of rollouts shown horizontally

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_rollouts = len(matrices)
    show_std = std_matrices is not None

    if show_std:
        cols_per_rollout = 2
        total_cols = rollouts_per_row * cols_per_rollout
        rows = (num_rollouts + rollouts_per_row - 1) // rollouts_per_row

        specs = []
        full_titles = []
        for i in range(rows):
            row_specs = []
            for j in range(rollouts_per_row):
                idx = i * rollouts_per_row + j
                if idx < num_rollouts:
                    row_specs.extend([{"type": "heatmap"}, {"type": "heatmap"}])
                    full_titles.extend([titles[idx], "Std(rollout t)"])
                else:
                    row_specs.extend([{"type": "heatmap"}, {"type": "heatmap"}])
                    full_titles.extend(["", ""])
            specs.append(row_specs)

        fig = make_subplots(
            rows=rows,
            cols=total_cols,
            subplot_titles=full_titles,
            specs=specs,
            vertical_spacing=0.12,
            horizontal_spacing=0.02,
            column_widths=[0.88, 0.12] * rollouts_per_row,
        )
    else:
        total_cols = rollouts_per_row
        rows = (num_rollouts + rollouts_per_row - 1) // rollouts_per_row
        fig = make_subplots(
            rows=rows,
            cols=total_cols,
            subplot_titles=titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

    for i, matrix in enumerate(matrices):
        row = i // rollouts_per_row + 1
        col_base = (i % rollouts_per_row) * (cols_per_rollout if show_std else 1) + 1

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=[(0, "red"), (0.5, "white"), (1, "green")],
                showscale=(i == 0),
                zmid=0,
                hovertemplate="Rollout t: %{y}<br>Demo t: %{x}<br>Summed Influence: %{z:.4f}<extra></extra>",
            ),
            row=row,
            col=col_base,
        )

        if show_std and std_matrices is not None and i < len(std_matrices):
            fig.add_trace(
                go.Heatmap(
                    z=std_matrices[i],
                    colorscale="Viridis",
                    showscale=False,
                    hovertemplate="Rollout t: %{y}<br>Std Dev: %{z:.4f}<extra></extra>",
                ),
                row=row,
                col=col_base + 1,
            )
            fig.update_xaxes(showticklabels=False, row=row, col=col_base + 1)
            fig.update_yaxes(showticklabels=False, row=row, col=col_base + 1)

    fig.update_layout(
        title=main_title,
        height=350 * rows,
        showlegend=False,
    )

    return fig


def create_influence_density_heatmaps(
    demo_timestep_influences_list: List[List[List[float]]],
    subplot_titles: List[str],
    title: str,
    cols: int = 4,
    nbinsy: int = 200,
) -> go.Figure:
    """Create a grid of density heatmaps for influence distributions.

    Args:
        demo_timestep_influences_list: List of demos, each containing lists of influence values per timestep
        subplot_titles: Titles for each subplot
        title: Main plot title
        cols: Number of columns in grid
        nbinsy: Number of bins along y-axis (influence values)

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_demos = len(demo_timestep_influences_list)
    rows = (num_demos + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, demo_timestep_influences in enumerate(demo_timestep_influences_list):
        demo_len = len(demo_timestep_influences)

        # Flatten data for histogram2d
        all_demo_timesteps = []
        all_influence_values = []

        for demo_t in range(demo_len):
            values = demo_timestep_influences[demo_t]
            all_demo_timesteps.extend([demo_t] * len(values))
            all_influence_values.extend(values)

        row = idx // cols + 1
        col = idx % cols + 1

        # Create 2D histogram heatmap
        fig.add_trace(
            go.Histogram2d(
                x=all_demo_timesteps,
                y=all_influence_values,
                colorscale="Viridis",
                showscale=(idx == 0),
                nbinsx=demo_len,
                nbinsy=nbinsy,
                colorbar=dict(
                    title="Count",
                    len=0.3,
                    y=0.85,
                )
                if idx == 0
                else None,
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_xaxes(title_text="Demo Timestep")
    fig.update_yaxes(title_text="Influence")

    fig.update_layout(
        height=300 * rows,
        title_text=title,
        showlegend=False,
    )

    return fig


def create_influence_distribution_lines(
    statistics_list: List[Dict[str, np.ndarray]],
    subplot_titles: List[str],
    title: str,
    show_mean: bool = True,
    show_midpoint: bool = False,
    show_std_bands: bool = True,
    show_percentile_bands: bool = False,
    show_minmax_bands: bool = True,
    cols: int = 4,
) -> go.Figure:
    """Create a grid of line plots showing influence distribution statistics.

    Args:
        statistics_list: List of dicts containing statistics arrays for each demo
            Each dict should have keys: 'demo_timesteps', 'means', 'stds', 'mins', 'maxs',
            'p1s', 'p99s', 'midpoints'
        subplot_titles: Titles for each subplot
        title: Main plot title
        show_mean: Whether to show mean line
        show_midpoint: Whether to show midpoint line
        show_std_bands: Whether to show ±1σ and ±2σ bands
        show_percentile_bands: Whether to show 1-99 percentile bands
        show_minmax_bands: Whether to show min-max bands
        cols: Number of columns in grid

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_demos = len(statistics_list)
    rows = (num_demos + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, stats in enumerate(statistics_list):
        demo_timesteps = stats["demo_timesteps"]
        means = stats["means"]
        stds = stats["stds"]
        mins = stats["mins"]
        maxs = stats["maxs"]
        p1s = stats.get("p1s")
        p99s = stats.get("p99s")
        midpoints = stats.get("midpoints")

        row = idx // cols + 1
        col = idx % cols + 1

        # Add min/max band (optional)
        if show_minmax_bands:
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=maxs,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=(idx == 0),
                    name="Max",
                    legendgroup="max",
                    hovertemplate="Demo t=%{x}<br>Max=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=mins,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(200, 200, 200, 0.2)",
                    line=dict(width=0),
                    showlegend=(idx == 0),
                    name="Min-Max",
                    legendgroup="minmax",
                    hovertemplate="Demo t=%{x}<br>Min=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        # Add 1-99 percentile band (optional)
        if show_percentile_bands and p1s is not None and p99s is not None:
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=p99s,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate="Demo t=%{x}<br>P99=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=p1s,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 200, 100, 0.2)",
                    line=dict(width=0),
                    showlegend=(idx == 0),
                    name="P1-P99",
                    legendgroup="percentile",
                    hovertemplate="Demo t=%{x}<br>P1=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        # Add std bands (optional)
        if show_std_bands:
            # ±2 std band
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=means + 2 * stds,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate="Demo t=%{x}<br>Mean+2σ=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=means - 2 * stds,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(100, 150, 255, 0.2)",
                    line=dict(width=0),
                    showlegend=(idx == 0),
                    name="±2σ",
                    legendgroup="2std",
                    hovertemplate="Demo t=%{x}<br>Mean-2σ=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # ±1 std band
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=means + stds,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate="Demo t=%{x}<br>Mean+σ=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=means - stds,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(100, 150, 255, 0.4)",
                    line=dict(width=0),
                    showlegend=(idx == 0),
                    name="±1σ",
                    legendgroup="1std",
                    hovertemplate="Demo t=%{x}<br>Mean-σ=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        # Add mean line (optional)
        if show_mean:
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=means,
                    mode="lines",
                    line=dict(color="rgb(255, 100, 100)", width=2),
                    showlegend=(idx == 0),
                    name="Mean",
                    legendgroup="mean",
                    hovertemplate="Demo t=%{x}<br>Mean=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        # Add midpoint line (optional)
        if show_midpoint and midpoints is not None:
            fig.add_trace(
                go.Scatter(
                    x=demo_timesteps,
                    y=midpoints,
                    mode="lines",
                    line=dict(color="rgb(150, 150, 150)", width=1.5, dash="dot"),
                    showlegend=(idx == 0),
                    name="Midpoint",
                    legendgroup="midpoint",
                    hovertemplate="Demo t=%{x}<br>Midpoint=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_xaxes(title_text="Demo Timestep")
    fig.update_yaxes(title_text="Influence")

    fig.update_layout(
        height=300 * rows,
        title_text=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_slice_influence_heatmap(
    influence_slice: np.ndarray,
    rollout_start: int,
    rollout_end: int,
    aggregated_scores: Optional[np.ndarray] = None,
    window_width: Optional[int] = None,
    title: str = "Local Influence Matrix",
    show_aggregation_line: bool = True,
    highlight_top_k: Optional[int] = None,
    top_k_indices: Optional[np.ndarray] = None,
    demo_episode_boundaries: Optional[List[int]] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Create a heatmap visualization of the local influence matrix for a rollout slice.

    This shows the influence submatrix for a selected slice of the rollout,
    with optional overlays for sliding window aggregation scores and top-k highlights.

    Args:
        influence_slice: 2D array of shape (slice_height, num_demo_samples)
        rollout_start: Start index of the rollout slice (for labeling)
        rollout_end: End index of the rollout slice (for labeling)
        aggregated_scores: Optional 1D array of aggregated influence scores per demo sample
        window_width: Width of the sliding window (for annotation)
        title: Plot title
        show_aggregation_line: Whether to show the aggregation scores as a line plot
        highlight_top_k: Number of top samples to highlight
        top_k_indices: Indices of top-k samples to highlight
        demo_episode_boundaries: List of demo sample indices where episodes start
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    slice_height, num_demo_samples = influence_slice.shape

    # Determine if we need subplots for aggregation line
    if show_aggregation_line and aggregated_scores is not None:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=["Influence Matrix", "Sliding Window Aggregation"],
        )
    else:
        fig = go.Figure()

    # Compute symmetric color limits centered at 0
    abs_max = max(
        abs(np.nanmin(influence_slice)),
        abs(np.nanmax(influence_slice)),
    )
    if abs_max == 0:
        abs_max = 1.0
    zmin, zmax = -abs_max, abs_max

    # Create y-axis labels (rollout timesteps)
    y_labels = [f"t={rollout_start + i}" for i in range(slice_height)]

    # Main heatmap
    heatmap_trace = go.Heatmap(
        z=influence_slice,
        x=np.arange(num_demo_samples),
        y=np.arange(slice_height),
        colorscale=get_influence_colorscale(),
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        hovertemplate=(
            "Demo sample: %{x}<br>"
            f"Rollout t={rollout_start}+%{{y}}<br>"
            "Influence: %{z:.4f}<extra></extra>"
        ),
        colorbar=dict(title="Influence", len=0.7 if show_aggregation_line else 1.0),
    )

    if show_aggregation_line and aggregated_scores is not None:
        fig.add_trace(heatmap_trace, row=1, col=1)
    else:
        fig.add_trace(heatmap_trace)

    # Add demo episode boundaries as vertical lines
    if demo_episode_boundaries is not None:
        for boundary in demo_episode_boundaries:
            if 0 < boundary < num_demo_samples:
                line_kwargs = dict(
                    type="line",
                    x0=boundary - 0.5,
                    x1=boundary - 0.5,
                    y0=-0.5,
                    y1=slice_height - 0.5,
                    line=dict(color="rgba(100, 100, 100, 0.5)", width=1, dash="dot"),
                )
                if show_aggregation_line and aggregated_scores is not None:
                    fig.add_shape(**line_kwargs, row=1, col=1)
                else:
                    fig.add_shape(**line_kwargs)

    # Highlight top-k samples
    if highlight_top_k is not None and top_k_indices is not None:
        for i, idx in enumerate(top_k_indices[:highlight_top_k]):
            if 0 <= idx < num_demo_samples:
                rect_kwargs = dict(
                    type="rect",
                    x0=idx - 0.5,
                    x1=idx + 0.5,
                    y0=-0.5,
                    y1=slice_height - 0.5,
                    line=dict(
                        color="rgba(0, 255, 0, 0.8)"
                        if i < 3
                        else "rgba(0, 200, 0, 0.5)",
                        width=2 if i < 3 else 1,
                    ),
                    fillcolor="rgba(0, 0, 0, 0)",
                )
                if show_aggregation_line and aggregated_scores is not None:
                    fig.add_shape(**rect_kwargs, row=1, col=1)
                else:
                    fig.add_shape(**rect_kwargs)

    # Add aggregation line plot
    if show_aggregation_line and aggregated_scores is not None:
        # Main aggregation line
        fig.add_trace(
            go.Scatter(
                x=np.arange(num_demo_samples),
                y=aggregated_scores,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=1.5),
                name="Aggregated Score",
                hovertemplate="Demo sample: %{x}<br>Score: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Highlight top-k points
        if highlight_top_k is not None and top_k_indices is not None:
            top_indices = top_k_indices[:highlight_top_k]
            top_scores = aggregated_scores[top_indices]
            fig.add_trace(
                go.Scatter(
                    x=top_indices,
                    y=top_scores,
                    mode="markers",
                    marker=dict(
                        color="rgba(0, 200, 0, 0.8)",
                        size=8,
                        symbol="circle",
                        line=dict(color="darkgreen", width=1),
                    ),
                    name=f"Top {highlight_top_k}",
                    hovertemplate="Demo sample: %{x}<br>Score: %{y:.4f}<br>(Top %{text})<extra></extra>",
                    text=[f"#{i + 1}" for i in range(len(top_indices))],
                ),
                row=2,
                col=1,
            )

        # Add zero line
        fig.add_hline(
            y=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=2,
            col=1,
        )

    # Calculate height
    if height is None:
        base_height = max(300, slice_height * 15 + 100)
        height = base_height + (
            150 if show_aggregation_line and aggregated_scores is not None else 0
        )

    # Update layout
    window_info = f" (window={window_width})" if window_width else ""
    full_title = f"{title}: Rollout [{rollout_start}:{rollout_end}]{window_info}"

    fig.update_layout(
        title=full_title,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    if show_aggregation_line and aggregated_scores is not None:
        fig.update_xaxes(title_text="Demo Sample Index", row=2, col=1)
        fig.update_yaxes(
            title_text="Rollout Timestep",
            tickmode="array",
            tickvals=list(range(slice_height)),
            ticktext=y_labels,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Agg. Score", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Demo Sample Index")
        fig.update_yaxes(
            title_text="Rollout Timestep",
            tickmode="array",
            tickvals=list(range(slice_height)),
            ticktext=y_labels,
        )

    return fig
