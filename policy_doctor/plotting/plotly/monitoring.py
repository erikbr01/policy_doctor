"""Pure Plotly plotting functions for runtime monitoring visualizations.

No Streamlit imports. All functions accept pre-processed data and return
plotly.Figure objects or HTML strings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from policy_doctor.plotting.common import EXTRA_COLORS, get_label_color


# Canonical color for intervention markers
INTERVENTION_COLOR = "#FF4444"
INTERVENTION_SYMBOL = "triangle-up"

# Color for timesteps with no assignment (noise / -1 cluster)
UNASSIGNED_COLOR = "#CCCCCC"


def _build_node_color_map(node_names: Sequence[str]) -> Dict[str, str]:
    """Assign a stable color to each unique node name."""
    color_map: Dict[str, str] = {}
    for name in node_names:
        get_label_color(name, color_map)
    return color_map


def create_monitoring_timeline(
    timesteps: np.ndarray,
    node_names: Sequence[str],
    intervention_mask: Optional[np.ndarray] = None,
    current_t: Optional[int] = None,
    distances: Optional[np.ndarray] = None,
    node_color_map: Optional[Dict[str, str]] = None,
    height: int = 120,
    title: str = "",
) -> go.Figure:
    """Horizontal segmented-bar timeline colored by behavior graph node.

    Args:
        timesteps: ``(T,)`` int array of timestep indices.
        node_names: ``(T,)`` sequence of node name strings (one per timestep).
        intervention_mask: ``(T,)`` bool array — True where intervention fires.
        current_t: Optional current timestep index (draws a vertical marker).
        distances: ``(T,)`` float array of distances to cluster centroid (for hover).
        node_color_map: Pre-built ``{node_name: hex_color}`` dict. Built on the fly
            when None (share the same dict across calls for consistent colors).
        height: Figure height in pixels.
        title: Optional figure title.

    Returns:
        Plotly Figure with a compact horizontal timeline.
    """
    if node_color_map is None:
        node_color_map = {}

    T = len(timesteps)
    unique_names = sorted(set(node_names))

    # Assign colors
    colors_per_step = []
    for name in node_names:
        if name in ("", "N/A", None) or str(name) == "-1":
            colors_per_step.append(UNASSIGNED_COLOR)
        else:
            colors_per_step.append(get_label_color(str(name), node_color_map))

    # Build hover text
    hover_texts = []
    for i, (t, name) in enumerate(zip(timesteps, node_names)):
        dist_str = f"  dist={distances[i]:.3f}" if distances is not None else ""
        intv_str = "  ⚡ INTERVENTION" if (intervention_mask is not None and intervention_mask[i]) else ""
        hover_texts.append(f"t={t}  {name}{dist_str}{intv_str}")

    fig = go.Figure()

    # One bar per timestep; each bar width = 1, y = 0..1
    fig.add_trace(go.Bar(
        x=timesteps,
        y=[1.0] * T,
        marker_color=colors_per_step,
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
        width=1.0,
    ))

    # Legend traces (one invisible scatter per unique node name)
    for name in unique_names:
        if name in ("", "N/A", None) or str(name) == "-1":
            continue
        color = get_label_color(str(name), node_color_map)
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=color, size=10, symbol="square"),
            name=str(name),
            showlegend=True,
        ))

    # Intervention markers
    if intervention_mask is not None:
        intv_t = timesteps[intervention_mask]
        if len(intv_t) > 0:
            fig.add_trace(go.Scatter(
                x=intv_t,
                y=[1.1] * len(intv_t),
                mode="markers",
                marker=dict(
                    color=INTERVENTION_COLOR,
                    size=8,
                    symbol=INTERVENTION_SYMBOL,
                ),
                name="Intervention",
                hovertext=[f"t={t}  ⚡ intervention" for t in intv_t],
                hoverinfo="text",
                showlegend=True,
            ))

    # Current timestep marker
    if current_t is not None:
        fig.add_vline(
            x=current_t,
            line_color="white",
            line_width=2,
            line_dash="solid",
            annotation_text=f"t={current_t}",
            annotation_position="top",
            annotation_font_color="white",
        )

    t_min = int(timesteps[0]) if T > 0 else 0
    t_max = int(timesteps[-1]) if T > 0 else 0

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=4, r=4, t=30 if title else 4, b=4),
        xaxis=dict(
            range=[t_min - 0.5, t_max + 0.5],
            title="Timestep",
            showgrid=False,
        ),
        yaxis=dict(
            visible=False,
            range=[0, 1.4],
        ),
        bargap=0,
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font_color="white",
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0,
        ),
    )
    return fig


def create_intervention_scatter(
    timesteps: np.ndarray,
    node_names: Sequence[str],
    values: Optional[np.ndarray] = None,
    distances: Optional[np.ndarray] = None,
    intervention_mask: Optional[np.ndarray] = None,
    node_color_map: Optional[Dict[str, str]] = None,
    height: int = 200,
    title: str = "Node assignments over time",
) -> go.Figure:
    """Scatter plot of node assignments over time, with optional value trace.

    Shows a scatter strip where each point is a timestep, colored by node.
    Optionally overlays the node state-value as a continuous line, making
    intervention thresholds visually apparent.

    Args:
        timesteps: ``(T,)`` int array.
        node_names: ``(T,)`` sequence of node name strings.
        values: ``(T,)`` float array of V(s) for each timestep (optional).
        distances: ``(T,)`` float distances to centroid (optional, for hover).
        intervention_mask: ``(T,)`` bool array (optional).
        node_color_map: Shared color dict (mutated in place).
        height: Figure height in pixels.
        title: Figure title.

    Returns:
        Plotly Figure.
    """
    if node_color_map is None:
        node_color_map = {}

    T = len(timesteps)
    colors = []
    for name in node_names:
        if name in ("", "N/A", None) or str(name) == "-1":
            colors.append(UNASSIGNED_COLOR)
        else:
            colors.append(get_label_color(str(name), node_color_map))

    hover_texts = []
    for i, (t, name) in enumerate(zip(timesteps, node_names)):
        v_str = f"  V={values[i]:.3f}" if values is not None else ""
        d_str = f"  dist={distances[i]:.3f}" if distances is not None else ""
        intv_str = "  ⚡" if (intervention_mask is not None and intervention_mask[i]) else ""
        hover_texts.append(f"t={t}  {name}{v_str}{d_str}{intv_str}")

    fig = go.Figure()

    # Node scatter strip
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=np.zeros(T),
        mode="markers",
        marker=dict(color=colors, size=6, symbol="square"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
        name="nodes",
    ))

    # Value line (on secondary y-axis)
    if values is not None:
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=values,
            mode="lines",
            line=dict(color="#88AAFF", width=1.5),
            name="V(s)",
            yaxis="y2",
        ))

    # Intervention markers
    if intervention_mask is not None:
        intv_t = timesteps[intervention_mask]
        intv_v = values[intervention_mask] if values is not None else np.zeros(len(intv_t))
        if len(intv_t) > 0:
            fig.add_trace(go.Scatter(
                x=intv_t,
                y=intv_v if values is not None else np.zeros(len(intv_t)),
                mode="markers",
                marker=dict(
                    color=INTERVENTION_COLOR,
                    size=10,
                    symbol=INTERVENTION_SYMBOL,
                    line=dict(color="white", width=1),
                ),
                name="Intervention",
                yaxis="y2" if values is not None else "y",
            ))

    layout_kwargs = dict(
        title=title,
        height=height,
        margin=dict(l=4, r=4, t=30 if title else 4, b=4),
        xaxis=dict(title="Timestep", showgrid=False),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font_color="white",
        legend=dict(orientation="h", y=-0.3, x=0),
    )

    if values is not None:
        layout_kwargs["yaxis"] = dict(visible=False)
        layout_kwargs["yaxis2"] = dict(
            title="V(s)",
            overlaying="y",
            side="right",
            showgrid=True,
            gridcolor="#333355",
            zeroline=True,
            zerolinecolor="#555577",
        )
    else:
        layout_kwargs["yaxis"] = dict(visible=False)

    fig.update_layout(**layout_kwargs)
    return fig


def create_demo_influence_bar(
    demo_indices: np.ndarray,
    scores: np.ndarray,
    demo_labels: Optional[List[str]] = None,
    top_k: int = 10,
    height: int = 280,
    title: str = "Top influential training demos",
) -> go.Figure:
    """Horizontal bar chart ranking training demos by influence score.

    Args:
        demo_indices: ``(N,)`` demo episode indices (or sample indices).
        scores: ``(N,)`` influence scores.
        demo_labels: Optional list of label strings (e.g. "ep12 t=34 ✓").
            If None, uses "Demo {idx}".
        top_k: Number of bars to show.
        height: Figure height in pixels.
        title: Figure title.

    Returns:
        Plotly Figure.
    """
    k = min(top_k, len(demo_indices))
    idx_k = demo_indices[:k]
    scores_k = scores[:k]
    labels = (
        [demo_labels[i] for i in range(k)]
        if demo_labels is not None
        else [f"Demo {idx}" for idx in idx_k]
    )

    # Normalize for color
    vmin, vmax = float(scores_k.min()), float(scores_k.max())
    norm = (scores_k - vmin) / max(vmax - vmin, 1e-9)
    bar_colors = [
        f"rgb({int(50 + 180 * v)},{int(120 + 80 * v)},{int(220 - 80 * v)})"
        for v in norm
    ]

    fig = go.Figure(go.Bar(
        x=scores_k[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        hovertext=[f"{l}  score={s:.4f}" for l, s in zip(labels[::-1], scores_k[::-1])],
        hoverinfo="text",
    ))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=4, r=4, t=30 if title else 4, b=4),
        xaxis=dict(title="Influence score"),
        yaxis=dict(automargin=True),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font_color="white",
    )
    return fig
