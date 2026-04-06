"""Pure Plotly plotting for training curve comparisons. No UI dependencies."""

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


_COLORS = [
    ("rgb(31, 119, 180)", "rgba(31, 119, 180, 0.2)"),
    ("rgb(255, 127, 14)", "rgba(255, 127, 14, 0.2)"),
    ("rgb(44, 160, 44)", "rgba(44, 160, 44, 0.2)"),
    ("rgb(214, 39, 40)", "rgba(214, 39, 40, 0.2)"),
    ("rgb(148, 103, 189)", "rgba(148, 103, 189, 0.2)"),
    ("rgb(140, 86, 75)", "rgba(140, 86, 75, 0.2)"),
]


def create_training_comparison_plot(
    series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    metric_key: str = "test/mean_score",
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """Create a training curve comparison plot.

    Args:
        series: List of (label, steps, mean, std) tuples, one per experiment group.
        metric_key: Metric name for y-axis label.
        title: Plot title (defaults to metric_key over training).
        height: Figure height in pixels.

    Returns:
        Plotly Figure with mean lines and shaded std bands.
    """
    fig = go.Figure()
    plot_title = title or f"{metric_key} over training (mean \u00b1 std across seeds)"

    for i, (label, steps, mean, std) in enumerate(series):
        line_c, fill_c = _COLORS[i % len(_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                name=label,
                line=dict(color=line_c, width=2),
                mode="lines",
            )
        )
        if std is not None and np.any(np.isfinite(std) & (std > 0)):
            upper = mean + std
            lower = np.maximum(mean - std, 0.0)  # clamp to [0, 1] — success rate can't be negative
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=fill_c,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=plot_title,
        xaxis_title="Step",
        yaxis_title=metric_key,
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    return fig
