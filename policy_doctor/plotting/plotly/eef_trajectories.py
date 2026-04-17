"""Plotly figures for end-effector trajectory distribution visualisation.

Two figures are produced:
* A **3D line figure** showing the seed trajectory and all generated
  trajectories, with a mean trajectory overlay.
* A **2D scatter figure** showing only the initial (t=0) XY positions of
  seed and generated demos — the key view for answering whether generated
  initial states are in-distribution.

No Streamlit imports here.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_SEED_COLOR = "#d62728"       # red — seed demo
_GEN_COLOR = "#1f77b4"        # blue — generated demos
_MEAN_COLOR = "#ff7f0e"       # orange — mean trajectory


def create_eef_trajectory_figure(
    seed_xyz: np.ndarray | None,
    generated_xyz_list: list[np.ndarray],
    *,
    show_mean: bool = True,
    generated_opacity: float = 0.25,
    title: str = "EEF Trajectory Distribution",
) -> go.Figure:
    """3D Plotly figure of seed + generated EEF trajectories.

    Args:
        seed_xyz:            ``(T, 3)`` array for the seed demo.  ``None`` if
                             the seed was not prepared (no ``datagen_info``).
        generated_xyz_list:  List of ``(T, 3)`` arrays — one per generated demo.
        show_mean:           Overlay mean trajectory of generated demos.
        generated_opacity:   Opacity for individual generated trajectories.
        title:               Figure title.

    Returns:
        :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    # --- Generated trajectories (faint) ---
    first_gen = True
    for traj in generated_xyz_list:
        if traj.shape[0] == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines",
                line=dict(color=_GEN_COLOR, width=2),
                opacity=generated_opacity,
                name="Generated" if first_gen else None,
                showlegend=first_gen,
                legendgroup="generated",
            )
        )
        first_gen = False

    # --- Initial positions of generated demos ---
    if generated_xyz_list:
        t0_xyz = np.stack([t[0] for t in generated_xyz_list if t.shape[0] > 0])
        fig.add_trace(
            go.Scatter3d(
                x=t0_xyz[:, 0],
                y=t0_xyz[:, 1],
                z=t0_xyz[:, 2],
                mode="markers",
                marker=dict(color=_GEN_COLOR, size=5, symbol="circle"),
                name="Generated t=0",
                legendgroup="gen_t0",
            )
        )

    # --- Mean trajectory ---
    if show_mean and generated_xyz_list:
        # Align to minimum length across all trajectories
        min_len = min(t.shape[0] for t in generated_xyz_list if t.shape[0] > 0)
        if min_len > 0:
            stacked = np.stack([t[:min_len] for t in generated_xyz_list if t.shape[0] >= min_len])
            mean_xyz = stacked.mean(axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=mean_xyz[:, 0],
                    y=mean_xyz[:, 1],
                    z=mean_xyz[:, 2],
                    mode="lines",
                    line=dict(color=_MEAN_COLOR, width=4),
                    name="Mean (generated)",
                )
            )

    # --- Seed trajectory (prominent) ---
    if seed_xyz is not None and seed_xyz.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=seed_xyz[:, 0],
                y=seed_xyz[:, 1],
                z=seed_xyz[:, 2],
                mode="lines+markers",
                line=dict(color=_SEED_COLOR, width=5),
                marker=dict(
                    color=list(range(seed_xyz.shape[0])),
                    colorscale="Reds",
                    size=4,
                ),
                name="Seed demo",
            )
        )
        # Seed initial position
        fig.add_trace(
            go.Scatter3d(
                x=[seed_xyz[0, 0]],
                y=[seed_xyz[0, 1]],
                z=[seed_xyz[0, 2]],
                mode="markers",
                marker=dict(color=_SEED_COLOR, size=10, symbol="diamond"),
                name="Seed t=0",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        ),
        legend=dict(orientation="v", x=1.0, y=1.0),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def create_initial_eef_scatter_2d(
    seed_xyz: np.ndarray | None,
    generated_xyz_list: list[np.ndarray],
    *,
    title: str = "Initial EEF Positions (t=0) — XY Plane",
) -> go.Figure:
    """2D XY scatter of initial end-effector positions (t=0).

    Shows whether the generated demos' starting positions span or match the
    seed demo's initial position.

    Args:
        seed_xyz:            ``(T, 3)`` seed trajectory — only t=0 is used.
        generated_xyz_list:  List of ``(T, 3)`` generated trajectories.
        title:               Figure title.

    Returns:
        :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    if generated_xyz_list:
        t0 = np.stack([t[0] for t in generated_xyz_list if t.shape[0] > 0])
        fig.add_trace(
            go.Scatter(
                x=t0[:, 0],
                y=t0[:, 1],
                mode="markers",
                marker=dict(
                    color=_GEN_COLOR,
                    size=8,
                    opacity=0.6,
                    line=dict(color="white", width=0.5),
                ),
                name=f"Generated (n={len(t0)})",
                text=[f"demo_{i}" for i in range(len(t0))],
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>%{text}<extra></extra>",
            )
        )

    if seed_xyz is not None and seed_xyz.shape[0] > 0:
        fig.add_trace(
            go.Scatter(
                x=[seed_xyz[0, 0]],
                y=[seed_xyz[0, 1]],
                mode="markers",
                marker=dict(color=_SEED_COLOR, size=14, symbol="star"),
                name="Seed t=0",
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<extra>Seed</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        legend=dict(orientation="h", y=-0.15),
        height=450,
    )
    return fig
