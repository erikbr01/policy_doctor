"""Plotly figures for end-effector trajectory distribution visualisation.

Two figures are produced:
* A **3D line figure** showing the seed trajectory and all generated
  trajectories, coloured by success (green) or failure (red).
* A **2D scatter figure** showing only the initial (t=0) XY positions of
  seed and generated demos — the key view for answering whether generated
  initial states are in-distribution.

No Streamlit imports here.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_SEED_COLOR = "#e377c2"        # pink/magenta — seed demo (distinct from success/failure)
_SUCCESS_COLOR = "#2ca02c"     # green — successful generated demos
_FAILURE_COLOR = "#d62728"     # red — failed generated demos
_FALLBACK_COLOR = "#1f77b4"    # blue — when success/failure unknown


def create_eef_trajectory_figure(
    seed_xyz: np.ndarray | None,
    generated_xyz_list: list[np.ndarray],
    *,
    failed_xyz_list: list[np.ndarray] | None = None,
    generated_opacity: float = 0.3,
    title: str = "EEF Trajectory Distribution",
) -> go.Figure:
    """3D Plotly figure of seed + generated EEF trajectories coloured by outcome.

    Args:
        seed_xyz:            ``(T, 3)`` array for the seed demo.  ``None`` if
                             the seed was not prepared (no ``datagen_info``).
        generated_xyz_list:  List of ``(T, 3)`` arrays — successful demos.
        failed_xyz_list:     List of ``(T, 3)`` arrays — failed demos.
                             When ``None``, all demos in ``generated_xyz_list``
                             are plotted in blue without a success/failure split.
        generated_opacity:   Opacity for individual generated trajectories.
        title:               Figure title.

    Returns:
        :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    if failed_xyz_list is not None:
        # --- Failed trajectories (red, faint) ---
        first = True
        for traj in failed_xyz_list:
            if traj.shape[0] == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line=dict(color=_FAILURE_COLOR, width=2),
                opacity=generated_opacity,
                name="Failed" if first else None,
                showlegend=first,
                legendgroup="failed",
            ))
            first = False

        # Initial positions of failed demos
        failed_valid = [t for t in failed_xyz_list if t.shape[0] > 0]
        if failed_valid:
            t0 = np.stack([t[0] for t in failed_valid])
            fig.add_trace(go.Scatter3d(
                x=t0[:, 0], y=t0[:, 1], z=t0[:, 2],
                mode="markers",
                marker=dict(color=_FAILURE_COLOR, size=4, symbol="circle"),
                name=f"Failed t=0 (n={len(t0)})",
                legendgroup="failed_t0",
            ))

        # --- Successful trajectories (green) ---
        first = True
        for traj in generated_xyz_list:
            if traj.shape[0] == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line=dict(color=_SUCCESS_COLOR, width=2),
                opacity=generated_opacity,
                name="Success" if first else None,
                showlegend=first,
                legendgroup="success",
            ))
            first = False

        # Initial positions of successful demos
        succ_valid = [t for t in generated_xyz_list if t.shape[0] > 0]
        if succ_valid:
            t0 = np.stack([t[0] for t in succ_valid])
            fig.add_trace(go.Scatter3d(
                x=t0[:, 0], y=t0[:, 1], z=t0[:, 2],
                mode="markers",
                marker=dict(color=_SUCCESS_COLOR, size=4, symbol="circle"),
                name=f"Success t=0 (n={len(t0)})",
                legendgroup="success_t0",
            ))

    else:
        # No success/failure split — single colour
        first = True
        for traj in generated_xyz_list:
            if traj.shape[0] == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line=dict(color=_FALLBACK_COLOR, width=2),
                opacity=generated_opacity,
                name="Generated" if first else None,
                showlegend=first,
                legendgroup="generated",
            ))
            first = False

        if generated_xyz_list:
            t0_xyz = np.stack([t[0] for t in generated_xyz_list if t.shape[0] > 0])
            fig.add_trace(go.Scatter3d(
                x=t0_xyz[:, 0], y=t0_xyz[:, 1], z=t0_xyz[:, 2],
                mode="markers",
                marker=dict(color=_FALLBACK_COLOR, size=5, symbol="circle"),
                name="Generated t=0",
                legendgroup="gen_t0",
            ))

    # --- Seed trajectory (prominent) ---
    if seed_xyz is not None and seed_xyz.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=seed_xyz[:, 0], y=seed_xyz[:, 1], z=seed_xyz[:, 2],
            mode="lines+markers",
            line=dict(color=_SEED_COLOR, width=5),
            marker=dict(color=list(range(seed_xyz.shape[0])),
                        colorscale="RdPu", size=4),
            name="Seed demo",
        ))
        fig.add_trace(go.Scatter3d(
            x=[seed_xyz[0, 0]], y=[seed_xyz[0, 1]], z=[seed_xyz[0, 2]],
            mode="markers",
            marker=dict(color=_SEED_COLOR, size=10, symbol="diamond"),
            name="Seed t=0",
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
        legend=dict(orientation="v", x=1.0, y=1.0),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def create_initial_eef_scatter_2d(
    seed_xyz: np.ndarray | None,
    generated_xyz_list: list[np.ndarray],
    *,
    failed_xyz_list: list[np.ndarray] | None = None,
    title: str = "Initial EEF Positions (t=0) — XY Plane",
) -> go.Figure:
    """2D XY scatter of initial end-effector positions (t=0), coloured by outcome.

    Args:
        seed_xyz:            ``(T, 3)`` seed trajectory — only t=0 is used.
        generated_xyz_list:  List of ``(T, 3)`` successful generated trajectories.
        failed_xyz_list:     List of ``(T, 3)`` failed generated trajectories.
                             When ``None``, all demos are plotted in blue.
        title:               Figure title.

    Returns:
        :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    if failed_xyz_list is not None:
        # Failed initial positions
        failed_valid = [t for t in failed_xyz_list if t.shape[0] > 0]
        if failed_valid:
            t0 = np.stack([t[0] for t in failed_valid])
            fig.add_trace(go.Scatter(
                x=t0[:, 0], y=t0[:, 1],
                mode="markers",
                marker=dict(color=_FAILURE_COLOR, size=8, opacity=0.55,
                            line=dict(color="white", width=0.5)),
                name=f"Failed (n={len(t0)})",
                text=[f"fail_{i}" for i in range(len(t0))],
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>%{text}<extra></extra>",
            ))

        # Success initial positions
        succ_valid = [t for t in generated_xyz_list if t.shape[0] > 0]
        if succ_valid:
            t0 = np.stack([t[0] for t in succ_valid])
            fig.add_trace(go.Scatter(
                x=t0[:, 0], y=t0[:, 1],
                mode="markers",
                marker=dict(color=_SUCCESS_COLOR, size=8, opacity=0.7,
                            line=dict(color="white", width=0.5)),
                name=f"Success (n={len(t0)})",
                text=[f"demo_{i}" for i in range(len(t0))],
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>%{text}<extra></extra>",
            ))

    else:
        if generated_xyz_list:
            t0 = np.stack([t[0] for t in generated_xyz_list if t.shape[0] > 0])
            fig.add_trace(go.Scatter(
                x=t0[:, 0], y=t0[:, 1],
                mode="markers",
                marker=dict(color=_FALLBACK_COLOR, size=8, opacity=0.6,
                            line=dict(color="white", width=0.5)),
                name=f"Generated (n={len(t0)})",
                text=[f"demo_{i}" for i in range(len(t0))],
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>%{text}<extra></extra>",
            ))

    if seed_xyz is not None and seed_xyz.shape[0] > 0:
        fig.add_trace(go.Scatter(
            x=[seed_xyz[0, 0]], y=[seed_xyz[0, 1]],
            mode="markers",
            marker=dict(color=_SEED_COLOR, size=16, symbol="star"),
            name="Seed t=0",
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<extra>Seed</extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        legend=dict(orientation="h", y=-0.15),
        height=450,
    )
    return fig
