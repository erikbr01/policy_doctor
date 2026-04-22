"""Plotly figures for end-effector trajectory distribution visualisation.

Two figures are produced:
* A **3D line figure** showing the seed trajectory and all generated
  trajectories, coloured by success (green) or failure (red).
* A **2D scatter figure** showing only the initial (t=0) XY positions of
  seed and generated demos — the key view for answering whether generated
  initial states are in-distribution.

Optional overlays (each controlled by a boolean flag):
* ``show_nut_pose``          — draw the square nut's initial pose as a
  coordinate frame (3D) or position marker + heading arrow (2D).
* ``show_gripper_at_lowest_z`` — draw the EEF coordinate frame at the
  timestep with the lowest Z in each trajectory (3D only).

No Streamlit imports here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

_SEED_COLOR = "#e377c2"        # pink/magenta — seed demo
_SUCCESS_COLOR = "#2ca02c"     # green — successful generated demos
_FAILURE_COLOR = "#d62728"     # red — failed generated demos
_FALLBACK_COLOR = "#1f77b4"    # blue — when success/failure unknown
_NUT_COLOR = "#ff7f0e"         # orange — square nut
_GRIPPER_LOW_COLOR = "#9467bd" # purple — gripper at lowest Z

# Axis colours for coordinate frames  (X=red, Y=green, Z=blue)
_FRAME_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]


# ---------------------------------------------------------------------------
# Coordinate-frame helpers
# ---------------------------------------------------------------------------

def _frame_traces_3d(
    pose: np.ndarray,
    scale: float = 0.015,
    name_prefix: str = "",
    group: str = "",
    showlegend_label: str | None = None,
) -> list[go.Scatter3d]:
    """Three Scatter3d line traces for the X/Y/Z axes of a 4×4 pose matrix."""
    pos = pose[:3, 3]
    rot = pose[:3, :3]
    labels = ["x", "y", "z"]
    traces: list[go.Scatter3d] = []
    for i in range(3):
        end = pos + rot[:, i] * scale
        show = (showlegend_label is not None) and (i == 0)
        traces.append(go.Scatter3d(
            x=[pos[0], end[0]],
            y=[pos[1], end[1]],
            z=[pos[2], end[2]],
            mode="lines",
            line=dict(color=_FRAME_COLORS[i], width=4),
            name=showlegend_label if show else None,
            showlegend=show,
            legendgroup=group or name_prefix,
        ))
    return traces


def _frame_arrow_2d(
    pose: np.ndarray,
    scale: float = 0.008,
    color: str = _NUT_COLOR,
) -> tuple[float, float, float, float]:
    """Return (x0, y0, dx, dy) for the XY-plane projection of the pose's X-axis."""
    pos = pose[:3, 3]
    x_axis = pose[:3, 0]  # first column = local X
    return pos[0], pos[1], x_axis[0] * scale, x_axis[1] * scale


# ---------------------------------------------------------------------------
# Main 3D figure
# ---------------------------------------------------------------------------

def create_eef_trajectory_figure(
    seed_xyz: np.ndarray | None,
    generated_xyz_list: list[np.ndarray],
    *,
    failed_xyz_list: list[np.ndarray] | None = None,
    nut_poses_t0: list[np.ndarray] | None = None,
    eef_poses_at_lowest_z: list[np.ndarray] | None = None,
    failed_eef_poses_at_lowest_z: list[np.ndarray] | None = None,
    seed_pose_at_lowest_z: np.ndarray | None = None,
    show_nut_pose: bool = True,
    show_gripper_at_lowest_z: bool = True,
    generated_opacity: float = 0.3,
    title: str = "EEF Trajectory Distribution",
) -> go.Figure:
    """3D Plotly figure of seed + generated EEF trajectories coloured by outcome.

    Args:
        seed_xyz:                    ``(T, 3)`` seed trajectory.
        generated_xyz_list:          Successful demo trajectories.
        failed_xyz_list:             Failed demo trajectories (or ``None``).
        nut_poses_t0:                List of ``(4, 4)`` initial nut poses per demo.
                                     Shown when *show_nut_pose* is ``True``.
        eef_poses_at_lowest_z:       List of ``(4, 4)`` EEF poses at min-Z, for
                                     successful demos.
        failed_eef_poses_at_lowest_z: Same for failed demos.
        seed_pose_at_lowest_z:       ``(4, 4)`` EEF pose at min-Z for the seed.
        show_nut_pose:               Draw initial nut pose as a coordinate frame.
        show_gripper_at_lowest_z:    Draw EEF coordinate frame at lowest Z.
        generated_opacity:           Opacity for individual trajectories.
        title:                       Figure title.
    """
    fig = go.Figure()

    if failed_xyz_list is not None:
        # Failed trajectories (red)
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

        # Success trajectories (green)
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

    # --- Seed trajectory ---
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

    # --- Nut initial pose (coordinate frame per demo) ---
    if show_nut_pose and nut_poses_t0:
        first_label: str | None = "Nut pose t=0"
        for pose in nut_poses_t0:
            for tr in _frame_traces_3d(
                pose, scale=0.015,
                name_prefix="nut",
                group="nut_pose",
                showlegend_label=first_label,
            ):
                fig.add_trace(tr)
            first_label = None  # only label the first frame

    # --- Gripper orientation at lowest Z ---
    if show_gripper_at_lowest_z:
        # Seed
        if seed_pose_at_lowest_z is not None:
            first_label = "Gripper @ min-Z (seed)"
            for tr in _frame_traces_3d(
                seed_pose_at_lowest_z, scale=0.02,
                name_prefix="grip_seed",
                group="grip_seed",
                showlegend_label=first_label,
            ):
                fig.add_trace(tr)

        # Successful demos
        if eef_poses_at_lowest_z:
            first_label = "Gripper @ min-Z (success)"
            for pose in eef_poses_at_lowest_z:
                for tr in _frame_traces_3d(
                    pose, scale=0.015,
                    name_prefix="grip_succ",
                    group="grip_succ",
                    showlegend_label=first_label,
                ):
                    fig.add_trace(tr)
                first_label = None

        # Failed demos
        if failed_eef_poses_at_lowest_z:
            first_label = "Gripper @ min-Z (failed)"
            for pose in failed_eef_poses_at_lowest_z:
                for tr in _frame_traces_3d(
                    pose, scale=0.015,
                    name_prefix="grip_fail",
                    group="grip_fail",
                    showlegend_label=first_label,
                ):
                    fig.add_trace(tr)
                first_label = None

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
        legend=dict(orientation="v", x=1.0, y=1.0),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 2D initial-position scatter
# ---------------------------------------------------------------------------

def create_initial_eef_scatter_2d(
    seed_xyz: np.ndarray | None,
    generated_xyz_list: list[np.ndarray],
    *,
    failed_xyz_list: list[np.ndarray] | None = None,
    nut_poses_t0: list[np.ndarray] | None = None,
    show_nut_pose: bool = True,
    title: str = "Initial EEF Positions (t=0) — XY Plane",
) -> go.Figure:
    """2D XY scatter of initial EEF positions (t=0), coloured by outcome.

    Args:
        seed_xyz:            ``(T, 3)`` seed trajectory — only t=0 used.
        generated_xyz_list:  Successful demo trajectories.
        failed_xyz_list:     Failed demo trajectories (or ``None``).
        nut_poses_t0:        List of ``(4, 4)`` initial nut poses.
                             Shown as orange squares + X-axis heading arrows
                             when *show_nut_pose* is ``True``.
        show_nut_pose:       Overlay nut initial position and orientation.
        title:               Figure title.
    """
    fig = go.Figure()

    if failed_xyz_list is not None:
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

    # --- Nut initial pose: position marker + X-axis heading arrow ---
    if show_nut_pose and nut_poses_t0:
        nut_xy = np.array([[p[0, 3], p[1, 3]] for p in nut_poses_t0])
        fig.add_trace(go.Scatter(
            x=nut_xy[:, 0], y=nut_xy[:, 1],
            mode="markers",
            marker=dict(color=_NUT_COLOR, size=10, symbol="square",
                        line=dict(color="white", width=1)),
            name=f"Nut t=0 (n={len(nut_poses_t0)})",
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<extra>Nut</extra>",
        ))
        # Draw X-axis heading as annotation arrows
        for pose in nut_poses_t0:
            x0, y0, dx, dy = _frame_arrow_2d(pose, scale=0.008, color=_NUT_COLOR)
            fig.add_annotation(
                x=x0 + dx, y=y0 + dy,
                ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1.2, arrowwidth=2,
                arrowcolor=_NUT_COLOR,
            )

    fig.update_layout(
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    return fig
