"""Unit tests for policy_doctor.plotting.plotly.eef_trajectories."""

from __future__ import annotations

import unittest

import numpy as np
import plotly.graph_objects as go

from policy_doctor.plotting.plotly.eef_trajectories import (
    create_eef_trajectory_figure,
    create_initial_eef_scatter_2d,
)


def _line(n: int = 10, offset: float = 0.0) -> np.ndarray:
    """Straight-line EEF trajectory (T, 3)."""
    t = np.linspace(0, 1, n)
    return np.column_stack([t + offset, np.zeros(n), np.zeros(n)]).astype(np.float32)


class TestCreate3dTrajectoryFigure(unittest.TestCase):
    def test_returns_figure(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=_line(10),
            generated_xyz_list=[_line(10, 0.1), _line(10, 0.2)],
        )
        self.assertIsInstance(fig, go.Figure)

    def test_empty_generated_still_renders(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=_line(5),
            generated_xyz_list=[],
        )
        self.assertIsInstance(fig, go.Figure)

    def test_none_seed_still_renders(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=None,
            generated_xyz_list=[_line(8), _line(8, 0.5)],
        )
        self.assertIsInstance(fig, go.Figure)

    def test_has_seed_trace(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=_line(5),
            generated_xyz_list=[_line(5)],
        )
        names = [t.name for t in fig.data]
        self.assertIn("Seed demo", names)

    def test_has_generated_trace(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=None,
            generated_xyz_list=[_line(5), _line(5)],
        )
        names = [t.name for t in fig.data]
        self.assertIn("Generated", names)

    def test_mean_trace_present_when_enabled(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=None,
            generated_xyz_list=[_line(5), _line(5)],
            show_mean=True,
        )
        names = [t.name for t in fig.data]
        self.assertIn("Mean (generated)", names)

    def test_mean_trace_absent_when_disabled(self):
        fig = create_eef_trajectory_figure(
            seed_xyz=None,
            generated_xyz_list=[_line(5)],
            show_mean=False,
        )
        names = [t.name for t in fig.data]
        self.assertNotIn("Mean (generated)", names)


class TestCreate2dInitialScatter(unittest.TestCase):
    def test_returns_figure(self):
        fig = create_initial_eef_scatter_2d(
            seed_xyz=_line(5),
            generated_xyz_list=[_line(5), _line(5, 0.3)],
        )
        self.assertIsInstance(fig, go.Figure)

    def test_seed_and_generated_traces(self):
        fig = create_initial_eef_scatter_2d(
            seed_xyz=_line(5),
            generated_xyz_list=[_line(5)],
        )
        names = [t.name for t in fig.data]
        self.assertTrue(any("Seed" in n for n in names))
        self.assertTrue(any("Generated" in n for n in names))

    def test_empty_generated(self):
        fig = create_initial_eef_scatter_2d(
            seed_xyz=_line(3),
            generated_xyz_list=[],
        )
        self.assertIsInstance(fig, go.Figure)
