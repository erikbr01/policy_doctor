"""Tests for policy_doctor.plotting.plotly.heatmaps."""

import unittest

import numpy as np

from policy_doctor.plotting.plotly import create_influence_heatmap


class TestPlotlyHeatmaps(unittest.TestCase):
    def test_create_influence_heatmap_returns_figure(self):
        mat = np.array([[1.0, -0.5], [0.0, 0.5]], dtype=np.float32)
        fig = create_influence_heatmap(
            mat,
            x_labels=["d0", "d1"],
            y_labels=["r0", "r1"],
            title="Test",
        )
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")
        # Plotly may expose Figure as plotly.graph_objects.Figure or plotly.graph_objs._figure
        self.assertIn("plotly", fig.__class__.__module__)

    def test_create_influence_heatmap_has_heatmap_trace(self):
        mat = np.array([[1.0, -0.5], [0.0, 0.5]], dtype=np.float32)
        fig = create_influence_heatmap(
            mat,
            x_labels=["d0", "d1"],
            y_labels=["r0", "r1"],
        )
        self.assertGreater(len(fig.data), 0)
        self.assertEqual(fig.data[0].type, "heatmap")
        np.testing.assert_array_almost_equal(fig.data[0].z, mat)

    def test_create_influence_heatmap_symmetric_z(self):
        mat = np.array([[-2.0, 1.0]], dtype=np.float32)
        fig = create_influence_heatmap(mat, x_labels=["a", "b"], y_labels=["r0"])
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertLessEqual(fig.data[0].zmin, -2.0)
        self.assertGreaterEqual(fig.data[0].zmax, 1.0)

    def test_create_influence_heatmap_no_label_bars(self):
        mat = np.ones((2, 2), dtype=np.float32)
        fig = create_influence_heatmap(
            mat,
            x_labels=["a", "b"],
            y_labels=["x", "y"],
            show_label_bars=False,
        )
        self.assertEqual(fig.data[0].type, "heatmap")
