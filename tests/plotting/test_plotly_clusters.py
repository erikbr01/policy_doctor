"""Tests for policy_doctor.plotting.plotly.clusters."""

import unittest

import numpy as np

from policy_doctor.plotting.plotly import create_cluster_scatter_2d


class TestPlotlyClusters(unittest.TestCase):
    def test_create_cluster_scatter_2d_returns_figure(self):
        emb = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
        labels = np.array([0, 0, 1], dtype=np.int64)
        meta = [
            {"rollout_idx": 0, "timestep": 0},
            {"rollout_idx": 0, "timestep": 1},
            {"rollout_idx": 1, "timestep": 0},
        ]
        fig = create_cluster_scatter_2d(emb, labels, meta, title="Test")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")

    def test_create_cluster_scatter_2d_has_scatter_traces(self):
        emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.int64)
        meta = [{"rollout_idx": 0, "timestep": i} for i in range(3)]
        fig = create_cluster_scatter_2d(emb, labels, meta)
        self.assertGreater(len(fig.data), 0)
        for t in fig.data:
            self.assertEqual(t.type, "scatter")

    def test_create_cluster_scatter_2d_with_noise(self):
        emb = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        labels = np.array([0, -1, 1], dtype=np.int64)  # -1 = noise
        meta = [{"rollout_idx": 0, "timestep": i} for i in range(3)]
        fig = create_cluster_scatter_2d(emb, labels, meta)
        self.assertGreater(len(fig.data), 0)
