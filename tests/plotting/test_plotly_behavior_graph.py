"""Tests for policy_doctor.plotting.plotly.behavior_graph."""

import unittest

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.plotting.plotly import create_behavior_graph_plot


def _minimal_behavior_graph():
    """Build a minimal BehaviorGraph (START -> 0 -> SUCCESS, 0 -> 1 -> FAILURE)."""
    cluster_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    metadata = [
        {"rollout_idx": 0, "timestep": 0},
        {"rollout_idx": 0, "timestep": 1},
        {"rollout_idx": 1, "timestep": 0},
        {"rollout_idx": 1, "timestep": 1},
        {"rollout_idx": 0, "timestep": 2},
    ]
    # Ep0: 0,0,0 -> success; Ep1: 1,1 -> failure
    cluster_labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    metadata = [
        {"rollout_idx": 0, "timestep": 0, "success": True},
        {"rollout_idx": 0, "timestep": 1, "success": True},
        {"rollout_idx": 0, "timestep": 2, "success": True},
        {"rollout_idx": 1, "timestep": 0, "success": False},
        {"rollout_idx": 1, "timestep": 1, "success": False},
    ]
    return BehaviorGraph.from_cluster_assignments(
        cluster_labels, metadata, level="rollout"
    )


class TestPlotlyBehaviorGraph(unittest.TestCase):
    def test_create_behavior_graph_plot_returns_figure(self):
        graph = _minimal_behavior_graph()
        fig = create_behavior_graph_plot(graph, min_probability=0.0, title="Test")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")

    def test_create_behavior_graph_plot_has_scatter_traces(self):
        graph = _minimal_behavior_graph()
        fig = create_behavior_graph_plot(graph)
        self.assertGreater(len(fig.data), 0)
        for t in fig.data:
            self.assertEqual(t.type, "scatter")
