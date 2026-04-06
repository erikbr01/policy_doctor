"""Tests for policy_doctor.plotting.pyvis (interactive behavior graph)."""

import unittest

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph


def _minimal_graph():
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


class TestPyvis(unittest.TestCase):
    def test_create_interactive_behavior_graph_returns_str(self):
        try:
            from policy_doctor.plotting.pyvis import create_interactive_behavior_graph
        except ImportError as e:
            self.skipTest(f"pyvis not installed: {e}")
        graph = _minimal_graph()
        html = create_interactive_behavior_graph(graph, min_probability=0.0)
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 100)

    def test_create_interactive_behavior_graph_contains_vis_network(self):
        try:
            from policy_doctor.plotting.pyvis import create_interactive_behavior_graph
        except ImportError as e:
            self.skipTest(f"pyvis not installed: {e}")
        graph = _minimal_graph()
        html = create_interactive_behavior_graph(graph)
        self.assertTrue(
            "vis" in html.lower() or "network" in html.lower() or "graph" in html.lower(),
            "Expected HTML to contain vis/network/graph (vis.js)",
        )

    def test_create_value_colored_interactive_graph_returns_str(self):
        try:
            from policy_doctor.plotting.pyvis import (
                create_value_colored_interactive_graph,
            )
        except ImportError as e:
            self.skipTest(f"pyvis not installed: {e}")
        graph = _minimal_graph()
        values = {nid: 0.5 for nid in graph.nodes}
        html = create_value_colored_interactive_graph(graph, values)
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 100)

    def test_create_timestep_colored_interactive_graph_returns_str(self):
        try:
            from policy_doctor.plotting.pyvis import (
                create_timestep_colored_interactive_graph,
            )
        except ImportError as e:
            self.skipTest(f"pyvis not installed: {e}")
        graph = _minimal_graph()
        html = create_timestep_colored_interactive_graph(graph, min_probability=0.0)
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 100)
