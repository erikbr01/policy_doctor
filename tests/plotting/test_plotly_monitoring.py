"""Tests for policy_doctor.plotting.plotly.monitoring."""

import unittest

import numpy as np
import plotly.graph_objects as go

from policy_doctor.plotting.plotly.monitoring import (
    create_demo_influence_bar,
    create_intervention_scatter,
    create_monitoring_timeline,
)


class TestCreateMonitoringTimeline(unittest.TestCase):

    def _make_inputs(self, T=20):
        rng = np.random.RandomState(42)
        timesteps = np.arange(T)
        node_names = [f"Behavior {rng.randint(0, 4)}" for _ in range(T)]
        return timesteps, node_names

    def test_returns_figure(self):
        ts, names = self._make_inputs()
        fig = create_monitoring_timeline(ts, names)
        self.assertIsInstance(fig, go.Figure)

    def test_has_bar_trace(self):
        ts, names = self._make_inputs()
        fig = create_monitoring_timeline(ts, names)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        self.assertGreater(len(bar_traces), 0)

    def test_bar_length_matches_timesteps(self):
        T = 15
        ts, names = self._make_inputs(T)
        fig = create_monitoring_timeline(ts, names)
        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        self.assertEqual(len(bar.x), T)

    def test_intervention_marker_trace_added(self):
        ts, names = self._make_inputs(10)
        mask = np.zeros(10, dtype=bool)
        mask[3] = True
        mask[7] = True
        fig = create_monitoring_timeline(ts, names, intervention_mask=mask)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        intv_traces = [t for t in scatter_traces if "Intervention" in (t.name or "")]
        self.assertEqual(len(intv_traces), 1)
        self.assertEqual(len(intv_traces[0].x), 2)

    def test_no_intervention_trace_when_mask_all_false(self):
        ts, names = self._make_inputs(10)
        mask = np.zeros(10, dtype=bool)
        fig = create_monitoring_timeline(ts, names, intervention_mask=mask)
        intv_traces = [t for t in fig.data if "Intervention" in (t.name or "")]
        self.assertEqual(len(intv_traces), 0)

    def test_legend_entries_for_unique_nodes(self):
        ts = np.arange(6)
        names = ["A", "B", "A", "C", "B", "A"]
        fig = create_monitoring_timeline(ts, names)
        legend_names = {t.name for t in fig.data if t.showlegend}
        self.assertIn("A", legend_names)
        self.assertIn("B", legend_names)
        self.assertIn("C", legend_names)

    def test_shared_color_map_is_mutated(self):
        ts, names = self._make_inputs(5)
        color_map: dict = {}
        create_monitoring_timeline(ts, names, node_color_map=color_map)
        self.assertGreater(len(color_map), 0)

    def test_unassigned_nodes_do_not_appear_in_legend(self):
        ts = np.arange(4)
        names = ["A", "N/A", "A", "-1"]
        fig = create_monitoring_timeline(ts, names)
        legend_names = {t.name for t in fig.data if t.showlegend}
        self.assertNotIn("N/A", legend_names)
        self.assertNotIn("-1", legend_names)

    def test_height_respected(self):
        ts, names = self._make_inputs()
        fig = create_monitoring_timeline(ts, names, height=200)
        self.assertEqual(fig.layout.height, 200)

    def test_title_set(self):
        ts, names = self._make_inputs()
        fig = create_monitoring_timeline(ts, names, title="My Timeline")
        self.assertEqual(fig.layout.title.text, "My Timeline")

    def test_distances_appear_in_hover(self):
        ts = np.arange(3)
        names = ["A", "B", "A"]
        dists = np.array([0.1, 0.2, 0.3])
        fig = create_monitoring_timeline(ts, names, distances=dists)
        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        self.assertTrue(any("dist=" in str(h) for h in bar.hovertext))


class TestCreateInterventionScatter(unittest.TestCase):

    def _make_inputs(self, T=12):
        rng = np.random.RandomState(7)
        ts = np.arange(T)
        names = [f"Node{rng.randint(0, 3)}" for _ in range(T)]
        vals = rng.randn(T).astype(np.float32)
        return ts, names, vals

    def test_returns_figure(self):
        ts, names, vals = self._make_inputs()
        fig = create_intervention_scatter(ts, names, values=vals)
        self.assertIsInstance(fig, go.Figure)

    def test_value_line_trace_present(self):
        ts, names, vals = self._make_inputs()
        fig = create_intervention_scatter(ts, names, values=vals)
        line_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.name == "V(s)"]
        self.assertEqual(len(line_traces), 1)

    def test_no_value_line_when_values_none(self):
        ts, names, _ = self._make_inputs()
        fig = create_intervention_scatter(ts, names, values=None)
        line_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.name == "V(s)"]
        self.assertEqual(len(line_traces), 0)

    def test_intervention_markers_with_values(self):
        T = 8
        ts = np.arange(T)
        names = ["A"] * T
        vals = np.zeros(T, dtype=np.float32)
        mask = np.zeros(T, dtype=bool)
        mask[2] = True
        fig = create_intervention_scatter(ts, names, values=vals, intervention_mask=mask)
        intv = [t for t in fig.data if "Intervention" in (t.name or "")]
        self.assertEqual(len(intv), 1)
        self.assertEqual(len(intv[0].x), 1)


class TestCreateDemoInfluenceBar(unittest.TestCase):

    def test_returns_figure(self):
        indices = np.array([5, 2, 8, 1, 3])
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)
        fig = create_demo_influence_bar(indices, scores)
        self.assertIsInstance(fig, go.Figure)

    def test_has_bar_trace(self):
        indices = np.arange(5)
        scores = np.linspace(0.1, 0.9, 5).astype(np.float32)
        fig = create_demo_influence_bar(indices, scores)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        self.assertGreater(len(bar_traces), 0)

    def test_top_k_limits_bars(self):
        indices = np.arange(10)
        scores = np.linspace(0.0, 1.0, 10).astype(np.float32)
        fig = create_demo_influence_bar(indices, scores, top_k=4)
        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        self.assertEqual(len(bar.x), 4)

    def test_custom_labels_used(self):
        indices = np.array([0, 1, 2])
        scores = np.array([0.8, 0.5, 0.2], dtype=np.float32)
        labels = ["ep0 t=10 ✓", "ep1 t=5 ✗", "ep2 t=20 ✓"]
        fig = create_demo_influence_bar(indices, scores, demo_labels=labels)
        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        for label in labels:
            self.assertTrue(any(label in str(y) for y in bar.y))

    def test_height_respected(self):
        indices = np.arange(3)
        scores = np.array([0.3, 0.2, 0.1], dtype=np.float32)
        fig = create_demo_influence_bar(indices, scores, height=250)
        self.assertEqual(fig.layout.height, 250)

    def test_single_entry(self):
        indices = np.array([7])
        scores = np.array([0.42], dtype=np.float32)
        fig = create_demo_influence_bar(indices, scores, top_k=1)
        self.assertIsInstance(fig, go.Figure)
