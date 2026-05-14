"""Tests for Layer 1 (graph topology) tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from tests.vlm.proposals.agents.conftest import build_fixture_context


class TestTopologyTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp)
        self.tools = build_tool_registry("A_G", self.ctx)

    def _call(self, name, args):
        return self.tools[name].func(args)

    def test_get_graph_summary(self):
        r = self._call("get_graph_summary", {})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertIn("n_cluster_nodes", payload)
        # Three real clusters in the fixture (0, 1, 2, 4 — but "3" is unused).
        self.assertEqual(payload["n_rollouts_in_pool"], 6)
        self.assertEqual(payload["rollout_outcomes"]["success"], 3)
        self.assertEqual(payload["rollout_outcomes"]["failure"], 3)

    def test_list_nodes_returns_each_cluster(self):
        r = self._call("list_nodes", {})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        ids = {n["node_id"] for n in payload["nodes"]}
        # Cluster ids present in the fixture metadata.
        self.assertEqual(ids, {0, 1, 2, 4})

    def test_list_nodes_filter_by_failure_likelihood(self):
        r = self._call("list_nodes", {"min_failure_likelihood": 0.99})
        payload = json.loads(r.content[0].text)
        # Cluster 4 leads to FAILURE with probability 1 in the fixture.
        ids = [n["node_id"] for n in payload["nodes"]]
        self.assertIn(4, ids)
        self.assertNotIn(2, ids)

    def test_list_paths_start_to_failure(self):
        r = self._call("list_paths", {"from_node": "START", "to_node": "FAILURE", "top_k": 5})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertGreaterEqual(len(payload["paths"]), 1)
        # Each path starts with START and ends with FAILURE.
        for p in payload["paths"]:
            self.assertEqual(p["path"][0], "START")
            self.assertEqual(p["path"][-1], "FAILURE")

    def test_get_node_includes_kinematic_summary(self):
        r = self._call("get_node", {"node_id": 1})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["node_id"], 1)
        self.assertIn("kinematic_summary", payload)
        # Kinematic summary should be non-empty (cluster_stats fallback path).
        self.assertGreater(len(payload["kinematic_summary"]), 10)

    def test_get_node_unknown_id(self):
        r = self._call("get_node", {"node_id": 999})
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "not_found")

    def test_get_edge_existing(self):
        r = self._call("get_edge", {"from_node": 0, "to_node": 1})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["from_id"], 0)
        self.assertEqual(payload["to_id"], 1)
        # All 6 fixture rollouts traverse 0 -> 1.
        self.assertGreater(payload["count"], 0)
        self.assertGreater(len(payload["example_rollouts"]), 0)

    def test_get_edge_missing(self):
        r = self._call("get_edge", {"from_node": 2, "to_node": 4})
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "not_found")


if __name__ == "__main__":
    unittest.main()
