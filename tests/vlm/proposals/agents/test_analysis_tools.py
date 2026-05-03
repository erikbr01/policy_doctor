"""Tests for Layer 3 (analysis) tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from tests.vlm.proposals.agents.conftest import build_fixture_context


class TestAnalysisTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp)
        self.tools = build_tool_registry("A_G", self.ctx)

    def _call(self, name, args):
        return self.tools[name].func(args)

    def test_find_failure_nodes_finds_cluster_4(self):
        r = self._call("find_failure_nodes", {"min_failure_prob": 0.5})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        ids = [n["node_id"] for n in payload["nodes"]]
        # In the fixture, cluster 4 leads directly to FAILURE.
        self.assertIn(4, ids)

    def test_find_recovery_paths(self):
        r = self._call("find_recovery_paths", {"from_node": 1, "top_k": 5})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        # From node 1 there's at least one path through 2 to SUCCESS.
        self.assertGreaterEqual(len(payload["paths"]), 1)
        for p in payload["paths"]:
            self.assertIn(1, p["node_ids"])
            self.assertEqual(p["path"][-1], "SUCCESS")

    def test_find_recovery_paths_unknown_node(self):
        r = self._call("find_recovery_paths", {"from_node": 999})
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "not_found")

    def test_find_underrepresented_modes_rollout_count(self):
        r = self._call(
            "find_underrepresented_modes",
            {"metric": "rollout_count", "threshold": 4},
        )
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        # Cluster 2 (success-only) and 4 (failure-only) each have 3 rollouts;
        # both fall below threshold=4.
        ids = {n["node_id"] for n in payload["nodes"]}
        self.assertIn(2, ids)
        self.assertIn(4, ids)

    def test_find_underrepresented_modes_bad_metric(self):
        r = self._call("find_underrepresented_modes", {"metric": "garbage"})
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "bad_arg")

    def test_compare_paths(self):
        # Success path vs failure path.
        success_path = [-2, 0, 1, 2, -4]   # START, 0, 1, 2, SUCCESS
        failure_path = [-2, 0, 1, 4, -5]   # START, 0, 1, 4, FAILURE
        r = self._call("compare_paths", {"path_a": success_path, "path_b": failure_path})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        # Shared prefix is START -> 0 -> 1.
        self.assertEqual(payload["shared_prefix"], ["START", "c0", "c1"])
        # Path A (success) outcome distribution: 100% success in fixture.
        self.assertEqual(payload["path_a_outcome_distribution"]["success"], 1.0)
        self.assertEqual(payload["path_b_outcome_distribution"]["failure"], 1.0)


if __name__ == "__main__":
    unittest.main()
