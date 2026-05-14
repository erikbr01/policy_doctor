"""Tests for Layer 2 (slice + rollout access) tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from tests.vlm.proposals.agents.conftest import build_fixture_context


class TestAccessTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp)
        self.tools = build_tool_registry("A_G", self.ctx)

    def _call(self, name, args):
        return self.tools[name].func(args)

    def test_list_slices_in_node(self):
        r = self._call("list_slices_in_node", {"node_id": 1})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["node_id"], 1)
        # Six rollouts each have a slice in node 1.
        self.assertEqual(payload["n_slices"], 6)
        # Slice ids round-trip.
        from policy_doctor.vlm.proposals.agents.tools.access import parse_slice_id

        for row in payload["slices"]:
            sid = row["slice_id"]
            parsed = parse_slice_id(sid)
            self.assertIsNotNone(parsed)

    def test_list_slices_in_node_missing(self):
        r = self._call("list_slices_in_node", {"node_id": 99})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["n_slices"], 0)

    def test_get_rollout_summary(self):
        r = self._call("get_rollout_summary", {"rollout_id": "r0001"})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["rollout_id"], "r0001")
        self.assertEqual(payload["outcome"], "success")
        # Fixture rollouts have cluster_path = [0, 1, 2] for successes.
        self.assertEqual(payload["cluster_path_ids"], [0, 1, 2])

    def test_list_rollouts_filters(self):
        r = self._call("list_rollouts", {"outcome": "failure"})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["n_rollouts"], 3)
        for row in payload["rollouts"]:
            self.assertEqual(row["outcome"], "failure")

    def test_list_rollouts_passes_through(self):
        r = self._call("list_rollouts", {"passes_through": [4]})
        payload = json.loads(r.content[0].text)
        # Failure cluster_path includes 4; success cluster_path doesn't.
        ids = [row["rollout_id"] for row in payload["rollouts"]]
        self.assertEqual(set(ids), {"r0003", "r0004", "r0005"})

    def test_get_rollout_video_no_storyboard(self):
        # No storyboard files exist in the fixture; tool should return error.
        r = self._call("get_rollout_video", {"rollout_id": "r0000", "format": "storyboard"})
        self.assertFalse(r.ok)
        self.assertIn(r.metadata["error_code"], {"no_frames", "not_found"})

    def test_get_slice_video_bad_id(self):
        r = self._call("get_slice_video", {"slice_id": "garbage"})
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "bad_arg")


if __name__ == "__main__":
    unittest.main()
