"""Tests for Layer 4 (strategy submission) tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from tests.vlm.proposals.agents.conftest import build_fixture_context


class TestSubmissionTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp)
        self.tools = build_tool_registry("A_G", self.ctx)

    def _call(self, name, args):
        return self.tools[name].func(args)

    def _good_args(self, **overrides):
        base = {
            "request_type": "full_trajectory",
            "initial_conditions": {"reference_rollout_id": "r0000", "reference_frame": 0},
            "target_behavior": "pick up the cube and place it on the platform",
            "prohibitions": ["do not push the cube off the table"],
            "success_criterion": "task_success",
            "target_cluster": 1,
            "reasoning": "this cluster has high failure likelihood; demonstrating a clean traversal "
            "should improve the policy's pre-grasp positioning.",
        }
        base.update(overrides)
        return base

    def test_propose_happy_path(self):
        r = self._call("propose_collection_request", self._good_args())
        self.assertTrue(r.ok, r.content[0].text)
        self.assertEqual(len(self.ctx.submitted), 1)
        self.assertEqual(self.ctx.submitted[0].request.target_cluster, 1)

    def test_propose_rejects_cluster_term_in_target_behavior(self):
        r = self._call(
            "propose_collection_request",
            self._good_args(target_behavior="re-enter cluster 3 from above"),
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "validation_failed")
        self.assertEqual(len(self.ctx.submitted), 0)

    def test_propose_rejects_unknown_rollout(self):
        r = self._call(
            "propose_collection_request",
            self._good_args(initial_conditions={"reference_rollout_id": "r9999"}),
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "validation_failed")

    def test_propose_rejects_empty_reasoning(self):
        r = self._call("propose_collection_request", self._good_args(reasoning=""))
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "bad_arg")

    def test_list_submitted(self):
        self._call("propose_collection_request", self._good_args())
        self._call("propose_collection_request", self._good_args(target_cluster=2))
        r = self._call("list_submitted_requests", {})
        self.assertTrue(r.ok)
        payload = json.loads(r.content[0].text)
        self.assertEqual(payload["n_submitted"], 2)

    def test_revise_request(self):
        self._call("propose_collection_request", self._good_args())
        rid = self.ctx.submitted[0].request.request_id
        r = self._call(
            "revise_request",
            {"request_id": rid, "target_behavior": "place the cube gently",
             "reasoning": "softer placement avoids bouncing"},
        )
        self.assertTrue(r.ok)
        self.assertEqual(self.ctx.submitted[0].request.target_behavior, "place the cube gently")
        self.assertEqual(self.ctx.submitted[0].reasoning, "softer placement avoids bouncing")
        self.assertEqual(len(self.ctx.submitted[0].revision_history), 1)

    def test_revise_rolls_back_on_validation_failure(self):
        self._call("propose_collection_request", self._good_args())
        rid = self.ctx.submitted[0].request.request_id
        original = self.ctx.submitted[0].request.target_behavior
        r = self._call(
            "revise_request",
            {"request_id": rid, "target_behavior": "head to cluster 7", "reasoning": "fix"},
        )
        self.assertFalse(r.ok)
        # Rolled back
        self.assertEqual(self.ctx.submitted[0].request.target_behavior, original)

    def test_delete_request(self):
        self._call("propose_collection_request", self._good_args())
        rid = self.ctx.submitted[0].request.request_id
        r = self._call("delete_request", {"request_id": rid})
        self.assertTrue(r.ok)
        self.assertEqual(len(self.ctx.submitted), 0)

    def test_finalize_strategy(self):
        self._call("propose_collection_request", self._good_args())
        r = self._call("finalize_strategy", {"rationale": "focus on high-failure transitions"})
        self.assertTrue(r.ok)
        self.assertTrue(self.ctx.finalized)
        # Subsequent submissions are blocked.
        r2 = self._call("propose_collection_request", self._good_args())
        self.assertFalse(r2.ok)
        self.assertEqual(r2.metadata["error_code"], "finalized")

    def test_finalize_rejects_empty_rationale(self):
        r = self._call("finalize_strategy", {"rationale": "  "})
        self.assertFalse(r.ok)
        self.assertFalse(self.ctx.finalized)


class TestANGSubmission(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp, condition="A_NG")
        self.tools = build_tool_registry("A_NG", self.ctx)

    def test_propose_no_target_cluster(self):
        # A_NG schema omits target_cluster — request still validates with cluster=None.
        args = {
            "request_type": "recovery",
            "initial_conditions": {"reference_rollout_id": "r0003", "reference_frame": 5},
            "target_behavior": "stabilize the cube and re-grasp it",
            "prohibitions": [],
            "success_criterion": "task_success",
            "reasoning": "this rollout failed mid-grasp; recovering from that state is high value",
        }
        r = self.tools["propose_collection_request"].func(args)
        self.assertTrue(r.ok, r.content[0].text)
        self.assertIsNone(self.ctx.submitted[0].request.target_cluster)


if __name__ == "__main__":
    unittest.main()
