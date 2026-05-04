"""Tests for Layer 4 (strategy submission) tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from tests.vlm.proposals.agents.conftest import (
    FIXTURE_SLICES_BY_CLUSTER,
    build_fixture_context,
    evidence_for_cluster,
)


class TestSubmissionTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Pre-inspect every fixture cluster so submission tests focus on
        # submission semantics (denylist, dedup, evidence) not on the
        # cluster-inspection or visual-evidence gates which have their own
        # focused tests.
        self.ctx = build_fixture_context(self.tmp, pre_inspect_clusters=[0, 1, 2, 4])
        self.tools = build_tool_registry("A_G", self.ctx)

    def _call(self, name, args):
        return self.tools[name].func(args)

    def _good_args(self, **overrides):
        target = overrides.get("target_cluster", 1)
        base = {
            "request_type": "full_trajectory",
            "initial_conditions": {"reference_rollout_id": "r0000", "reference_frame": 0},
            "target_behavior": "pick up the cube and place it on the platform",
            "prohibitions": ["do not push the cube off the table"],
            "success_criterion": "task_success",
            "target_cluster": 1,
            "evidence_slice_ids": evidence_for_cluster(target, n=3),
            "reasoning": "this cluster has high failure likelihood; demonstrating a clean traversal "
            "should improve the policy's pre-grasp positioning.",
        }
        base.update(overrides)
        # If the override changed target_cluster, refresh evidence to match
        # unless the test explicitly overrode evidence_slice_ids too.
        if "target_cluster" in overrides and "evidence_slice_ids" not in overrides:
            base["evidence_slice_ids"] = evidence_for_cluster(overrides["target_cluster"], n=3)
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
        # Second request needs different target_behavior (dedup gate).
        self._call(
            "propose_collection_request",
            self._good_args(
                target_cluster=2,
                target_behavior="approach the cube from the side and pinch grasp it",
            ),
        )
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

    def test_revise_rejects_duplicate_target_behavior(self):
        # Two submissions with distinct prose.
        self._call("propose_collection_request", self._good_args())
        first_rid = self.ctx.submitted[0].request.request_id
        first_text = self.ctx.submitted[0].request.target_behavior
        self._call(
            "propose_collection_request",
            self._good_args(
                target_cluster=2,
                target_behavior="approach the cube from the side and pinch grasp it",
            ),
        )
        second_rid = self.ctx.submitted[1].request.request_id

        # Try to revise the second one to match the first — must fail.
        r = self._call(
            "revise_request",
            {"request_id": second_rid, "target_behavior": first_text,
             "reasoning": "trying to dodge the dedup gate"},
        )
        self.assertFalse(r.ok, "revise should reject duplicate target_behavior")
        self.assertEqual(r.metadata["error_code"], "duplicate_target_behavior")
        # Rolled back — second request keeps its original prose.
        self.assertNotEqual(
            self.ctx.submitted[1].request.target_behavior, first_text,
            "rollback should restore the original target_behavior",
        )
        # First request is also untouched.
        self.assertEqual(self.ctx.submitted[0].request.target_behavior, first_text)

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


class TestSubmissionGates(unittest.TestCase):
    """Agentic-experiment validation gates layered on top of the schema check."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp)
        self.tools = build_tool_registry("A_G", self.ctx)

    def _good_args(self, **overrides):
        target = overrides.get("target_cluster", 1)
        base = {
            "request_type": "full_trajectory",
            "initial_conditions": {"reference_rollout_id": "r0000", "reference_frame": 0},
            "target_behavior": "approach the cube from above and grasp it firmly",
            "prohibitions": [],
            "success_criterion": "task_success",
            "target_cluster": 1,
            "evidence_slice_ids": evidence_for_cluster(target, n=3),
            "reasoning": "demonstrating a clean traversal through the high-failure region",
        }
        base.update(overrides)
        if "target_cluster" in overrides and "evidence_slice_ids" not in overrides:
            base["evidence_slice_ids"] = evidence_for_cluster(overrides["target_cluster"], n=3)
        return base

    def _setup_with_clusters(self, *cluster_ids):
        """Re-create ctx with pre-inspected clusters; used by tests below."""
        self.ctx = build_fixture_context(self.tmp, pre_inspect_clusters=list(cluster_ids))
        self.tools = build_tool_registry("A_G", self.ctx)

    def test_a_g_rejects_missing_target_cluster(self):
        self._setup_with_clusters(1)
        # Drop target_cluster from A_G submission — gate must reject.
        args = self._good_args()
        del args["target_cluster"]
        r = self.tools["propose_collection_request"].func(args)
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "missing_target_cluster")

    def test_recovery_with_frame_zero_rejected(self):
        self._setup_with_clusters(1)
        r = self.tools["propose_collection_request"].func(
            self._good_args(request_type="recovery", initial_conditions={
                "reference_rollout_id": "r0000", "reference_frame": 0,
            })
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "recovery_frame_zero")

    def test_recovery_with_positive_frame_accepted(self):
        self._setup_with_clusters(1)
        r = self.tools["propose_collection_request"].func(
            self._good_args(request_type="recovery", initial_conditions={
                "reference_rollout_id": "r0000", "reference_frame": 8,
            })
        )
        self.assertTrue(r.ok, r.content[0].text)

    def test_duplicate_target_behavior_rejected(self):
        self._setup_with_clusters(1, 2)
        first = self.tools["propose_collection_request"].func(self._good_args(target_cluster=1))
        self.assertTrue(first.ok, first.content[0].text)
        # Same target_behavior, different target_cluster → still duplicate.
        second = self.tools["propose_collection_request"].func(self._good_args(target_cluster=2))
        self.assertFalse(second.ok)
        self.assertEqual(second.metadata["error_code"], "duplicate_target_behavior")

    def test_normalized_dedup_catches_whitespace_and_case_variants(self):
        self._setup_with_clusters(1, 2)
        self.tools["propose_collection_request"].func(self._good_args(target_cluster=1))
        r = self.tools["propose_collection_request"].func(self._good_args(
            target_cluster=2,
            target_behavior="  APPROACH the   CUBE from above and GRASP it firmly\n",
        ))
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "duplicate_target_behavior")

    def test_uninspected_cluster_rejected(self):
        # No pre-inspection: cluster_not_inspected fires before evidence checks.
        r = self.tools["propose_collection_request"].func(self._good_args(target_cluster=1))
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "cluster_not_inspected")

    def test_insufficient_evidence_rejected(self):
        # Pre-inspect cluster but pass fewer than 3 evidence slice_ids.
        self._setup_with_clusters(1)
        r = self.tools["propose_collection_request"].func(
            self._good_args(target_cluster=1, evidence_slice_ids=evidence_for_cluster(1, n=2))
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "insufficient_evidence")

    def test_evidence_not_inspected_rejected(self):
        # Pre-inspect node + valid evidence_slice_ids in cluster — but the
        # specific slice_ids referenced were never visually fetched.
        self.ctx.inspected_nodes.add(1)
        # Don't populate inspected_slices.
        r = self.tools["propose_collection_request"].func(
            self._good_args(target_cluster=1, evidence_slice_ids=evidence_for_cluster(1, n=3))
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "evidence_not_inspected")

    def test_evidence_wrong_cluster_rejected(self):
        # Pre-inspect cluster 1 but submit slice_ids that belong to cluster 2.
        self._setup_with_clusters(1, 2)
        r = self.tools["propose_collection_request"].func(
            self._good_args(target_cluster=1, evidence_slice_ids=evidence_for_cluster(2, n=3))
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "evidence_wrong_cluster")

    def test_full_evidence_flow_succeeds(self):
        # The happy-path: inspect cluster + provide 3 valid evidence slices.
        self._setup_with_clusters(1)
        r = self.tools["propose_collection_request"].func(
            self._good_args(target_cluster=1, evidence_slice_ids=evidence_for_cluster(1, n=3))
        )
        self.assertTrue(r.ok, r.content[0].text)
        self.assertEqual(self.ctx.submitted[0].request.target_cluster, 1)

class TestANGSubmission(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Pre-inspect rollouts so the evidence_rollout_ids gate doesn't
        # short-circuit; the test asserts schema-level (no target_cluster).
        self.ctx = build_fixture_context(
            self.tmp, condition="A_NG",
            pre_inspect_rollouts=["r0003", "r0004", "r0005"],
        )
        self.tools = build_tool_registry("A_NG", self.ctx)

    def test_propose_no_target_cluster(self):
        # A_NG schema omits target_cluster — request validates with cluster=None
        # and uses evidence_rollout_ids instead of evidence_slice_ids.
        args = {
            "request_type": "recovery",
            "initial_conditions": {"reference_rollout_id": "r0003", "reference_frame": 5},
            "target_behavior": "stabilize the cube and re-grasp it",
            "prohibitions": [],
            "success_criterion": "task_success",
            "evidence_rollout_ids": ["r0003", "r0004", "r0005"],
            "reasoning": "all three failure rollouts I viewed showed the same slipping pattern; "
            "demonstrating a stable recovery should give the policy a counter-example",
        }
        r = self.tools["propose_collection_request"].func(args)
        self.assertTrue(r.ok, r.content[0].text)
        self.assertIsNone(self.ctx.submitted[0].request.target_cluster)

    def test_a_ng_insufficient_evidence_rejected(self):
        args = {
            "request_type": "recovery",
            "initial_conditions": {"reference_rollout_id": "r0003", "reference_frame": 5},
            "target_behavior": "stabilize the cube and re-grasp it",
            "evidence_rollout_ids": ["r0003"],  # only 1 — needs 3
            "reasoning": "stable recovery is high value",
        }
        r = self.tools["propose_collection_request"].func(args)
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "insufficient_evidence")

    def test_a_ng_evidence_not_inspected_rejected(self):
        args = {
            "request_type": "recovery",
            "initial_conditions": {"reference_rollout_id": "r0003", "reference_frame": 5},
            "target_behavior": "stabilize the cube and re-grasp it",
            "evidence_rollout_ids": ["r0000", "r0001", "r0002"],  # not in inspected_rollouts
            "reasoning": "stable recovery is high value",
        }
        r = self.tools["propose_collection_request"].func(args)
        self.assertFalse(r.ok)
        self.assertEqual(r.metadata["error_code"], "evidence_not_inspected")


if __name__ == "__main__":
    unittest.main()
