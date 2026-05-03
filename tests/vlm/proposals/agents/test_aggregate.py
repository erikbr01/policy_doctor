"""Cross-session aggregation tests."""

from __future__ import annotations

import unittest
from typing import Any, Dict

from policy_doctor.vlm.proposals.agents.aggregate import aggregate_agent_sessions
from policy_doctor.vlm.proposals.agents.session import SessionResult


def _req(rid_suffix: str, *, ref_rollout: str, target_behavior: str,
         request_type: str = "full_trajectory", target_cluster: int | None = None) -> Dict[str, Any]:
    return {
        "request": {
            "request_id": rid_suffix,
            "request_type": request_type,
            "initial_conditions": {
                "reference_rollout_id": ref_rollout,
                "reference_frame": 0,
                "object_poses": {},
                "gripper_state": {},
                "robot_pose": {},
                "tolerances": {},
            },
            "target_behavior": target_behavior,
            "prohibitions": [],
            "success_criterion": "task_success",
            "target_cluster": target_cluster,
            "source_condition": "A_G",
        },
        "reasoning": "scripted reasoning text",
        "revision_history": [],
    }


def _session(seed: int, submissions) -> SessionResult:
    r = SessionResult(condition="A_G", seed=seed)
    r.submitted_requests = list(submissions)
    return r


class TestAggregateAgentSessions(unittest.TestCase):
    def test_best_consistency_run_picks_overlapping_seed(self):
        # Three sessions: seeds 0 and 1 propose overlapping demos; seed 2 is divergent.
        s0 = _session(0, [
            _req("a", ref_rollout="r0001", target_behavior="approach the cube and grasp it"),
            _req("b", ref_rollout="r0002", target_behavior="recover from a slipping grasp"),
        ])
        s1 = _session(1, [
            _req("c", ref_rollout="r0001", target_behavior="approach the cube and grasp it firmly"),
            _req("d", ref_rollout="r0002", target_behavior="recover from a slipping grasp"),
        ])
        s2 = _session(2, [
            _req("e", ref_rollout="r0005", target_behavior="something completely different"),
        ])
        result = aggregate_agent_sessions([s0, s1, s2])

        # Either seed 0 or 1 wins (both have 2 matches).
        self.assertIn(result.selected_seed, {0, 1})
        self.assertEqual(len(result.selected_requests), 2)
        # Metrics include consistency_rate and per-rep counts.
        self.assertIn("consistency_rate", result.consistency_metrics)
        self.assertIn("per_rep_match_counts", result.consistency_metrics)

    def test_union_method_dedupes(self):
        s0 = _session(0, [
            _req("a", ref_rollout="r0001", target_behavior="approach and grasp the cube"),
        ])
        s1 = _session(1, [
            _req("b", ref_rollout="r0001", target_behavior="approach and grasp the cube"),
            _req("c", ref_rollout="r0002", target_behavior="recover from slipping"),
        ])
        result = aggregate_agent_sessions([s0, s1], method="union")
        self.assertEqual(result.selected_seed, -1)
        self.assertEqual(len(result.selected_requests), 2)
        self.assertEqual(len(result.union_requests), 2)

    def test_handles_empty_sessions(self):
        result = aggregate_agent_sessions([_session(0, [])])
        self.assertEqual(len(result.selected_requests), 0)


if __name__ == "__main__":
    unittest.main()
