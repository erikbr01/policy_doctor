"""End-to-end test of AgentSession with the mock backend (Tier 0 milestone).

The mock backend walks a fixed exploration script; this test verifies the
loop wires up correctly: tools dispatch, budget charges, results land in the
trace, and a final strategy + rationale are produced. No API key needed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_doctor.vlm.backends.mock import MockVLMBackend
from policy_doctor.vlm.proposals.agents.session import AgentSession
from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from policy_doctor.vlm.proposals.agents.trace import SessionTrace
from tests.vlm.proposals.agents.conftest import build_fixture_context


SYSTEM_PROMPT = "You are an exploration agent. Use the tools to propose data collection requests."
USER_MESSAGE = (
    "Explore the available rollouts and submit demonstration requests. "
    "Pool ids: r0000 r0001 r0002 r0003 r0004 r0005."
)


class TestAgentSessionMockAG(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Pre-inspect the clusters the mock script will target (0, 1, 2):
        # the fixture pool has no real frames so get_slice_video would error.
        # Pre-population mimics the post-inspection state for end-to-end tests.
        self.ctx = build_fixture_context(
            self.tmp, condition="A_G", pre_inspect_clusters=[0, 1, 2],
        )
        self.tools = build_tool_registry("A_G", self.ctx)
        self.out_dir = self.tmp / "session_out"

    def test_end_to_end_submits_and_finalizes(self):
        trace_path = self.tmp / "trace.jsonl"
        with SessionTrace(out_path=trace_path) as trace:
            session = AgentSession(
                backend=MockVLMBackend(),
                ctx=self.ctx,
                tools=self.tools,
                system_prompt=SYSTEM_PROMPT,
                user_message=USER_MESSAGE,
                seed=0,
                max_turns=20,
                trace=trace,
                out_dir=self.out_dir,
            )
            result = session.run()

        # Mock script ends with finalize_strategy.
        self.assertEqual(result.stop_reason, "finalize")
        self.assertTrue(self.ctx.finalized)

        # Three propose_collection_request calls + finalize → 3 submitted requests.
        self.assertEqual(len(result.submitted_requests), 3)
        for req in result.submitted_requests:
            self.assertIn("request", req)
            self.assertIn("reasoning", req)

        # Budget was charged for at least the budgeted tools (get_graph_summary,
        # list_nodes, list_paths). Submission tools and read-only ID lookups
        # bypass the budget so they don't increment the counter.
        self.assertGreaterEqual(result.budget_summary["n_tool_calls"], 1)

        # Trace contains assistant_turn + tool_call + tool_result events.
        kinds = [ev.kind for ev in trace.events]
        self.assertIn("session_start", kinds)
        self.assertIn("session_end", kinds)
        self.assertIn("assistant_turn", kinds)
        self.assertIn("tool_result", kinds)

        # Persistence artefacts written.
        self.assertTrue((self.out_dir / "conversation.json").exists())
        self.assertTrue((self.out_dir / "submitted_requests.json").exists())
        self.assertTrue((self.out_dir / "rationale.txt").exists())
        self.assertTrue((self.out_dir / "budget_summary.json").exists())
        self.assertTrue((self.out_dir / "session_summary.json").exists())
        self.assertTrue(trace_path.exists())

        # JSONL trace is parseable line-by-line.
        with open(trace_path) as f:
            for line in f:
                json.loads(line)


class TestAgentSessionMockANG(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Mock A_NG script cites rollout_ids r0000–r0002 as evidence; pre-mark
        # them inspected (fixture pkls have no real frames).
        self.ctx = build_fixture_context(
            self.tmp, condition="A_NG",
            pre_inspect_rollouts=["r0000", "r0001", "r0002", "r0003", "r0004", "r0005"],
        )
        self.tools = build_tool_registry("A_NG", self.ctx)

    def test_ang_path_runs_to_finalize(self):
        session = AgentSession(
            backend=MockVLMBackend(),
            ctx=self.ctx,
            tools=self.tools,
            system_prompt="A_NG agent.",
            user_message="Pool ids: r0000 r0001 r0002 r0003 r0004 r0005.",
            seed=0,
            max_turns=20,
        )
        result = session.run()
        self.assertEqual(result.stop_reason, "finalize")
        self.assertEqual(len(result.submitted_requests), 3)
        for req in result.submitted_requests:
            # A_NG must NOT include target_cluster.
            self.assertIsNone(req["request"]["target_cluster"])


if __name__ == "__main__":
    unittest.main()
