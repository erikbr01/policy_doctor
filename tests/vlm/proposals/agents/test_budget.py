"""Budget exhaustion + cache deduplication."""

from __future__ import annotations

import time
import unittest

from policy_doctor.vlm.proposals.agents.budget import (
    BudgetConfig,
    BudgetTracker,
    ResultCache,
)
from policy_doctor.vlm.proposals.agents.tools.types import ToolResult


class TestBudgetTracker(unittest.TestCase):
    def test_charges_increment_correctly(self):
        b = BudgetTracker(config=BudgetConfig(max_tool_calls=5, max_visual_calls=2, max_video_calls=1))
        self.assertIsNone(b.check("get_node", "cheap"))
        b.charge("cheap")
        self.assertEqual(b.state.n_tool_calls, 1)

        b.charge("visual")
        self.assertEqual(b.state.n_visual_calls, 1)
        self.assertEqual(b.state.n_tool_calls, 2)

        # Video charge increments visual too (videos strictly more expensive).
        b.charge("video")
        self.assertEqual(b.state.n_video_calls, 1)
        self.assertEqual(b.state.n_visual_calls, 2)

    def test_total_budget_exhaustion(self):
        b = BudgetTracker(config=BudgetConfig(max_tool_calls=2, max_visual_calls=10, max_video_calls=10))
        b.charge("cheap")
        b.charge("cheap")
        err = b.check("get_node", "cheap")
        self.assertIsNotNone(err)
        self.assertFalse(err.ok)
        self.assertEqual(err.metadata["error_code"], "budget_exhausted")
        self.assertEqual(err.metadata["budget_kind"], "cheap")

    def test_visual_subbudget(self):
        b = BudgetTracker(config=BudgetConfig(max_tool_calls=100, max_visual_calls=1, max_video_calls=10))
        b.charge("visual")
        err = b.check("get_slice_video", "visual")
        self.assertIsNotNone(err)
        self.assertEqual(err.metadata["budget_kind"], "visual")

    def test_video_subbudget(self):
        b = BudgetTracker(config=BudgetConfig(max_tool_calls=100, max_visual_calls=10, max_video_calls=1))
        b.charge("video")
        err = b.check("get_rollout_video", "video")
        self.assertIsNotNone(err)
        self.assertEqual(err.metadata["budget_kind"], "video")

    def test_terminal_calls_bypass_budget(self):
        b = BudgetTracker(config=BudgetConfig(max_tool_calls=0, max_visual_calls=0, max_video_calls=0))
        # Even with zero budget, finalize_strategy can be called.
        self.assertIsNone(b.check("finalize_strategy", "cheap", is_terminal=True))

    def test_session_timeout(self):
        b = BudgetTracker(config=BudgetConfig(max_session_duration_s=0.0))
        # Force enough wall-clock to elapse.
        time.sleep(0.001)
        err = b.check("get_node", "cheap")
        self.assertIsNotNone(err)
        self.assertEqual(err.metadata["error_code"], "session_timeout")

    def test_warning_emitted_near_exhaustion(self):
        b = BudgetTracker(config=BudgetConfig(max_visual_calls=3, warning_remaining_threshold=2))
        b.charge("visual")
        # 2 remain — at threshold, warning fires.
        warn = b.warning_for("visual")
        self.assertIsNotNone(warn)
        self.assertEqual(warn["remaining"], 2)
        self.assertEqual(warn["warning"], "approaching_visual_budget")

    def test_warning_silent_when_far_from_exhaustion(self):
        b = BudgetTracker(config=BudgetConfig(max_visual_calls=30, warning_remaining_threshold=5))
        self.assertIsNone(b.warning_for("visual"))


class TestResultCache(unittest.TestCase):
    def test_cache_round_trip(self):
        c = ResultCache()
        r = ToolResult.text("get_graph_summary", "ok")
        c.put("get_graph_summary", {}, r)
        hit = c.get("get_graph_summary", {})
        self.assertIsNotNone(hit)
        self.assertEqual(hit.content[0].text, "ok")

    def test_args_change_changes_key(self):
        c = ResultCache()
        c.put("get_node", {"node_id": 1}, ToolResult.text("get_node", "n1"))
        # Different args → cache miss.
        self.assertIsNone(c.get("get_node", {"node_id": 2}))
        # Same args (different dict order) → cache hit (sorted normalization).
        self.assertIsNotNone(c.get("get_node", {"node_id": 1}))

    def test_failed_results_not_cached(self):
        c = ResultCache()
        c.put("get_node", {"node_id": 99}, ToolResult.error("get_node", "missing"))
        self.assertIsNone(c.get("get_node", {"node_id": 99}))

    def test_disabled_cache_returns_nothing(self):
        c = ResultCache(enabled=False)
        c.put("get_node", {"node_id": 1}, ToolResult.text("get_node", "n1"))
        self.assertIsNone(c.get("get_node", {"node_id": 1}))


if __name__ == "__main__":
    unittest.main()
