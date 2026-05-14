"""Unit tests for intervention rules (no diffusion_policy dependency)."""

import unittest

import numpy as np

from policy_doctor.monitoring.base import AssignmentResult, MonitorResult
from policy_doctor.monitoring.intervention import (
    InterventionDecision,
    InterventionRule,
    NodeValueThresholdRule,
)


def _result_with_assignment(node_id=0):
    return MonitorResult(
        embedding=np.zeros(50, dtype=np.float32),
        influence_scores=np.zeros(100, dtype=np.float32),
        assignment=AssignmentResult(
            cluster_id=node_id, node_id=node_id, distance=0.1, node_name=f"B{node_id}"
        ),
        timing_ms={"total_ms": 1.0},
    )


def _result_no_assignment():
    return MonitorResult(
        embedding=np.zeros(50, dtype=np.float32),
        influence_scores=np.zeros(100, dtype=np.float32),
        assignment=None,
        timing_ms={"total_ms": 1.0},
    )


class TestInterventionDecision(unittest.TestCase):

    def test_triggered_true(self):
        d = InterventionDecision(triggered=True)
        self.assertTrue(d.triggered)

    def test_triggered_false(self):
        d = InterventionDecision(triggered=False)
        self.assertFalse(d.triggered)

    def test_defaults(self):
        d = InterventionDecision(triggered=False)
        self.assertIsNone(d.node_id)
        self.assertIsNone(d.node_value)
        self.assertEqual(d.reason, "")

    def test_all_fields_stored(self):
        d = InterventionDecision(triggered=True, node_id=3, node_value=0.5, reason="low")
        self.assertEqual(d.node_id, 3)
        self.assertAlmostEqual(d.node_value, 0.5)
        self.assertEqual(d.reason, "low")


class TestNodeValueThresholdRule(unittest.TestCase):

    def _rule(self, values=None, threshold=0.0):
        if values is None:
            values = {0: 0.5, 1: -0.5, 2: 0.0}
        return NodeValueThresholdRule(node_values=values, threshold=threshold)

    def test_no_assignment_not_triggered(self):
        rule = self._rule()
        decision = rule.check(_result_no_assignment(), [])
        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "no_assignment")

    def test_no_assignment_node_id_is_none(self):
        rule = self._rule()
        decision = rule.check(_result_no_assignment(), [])
        self.assertIsNone(decision.node_id)
        self.assertIsNone(decision.node_value)

    def test_node_not_in_values_not_triggered(self):
        rule = NodeValueThresholdRule(node_values={0: 0.5}, threshold=0.0)
        decision = rule.check(_result_with_assignment(node_id=99), [])
        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "node_not_in_values")

    def test_node_not_in_values_includes_node_id(self):
        rule = NodeValueThresholdRule(node_values={0: 0.5}, threshold=0.0)
        decision = rule.check(_result_with_assignment(node_id=99), [])
        self.assertEqual(decision.node_id, 99)
        self.assertIsNone(decision.node_value)

    def test_value_above_threshold_not_triggered(self):
        rule = NodeValueThresholdRule(node_values={0: 0.8}, threshold=0.5)
        decision = rule.check(_result_with_assignment(node_id=0), [])
        self.assertFalse(decision.triggered)

    def test_value_below_threshold_triggered(self):
        rule = NodeValueThresholdRule(node_values={0: 0.2}, threshold=0.5)
        decision = rule.check(_result_with_assignment(node_id=0), [])
        self.assertTrue(decision.triggered)

    def test_value_equal_threshold_not_triggered(self):
        # value < threshold is the condition, so equal should not trigger
        rule = NodeValueThresholdRule(node_values={0: 0.5}, threshold=0.5)
        decision = rule.check(_result_with_assignment(node_id=0), [])
        self.assertFalse(decision.triggered)

    def test_negative_value_below_zero_threshold_triggered(self):
        rule = NodeValueThresholdRule(node_values={0: -0.1}, threshold=0.0)
        decision = rule.check(_result_with_assignment(node_id=0), [])
        self.assertTrue(decision.triggered)

    def test_decision_carries_node_id_and_value(self):
        rule = NodeValueThresholdRule(node_values={2: 0.7}, threshold=0.5)
        decision = rule.check(_result_with_assignment(node_id=2), [])
        self.assertEqual(decision.node_id, 2)
        self.assertAlmostEqual(decision.node_value, 0.7)

    def test_reason_contains_formatted_value_and_threshold(self):
        rule = NodeValueThresholdRule(node_values={0: 0.2}, threshold=0.5)
        decision = rule.check(_result_with_assignment(node_id=0), [])
        self.assertIn("0.2000", decision.reason)
        self.assertIn("0.5000", decision.reason)

    def test_history_ignored_does_not_raise(self):
        rule = self._rule()
        history = [_result_with_assignment(i % 3) for i in range(5)]
        decision = rule.check(_result_with_assignment(node_id=0), history)
        self.assertIsInstance(decision, InterventionDecision)

    def test_reset_is_no_op(self):
        rule = self._rule()
        rule.reset()  # should not raise


class TestInterventionRuleAbstract(unittest.TestCase):

    def test_concrete_subclass_must_implement_check(self):
        class NoCheck(InterventionRule):
            pass
        with self.assertRaises(TypeError):
            NoCheck()

    def test_base_reset_is_no_op(self):
        class MinimalRule(InterventionRule):
            def check(self, result, history):
                return InterventionDecision(triggered=False)
        rule = MinimalRule()
        rule.reset()  # inherited no-op; should not raise

    def test_concrete_check_is_callable(self):
        class AlwaysIntervene(InterventionRule):
            def check(self, result, history):
                return InterventionDecision(triggered=True, reason="always")
        decision = AlwaysIntervene().check(_result_with_assignment(), [])
        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "always")


if __name__ == "__main__":
    unittest.main()
