"""Unit tests for MonitoredPolicy (no diffusion_policy dependency)."""

import unittest
from unittest.mock import MagicMock, call

import numpy as np
import torch

from policy_doctor.monitoring.base import AssignmentResult, MonitorResult
from policy_doctor.monitoring.intervention import InterventionDecision, InterventionRule
from policy_doctor.monitoring.monitored_policy import MonitoredPolicy
from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier


def _dummy_result(cluster_id=0):
    return MonitorResult(
        embedding=np.zeros(50, dtype=np.float32),
        influence_scores=np.zeros(100, dtype=np.float32),
        assignment=AssignmentResult(cluster_id=cluster_id, node_id=cluster_id,
                                    distance=0.1, node_name=f"Behavior {cluster_id}"),
        timing_ms={"total_ms": 1.5, "gradient_project_ms": 1.0,
                   "score_ms": 0.3, "assign_ms": 0.2},
    )


def _make_mock_policy(B=1, n_action_steps=8, action_dim=14, use_action_pred=False):
    policy = MagicMock()
    action = torch.randn(B, n_action_steps, action_dim)
    if use_action_pred:
        policy.predict_action.return_value = {"action": torch.zeros(B, n_action_steps, action_dim),
                                               "action_pred": action}
    else:
        policy.predict_action.return_value = {"action": action}
    return policy


def _dummy_embed_only_result(cluster_id=0):
    return MonitorResult(
        embedding=np.zeros(50, dtype=np.float32),
        influence_scores=None,
        assignment=AssignmentResult(cluster_id=cluster_id, node_id=cluster_id,
                                    distance=0.1, node_name=f"Behavior {cluster_id}"),
        timing_ms={"total_ms": 1.0, "gradient_project_ms": 0.8, "assign_ms": 0.2},
    )


def _make_mock_classifier():
    classifier = MagicMock(spec=TrajectoryClassifier)
    classifier.classify_sample.return_value = _dummy_result()
    classifier.classify_sample_embed_only.return_value = _dummy_embed_only_result()
    classifier.score_embedding.return_value = np.zeros(100, dtype=np.float32)
    return classifier


def _obs_dict(B=1, n_obs_steps=2, obs_dim=20):
    return {"obs": torch.randn(B, n_obs_steps, obs_dim)}


class TestMonitoredPolicyPredictAction(unittest.TestCase):

    def test_returns_original_action_dict(self):
        policy = _make_mock_policy()
        mp = MonitoredPolicy(policy, _make_mock_classifier())
        result = mp.predict_action(_obs_dict())
        self.assertIn("action", result)
        policy.predict_action.assert_called_once()

    def test_accumulates_one_result_per_timestep(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        obs = _obs_dict()
        mp.predict_action(obs)
        mp.predict_action(obs)
        self.assertEqual(len(mp.episode_results), 2)

    def test_result_has_correct_fields(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        entry = mp.episode_results[0]
        for key in ("episode", "timestep", "env_idx", "cluster_id", "node_id",
                    "node_name", "distance", "total_ms", "result"):
            self.assertIn(key, entry)

    def test_timestep_increments(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        obs = _obs_dict()
        mp.predict_action(obs)
        mp.predict_action(obs)
        self.assertEqual(mp.episode_results[0]["timestep"], 0)
        self.assertEqual(mp.episode_results[1]["timestep"], 1)

    def test_uses_action_pred_key_when_present(self):
        policy = _make_mock_policy(use_action_pred=True)
        classifier = _make_mock_classifier()
        mp = MonitoredPolicy(policy, classifier)
        mp.predict_action(_obs_dict())
        action_arg = classifier.classify_sample_embed_only.call_args[0][1]
        # action_pred is all randn, action is all zeros — verify non-zero was used
        self.assertFalse(np.allclose(action_arg, 0.0))

    def test_uses_action_key_when_action_pred_absent(self):
        policy = _make_mock_policy(use_action_pred=False)
        classifier = _make_mock_classifier()
        mp = MonitoredPolicy(policy, classifier)
        mp.predict_action(_obs_dict())
        classifier.classify_sample_embed_only.assert_called_once()

    def test_batch_size_gt1_classifies_each_env(self):
        B = 3
        policy = _make_mock_policy(B=B)
        classifier = _make_mock_classifier()
        mp = MonitoredPolicy(policy, classifier)
        mp.predict_action(_obs_dict(B=B))
        self.assertEqual(classifier.classify_sample_embed_only.call_count, B)
        self.assertEqual(len(mp.episode_results), B)

    def test_batch_env_idx_stored_correctly(self):
        B = 3
        policy = _make_mock_policy(B=B)
        mp = MonitoredPolicy(policy, _make_mock_classifier())
        mp.predict_action(_obs_dict(B=B))
        env_ids = [e["env_idx"] for e in mp.episode_results]
        self.assertEqual(env_ids, list(range(B)))

    def test_numpy_obs_also_accepted(self):
        policy = MagicMock()
        policy.predict_action.return_value = {
            "action": np.random.randn(1, 8, 14).astype(np.float32)
        }
        mp = MonitoredPolicy(policy, _make_mock_classifier())
        obs_dict = {"obs": np.random.randn(1, 2, 20).astype(np.float32)}
        mp.predict_action(obs_dict)
        self.assertEqual(len(mp.episode_results), 1)


class TestMonitoredPolicyReset(unittest.TestCase):

    def test_first_reset_does_not_increment_episode(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.reset()
        self.assertEqual(mp._episode_idx, 0)

    def test_reset_after_steps_increments_episode(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        mp.reset()
        self.assertEqual(mp._episode_idx, 1)

    def test_reset_resets_timestep(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        mp.predict_action(_obs_dict())
        mp.reset()
        self.assertEqual(mp._timestep, 0)

    def test_episode_idx_increments_across_multiple_resets(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        for _ in range(3):
            mp.predict_action(_obs_dict())
            mp.reset()
        self.assertEqual(mp._episode_idx, 3)

    def test_reset_calls_wrapped_policy_reset(self):
        policy = _make_mock_policy()
        mp = MonitoredPolicy(policy, _make_mock_classifier())
        mp.reset()
        policy.reset.assert_called_once()

    def test_episode_idx_stored_in_results(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        mp.reset()
        mp.predict_action(_obs_dict())
        episodes = [e["episode"] for e in mp.episode_results]
        self.assertEqual(episodes, [0, 1])


class TestMonitoredPolicyDelegation(unittest.TestCase):

    def test_getattr_delegates_to_wrapped_policy(self):
        policy = _make_mock_policy()
        policy.some_custom_attr = "hello"
        mp = MonitoredPolicy(policy, _make_mock_classifier())
        self.assertEqual(mp.some_custom_attr, "hello")

    def test_getattr_delegates_method_calls(self):
        policy = _make_mock_policy()
        policy.set_normalizer = MagicMock(return_value=None)
        mp = MonitoredPolicy(policy, _make_mock_classifier())
        mp.set_normalizer("something")
        policy.set_normalizer.assert_called_once_with("something")


def _make_mock_classifier_with_scores(scores_array: np.ndarray):
    classifier = MagicMock(spec=TrajectoryClassifier)
    classifier.classify_sample.return_value = _dummy_result()
    classifier.classify_sample_embed_only.return_value = _dummy_embed_only_result()
    classifier.score_embedding.return_value = scores_array
    return classifier


class TestMonitoredPolicyInterventionRule(unittest.TestCase):

    def test_intervention_key_none_when_no_rule(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        self.assertIsNone(mp.episode_results[0]["intervention"])

    def test_intervention_key_is_decision_when_rule_set(self):
        rule = MagicMock(spec=InterventionRule)
        rule.check.return_value = InterventionDecision(triggered=False, reason="ok")
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.predict_action(_obs_dict())
        decision = mp.episode_results[0]["intervention"]
        self.assertIsInstance(decision, InterventionDecision)

    def test_rule_check_called_once_per_timestep(self):
        rule = MagicMock(spec=InterventionRule)
        rule.check.return_value = InterventionDecision(triggered=False)
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.predict_action(_obs_dict())
        mp.predict_action(_obs_dict())
        self.assertEqual(rule.check.call_count, 2)

    def test_rule_check_receives_monitor_result(self):
        rule = MagicMock(spec=InterventionRule)
        rule.check.return_value = InterventionDecision(triggered=False)
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.predict_action(_obs_dict())
        result_arg = rule.check.call_args[0][0]
        self.assertIsInstance(result_arg, MonitorResult)
        self.assertIsNone(result_arg.influence_scores)  # embed-only result

    def test_rule_check_receives_growing_history(self):
        rule = MagicMock(spec=InterventionRule)
        rule.check.return_value = InterventionDecision(triggered=False)
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.predict_action(_obs_dict())
        mp.predict_action(_obs_dict())
        # second call: history contains both results (current was appended before check)
        history_arg = rule.check.call_args[0][1]
        self.assertIsInstance(history_arg, list)
        self.assertEqual(len(history_arg), 2)

    def test_rule_reset_called_on_policy_reset(self):
        rule = MagicMock(spec=InterventionRule)
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.predict_action(_obs_dict())
        mp.reset()
        rule.reset.assert_called_once()

    def test_rule_reset_called_even_without_prior_steps(self):
        rule = MagicMock(spec=InterventionRule)
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.reset()
        rule.reset.assert_called_once()

    def test_triggered_decision_stored_in_results(self):
        rule = MagicMock(spec=InterventionRule)
        rule.check.return_value = InterventionDecision(triggered=True, reason="low_value")
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), intervention_rule=rule)
        mp.predict_action(_obs_dict())
        self.assertTrue(mp.episode_results[0]["intervention"].triggered)


class TestMonitoredPolicyBuffers(unittest.TestCase):

    def test_embedding_buffer_maxlen_equals_window(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), max_influence_window=3)
        self.assertEqual(mp._embedding_buffer.maxlen, 3)

    def test_monitor_history_maxlen_equals_window(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), max_influence_window=3)
        self.assertEqual(mp._monitor_history.maxlen, 3)

    def test_buffers_cleared_on_reset(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        mp.reset()
        self.assertEqual(len(mp._embedding_buffer), 0)
        self.assertEqual(len(mp._monitor_history), 0)

    def test_buffer_does_not_exceed_max_influence_window(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), max_influence_window=3)
        for _ in range(10):
            mp.predict_action(_obs_dict())
        self.assertLessEqual(len(mp._embedding_buffer), 3)
        self.assertLessEqual(len(mp._monitor_history), 3)

    def test_embedding_buffered_every_step(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier(), max_influence_window=5)
        mp.predict_action(_obs_dict())
        mp.predict_action(_obs_dict())
        self.assertEqual(len(mp._embedding_buffer), 2)
        self.assertEqual(len(mp._monitor_history), 2)


class TestMonitoredPolicyGetSliceInfluence(unittest.TestCase):

    def test_empty_buffer_returns_empty_arrays(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        indices, scores = mp.get_slice_influence()
        self.assertEqual(len(indices), 0)
        self.assertEqual(len(scores), 0)

    def test_returns_top_k_results(self):
        # score_embedding returns 100 scores by default in _make_mock_classifier
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        indices, scores = mp.get_slice_influence(top_k=20)
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(scores), 20)

    def test_top_k_capped_at_n_demo(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        indices, scores = mp.get_slice_influence(top_k=500)  # N_demo=100
        self.assertEqual(len(indices), 100)
        self.assertEqual(len(scores), 100)

    def test_scores_descending_by_default(self):
        scores_array = np.arange(100, dtype=np.float32)  # distinct values
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier_with_scores(scores_array))
        mp.predict_action(_obs_dict())
        _, scores = mp.get_slice_influence(top_k=10)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_ascending_returns_lowest_scores_first(self):
        scores_array = np.arange(100, dtype=np.float32)
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier_with_scores(scores_array))
        mp.predict_action(_obs_dict())
        _, scores_desc = mp.get_slice_influence(top_k=5, ascending=False)
        _, scores_asc = mp.get_slice_influence(top_k=5, ascending=True)
        self.assertGreater(scores_desc[0], scores_asc[0])

    def test_indices_and_scores_same_length(self):
        mp = MonitoredPolicy(_make_mock_policy(), _make_mock_classifier())
        mp.predict_action(_obs_dict())
        indices, scores = mp.get_slice_influence(top_k=15)
        self.assertEqual(len(indices), len(scores))

    def test_multi_step_buffer_aggregated(self):
        # Two timesteps: buffer contains 2 rows; result shape should still be (top_k,)
        scores_array = np.arange(100, dtype=np.float32)
        mp = MonitoredPolicy(
            _make_mock_policy(), _make_mock_classifier_with_scores(scores_array), max_influence_window=5
        )
        mp.predict_action(_obs_dict())
        mp.predict_action(_obs_dict())
        indices, scores = mp.get_slice_influence(top_k=10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(scores), 10)


if __name__ == "__main__":
    unittest.main()
