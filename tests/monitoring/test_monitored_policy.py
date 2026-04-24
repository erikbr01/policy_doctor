"""Unit tests for MonitoredPolicy (no diffusion_policy dependency)."""

import unittest
from unittest.mock import MagicMock, call

import numpy as np
import torch

from policy_doctor.monitoring.base import AssignmentResult, MonitorResult
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


def _make_mock_classifier():
    classifier = MagicMock(spec=TrajectoryClassifier)
    classifier.classify_sample.return_value = _dummy_result()
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
        action_arg = classifier.classify_sample.call_args[0][1]
        # action_pred is all randn, action is all zeros — verify non-zero was used
        self.assertFalse(np.allclose(action_arg, 0.0))

    def test_uses_action_key_when_action_pred_absent(self):
        policy = _make_mock_policy(use_action_pred=False)
        classifier = _make_mock_classifier()
        mp = MonitoredPolicy(policy, classifier)
        mp.predict_action(_obs_dict())
        classifier.classify_sample.assert_called_once()

    def test_batch_size_gt1_classifies_each_env(self):
        B = 3
        policy = _make_mock_policy(B=B)
        classifier = _make_mock_classifier()
        mp = MonitoredPolicy(policy, classifier)
        mp.predict_action(_obs_dict(B=B))
        self.assertEqual(classifier.classify_sample.call_count, B)
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


if __name__ == "__main__":
    unittest.main()
