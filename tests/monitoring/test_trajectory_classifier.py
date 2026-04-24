"""Unit tests for TrajectoryClassifier (no diffusion_policy dependency)."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from policy_doctor.monitoring.base import AssignmentResult, MonitorResult
from policy_doctor.monitoring.stream_monitor import StreamMonitor
from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier


def _make_mock_monitor(proj_dim=50, n_train=100, cluster_id=0):
    monitor = MagicMock(spec=StreamMonitor)
    monitor.process_sample.return_value = MonitorResult(
        embedding=np.zeros(proj_dim, dtype=np.float32),
        influence_scores=np.zeros(n_train, dtype=np.float32),
        assignment=AssignmentResult(cluster_id=cluster_id, node_id=cluster_id,
                                    distance=0.1, node_name=f"Behavior {cluster_id}"),
        timing_ms={"total_ms": 1.0, "gradient_project_ms": 0.5,
                   "score_ms": 0.3, "assign_ms": 0.2},
    )
    return monitor


def _make_classifier(n_obs_steps=2, n_action_steps=8, mode="rollout"):
    monitor = _make_mock_monitor()
    return TrajectoryClassifier(
        monitor=monitor, mode=mode,
        n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
    )


class TestClassifySequence(unittest.TestCase):

    def setUp(self):
        self.To, self.Ta = 2, 8
        self.Do, self.Da = 20, 7
        self.T = 15
        self.classifier = _make_classifier(n_obs_steps=self.To, n_action_steps=self.Ta)
        rng = np.random.RandomState(0)
        self.obs_seq = rng.randn(self.T, self.Do).astype(np.float32)
        self.action_seq = rng.randn(self.T, self.Da).astype(np.float32)

    def test_result_length(self):
        results = self.classifier.classify_sequence(self.obs_seq, self.action_seq)
        # starts at t = To - 1
        self.assertEqual(len(results), self.T - (self.To - 1))

    def test_timestep_indices(self):
        results = self.classifier.classify_sequence(self.obs_seq, self.action_seq)
        expected = list(range(self.To - 1, self.T))
        self.assertEqual([t for t, _ in results], expected)

    def test_obs_window_shape(self):
        self.classifier.classify_sequence(self.obs_seq, self.action_seq)
        for call in self.classifier.monitor.process_sample.call_args_list:
            obs_arg = call[0][0]
            self.assertEqual(obs_arg.shape, (self.To, self.Do))

    def test_action_window_shape(self):
        self.classifier.classify_sequence(self.obs_seq, self.action_seq)
        for call in self.classifier.monitor.process_sample.call_args_list:
            action_arg = call[0][1]
            self.assertEqual(action_arg.shape, (self.Ta, self.Da))

    def test_end_of_episode_padding(self):
        # At the last timestep, action window is short and must be repeat-padded.
        T = self.To  # just enough for one window
        obs_seq = np.random.randn(T, self.Do).astype(np.float32)
        action_seq = np.random.randn(T, self.Da).astype(np.float32)
        results = self.classifier.classify_sequence(obs_seq, action_seq)
        self.assertEqual(len(results), 1)
        last_action_arg = self.classifier.monitor.process_sample.call_args[0][1]
        self.assertEqual(last_action_arg.shape[0], self.Ta)
        # All rows after the last real action should be the same (last action repeated)
        np.testing.assert_array_equal(last_action_arg[1:], last_action_arg[0:1].repeat(self.Ta - 1, axis=0))

    def test_obs_window_uses_correct_history(self):
        # The obs at timestep t should be obs_seq[t-To+1 : t+1]
        T = 5
        obs_seq = np.arange(T * self.Do, dtype=np.float32).reshape(T, self.Do)
        action_seq = np.zeros((T, self.Da), dtype=np.float32)
        self.classifier.classify_sequence(obs_seq, action_seq)
        first_call = self.classifier.monitor.process_sample.call_args_list[0]
        obs_arg = first_call[0][0]
        # t=1 (To-1=1): window is obs_seq[0:2]
        np.testing.assert_array_equal(obs_arg, obs_seq[0:self.To])

    def test_returns_monitor_result_instances(self):
        results = self.classifier.classify_sequence(self.obs_seq, self.action_seq)
        for _, r in results:
            self.assertIsInstance(r, MonitorResult)

    def test_single_timestep_sequence(self):
        # T == n_obs_steps: exactly one result
        obs_seq = np.random.randn(self.To, self.Do).astype(np.float32)
        action_seq = np.random.randn(self.To, self.Da).astype(np.float32)
        results = self.classifier.classify_sequence(obs_seq, action_seq)
        self.assertEqual(len(results), 1)


class TestClassifyEpisodeFromPkl(unittest.TestCase):

    def test_classifies_each_row(self):
        classifier = _make_classifier()
        T = 10
        df = pd.DataFrame({
            "obs": [np.random.randn(2, 20).astype(np.float32) for _ in range(T)],
            "action": [np.random.randn(8, 7).astype(np.float32) for _ in range(T)],
        })
        results = classifier.classify_episode_from_pkl(df)
        self.assertEqual(len(results), T)
        self.assertEqual(classifier.monitor.process_sample.call_count, T)

    def test_timestep_indices_match_dataframe(self):
        classifier = _make_classifier()
        df = pd.DataFrame({
            "obs": [np.random.randn(2, 20).astype(np.float32) for _ in range(5)],
            "action": [np.random.randn(8, 7).astype(np.float32) for _ in range(5)],
        })
        results = classifier.classify_episode_from_pkl(df)
        timesteps = [t for t, _ in results]
        self.assertEqual(timesteps, list(df.index))

    def test_no_transform_in_rollout_mode(self):
        # Even if _rotation_transformer is set, pkl data is never transformed
        classifier = _make_classifier(mode="rollout")
        mock_transformer = MagicMock()
        classifier._rotation_transformer = mock_transformer
        df = pd.DataFrame({
            "obs": [np.random.randn(2, 20).astype(np.float32)],
            "action": [np.random.randn(8, 7).astype(np.float32)],
        })
        # classify_episode_from_pkl skips _apply_action_transform entirely
        classifier.classify_episode_from_pkl(df)
        mock_transformer.forward.assert_not_called()


class TestClassifyDemoFromHdf5(unittest.TestCase):

    def _make_demo_group(self, T=10, obs_dim=20, action_dim=7):
        """Build a fake h5py-like group using nested dicts + numpy arrays."""
        rng = np.random.RandomState(0)
        demo = {
            "obs": {
                "robot0_eef_pos": rng.randn(T, 3).astype(np.float32),
                "robot0_eef_quat": rng.randn(T, 4).astype(np.float32),
                "robot0_gripper_qpos": rng.randn(T, 2).astype(np.float32),
            },
            "actions": rng.randn(T, action_dim).astype(np.float32),
        }
        # Wrap leaf arrays so [:] works
        class _ArrayProxy:
            def __init__(self, arr):
                self._arr = arr
            def __getitem__(self, key):
                return self._arr[key]

        class _DictGroup:
            def __init__(self, d):
                self._d = {k: (_DictGroup(v) if isinstance(v, dict) else _ArrayProxy(v)) for k, v in d.items()}
            def __getitem__(self, k):
                return self._d[k]

        return _DictGroup(demo), T, obs_dim

    def test_classify_demo_result_length(self):
        classifier = _make_classifier(n_obs_steps=2, n_action_steps=4)
        obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        classifier._obs_keys = obs_keys
        demo_group, T, _ = self._make_demo_group(T=10)
        results = classifier.classify_demo_from_hdf5(demo_group)
        self.assertEqual(len(results), T - (classifier.n_obs_steps - 1))

    def test_classify_demo_raises_without_obs_keys(self):
        classifier = _make_classifier()
        classifier._obs_keys = None
        demo_group, _, _ = self._make_demo_group()
        with self.assertRaises(ValueError):
            classifier.classify_demo_from_hdf5(demo_group)

    def test_classify_demo_obs_keys_override(self):
        classifier = _make_classifier()
        classifier._obs_keys = None  # not set on object
        demo_group, T, _ = self._make_demo_group(T=8)
        obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        results = classifier.classify_demo_from_hdf5(demo_group, obs_keys=obs_keys)
        self.assertEqual(len(results), T - (classifier.n_obs_steps - 1))


class TestActionTransform(unittest.TestCase):

    def _make_classifier_with_mock_transformer(self):
        """Return a classifier with a mock rotation transformer injected."""
        monitor = _make_mock_monitor()
        classifier = TrajectoryClassifier(monitor=monitor, mode="demo", abs_action=False)
        mock_transformer = MagicMock()
        # forward: (T, 3) → (T, 6)
        mock_transformer.forward.side_effect = lambda x: np.zeros((*x.shape[:-1], 6), dtype=np.float32)
        classifier._rotation_transformer = mock_transformer
        return classifier, mock_transformer

    def test_rollout_mode_no_transform(self):
        classifier = _make_classifier(mode="rollout")
        action = np.ones((8, 10), dtype=np.float32)
        result = classifier._apply_action_transform(action)
        np.testing.assert_array_equal(result, action)

    def test_demo_mode_no_abs_action_no_transform(self):
        classifier = _make_classifier(mode="demo")
        action = np.ones((8, 10), dtype=np.float32)
        result = classifier._apply_action_transform(action)
        np.testing.assert_array_equal(result, action)

    def test_single_arm_output_shape(self):
        # 7-dim raw: pos(3) + rot_aa(3) + gripper(1) → 10-dim: pos(3) + rot_6d(6) + gripper(1)
        classifier, transformer = self._make_classifier_with_mock_transformer()
        action = np.ones((8, 7), dtype=np.float32)
        result = classifier._apply_action_transform(action)
        self.assertEqual(result.shape, (8, 10))
        transformer.forward.assert_called_once()

    def test_dual_arm_output_shape(self):
        # 14-dim raw (2×7) → 20-dim (2×10)
        classifier, transformer = self._make_classifier_with_mock_transformer()
        # For dual-arm, forward is called on (8, 2, 3) → returns (8, 2, 6)
        transformer.forward.side_effect = lambda x: np.zeros((*x.shape[:-1], 6), dtype=np.float32)
        action = np.ones((8, 14), dtype=np.float32)
        result = classifier._apply_action_transform(action)
        self.assertEqual(result.shape, (8, 20))

    def test_classify_sample_applies_transform_in_demo_mode(self):
        classifier, transformer = self._make_classifier_with_mock_transformer()
        obs = np.random.randn(2, 20).astype(np.float32)
        action = np.random.randn(8, 7).astype(np.float32)
        classifier.classify_sample(obs, action)
        transformer.forward.assert_called_once()

    def test_classify_sample_skips_transform_in_rollout_mode(self):
        classifier = _make_classifier(mode="rollout")
        mock_transformer = MagicMock()
        classifier._rotation_transformer = mock_transformer
        obs = np.random.randn(2, 20).astype(np.float32)
        action = np.random.randn(8, 7).astype(np.float32)
        classifier.classify_sample(obs, action)
        mock_transformer.forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
