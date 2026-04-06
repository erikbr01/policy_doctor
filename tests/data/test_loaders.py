"""Tests for loaders (create_global_influence_from_array)."""

import unittest
import numpy as np

from policy_doctor.data.loaders import create_global_influence_from_array


class TestLoaders(unittest.TestCase):
    def test_create_global_influence_from_array(self):
        mat = np.random.randn(5, 7).astype(np.float32)
        global_mat = create_global_influence_from_array(
            mat,
            rollout_episode_lengths=[2, 3],
            demo_episode_lengths=[3, 4],
            rollout_success=[True, False],
        )
        self.assertEqual(global_mat.shape, (5, 7))
        self.assertEqual(len(global_mat.rollout_episodes), 2)
        self.assertEqual(len(global_mat.demo_episodes), 2)
        np.testing.assert_array_almost_equal(
            global_mat.get_slice(0, 5, 0, 7),
            mat,
        )
