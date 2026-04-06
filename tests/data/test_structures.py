"""Tests for Trajectory, Segment, Sample, GlobalInfluenceMatrix, LocalInfluenceMatrix."""

import unittest
import numpy as np

from policy_doctor.data.structures import (
    ActionInfluence,
    EpisodeInfo,
    GlobalInfluenceMatrix,
    LocalInfluenceMatrix,
    Sample,
    Segment,
    Trajectory,
)
from policy_doctor.data.loaders import create_global_influence_from_array


class TestStructures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        mat = np.random.randn(6, 8).astype(np.float32)
        cls.global_matrix = create_global_influence_from_array(
            mat,
            rollout_episode_lengths=[3, 3],
            demo_episode_lengths=[4, 4],
            rollout_success=[True, False],
        )
        cls.local_matrix = cls.global_matrix.get_local_matrix(0, 0)

    def test_trajectory_segment_sample_hierarchy(self):
        s0 = Sample(global_idx=0, timestep=0, horizon=1)
        s1 = Sample(global_idx=1, timestep=1, horizon=1)
        seg = Segment(label="0", samples=[s0, s1], start_global_idx=0, end_global_idx=2)
        traj = Trajectory(index=0, segments=[seg], success=True, raw_length=2)
        self.assertEqual(traj.num_samples, 2)
        self.assertEqual(seg.num_samples, 2)

    def test_global_influence_matrix_shape(self):
        self.assertEqual(self.global_matrix.shape, (6, 8))
        self.assertEqual(self.global_matrix.num_rollout_samples, 6)
        self.assertEqual(self.global_matrix.num_demo_samples, 8)

    def test_global_get_slice(self):
        block = self.global_matrix.get_slice(1, 3, 2, 5)
        self.assertEqual(block.shape, (2, 3))

    def test_global_get_local_matrix(self):
        local = self.global_matrix.get_local_matrix(0, 0)
        self.assertIsInstance(local, LocalInfluenceMatrix)
        self.assertEqual(local.shape, (3, 4))

    def test_global_get_action_influence(self):
        a = self.global_matrix.get_action_influence(0, 0)
        self.assertIsInstance(a, ActionInfluence)
        self.assertIsInstance(float(a), float)

    def test_global_aggregate(self):
        row_sums = self.global_matrix.aggregate(axis=1, agg_fn="sum")
        self.assertEqual(row_sums.shape, (6,))
        total = self.global_matrix.aggregate(axis=None, agg_fn="sum")
        self.assertTrue(np.isscalar(total) or total.ndim == 0)

    def test_local_matrix_get_slice(self):
        block = self.local_matrix.get_slice(0, 2, 0, 3)
        self.assertEqual(block.shape, (2, 3))

    def test_local_matrix_get_action_influence(self):
        a = self.local_matrix.get_action_influence(0, 0)
        self.assertIsInstance(a, ActionInfluence)

    def test_local_matrix_aggregate(self):
        # local_matrix shape (3, 4); axis=1 -> sum over columns -> shape (3,)
        row_sums = self.local_matrix.aggregate(axis=1, agg_fn="sum")
        self.assertEqual(row_sums.shape, (3,))
