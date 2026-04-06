"""Tests for trajectory-level influence explanations."""

import unittest
import numpy as np

from policy_doctor.computations.explanations import (
    mean_of_mean_influence,
    sum_of_sum_influence,
    trajectory_scores,
    AGGREGATION_FUNCTIONS,
)


class TestExplanations(unittest.TestCase):
    def test_mean_of_mean_influence(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(mean_of_mean_influence(scores, True), 2.5)

    def test_sum_of_sum_influence(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(sum_of_sum_influence(scores, True), 10.0)

    def test_trajectory_scores(self):
        pairwise = np.random.randn(4, 4).astype(np.float32)
        test_ep_lens = np.array([2, 2])
        train_ep_lens = np.array([2, 2])
        success_mask = np.array([True, False])
        out = trajectory_scores(
            pairwise, test_ep_lens, train_ep_lens, success_mask,
            aggr_fn=mean_of_mean_influence,
        )
        self.assertEqual(out.shape, (2, 2))

    def test_aggregation_functions_exist(self):
        self.assertIn("mean_of_mean", AGGREGATION_FUNCTIONS)
        self.assertIn("sum_of_sum", AGGREGATION_FUNCTIONS)
