"""Tests for slice influence (rank demo slices inside local matrix)."""

import unittest
import numpy as np

from policy_doctor.data.structures import LocalInfluenceMatrix
from policy_doctor.data.backing import InMemoryBackingStore
from policy_doctor.computations.slice_influence import rank_demo_slices_by_influence


class TestSliceInfluence(unittest.TestCase):
    def setUp(self):
        arr = np.array(
            [[1.0, 0.0, 2.0, 0.5],
             [0.0, 3.0, 0.0, 1.0],
             [2.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        store = InMemoryBackingStore(arr)
        self.local_matrix_3x4 = LocalInfluenceMatrix(store, 0, 0, 0, 3, 0, 4)

    def test_rank_demo_slices_by_influence(self):
        sorted_idx, sorted_scores, raw = rank_demo_slices_by_influence(
            self.local_matrix_3x4,
            rollout_sample_lo=0,
            rollout_sample_hi=3,
            window_width_demo=1,
            ascending=False,
        )
        self.assertEqual(len(sorted_idx), 4)
        self.assertEqual(len(sorted_scores), 4)
        self.assertEqual(raw.shape, (4,))
        np.testing.assert_array_almost_equal(np.sort(sorted_scores)[::-1], sorted_scores)
