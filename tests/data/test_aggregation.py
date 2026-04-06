"""Tests for aggregation (sum/mean, sliding window)."""

import unittest
import numpy as np

from policy_doctor.data.aggregation import aggregate_axis, sliding_window_sum


class TestAggregation(unittest.TestCase):
    def test_aggregate_axis_sum(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = aggregate_axis(data, axis=0, agg_fn="sum")
        np.testing.assert_array_almost_equal(out, [4.0, 6.0])
        out1 = aggregate_axis(data, axis=1, agg_fn="sum")
        np.testing.assert_array_almost_equal(out1, [3.0, 7.0])
        out2 = aggregate_axis(data, axis=None, agg_fn="sum")
        self.assertEqual(float(out2), 10.0)

    def test_aggregate_axis_mean(self):
        data = np.array([[2.0, 4.0], [6.0, 8.0]])
        out = aggregate_axis(data, axis=0, agg_fn="mean")
        np.testing.assert_array_almost_equal(out, [4.0, 6.0])

    def test_sliding_window_sum(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = sliding_window_sum(data, window_width=3, axis=1)
        self.assertEqual(out.shape, (1, 5))
        self.assertEqual(out[0, 1], 6.0)
