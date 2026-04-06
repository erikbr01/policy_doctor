"""Tests for curation config (CurationSlice, save/load, fingerprint)."""

import unittest
import numpy as np

from policy_doctor.curation.config import (
    CurationConfig,
    CurationSlice,
    compute_dataset_fingerprint,
    merge_overlapping_slices,
)


class TestCurationConfig(unittest.TestCase):
    def test_curation_slice_roundtrip(self):
        s = CurationSlice(episode_idx=0, start=5, end=10, label="reach", source="manual")
        d = s.to_dict()
        s2 = CurationSlice.from_dict(d)
        self.assertEqual(s2.episode_idx, s.episode_idx)
        self.assertEqual(s2.start, s.start)
        self.assertEqual(s2.end, s.end)

    def test_compute_dataset_fingerprint(self):
        ends = np.array([10, 25, 40])
        fp = compute_dataset_fingerprint(ends)
        self.assertIsInstance(fp, str)
        self.assertEqual(len(fp), 64)
        self.assertEqual(compute_dataset_fingerprint(ends), compute_dataset_fingerprint(ends))

    def test_merge_overlapping_empty(self):
        self.assertEqual(merge_overlapping_slices([]), [])

    def test_merge_overlapping_no_overlap(self):
        slices = [
            CurationSlice(0, 0, 4, "a", "s"),
            CurationSlice(0, 10, 14, "a", "s"),
        ]
        merged = merge_overlapping_slices(slices)
        tuples = [(s.episode_idx, s.start, s.end) for s in merged]
        self.assertEqual(tuples, [(0, 0, 4), (0, 10, 14)])

    def test_merge_overlapping_adjacent(self):
        slices = [
            CurationSlice(0, 0, 4, "a", "s"),
            CurationSlice(0, 5, 9, "a", "s"),
        ]
        merged = merge_overlapping_slices(slices)
        tuples = [(s.episode_idx, s.start, s.end) for s in merged]
        self.assertEqual(tuples, [(0, 0, 9)])

    def test_merge_overlapping_windows(self):
        """Sliding window scenario: windows [0,4], [1,5], [2,6], [10,14] -> [0,6], [10,14]."""
        slices = [
            CurationSlice(0, 0, 4, "a", "s"),
            CurationSlice(0, 1, 5, "a", "s"),
            CurationSlice(0, 2, 6, "a", "s"),
            CurationSlice(0, 10, 14, "a", "s"),
        ]
        merged = merge_overlapping_slices(slices)
        tuples = [(s.episode_idx, s.start, s.end) for s in merged]
        self.assertEqual(tuples, [(0, 0, 6), (0, 10, 14)])

    def test_merge_overlapping_multiple_episodes(self):
        slices = [
            CurationSlice(1, 5, 9, "a", "s"),
            CurationSlice(0, 0, 4, "a", "s"),
            CurationSlice(0, 3, 7, "a", "s"),
            CurationSlice(1, 8, 12, "a", "s"),
        ]
        merged = merge_overlapping_slices(slices)
        tuples = [(s.episode_idx, s.start, s.end) for s in merged]
        self.assertEqual(tuples, [(0, 0, 7), (1, 5, 12)])

    def test_curation_config_to_from_dict(self):
        config = CurationConfig(
            slices=[
                CurationSlice(0, 0, 5, "a", "manual"),
                CurationSlice(1, 10, 20, "b", "manual"),
            ],
            metadata={"task": "test"},
            episode_lengths={0: 50, 1: 50},
        )
        d = config.to_dict()
        loaded = CurationConfig.from_dict(d)
        self.assertEqual(len(loaded.slices), 2)
        self.assertEqual(loaded.metadata.get("task"), "test")
        self.assertEqual(loaded.episode_lengths[0], 50)
