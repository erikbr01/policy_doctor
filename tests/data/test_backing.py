"""Tests for backing store (InMemoryBackingStore)."""

import unittest
import numpy as np

from policy_doctor.data.backing import InMemoryBackingStore
from policy_doctor.data.structures import GlobalInfluenceMatrix, EpisodeInfo


class TestBacking(unittest.TestCase):
    def test_in_memory_backing_store(self):
        arr = np.random.randn(4, 6).astype(np.float32)
        store = InMemoryBackingStore(arr)
        self.assertEqual(store.shape, (4, 6))
        block = store.read_slice(1, 3, 2, 5)
        self.assertEqual(block.shape, (2, 3))
        np.testing.assert_array_almost_equal(block, arr[1:3, 2:5])
        c = store.read_cell(2, 3)
        self.assertEqual(c, arr[2, 3])

    def test_global_matrix_with_in_memory_backing(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        store = InMemoryBackingStore(arr)
        episodes_r = [
            EpisodeInfo(0, 2, 0, 2, None, None),
            EpisodeInfo(1, 1, 2, 3, None, None),
        ]
        episodes_d = [
            EpisodeInfo(0, 2, 0, 2, None, None),
            EpisodeInfo(1, 2, 2, 4, None, None),
        ]
        global_mat = GlobalInfluenceMatrix(store, episodes_r, episodes_d)
        local = global_mat.get_local_matrix(0, 0)
        self.assertEqual(local.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            local._read_block(),
            arr[0:2, 0:2],
        )
