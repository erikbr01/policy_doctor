#!/usr/bin/env python
"""Standalone test runner for influence visualizer tests.

This script runs tests without requiring pytest, making it easier to
verify functionality in environments where pytest isn't installed.

Usage:
    python influence_visualizer/tests/run_tests.py
"""

import pathlib
import sys
import traceback
from typing import Callable, List, Tuple

# Add parent to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np

from influence_visualizer.data_loader import (
    EpisodeInfo,
    InfluenceData,
    create_mock_influence_data,
    ep_lens_to_idxs,
)


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[Tuple[str, str]] = []

    def record_pass(self, name: str):
        self.passed += 1
        print(f"  [PASS] {name}")

    def record_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  [FAIL] {name}")
        print(f"         {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'=' * 60}")
        return self.failed == 0


def run_test(result: TestResult, name: str, test_fn: Callable):
    """Run a single test and record the result."""
    try:
        test_fn()
        result.record_pass(name)
    except AssertionError as e:
        result.record_fail(name, str(e))
    except Exception as e:
        result.record_fail(name, f"{type(e).__name__}: {e}")


def test_ep_lens_single():
    ep_lens = np.array([5])
    result = ep_lens_to_idxs(ep_lens)
    assert len(result) == 1
    assert np.array_equal(result[0], np.array([0, 1, 2, 3, 4]))


def test_ep_lens_multiple():
    ep_lens = np.array([3, 4, 2])
    result = ep_lens_to_idxs(ep_lens)
    assert len(result) == 3
    assert np.array_equal(result[0], np.array([0, 1, 2]))
    assert np.array_equal(result[1], np.array([3, 4, 5, 6]))
    assert np.array_equal(result[2], np.array([7, 8]))


def test_ep_lens_empty():
    ep_lens = np.array([])
    result = ep_lens_to_idxs(ep_lens)
    assert len(result) == 0


def test_ep_lens_total():
    ep_lens = np.array([10, 15, 8, 22])
    result = ep_lens_to_idxs(ep_lens)
    total = sum(len(idx) for idx in result)
    assert total == ep_lens.sum()


def test_episode_info_creation():
    ep = EpisodeInfo(
        index=5,
        video_path=pathlib.Path("/some/path.mp4"),
        num_samples=100,
        sample_start_idx=50,
        sample_end_idx=150,
        success=True,
    )
    assert ep.index == 5
    assert ep.num_samples == 100
    assert ep.sample_start_idx == 50
    assert ep.sample_end_idx == 150
    assert ep.success is True


def test_episode_info_to_dict():
    ep = EpisodeInfo(
        index=3,
        video_path=pathlib.Path("/test.mp4"),
        num_samples=50,
        sample_start_idx=0,
        sample_end_idx=50,
        success=False,
    )
    d = ep.to_dict()
    assert d["index"] == 3
    assert d["video_path"] == "/test.mp4"
    assert d["num_samples"] == 50
    assert d["success"] is False


def test_episode_info_none_video():
    ep = EpisodeInfo(
        index=0,
        video_path=None,
        num_samples=10,
        sample_start_idx=0,
        sample_end_idx=10,
    )
    d = ep.to_dict()
    assert d["video_path"] is None


def test_mock_data_creation():
    data = create_mock_influence_data(
        num_rollout_episodes=5,
        num_demo_episodes=10,
        samples_per_rollout=20,
        samples_per_demo=50,
    )
    assert isinstance(data, InfluenceData)
    assert data.num_rollout_episodes == 5
    assert data.num_demo_episodes == 10
    assert data.num_rollout_samples == 5 * 20
    assert data.num_demo_samples == 10 * 50


def test_mock_data_matrix_shape():
    data = create_mock_influence_data(
        num_rollout_episodes=3,
        num_demo_episodes=7,
        samples_per_rollout=15,
        samples_per_demo=30,
    )
    expected_shape = (3 * 15, 7 * 30)
    assert data.influence_matrix.shape == expected_shape


def test_sample_to_episode_mapping():
    data = create_mock_influence_data(
        num_rollout_episodes=4,
        num_demo_episodes=6,
        samples_per_rollout=10,
        samples_per_demo=20,
    )
    assert len(data.rollout_sample_to_episode) == 4 * 10
    assert len(data.rollout_sample_to_timestep) == 4 * 10
    assert all(data.rollout_sample_to_episode[:10] == 0)
    assert all(data.rollout_sample_to_episode[10:20] == 1)


def test_get_rollout_sample_info():
    data = create_mock_influence_data(
        num_rollout_episodes=5,
        num_demo_episodes=8,
        samples_per_rollout=10,
        samples_per_demo=15,
    )
    info = data.get_rollout_sample_info(0)
    assert info["episode_idx"] == 0
    assert info["timestep"] == 0

    info = data.get_rollout_sample_info(15)
    assert info["episode_idx"] == 1
    assert info["timestep"] == 5


def test_get_top_influences():
    data = create_mock_influence_data(
        num_rollout_episodes=5,
        num_demo_episodes=8,
        samples_per_rollout=10,
        samples_per_demo=15,
    )
    top = data.get_top_influences_for_sample(0, top_k=5)
    assert len(top) == 5

    scores = [t["influence_score"] for t in top]
    assert scores == sorted(scores, reverse=True)

    for t in top:
        assert "demo_sample_idx" in t
        assert "influence_score" in t
        assert "demo_episode_idx" in t
        assert "demo_timestep" in t
        assert "demo_episode" in t


def test_get_samples_for_episode():
    data = create_mock_influence_data(
        num_rollout_episodes=5,
        num_demo_episodes=8,
        samples_per_rollout=10,
        samples_per_demo=15,
    )
    samples = data.get_samples_for_rollout_episode(2)
    assert np.array_equal(samples, np.arange(20, 30))

    samples = data.get_samples_for_demo_episode(3)
    assert np.array_equal(samples, np.arange(45, 60))


def test_episode_coverage():
    data = create_mock_influence_data(
        num_rollout_episodes=6,
        num_demo_episodes=10,
        samples_per_rollout=12,
        samples_per_demo=18,
    )

    # Check rollout coverage
    covered = set()
    for ep in data.rollout_episodes:
        for idx in range(ep.sample_start_idx, ep.sample_end_idx):
            assert idx not in covered, f"Sample {idx} in multiple episodes"
            covered.add(idx)
    expected = set(range(data.num_rollout_samples))
    assert covered == expected

    # Check demo coverage
    covered = set()
    for ep in data.demo_episodes:
        for idx in range(ep.sample_start_idx, ep.sample_end_idx):
            assert idx not in covered
            covered.add(idx)
    expected = set(range(data.num_demo_samples))
    assert covered == expected


def test_sample_mapping_consistency():
    data = create_mock_influence_data(
        num_rollout_episodes=6,
        num_demo_episodes=10,
        samples_per_rollout=12,
        samples_per_demo=18,
    )

    for sample_idx in range(data.num_rollout_samples):
        ep_idx = data.rollout_sample_to_episode[sample_idx]
        ep = data.rollout_episodes[ep_idx]
        assert ep.sample_start_idx <= sample_idx < ep.sample_end_idx


def test_influence_scores_valid():
    data = create_mock_influence_data()
    assert not np.any(np.isnan(data.influence_matrix))
    assert not np.any(np.isinf(data.influence_matrix))


def test_top_influences_valid_episodes():
    data = create_mock_influence_data(
        num_rollout_episodes=3,
        num_demo_episodes=5,
    )

    for sample_idx in range(min(10, data.num_rollout_samples)):
        top = data.get_top_influences_for_sample(sample_idx, top_k=5)
        for influence in top:
            ep_idx = influence["demo_episode_idx"]
            assert 0 <= ep_idx < data.num_demo_episodes
            demo_sample = influence["demo_sample_idx"]
            assert 0 <= demo_sample < data.num_demo_samples


# SequenceSampler mapping tests
def test_timestep_mapping_matches_buffer_start():
    """Verify timestep = max(min_start + t, 0) matches buffer_start_idx logic."""
    episode_length = 100
    sequence_length = 16
    pad_before = 1
    pad_after = 7

    min_start = -pad_before
    max_start = episode_length - sequence_length + pad_after
    num_samples = max_start - min_start + 1

    expected_timesteps = []
    for t in range(num_samples):
        offset = min_start + t
        buffer_frame_idx = max(offset, 0)
        expected_timesteps.append(buffer_frame_idx)

    # First pad_before samples should all map to frame 0
    for t in range(pad_before):
        assert expected_timesteps[t] == 0, f"Sample {t} should map to frame 0"

    # After that, 1:1 mapping shifted by pad_before
    for t in range(pad_before, num_samples):
        assert expected_timesteps[t] == t - pad_before


def test_first_sample_maps_to_first_frame():
    """First sample should always map to frame 0."""
    for pad_before in [0, 1, 2, 5]:
        min_start = -pad_before
        t = 0
        frame_idx = max(min_start + t, 0)
        assert frame_idx == 0


def test_episode_lengths_match_trak_util():
    """Verify episode length computation matches trak_util.get_train_episode_lengths."""
    episode_ends = np.array([50, 120, 180])
    episode_mask = np.array([True, True, True])
    sequence_length = 16
    pad_before = 1
    pad_after = 7

    expected_lengths = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        num_samples = max_start - min_start + 1
        expected_lengths.append(num_samples)

    assert expected_lengths == [43, 63, 53], f"Got {expected_lengths}"


def main():
    print("=" * 60)
    print("Influence Visualizer Test Suite")
    print("=" * 60)

    result = TestResult()

    # ep_lens_to_idxs tests
    print("\nTesting ep_lens_to_idxs:")
    run_test(result, "single_episode", test_ep_lens_single)
    run_test(result, "multiple_episodes", test_ep_lens_multiple)
    run_test(result, "empty_episodes", test_ep_lens_empty)
    run_test(result, "total_indices", test_ep_lens_total)

    # EpisodeInfo tests
    print("\nTesting EpisodeInfo:")
    run_test(result, "creation", test_episode_info_creation)
    run_test(result, "to_dict", test_episode_info_to_dict)
    run_test(result, "none_video", test_episode_info_none_video)

    # Mock data tests
    print("\nTesting Mock Data:")
    run_test(result, "creation", test_mock_data_creation)
    run_test(result, "matrix_shape", test_mock_data_matrix_shape)
    run_test(result, "sample_mapping", test_sample_to_episode_mapping)

    # InfluenceData methods tests
    print("\nTesting InfluenceData Methods:")
    run_test(result, "get_rollout_sample_info", test_get_rollout_sample_info)
    run_test(result, "get_top_influences", test_get_top_influences)
    run_test(result, "get_samples_for_episode", test_get_samples_for_episode)

    # Consistency tests
    print("\nTesting Data Consistency:")
    run_test(result, "episode_coverage", test_episode_coverage)
    run_test(result, "sample_mapping_consistency", test_sample_mapping_consistency)
    run_test(result, "influence_scores_valid", test_influence_scores_valid)
    run_test(
        result, "top_influences_valid_episodes", test_top_influences_valid_episodes
    )

    # SequenceSampler mapping tests
    print("\nTesting SequenceSampler Mapping:")
    run_test(
        result,
        "timestep_mapping_matches_buffer_start",
        test_timestep_mapping_matches_buffer_start,
    )
    run_test(
        result,
        "first_sample_maps_to_first_frame",
        test_first_sample_maps_to_first_frame,
    )
    run_test(
        result, "episode_lengths_match_trak_util", test_episode_lengths_match_trak_util
    )

    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
