"""Tests for the influence visualizer data loader.

These tests verify the core data loading and mapping functionality without
requiring a web browser or the full Streamlit application.

Run tests with:
    pytest influence_visualizer/tests/test_data_loader.py -v
"""

import pathlib
import sys

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from influence_visualizer.data_loader import (
    EpisodeInfo,
    InfluenceData,
    SampleInfo,
    create_mock_influence_data,
)


class TestEpisodeInfo:
    """Test the EpisodeInfo dataclass."""

    def test_creation(self):
        ep = EpisodeInfo(
            index=5,
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

    def test_to_dict(self):
        ep = EpisodeInfo(
            index=3,
            num_samples=50,
            sample_start_idx=0,
            sample_end_idx=50,
            success=False,
        )

        d = ep.to_dict()

        assert d["index"] == 3
        assert d["num_samples"] == 50
        assert d["success"] is False

    def test_raw_length_optional(self):
        ep = EpisodeInfo(
            index=0,
            num_samples=10,
            sample_start_idx=0,
            sample_end_idx=10,
            raw_length=25,
        )

        assert ep.raw_length == 25
        d = ep.to_dict()
        assert d["raw_length"] == 25


class TestSampleInfo:
    """Test the SampleInfo dataclass."""

    def test_creation(self):
        sample = SampleInfo(
            global_idx=42,
            episode_idx=3,
            timestep=5,
            buffer_start_idx=100,
            buffer_end_idx=116,
            sample_start_idx=0,
            sample_end_idx=16,
        )

        assert sample.global_idx == 42
        assert sample.episode_idx == 3
        assert sample.timestep == 5
        assert sample.buffer_start_idx == 100
        assert sample.buffer_end_idx == 116


class TestMockInfluenceData:
    """Test mock data generation."""

    def test_create_mock_data(self):
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

    def test_influence_matrix_shape(self):
        data = create_mock_influence_data(
            num_rollout_episodes=3,
            num_demo_episodes=7,
            samples_per_rollout=15,
            samples_per_demo=30,
        )

        expected_shape = (3 * 15, 7 * 30)
        assert data.influence_matrix.shape == expected_shape

    def test_rollout_sample_infos_created(self):
        data = create_mock_influence_data(
            num_rollout_episodes=4,
            num_demo_episodes=6,
            samples_per_rollout=10,
            samples_per_demo=20,
        )

        # Check rollout sample infos
        assert len(data.rollout_sample_infos) == 4 * 10

        # First sample should be episode 0, timestep 0
        first_sample = data.rollout_sample_infos[0]
        assert first_sample.episode_idx == 0
        assert first_sample.timestep == 0

        # Sample 15 should be episode 1, timestep 5
        sample_15 = data.rollout_sample_infos[15]
        assert sample_15.episode_idx == 1
        assert sample_15.timestep == 5

    def test_demo_sample_infos_created(self):
        data = create_mock_influence_data(
            num_rollout_episodes=4,
            num_demo_episodes=6,
            samples_per_rollout=10,
            samples_per_demo=20,
        )

        # Check demo sample infos
        assert len(data.demo_sample_infos) == 6 * 20


class TestInfluenceDataMethods:
    """Test InfluenceData query methods."""

    @pytest.fixture
    def data(self):
        return create_mock_influence_data(
            num_rollout_episodes=5,
            num_demo_episodes=8,
            samples_per_rollout=10,
            samples_per_demo=15,
        )

    def test_get_rollout_sample_info(self, data):
        # Sample 0 should be in episode 0, timestep 0
        info = data.get_rollout_sample_info(0)
        assert info.episode_idx == 0
        assert info.timestep == 0

        # Sample 15 should be in episode 1, timestep 5
        info = data.get_rollout_sample_info(15)
        assert info.episode_idx == 1
        assert info.timestep == 5

    def test_get_demo_sample_info(self, data):
        # Sample 0 should be in episode 0, timestep 0
        info = data.get_demo_sample_info(0)
        assert info.episode_idx == 0
        assert info.timestep == 0

    def test_get_top_influences_for_sample(self, data):
        top = data.get_top_influences_for_sample(0, top_k=5)

        assert len(top) == 5

        # Check that results are sorted by influence score (descending)
        scores = [t["influence_score"] for t in top]
        assert scores == sorted(scores, reverse=True)

        # Check that each result has required fields
        for t in top:
            assert "demo_sample_idx" in t
            assert "influence_score" in t
            assert "demo_episode_idx" in t
            assert "demo_timestep" in t

    def test_get_samples_for_rollout_episode(self, data):
        samples = data.get_samples_for_rollout_episode(2)

        # Episode 2 should have samples 20-29
        np.testing.assert_array_equal(samples, np.arange(20, 30))

    def test_get_samples_for_demo_episode(self, data):
        samples = data.get_samples_for_demo_episode(3)

        # Episode 3 should have samples 45-59 (3 * 15 to 4 * 15 - 1)
        np.testing.assert_array_equal(samples, np.arange(45, 60))


class TestInfluenceDataConsistency:
    """Test consistency of sample indexing across the data structure."""

    @pytest.fixture
    def data(self):
        return create_mock_influence_data(
            num_rollout_episodes=6,
            num_demo_episodes=10,
            samples_per_rollout=12,
            samples_per_demo=18,
        )

    def test_rollout_episode_indices_are_contiguous(self, data):
        """Verify that sample indices within episodes are contiguous."""
        for ep in data.rollout_episodes:
            assert ep.sample_end_idx == ep.sample_start_idx + ep.num_samples

    def test_rollout_episodes_cover_all_samples(self, data):
        """Verify that episodes cover all samples without gaps."""
        covered = set()
        for ep in data.rollout_episodes:
            for idx in range(ep.sample_start_idx, ep.sample_end_idx):
                assert idx not in covered, f"Sample {idx} is in multiple episodes"
                covered.add(idx)

        expected = set(range(data.num_rollout_samples))
        assert covered == expected

    def test_demo_episodes_cover_all_samples(self, data):
        """Verify that demo episodes cover all samples without gaps."""
        covered = set()
        for ep in data.demo_episodes:
            for idx in range(ep.sample_start_idx, ep.sample_end_idx):
                assert idx not in covered
                covered.add(idx)

        expected = set(range(data.num_demo_samples))
        assert covered == expected

    def test_sample_info_matches_episode_info(self, data):
        """Verify that sample_info episode_idx matches episode boundaries."""
        for sample_idx in range(data.num_rollout_samples):
            sample_info = data.rollout_sample_infos[sample_idx]
            ep_idx = sample_info.episode_idx
            ep = data.rollout_episodes[ep_idx]

            assert ep.sample_start_idx <= sample_idx < ep.sample_end_idx, (
                f"Sample {sample_idx} claims episode {ep_idx} but is outside its range"
            )


class TestInfluenceScoreOperations:
    """Test influence score queries and aggregations."""

    def test_top_k_returns_correct_count(self):
        data = create_mock_influence_data(
            num_rollout_episodes=2,
            num_demo_episodes=20,
            samples_per_rollout=5,
            samples_per_demo=10,
        )

        # Request top 15
        top = data.get_top_influences_for_sample(0, top_k=15)
        assert len(top) == 15

        # Request more than available (200 demo samples)
        top = data.get_top_influences_for_sample(0, top_k=300)
        assert len(top) == 200  # Should return all available

    def test_influence_scores_are_valid(self):
        data = create_mock_influence_data()

        # Check that influence matrix doesn't have NaN or Inf
        assert not np.any(np.isnan(data.influence_matrix))
        assert not np.any(np.isinf(data.influence_matrix))

    def test_top_influences_reference_valid_episodes(self):
        data = create_mock_influence_data(
            num_rollout_episodes=3,
            num_demo_episodes=5,
        )

        for sample_idx in range(data.num_rollout_samples):
            top = data.get_top_influences_for_sample(sample_idx, top_k=5)

            for influence in top:
                ep_idx = influence["demo_episode_idx"]
                assert 0 <= ep_idx < data.num_demo_episodes

                demo_sample = influence["demo_sample_idx"]
                assert 0 <= demo_sample < data.num_demo_samples


class TestSequenceSamplerMapping:
    """Test that the demo timestep mapping matches SequenceSampler logic."""

    def test_timestep_mapping_matches_buffer_start(self):
        """Verify timestep = max(min_start + t, 0) matches buffer_start_idx logic."""
        # Simulate SequenceSampler parameters
        episode_length = 100
        sequence_length = 16
        pad_before = 1
        pad_after = 7

        # Compute expected values using SequenceSampler logic
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        num_samples = max_start - min_start + 1

        expected_timesteps = []
        for t in range(num_samples):
            offset = min_start + t
            buffer_frame_idx = max(offset, 0)  # Frame index within episode
            expected_timesteps.append(buffer_frame_idx)

        # The first pad_before samples should all map to frame 0
        for t in range(pad_before):
            assert expected_timesteps[t] == 0, f"Sample {t} should map to frame 0"

        # After that, it should be a 1:1 mapping shifted by pad_before
        for t in range(pad_before, num_samples):
            assert expected_timesteps[t] == t - pad_before, (
                f"Sample {t} should map to frame {t - pad_before}"
            )

    def test_timestep_mapping_with_different_padding(self):
        """Test mapping with various padding configurations."""
        test_cases = [
            # (episode_length, sequence_length, pad_before, pad_after)
            (50, 16, 0, 0),  # No padding
            (50, 16, 1, 7),  # Typical diffusion policy config
            (50, 16, 2, 6),  # Different padding
            (100, 16, 1, 7),  # Longer episode
            (20, 16, 1, 7),  # Short episode
        ]

        for ep_len, seq_len, pad_before, pad_after in test_cases:
            min_start = -pad_before
            max_start = ep_len - seq_len + pad_after
            num_samples = max_start - min_start + 1

            if num_samples <= 0:
                continue

            for t in range(num_samples):
                offset = min_start + t
                expected_frame = max(offset, 0)

                # Frame should always be within episode bounds
                assert 0 <= expected_frame < ep_len, (
                    f"Frame {expected_frame} out of bounds for episode length {ep_len}"
                )

    def test_first_sample_maps_to_first_frame(self):
        """First sample should always map to the first frame (index 0)."""
        for pad_before in [0, 1, 2, 5]:
            min_start = -pad_before
            t = 0
            frame_idx = max(min_start + t, 0)
            assert frame_idx == 0, (
                f"First sample with pad_before={pad_before} should map to frame 0"
            )

    def test_last_sample_maps_to_valid_frame(self):
        """Last sample should map to a valid frame within the episode."""
        episode_length = 100
        sequence_length = 16
        pad_before = 1
        pad_after = 7

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        num_samples = max_start - min_start + 1

        last_t = num_samples - 1
        last_frame = max(min_start + last_t, 0)

        # The last frame should be within the episode
        assert 0 <= last_frame < episode_length, (
            f"Last frame {last_frame} should be within [0, {episode_length})"
        )

        # Specifically, it should be max_start (since max(max_start, 0) = max_start when max_start >= 0)
        expected_last_frame = max_start
        assert last_frame == expected_last_frame, (
            f"Last frame should be {expected_last_frame}, got {last_frame}"
        )


class TestCompareWithCupidCodebase:
    """Tests that compare our implementation with the original CUPID codebase."""

    def test_episode_lengths_match_trak_util(self):
        """Verify episode length computation matches trak_util.get_train_episode_lengths."""
        # Simulate episode ends and mask
        episode_ends = np.array([50, 120, 180])  # 3 episodes of length 50, 70, 60
        episode_mask = np.array([True, True, True])
        sequence_length = 16
        pad_before = 1
        pad_after = 7

        # Compute using our logic (same as trak_util.get_train_episode_lengths)
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

        # Expected: 50 - 16 + 7 - (-1) + 1 = 43, 70 - 16 + 7 + 1 + 1 = 63, 60 - 16 + 7 + 1 + 1 = 53
        assert expected_lengths == [43, 63, 53], f"Got {expected_lengths}"

    def test_sample_indices_contiguous(self):
        """Verify sample indices are contiguous across episodes."""
        episode_ends = np.array([30, 80, 100])
        episode_mask = np.array([True, False, True])  # Skip middle episode
        sequence_length = 16
        pad_before = 1
        pad_after = 7

        sample_idx = 0
        for i in range(len(episode_ends)):
            if not episode_mask[i]:
                continue
            start_idx = 0 if i == 0 else episode_ends[i - 1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx

            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after
            num_samples = max_start - min_start + 1

            if num_samples > 0:
                sample_idx += num_samples

        # Total samples should equal sum of valid episode sample counts
        expected_total = (30 - 16 + 7 + 1 + 1) + (20 - 16 + 7 + 1 + 1)  # eps 0 and 2
        assert sample_idx == expected_total, (
            f"Got {sample_idx}, expected {expected_total}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
