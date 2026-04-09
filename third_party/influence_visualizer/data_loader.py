"""Data loading utilities for the influence visualizer.

This module loads data using the EXACT same code paths as train_trak_diffusion.py
to ensure perfect alignment between influence scores and displayed data.

Key design principles:
1. Demo samples are loaded via hydra.utils.instantiate(cfg.task.dataset), matching TRAK featurization
2. Rollout samples are loaded via BatchEpisodeDataset, matching TRAK scoring
3. Images come directly from the dataset's replay_buffer (or equivalent image dataset)
4. All config parameters are extracted from the training checkpoint
"""

import abc
import json
import os
import pathlib
import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import dill
import h5py
import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion_policy.common.error_util import (
    compute_demo_quality_scores,
    mean_of_mean_influence,
    pairwise_sample_to_trajectory_scores,
)
from diffusion_policy.common.sampler import SequenceSampler, create_indices
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset, EpisodeDataset


@dataclass
class SampleInfo:
    """Information about a single sample in the influence matrix."""

    global_idx: int  # Index in the influence matrix
    episode_idx: int  # Episode index (original demo index or rollout index)
    timestep: int  # Timestep within the episode (buffer_start_idx for demos)
    # For demos: indices into the replay buffer that this sample covers
    buffer_start_idx: int = 0
    buffer_end_idx: int = 0
    # The actual sequence this sample corresponds to
    sample_start_idx: int = 0  # Start index within the horizon (for padding)
    sample_end_idx: int = 0  # End index within the horizon


@dataclass
class EpisodeInfo:
    """Information about a single episode (rollout or demonstration)."""

    index: int  # Episode index (original demo index or rollout episode number)
    num_samples: int  # Number of samples in this episode (in the influence matrix)
    sample_start_idx: int  # First sample index in the influence matrix
    sample_end_idx: int  # Last sample index + 1 in the influence matrix
    success: Optional[bool] = None  # For rollouts: whether the episode succeeded
    # Raw episode length in the replay buffer (for demos)
    raw_length: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "num_samples": self.num_samples,
            "sample_start_idx": self.sample_start_idx,
            "sample_end_idx": self.sample_end_idx,
            "success": self.success,
            "raw_length": self.raw_length,
        }


@dataclass
class InfluenceData:
    """Container for influence visualization data.

    This class holds all data needed to visualize influences, with data loaded
    using the exact same code paths as train_trak_diffusion.py.
    """

    # Influence matrix: shape (num_rollout_samples, num_demo_samples)
    influence_matrix: np.ndarray

    # Training config from checkpoint
    cfg: DictConfig

    # Demo dataset (same as used in TRAK featurization)
    demo_dataset: Any  # The actual dataset object
    demo_episodes: List[EpisodeInfo] = field(default_factory=list)
    demo_sample_infos: List[SampleInfo] = field(default_factory=list)

    # Holdout dataset (if included in TRAK)
    holdout_dataset: Optional[Any] = None
    holdout_episodes: List[EpisodeInfo] = field(default_factory=list)
    holdout_sample_infos: List[SampleInfo] = field(default_factory=list)

    # Rollout dataset (same as used in TRAK scoring)
    rollout_dataset: Optional[EpisodeDataset] = None
    rollout_episodes: List[EpisodeInfo] = field(default_factory=list)
    rollout_sample_infos: List[SampleInfo] = field(default_factory=list)

    # Image dataset for visualization (if training used lowdim, this is the image equivalent)
    image_dataset: Optional[Any] = None

    # Paths
    eval_dir: Optional[pathlib.Path] = None
    train_dir: Optional[pathlib.Path] = None

    # TRAK experiment name (e.g. "default_trak_results-...") for locating InfEmbed files
    trak_exp_name: Optional[str] = None

    # Demo quality labels (for MH datasets): maps episode index -> quality tier string
    demo_quality_labels: Optional[Dict[int, str]] = None

    @property
    def num_rollout_samples(self) -> int:
        return self.influence_matrix.shape[0]

    @property
    def num_demo_samples(self) -> int:
        return self.influence_matrix.shape[1]

    @property
    def num_rollout_episodes(self) -> int:
        return len(self.rollout_episodes)

    @property
    def num_demo_episodes(self) -> int:
        return len(self.demo_episodes) + len(self.holdout_episodes)

    @property
    def all_demo_episodes(self) -> List[EpisodeInfo]:
        """All demo episodes including holdout."""
        return self.demo_episodes + self.holdout_episodes

    @property
    def all_demo_sample_infos(self) -> List[SampleInfo]:
        """All demo sample infos including holdout."""
        return self.demo_sample_infos + self.holdout_sample_infos

    @property
    def horizon(self) -> int:
        """Action prediction horizon."""
        if hasattr(self.cfg, "task") and hasattr(self.cfg.task, "dataset"):
            return self.cfg.task.dataset.get("horizon", 16)
        if hasattr(self.cfg, "horizon"):
            return self.cfg.horizon
        raise ValueError(
            "Missing 'horizon' in configuration (checked cfg.task.dataset and cfg root)"
        )

    @property
    def pad_before(self) -> int:
        """Padding before the sequence."""
        if hasattr(self.cfg, "task") and hasattr(self.cfg.task, "dataset"):
            return self.cfg.task.dataset.get("pad_before", 0)
        if hasattr(self.cfg, "pad_before"):
            return self.cfg.pad_before
        raise ValueError(
            "Missing 'pad_before' in configuration (checked cfg.task.dataset and cfg root)"
        )

    @property
    def pad_after(self) -> int:
        """Padding after the sequence."""
        if hasattr(self.cfg, "task") and hasattr(self.cfg.task, "dataset"):
            return self.cfg.task.dataset.get("pad_after", 0)
        if hasattr(self.cfg, "pad_after"):
            return self.cfg.pad_after
        raise ValueError(
            "Missing 'pad_after' in configuration (checked cfg.task.dataset and cfg root)"
        )

    @property
    def n_obs_steps(self) -> int:
        """Number of observation steps the policy uses."""
        # Try different possible locations for n_obs_steps
        if hasattr(self.cfg, "policy") and hasattr(self.cfg.policy, "n_obs_steps"):
            return self.cfg.policy.n_obs_steps
        if hasattr(self.cfg, "n_obs_steps"):
            return self.cfg.n_obs_steps
        if (
            hasattr(self.cfg, "task")
            and hasattr(self.cfg.task, "dataset")
            and hasattr(self.cfg.task.dataset, "n_obs_steps")
        ):
            return self.cfg.task.dataset.n_obs_steps

        raise ValueError(
            "Missing 'n_obs_steps' in configuration (checked cfg.policy, cfg root, and cfg.task.dataset)"
        )

    @property
    def n_action_steps(self) -> int:
        """Number of action steps executed per policy call.

        This is the key parameter for understanding the temporal scaling between
        rollout samples (policy calls) and demo samples (environment steps).
        Each rollout sample represents n_action_steps environment steps.
        """
        # Try different possible locations for n_action_steps
        if hasattr(self.cfg, "policy") and hasattr(self.cfg.policy, "n_action_steps"):
            return self.cfg.policy.n_action_steps
        if hasattr(self.cfg, "n_action_steps"):
            return self.cfg.n_action_steps
        if (
            hasattr(self.cfg, "task")
            and hasattr(self.cfg.task, "dataset")
            and hasattr(self.cfg.task.dataset, "n_action_steps")
        ):
            return self.cfg.task.dataset.n_action_steps

        # Default to 8 if not found (common value for diffusion policies)
        # Print warning
        import warnings

        warnings.warn(
            "Could not find 'n_action_steps' in configuration. "
            "Defaulting to 8 (common for diffusion policies). "
            "Checked cfg.policy.n_action_steps, cfg.n_action_steps, and cfg.task.dataset.n_action_steps"
        )
        return 8

    def get_available_image_keys(self) -> List[str]:
        """Get available image observation keys from the datasets.

        Returns:
            List of available image observation keys
        """
        # Try to get keys from demo dataset first (or image dataset if available)
        dataset = (
            self.image_dataset if self.image_dataset is not None else self.demo_dataset
        )

        if dataset is None:
            return []

        try:
            replay_buffer = dataset.replay_buffer
            available_keys = list(replay_buffer.keys())

            # Filter for image keys
            image_keys = [k for k in available_keys if "image" in k.lower()]
            return image_keys
        except Exception:
            return []

    def _apply_rollout_state_transform(self, obs_flat: np.ndarray) -> np.ndarray:
        """Apply task-specific transformations to rollout state observations.

        Args:
            obs_flat: Flattened observation array (may contain multiple timesteps)

        Returns:
            Transformed observation array
        """
        # Get task name from config
        task_name = self.cfg.get("task_name", "")

        # PushT: Convert angle from degrees to radians
        # State format: [agent_x, agent_y, block_x, block_y, block_angle]
        # With observation history, this repeats: [t0_state, t1_state, ...]
        if "pusht" in task_name.lower():
            # Determine base state dimension (5 for pusht without velocities)
            # The angle is always at position 4 within each timestep
            state_dim = 5  # Base pusht state dimension

            # Convert angles for each timestep in the observation history
            for i in range(0, len(obs_flat), state_dim):
                angle_idx = i + 4  # 5th element in each state (0-indexed)
                if angle_idx < len(obs_flat):
                    obs_flat[angle_idx] = np.deg2rad(obs_flat[angle_idx])

        return obs_flat

    def get_demo_sample_info(self, sample_idx: int) -> SampleInfo:
        """Get sample info for a demo sample index."""
        return self.all_demo_sample_infos[sample_idx]

    def get_rollout_sample_info(self, sample_idx: int) -> SampleInfo:
        """Get sample info for a rollout sample index."""
        return self.rollout_sample_infos[sample_idx]

    def get_demo_frame(
        self,
        sample_idx: int,
        obs_key: str = "agentview_image",
        timestep_in_horizon: int = 0,
    ) -> Optional[np.ndarray]:
        """Get an image frame for a demo sample.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)
            obs_key: Observation key for the image (e.g., "agentview_image")
            timestep_in_horizon: Which timestep within the horizon to get (0 = first obs)

        Returns:
            RGB image as numpy array (H, W, 3) uint8, or None if not available.
        """
        sample_info = self.get_demo_sample_info(sample_idx)

        # The image_dataset (if provided) shares the same replay_buffer structure
        # as the demo_dataset, so we can use buffer indices directly
        dataset = (
            self.image_dataset if self.image_dataset is not None else self.demo_dataset
        )

        if dataset is None:
            return None

        try:
            replay_buffer = dataset.replay_buffer

            # Calculate the frame index within the buffer
            frame_idx = sample_info.buffer_start_idx + timestep_in_horizon
            # Clamp to valid range
            frame_idx = min(frame_idx, sample_info.buffer_end_idx - 1)

            # Check available keys
            available_keys = list(replay_buffer.keys())

            # Try the requested key first
            if obs_key in available_keys:
                img = replay_buffer[obs_key][frame_idx]
            else:
                # Try to find any image key
                image_keys = [k for k in available_keys if "image" in k.lower()]
                if image_keys:
                    img = replay_buffer[image_keys[0]][frame_idx]
                else:
                    return None

            # Convert to proper format
            img = np.array(img)

            # Convert from (H, W, C) to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img
        except Exception as e:
            print(f"Error loading demo frame: {e}")
            import traceback

            traceback.print_exc()

        return None

    def get_demo_action_chunk(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the action chunk for a demo sample.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)

        Returns:
            Action chunk as numpy array (horizon, action_dim), or None if not available.
        """
        sample_info = self.get_demo_sample_info(sample_idx)

        try:
            # Determine which dataset to use
            if sample_idx < len(self.demo_sample_infos):
                dataset = self.demo_dataset
            else:
                dataset = self.holdout_dataset

            if dataset is None:
                return None

            replay_buffer = dataset.replay_buffer

            # Get actions from buffer
            actions = replay_buffer["action"][
                sample_info.buffer_start_idx : sample_info.buffer_end_idx
            ]

            # Pad if necessary (same logic as SequenceSampler)
            if (
                sample_info.sample_start_idx > 0
                or sample_info.sample_end_idx < self.horizon
            ):
                padded_actions = np.zeros(
                    (self.horizon,) + actions.shape[1:], dtype=actions.dtype
                )
                if sample_info.sample_start_idx > 0:
                    padded_actions[: sample_info.sample_start_idx] = actions[0]
                if sample_info.sample_end_idx < self.horizon:
                    padded_actions[sample_info.sample_end_idx :] = actions[-1]
                padded_actions[
                    sample_info.sample_start_idx : sample_info.sample_end_idx
                ] = actions
                return padded_actions

            return actions
        except Exception as e:
            print(f"Error loading demo action chunk: {e}")

        return None

    def get_demo_action(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the action for a demo sample (single timestep).

        Returns the first action of the chunk at this sample index, for consistency
        with get_rollout_action(sample_idx) which returns a single timestep.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)

        Returns:
            Action as numpy array (action_dim,), or None if not available.
        """
        chunk = self.get_demo_action_chunk(sample_idx)
        if chunk is None or len(chunk) == 0:
            return None
        return chunk[0]

    def get_rollout_frame(
        self,
        sample_idx: int,
        obs_key: str = "img",
    ) -> Optional[np.ndarray]:
        """Get an image frame for a rollout sample.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)
            obs_key: Observation key for the image in the pickle file

        Returns:
            RGB image as numpy array (H, W, 3) uint8, or None if not available.
        """
        sample_info = self.get_rollout_sample_info(sample_idx)
        episode = self.rollout_episodes[sample_info.episode_idx]

        # Load from pickle file
        if self.eval_dir is None:
            return None

        episodes_dir = self.eval_dir / "episodes"
        episode_files = sorted(episodes_dir.glob("ep*.pkl"))

        if sample_info.episode_idx >= len(episode_files):
            return None

        pickle_path = episode_files[sample_info.episode_idx]

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            if obs_key not in data:
                return None

            img_series = data[obs_key]
            timestep = sample_info.timestep

            if timestep < 0 or timestep >= len(img_series):
                return None

            img = (
                img_series.iloc[timestep]
                if hasattr(img_series, "iloc")
                else img_series[timestep]
            )
            return img
        except Exception as e:
            print(f"Error loading rollout frame: {e}")

        return None

    def get_rollout_action(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the action for a rollout sample.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)

        Returns:
            Action as numpy array, or None if not available.
        """
        sample_info = self.get_rollout_sample_info(sample_idx)

        if self.eval_dir is None:
            return None

        episodes_dir = self.eval_dir / "episodes"
        episode_files = sorted(episodes_dir.glob("ep*.pkl"))

        if sample_info.episode_idx >= len(episode_files):
            return None

        pickle_path = episode_files[sample_info.episode_idx]

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            if "action" not in data:
                return None

            action_series = data["action"]
            timestep = sample_info.timestep

            if timestep < 0 or timestep >= len(action_series):
                return None

            action = (
                action_series.iloc[timestep]
                if hasattr(action_series, "iloc")
                else action_series[timestep]
            )
            return action
        except Exception as e:
            print(f"Error loading rollout action: {e}")

        return None

    def get_rollout_action_chunk(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the action chunk for a rollout sample.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)

        Returns:
            Action chunk as numpy array (horizon, action_dim), or None if not available.
        """
        sample_info = self.get_rollout_sample_info(sample_idx)

        if self.eval_dir is None:
            return None

        episodes_dir = self.eval_dir / "episodes"
        episode_files = sorted(episodes_dir.glob("ep*.pkl"))

        if sample_info.episode_idx >= len(episode_files):
            return None

        pickle_path = episode_files[sample_info.episode_idx]

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            if "action" not in data:
                return None

            action_series = data["action"]
            timestep = sample_info.timestep

            if timestep < 0 or timestep >= len(action_series):
                return None

            # The action at each timestep is already the full predicted action chunk
            # (stored as action_pred from the policy's predict_action output)
            action_chunk = (
                action_series.iloc[timestep]
                if hasattr(action_series, "iloc")
                else action_series[timestep]
            )

            # Convert to numpy array
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)

            return action_chunk
        except Exception as e:
            print(f"Error loading rollout action chunk: {e}")
            import traceback

            traceback.print_exc()

        return None

    def get_top_influences_for_sample(
        self, rollout_sample_idx: int, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top-k influential demonstration samples for a rollout sample.

        Args:
            rollout_sample_idx: Index of the rollout sample in the influence matrix.
            top_k: Number of top influences to return.

        Returns:
            List of dicts with keys: demo_sample_idx, influence_score,
            demo_episode_idx, demo_timestep, sample_info
        """
        influences = self.influence_matrix[rollout_sample_idx]
        top_indices = np.argsort(influences)[::-1][:top_k]

        results = []
        for demo_sample_idx in top_indices:
            if demo_sample_idx >= len(self.all_demo_sample_infos):
                continue

            score = float(influences[demo_sample_idx])
            sample_info = self.get_demo_sample_info(demo_sample_idx)

            # Find the episode
            episode = None
            for ep in self.all_demo_episodes:
                if ep.sample_start_idx <= demo_sample_idx < ep.sample_end_idx:
                    episode = ep
                    break

            results.append(
                {
                    "demo_sample_idx": int(demo_sample_idx),
                    "influence_score": score,
                    "demo_episode_idx": sample_info.episode_idx,
                    "demo_timestep": sample_info.timestep,
                    "buffer_start_idx": sample_info.buffer_start_idx,
                    "buffer_end_idx": sample_info.buffer_end_idx,
                    "sample_info": sample_info,
                    "episode": episode,
                }
            )

        return results

    def get_samples_for_rollout_episode(self, episode_idx: int) -> np.ndarray:
        """Get all sample indices for a rollout episode."""
        ep = self.rollout_episodes[episode_idx]
        return np.arange(ep.sample_start_idx, ep.sample_end_idx)

    def get_samples_for_demo_episode(self, episode_idx: int) -> np.ndarray:
        """Get all sample indices for a demonstration episode."""
        ep = self.all_demo_episodes[episode_idx]
        return np.arange(ep.sample_start_idx, ep.sample_end_idx)

    def compute_performance_influence(
        self,
        metric: str = "net",
        aggr_fn: Any = mean_of_mean_influence,
    ) -> np.ndarray:
        """Compute performance influence for each demonstration trajectory.

        Performance influence is defined as the weighted average of influences
        for all rollouts, where the weight is:
        - +1 for a successful rollout (positive contribution to quality)
        - -1 for a failed rollout (negative contribution to quality)

        This uses the same infrastructure as error_util.py for consistency.

        Args:
            metric: Quality metric to compute ("net", "succ", or "fail")
                - "net": Success influence - failure influence
                - "succ": Only success influence (sum over successful rollouts)
                - "fail": Only failure influence (negative sum over failed rollouts)
            aggr_fn: Aggregation function for converting sample-level to trajectory-level
                     scores. Should have signature (scores_ij: np.ndarray, is_success: bool) -> float

        Returns:
            Array of shape (num_demo_episodes,) containing performance influence
            for each demonstration trajectory.
        """
        # Build episode indices and lengths
        test_ep_lens = np.array([ep.num_samples for ep in self.rollout_episodes])
        test_ep_idxs = [
            np.arange(ep.sample_start_idx, ep.sample_end_idx)
            for ep in self.rollout_episodes
        ]

        train_ep_lens = np.array([ep.num_samples for ep in self.all_demo_episodes])
        train_ep_idxs = [
            np.arange(ep.sample_start_idx, ep.sample_end_idx)
            for ep in self.all_demo_episodes
        ]

        success_mask = np.array(
            [ep.success for ep in self.rollout_episodes], dtype=bool
        )

        # Convert sample-level influences to trajectory-level influences
        # Shape: (num_rollout_episodes, num_demo_episodes)
        trajectory_influences = pairwise_sample_to_trajectory_scores(
            pairwise_sample_scores=self.influence_matrix,
            num_test_eps=len(self.rollout_episodes),
            num_train_eps=len(self.all_demo_episodes),
            test_ep_idxs=test_ep_idxs,
            train_ep_idxs=train_ep_idxs,
            test_ep_lens=test_ep_lens,
            train_ep_lens=train_ep_lens,
            success_mask=success_mask,
            aggr_fn=aggr_fn,
            return_dtype=np.float32,
        )

        # Compute performance influence (demo quality scores)
        # This weights by success (+1) and failure (-1)
        performance_influence = compute_demo_quality_scores(
            traj_scores=trajectory_influences,
            success_mask=success_mask,
            metric=metric,
        )

        if performance_influence is None:
            # If no successes or failures, return zeros
            performance_influence = np.zeros(
                len(self.all_demo_episodes), dtype=np.float32
            )

        return performance_influence

    def get_demo_obs(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the observation vector for a demo sample.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)

        Returns:
            Observation as numpy array (flattened if necessary), or None if not available.
        """
        sample_info = self.get_demo_sample_info(sample_idx)

        try:
            # Determine which dataset to use and adjust index
            if sample_idx < len(self.demo_sample_infos):
                dataset = self.demo_dataset
                local_idx = sample_idx
            else:
                dataset = self.holdout_dataset
                # Adjust index for holdout dataset
                local_idx = sample_idx - len(self.demo_sample_infos)

            if dataset is None:
                return None

            # Use the dataset's sampler to get the data properly
            # This ensures we get data in the same format as used for training
            if local_idx >= len(dataset.sampler):
                print(
                    f"Index {local_idx} out of range for sampler with {len(dataset.sampler)} samples"
                )
                return None

            data = dataset.sampler.sample_sequence(local_idx)

            # Handle different observation key names (obs, state, keypoint)
            obs_key = None
            if "obs" in data:
                obs_key = "obs"
            elif "state" in data:
                obs_key = "state"
            elif "keypoint" in data:
                obs_key = "keypoint"

            if obs_key is not None:
                obs = data[obs_key]

                # Extract only n_obs_steps if the sequence is longer
                # The lowdim dataset doesn't use key_first_k, so it returns full horizon
                # We need to extract just the observation history that the policy uses
                n_obs_steps = self.n_obs_steps
                if len(obs.shape) > 1 and obs.shape[0] > n_obs_steps:
                    obs = obs[:n_obs_steps]  # Take first n_obs_steps timesteps

                # Flatten the observation history
                return obs.flatten()

            return None
        except Exception as e:
            print(
                f"Error loading demo observation (idx={sample_idx}, local={local_idx}): {e}"
            )
            import traceback

            traceback.print_exc()

        return None

    def get_demo_state_action(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the concatenated state-action vector for a demo sample.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)

        Returns:
            Concatenated [obs, action] vector, or None if not available.
        """
        try:
            # Determine which dataset to use and adjust index
            if sample_idx < len(self.demo_sample_infos):
                dataset = self.demo_dataset
                local_idx = sample_idx
            else:
                dataset = self.holdout_dataset
                # Adjust index for holdout dataset
                local_idx = sample_idx - len(self.demo_sample_infos)

            if dataset is None:
                return None

            # Use the dataset's sampler to get the data properly
            if local_idx >= len(dataset.sampler):
                print(
                    f"Index {local_idx} out of range for sampler with {len(dataset.sampler)} samples"
                )
                return None

            data = dataset.sampler.sample_sequence(local_idx)

            # Handle different observation key names (obs, state, keypoint)
            obs_key = None
            if "obs" in data:
                obs_key = "obs"
            elif "state" in data:
                obs_key = "state"
            elif "keypoint" in data:
                obs_key = "keypoint"

            if obs_key is not None and "action" in data:
                obs = data[obs_key]
                action = data["action"]

                # Take first n_obs_steps observations and flatten
                # This is what the policy actually uses as input
                n_obs_steps = self.n_obs_steps
                if len(obs.shape) > 1:
                    obs = obs[:n_obs_steps]

                # Use the full action chunk as stored
                # This represents the full prediction horizon from the policy
                # (Note: n_action_steps is the execution horizon, but we want the full prediction)

                # Flatten and concatenate
                obs_flat = obs.flatten()
                action_flat = action.flatten()
                state_action = np.concatenate([obs_flat, action_flat])

                return state_action
            else:
                pass
                return None
        except Exception as e:
            print(
                f"Error loading demo state-action (idx={sample_idx}, local={local_idx}): {e}"
            )
            import traceback

            traceback.print_exc()

        return None

    def get_rollout_obs(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the observation vector for a rollout sample.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)

        Returns:
            Observation as numpy array (flattened if necessary), or None if not available.
        """
        sample_info = self.get_rollout_sample_info(sample_idx)

        if self.eval_dir is None:
            return None

        episodes_dir = self.eval_dir / "episodes"
        episode_files = sorted(episodes_dir.glob("ep*.pkl"))

        if sample_info.episode_idx >= len(episode_files):
            return None

        pickle_path = episode_files[sample_info.episode_idx]

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            # Try to get observation state
            if "obs" in data:
                obs_series = data["obs"]
            elif "state" in data:
                obs_series = data["state"]
            else:
                return None

            timestep = sample_info.timestep

            if timestep < 0 or timestep >= len(obs_series):
                return None

            obs = (
                obs_series.iloc[timestep]
                if hasattr(obs_series, "iloc")
                else obs_series[timestep]
            )

            # Convert to numpy array and make a copy
            if not isinstance(obs, np.ndarray):
                obs = np.array([obs])
            else:
                obs = obs.copy()

            # Flatten (preserve full observation history if present)
            # This matches what the policy actually uses during training/inference
            obs_flat = obs.flatten()

            # Apply task-specific transformations (e.g., unit conversions)
            obs_flat = self._apply_rollout_state_transform(obs_flat)

            return obs_flat
        except Exception as e:
            print(f"Error loading rollout observation: {e}")

        return None

    def get_rollout_state_action(self, sample_idx: int) -> Optional[np.ndarray]:
        """Get the concatenated state-action vector for a rollout sample.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)

        Returns:
            Concatenated [obs, action] vector, or None if not available.
        """
        obs = self.get_rollout_obs(sample_idx)
        action = self.get_rollout_action(sample_idx)

        if obs is None or action is None:
            return None

        # Use the full action chunk as stored
        # This represents the full prediction horizon from the policy
        # (Note: n_action_steps is the execution horizon, but we want the full prediction)

        # Flatten action and concatenate with observation
        action_flat = (
            action.flatten()
            if isinstance(action, np.ndarray)
            else np.array([action]).flatten()
        )
        state_action = np.concatenate([obs, action_flat])

        return state_action


def load_checkpoint_config(checkpoint_path: pathlib.Path) -> Tuple[DictConfig, dict]:
    """Load config and state dict from a checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file

    Returns:
        Tuple of (config, payload)
    """
    payload = torch.load(open(str(checkpoint_path), "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    return cfg, payload


def get_checkpoint_path(
    train_dir: pathlib.Path,
    train_ckpt: str = "latest",
) -> pathlib.Path:
    """Get the checkpoint path from a training directory.

    Args:
        train_dir: Path to training output directory
        train_ckpt: Checkpoint name ("latest", "best", or epoch number)

    Returns:
        Path to the checkpoint file
    """
    from diffusion_policy.common.trak_util import (
        get_best_checkpoint,
        get_index_checkpoint,
    )

    checkpoint_dir = train_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())

    if train_ckpt == "latest":
        return checkpoint_dir / "latest.ckpt"
    elif train_ckpt == "best":
        return get_best_checkpoint(checkpoints)
    elif train_ckpt.isdigit():
        return get_index_checkpoint(checkpoints, int(train_ckpt))
    else:
        return checkpoint_dir / f"{train_ckpt}.ckpt"


def load_demo_quality_labels(
    dataset_path: str,
    quality_label_tiers: Optional[List[str]] = None,
) -> Optional[Dict[int, str]]:
    """Load demo quality labels from HDF5 mask groups (for MH datasets).

    Loads quality tier masks specified in the config. If no tiers are specified,
    returns None (quality labels disabled).

    Args:
        dataset_path: Path to the HDF5 dataset file.
        quality_label_tiers: List of quality tier names to load (in priority order).
                           If None or empty, returns None.

    Returns:
        Dictionary mapping demo index -> quality tier string,
        or None if no quality tiers specified or masks not present.
    """
    # If no quality tiers specified, skip loading
    if quality_label_tiers is None or len(quality_label_tiers) == 0:
        return None

    quality_labels = {}

    try:
        with h5py.File(dataset_path, "r") as f:
            # Check if mask group exists
            if "mask" not in f:
                return None

            mask_group = f["mask"]

            # Load only the specified quality tier masks (in priority order)
            found_any = False
            for tier in quality_label_tiers:
                if tier in mask_group:
                    found_any = True
                    demo_keys = mask_group[tier][:]
                    # Decode bytes to strings if necessary
                    if len(demo_keys) > 0 and isinstance(demo_keys[0], bytes):
                        demo_keys = [k.decode("utf-8") for k in demo_keys]

                    # Extract demo indices (e.g., "demo_0" -> 0)
                    for key in demo_keys:
                        if isinstance(key, str) and key.startswith("demo_"):
                            demo_idx = int(key.split("_")[1])
                            # Only assign if not already labeled (earlier tiers take precedence)
                            if demo_idx not in quality_labels:
                                quality_labels[demo_idx] = tier

            if not found_any:
                return None

            print(f"  Loaded quality labels for {len(quality_labels)} demos")
            tier_counts = {}
            for tier in quality_label_tiers:
                count = sum(1 for v in quality_labels.values() if v == tier)
                if count > 0:
                    tier_counts[tier] = count
            print(f"  Quality distribution: {tier_counts}")

            return quality_labels

    except Exception as e:
        print(f"  Could not load quality labels: {e}")
        return None


def build_demo_sample_infos(
    dataset: Any,
    cfg: DictConfig,
    sample_offset: int = 0,
) -> Tuple[List[EpisodeInfo], List[SampleInfo]]:
    """Build episode and sample info from a demo dataset.

    This replicates the exact indexing used by SequenceSampler during TRAK featurization.

    Args:
        dataset: The demo dataset (RobomimicReplayImageDataset or similar)
        cfg: Training config
        sample_offset: Offset to add to global sample indices (for combining train + holdout)

    Returns:
        Tuple of (episode_infos, sample_infos)
    """
    replay_buffer = dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    train_mask = dataset.train_mask

    horizon = cfg.task.dataset.horizon
    pad_before = cfg.task.dataset.pad_before
    pad_after = cfg.task.dataset.pad_after

    # Use the exact same create_indices function as SequenceSampler
    indices = create_indices(
        episode_ends=episode_ends,
        sequence_length=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        episode_mask=train_mask,
        debug=True,
    )

    # Build episode and sample infos
    episodes = []
    sample_infos = []

    # Track which episodes are included
    included_episodes = np.where(train_mask)[0]
    episode_sample_counts = {}

    # Count samples per episode
    for idx, (buffer_start, buffer_end, sample_start, sample_end) in enumerate(indices):
        # Determine which episode this sample belongs to
        ep_idx = np.searchsorted(episode_ends, buffer_start, side="right")
        orig_ep_idx = (
            included_episodes[
                np.searchsorted(included_episodes, ep_idx, side="right") - 1
            ]
            if len(included_episodes) > 0
            else ep_idx
        )

        # Find actual episode index
        for i, orig_idx in enumerate(included_episodes):
            ep_start = 0 if orig_idx == 0 else episode_ends[orig_idx - 1]
            ep_end = episode_ends[orig_idx]
            if ep_start <= buffer_start < ep_end:
                orig_ep_idx = orig_idx
                break

        if orig_ep_idx not in episode_sample_counts:
            episode_sample_counts[orig_ep_idx] = 0
        episode_sample_counts[orig_ep_idx] += 1

        # The "timestep" for visualization is buffer_start_idx relative to episode start
        ep_start = 0 if orig_ep_idx == 0 else episode_ends[orig_ep_idx - 1]
        timestep = buffer_start - ep_start

        sample_infos.append(
            SampleInfo(
                global_idx=idx + sample_offset,
                episode_idx=int(orig_ep_idx),
                timestep=timestep,
                buffer_start_idx=int(buffer_start),
                buffer_end_idx=int(buffer_end),
                sample_start_idx=int(sample_start),
                sample_end_idx=int(sample_end),
            )
        )

    # Build episode infos
    current_sample_idx = sample_offset
    for orig_ep_idx in included_episodes:
        if orig_ep_idx not in episode_sample_counts:
            continue

        num_samples = episode_sample_counts[orig_ep_idx]
        ep_start = 0 if orig_ep_idx == 0 else episode_ends[orig_ep_idx - 1]
        ep_end = episode_ends[orig_ep_idx]

        episodes.append(
            EpisodeInfo(
                index=int(orig_ep_idx),
                num_samples=num_samples,
                sample_start_idx=current_sample_idx,
                sample_end_idx=current_sample_idx + num_samples,
                raw_length=int(ep_end - ep_start),
            )
        )
        current_sample_idx += num_samples

    return episodes, sample_infos


def load_rollout_success_stats(eval_dir: pathlib.Path) -> Tuple[int, int]:
    """Load rollout success counts from eval_dir without loading full influence data.

    Args:
        eval_dir: Path to evaluation directory (contains episodes/metadata.yaml).

    Returns:
        Tuple of (n_success, n_total). Episodes with success=None are counted as failure.
    """
    episodes_dir = eval_dir / "episodes"
    metadata_path = episodes_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Rollout metadata not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    episode_successes = metadata.get("episode_successes", [None] * len(metadata.get("episode_lengths", [])))
    n_total = len(episode_successes)
    n_success = sum(1 for s in episode_successes if s is True)
    return n_success, n_total


def get_eval_dir_for_seed(eval_dir: str, seed: str, reference_seed: str) -> str:
    """Derive eval_dir path for a given seed from the reference eval_dir.

    Assumes the path has a segment ending with _reference_seed (e.g. square_mh_0).
    Replaces that segment so it ends with _seed (e.g. square_mh_1).

    Args:
        eval_dir: Path for the reference seed (e.g. .../square_mh_0/latest).
        seed: Target seed string (e.g. "1").
        reference_seed: Seed string that eval_dir is for (e.g. "0").

    Returns:
        Path with the seed segment replaced for the target seed.
    """
    if seed == reference_seed:
        return eval_dir
    path = eval_dir.rstrip("/")
    parts = path.split("/")
    suffix = "_" + reference_seed
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(suffix):
            parts[i] = parts[i][: -len(suffix)] + "_" + seed
            return "/".join(parts)
    return eval_dir


def get_train_dir_for_seed(train_dir: str, seed: str, reference_seed: str) -> str:
    """Derive train_dir path for a given seed from the reference train_dir.

    Same logic as get_eval_dir_for_seed but for train_dir (e.g. .../square_mh_0 -> .../square_mh_1).
    """
    if seed == reference_seed:
        return train_dir
    path = train_dir.rstrip("/")
    parts = path.split("/")
    suffix = "_" + reference_seed
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(suffix):
            parts[i] = parts[i][: -len(suffix)] + "_" + seed
            return "/".join(parts)
    return train_dir


def build_rollout_sample_infos(
    eval_dir: pathlib.Path,
) -> Tuple[List[EpisodeInfo], List[SampleInfo]]:
    """Build episode and sample info from rollout episodes.

    This matches the exact loading used by BatchEpisodeDataset during TRAK scoring.

    Args:
        eval_dir: Path to evaluation directory

    Returns:
        Tuple of (episode_infos, sample_infos)
    """
    episodes_dir = eval_dir / "episodes"
    metadata_path = episodes_dir / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Rollout metadata not found at {metadata_path}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    episode_lengths = metadata["episode_lengths"]
    episode_successes = metadata.get("episode_successes", [None] * len(episode_lengths))

    episodes = []
    sample_infos = []
    current_sample_idx = 0

    for ep_idx, (ep_len, ep_success) in enumerate(
        zip(episode_lengths, episode_successes)
    ):
        episodes.append(
            EpisodeInfo(
                index=ep_idx,
                num_samples=ep_len,
                sample_start_idx=current_sample_idx,
                sample_end_idx=current_sample_idx + ep_len,
                success=ep_success,
                raw_length=ep_len,
            )
        )

        for t in range(ep_len):
            sample_infos.append(
                SampleInfo(
                    global_idx=current_sample_idx + t,
                    episode_idx=ep_idx,
                    timestep=t,
                    buffer_start_idx=t,
                    buffer_end_idx=t + 1,
                    sample_start_idx=0,
                    sample_end_idx=1,
                )
            )

        current_sample_idx += ep_len

    return episodes, sample_infos


def load_influence_matrix(
    eval_dir: pathlib.Path,
    exp_name: str,
) -> Tuple[np.ndarray, int, int]:
    """Load the TRAK influence matrix.

    Args:
        eval_dir: Directory containing TRAK results.
        exp_name: Full experiment name (e.g., "default_trak_results-proj_dim=4000-...")

    Returns:
        Tuple of (influence_matrix, train_set_size, test_set_size).
        Influence matrix has shape (test_set_size, train_set_size).
    """
    exp_dir = eval_dir / exp_name

    metadata_path = exp_dir / "metadata.json"
    experiments_path = exp_dir / "experiments.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"TRAK metadata not found at {metadata_path}")
    if not experiments_path.exists():
        raise FileNotFoundError(f"TRAK experiments not found at {experiments_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(experiments_path) as f:
        experiments = json.load(f)

    train_set_size = metadata["train set size"]
    test_set_size = experiments["all_episodes"]["num_targets"]

    scores_path = exp_dir / "scores" / "all_episodes.mmap"
    if not scores_path.exists():
        raise FileNotFoundError(f"TRAK scores not found at {scores_path}")

    from numpy.lib.format import open_memmap

    scores = np.array(
        open_memmap(
            filename=str(scores_path),
            shape=(train_set_size, test_set_size),
            dtype=np.float32,
            mode="r",
        )
    )

    # Transpose to (test_set_size, train_set_size) = (rollouts, demos)
    return scores.T.astype(np.float32), train_set_size, test_set_size


def find_trak_experiment(eval_dir: pathlib.Path, exp_date: str = "default") -> str:
    """Find the TRAK experiment directory name.

    Args:
        eval_dir: Evaluation directory
        exp_date: Experiment date prefix

    Returns:
        Full experiment name
    """
    # Look for directories matching the pattern
    pattern = f"{exp_date}_trak_results-*"
    matches = list(eval_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No TRAK experiment found matching {pattern} in {eval_dir}"
        )

    if len(matches) > 1:
        # Return the most recent (by name)
        matches = sorted(matches)

    return matches[-1].name


def create_image_dataset_from_config(
    cfg: DictConfig,
    dataset_path: Optional[str] = None,
) -> Optional[Any]:
    """Create an image dataset from a lowdim config.

    If the training config used lowdim observations, this creates the equivalent
    image dataset for visualization purposes.

    Args:
        cfg: Training config
        dataset_path: Optional override for dataset path

    Returns:
        Image dataset or None if already using images
    """
    # Check if already using images
    dataset_cfg = cfg.task.dataset
    shape_meta = dataset_cfg.get("shape_meta", {})
    obs_meta = shape_meta.get("obs", {})

    # Check if any observation is an image
    has_image = any(attr.get("type") == "rgb" for attr in obs_meta.values())

    if has_image:
        # Already using images, no need for separate image dataset
        return None

    # Try to create image dataset
    # This assumes the dataset path follows the pattern: .../low_dim_abs.hdf5 -> .../image_abs.hdf5
    if dataset_path is None:
        dataset_path = dataset_cfg.get("dataset_path", "")

    if "low_dim" in str(dataset_path):
        image_path = str(dataset_path).replace("low_dim", "image")
        if pathlib.Path(image_path).exists():
            # Create a minimal config for image dataset
            image_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
            image_cfg["dataset_path"] = image_path

            # Update shape_meta for images
            # This is a simplification - you may need to adjust based on your data
            if "agentview_image" not in image_cfg.get("shape_meta", {}).get("obs", {}):
                # Add image observation
                if "shape_meta" not in image_cfg:
                    image_cfg["shape_meta"] = {
                        "obs": {},
                        "action": shape_meta.get("action", {}),
                    }
                image_cfg["shape_meta"]["obs"]["agentview_image"] = {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                }

            try:
                # Try to instantiate
                image_dataset = hydra.utils.instantiate(OmegaConf.create(image_cfg))
                return image_dataset
            except Exception as e:
                print(f"Could not create image dataset: {e}")

    return None


class BaseTaskLoader(ABC):
    """Abstract base class for task-specific data loading logic."""

    def __init__(self, task_name: str):
        self.task_name = task_name

    @abstractmethod
    def can_handle(self, cfg: DictConfig) -> bool:
        """Check if this loader can handle the given config."""
        pass

    @abstractmethod
    def fix_config(self, cfg: DictConfig, train_dir: pathlib.Path) -> None:
        """Fix the config (e.g., dataset paths) to ensure it can be loaded."""
        pass

    def get_cupid_root(self, train_dir: pathlib.Path) -> pathlib.Path:
        """Shared helper to find the 'cupid' project root."""
        cupid_root = train_dir
        while cupid_root.name != "cupid" and cupid_root.parent != cupid_root:
            cupid_root = cupid_root.parent
        return cupid_root


class PushTTaskLoader(BaseTaskLoader):
    """Task loader for PushT tasks using Zarr datasets."""

    def can_handle(self, cfg: DictConfig) -> bool:
        task_name = cfg.get("task_name", "").lower()
        return "pusht" in task_name

    def fix_config(self, cfg: DictConfig, train_dir: pathlib.Path) -> None:
        zarr_path_value = cfg.task.dataset.get("zarr_path", "")
        needs_fix = not zarr_path_value or str(zarr_path_value).strip() == ""
        if (
            not needs_fix
            and not pathlib.Path(zarr_path_value).is_absolute()
            and not pathlib.Path(zarr_path_value).exists()
        ):
            needs_fix = True

        if needs_fix:
            cupid_root = self.get_cupid_root(train_dir)
            inferred_path = cupid_root / "data/pusht/pusht_cchi_v7_replay.zarr"
            if inferred_path.exists():
                relative_path = "data/pusht/pusht_cchi_v7_replay.zarr"
                print(f"  ✓ Fixed PushT dataset path (using relative): {relative_path}")
                os.chdir(str(cupid_root))
                cfg.task.dataset.zarr_path = relative_path
            else:
                raise ValueError(
                    f"Could not find PushT dataset at {inferred_path}. Please set zarr_path manually."
                )


class RobomimicTaskLoader(BaseTaskLoader):
    """Task loader for Robomimic tasks using HDF5 datasets."""

    def can_handle(self, cfg: DictConfig) -> bool:
        task_name = cfg.get("task_name", "").lower()
        robomimic_tasks = ["transport", "lift", "can", "square", "tool_hang"]
        return any(t in task_name for t in robomimic_tasks)

    def fix_config(self, cfg: DictConfig, train_dir: pathlib.Path) -> None:
        # Robomimic datasets use 'dataset_path' instead of 'zarr_path'
        dataset_path_value = cfg.task.dataset.get("dataset_path", "")
        if not dataset_path_value:
            # Check if it's at the task level instead of dataset level
            dataset_path_value = cfg.task.get("dataset_path", "")

        needs_fix = not dataset_path_value or str(dataset_path_value).strip() == ""
        if (
            not needs_fix
            and not pathlib.Path(dataset_path_value).is_absolute()
            and not pathlib.Path(dataset_path_value).exists()
        ):
            needs_fix = True

        if needs_fix:
            cupid_root = self.get_cupid_root(train_dir)
            task_name = cfg.get("task_name", "").lower()

            # Infer task type and data type (mh, ph, etc)
            robomimic_tasks = ["transport", "lift", "can", "square", "tool_hang"]
            matched_task = next((t for t in robomimic_tasks if t in task_name), None)

            if not matched_task:
                raise ValueError(f"Could not infer Robomimic task from '{task_name}'")

            # Try to infer if it's mh or ph (default to mh)
            data_type = "mh"
            if "ph" in task_name:
                data_type = "ph"

            # Inferred path: data/robomimic/datasets/{task}/{data_type}/low_dim_abs.hdf5
            # Adjust based on whether 'lowdim' or 'image' is in task_name
            filename = "low_dim_abs.hdf5"
            if "image" in task_name:
                filename = "image_abs.hdf5"

            relative_path = (
                f"data/robomimic/datasets/{matched_task}/{data_type}/{filename}"
            )
            inferred_path = cupid_root / relative_path

            if inferred_path.exists():
                print(
                    f"  ✓ Fixed '{matched_task}' dataset path (using relative): {relative_path}"
                )
                os.chdir(str(cupid_root))
                # Update both locations if they exist
                if "dataset_path" in cfg.task.dataset:
                    cfg.task.dataset.dataset_path = relative_path
                if "dataset_path" in cfg.task:
                    cfg.task.dataset_path = relative_path
            else:
                raise ValueError(
                    f"Could not find Robomimic dataset at {inferred_path}. Please set dataset_path manually."
                )


def get_task_loaders() -> List[BaseTaskLoader]:
    """Get all available task loaders."""
    return [
        PushTTaskLoader("PushT"),
        RobomimicTaskLoader("Robomimic"),
    ]


def load_influence_data(
    eval_dir: str,
    train_dir: str,
    train_ckpt: str = "latest",
    exp_date: str = "default",
    include_holdout: bool = True,
    image_dataset_path: Optional[str] = None,
    lazy_load_images: bool = True,
    quality_labels: Optional[List[str]] = None,
) -> InfluenceData:
    """Load all data needed for the influence visualizer.

    This function loads data using the EXACT same code paths as train_trak_diffusion.py:
    1. Demo dataset via hydra.utils.instantiate(cfg.task.dataset)
    2. Rollout dataset via BatchEpisodeDataset
    3. Config parameters from the checkpoint

    Args:
        eval_dir: Path to evaluation output directory (contains TRAK results and rollout episodes)
        train_dir: Path to training output directory (contains checkpoints/)
        train_ckpt: Checkpoint name ("latest", "best", or epoch number)
        exp_date: Experiment date prefix for TRAK results
        include_holdout: Whether to include holdout episodes (as TRAK does)
        image_dataset_path: Optional path to image dataset (if training used lowdim)
        lazy_load_images: If True, use lazy HDF5 loading (recommended for large datasets)
        quality_labels: List of quality tier mask names to load from the HDF5 dataset

    Returns:
        InfluenceData object with all loaded data
    """
    eval_dir = pathlib.Path(eval_dir)
    train_dir = pathlib.Path(train_dir)

    # Load checkpoint and config
    checkpoint_path = get_checkpoint_path(train_dir, train_ckpt)
    cfg, payload = load_checkpoint_config(checkpoint_path)

    print(f"Loaded config from {checkpoint_path}")
    print(f"  horizon: {cfg.task.dataset.horizon}")
    print(f"  pad_before: {cfg.task.dataset.pad_before}")
    print(f"  pad_after: {cfg.task.dataset.pad_after}")

    # WORKAROUND: Fix dataset paths in checkpoint config
    task_name = cfg.get("task_name", "")
    print(f"  Task name from config: '{task_name}'")

    loaders = get_task_loaders()
    matched_loader = next((l for l in loaders if l.can_handle(cfg)), None)

    if matched_loader:
        print(f"  Using {matched_loader.__class__.__name__} to fix config")
        matched_loader.fix_config(cfg, train_dir)
    else:
        print(f"  Warning: No specialized TaskLoader found for '{task_name}'")

    # Load demo dataset using exact same instantiation as train_trak_diffusion.py
    print("Loading demo dataset (same as TRAK featurization)...")
    demo_dataset = hydra.utils.instantiate(cfg.task.dataset)
    demo_set_size = len(demo_dataset)
    print(f"  Demo dataset size: {demo_set_size}")

    # Build demo sample infos
    demo_episodes, demo_sample_infos = build_demo_sample_infos(
        demo_dataset, cfg, sample_offset=0
    )
    print(f"  Demo episodes: {len(demo_episodes)}")
    print(f"  Demo samples: {len(demo_sample_infos)}")

    # Load holdout dataset if requested
    holdout_dataset = None
    holdout_episodes = []
    holdout_sample_infos = []
    holdout_set_size = 0

    if include_holdout:
        holdout_dataset = demo_dataset.get_holdout_dataset()
        if holdout_dataset is not None and len(holdout_dataset) > 0:
            holdout_set_size = len(holdout_dataset)
            holdout_episodes, holdout_sample_infos = build_demo_sample_infos(
                holdout_dataset, cfg, sample_offset=len(demo_sample_infos)
            )
            print(f"  Holdout dataset size: {holdout_set_size}")
            print(f"  Holdout episodes: {len(holdout_episodes)}")
            print(f"  Holdout samples: {len(holdout_sample_infos)}")

    # Find and load influence matrix
    exp_name = find_trak_experiment(eval_dir, exp_date)
    print(f"Loading influence matrix from {exp_name}...")
    influence_matrix, train_set_size, test_set_size = load_influence_matrix(
        eval_dir, exp_name
    )
    print(f"  Influence matrix shape: {influence_matrix.shape}")
    print(f"  TRAK train_set_size: {train_set_size}")
    print(f"  TRAK test_set_size: {test_set_size}")

    # Validate sizes
    expected_demo_samples = len(demo_sample_infos) + len(holdout_sample_infos)
    if expected_demo_samples != train_set_size:
        print(
            f"  WARNING: Demo sample count mismatch! Expected {expected_demo_samples}, got {train_set_size}"
        )

    # Load rollout episodes (same as TRAK scoring)
    print("Loading rollout episodes (same as TRAK scoring)...")
    rollout_episodes, rollout_sample_infos = build_rollout_sample_infos(eval_dir)
    print(f"  Rollout episodes: {len(rollout_episodes)}")
    print(f"  Rollout samples: {len(rollout_sample_infos)}")

    if len(rollout_sample_infos) != test_set_size:
        print(
            f"  WARNING: Rollout sample count mismatch! Expected {len(rollout_sample_infos)}, got {test_set_size}"
        )

    # Create image dataset for visualization if needed
    image_dataset = None
    if image_dataset_path is not None:
        if lazy_load_images:
            # Use lazy HDF5 loading to avoid loading entire dataset into memory
            print(f"Using lazy HDF5 loading for image dataset: {image_dataset_path}")
            from influence_visualizer.lazy_hdf5 import (
                LazyHDF5ImageDataset,
                LazyReplayBuffer,
            )

            try:
                lazy_dataset = LazyHDF5ImageDataset(image_dataset_path)
                lazy_dataset.open()
                print(f"  Lazy dataset opened successfully")
                print(f"  Number of episodes: {lazy_dataset.num_episodes}")
                print(f"  Available obs keys: {lazy_dataset.get_available_obs_keys()}")

                # Create a minimal wrapper with replay_buffer interface
                class LazyImageDataset:
                    def __init__(self, lazy_ds):
                        self.replay_buffer = LazyReplayBuffer(lazy_ds)
                        self._lazy_dataset = lazy_ds

                image_dataset = LazyImageDataset(lazy_dataset)
                print(f"  Lazy image dataset ready")
            except Exception as e:
                print(f"  Could not create lazy image dataset: {e}")
                import traceback

                traceback.print_exc()
                image_dataset = None
        else:
            # Load explicit image dataset (full loading - may cause memory issues)
            print(f"Loading image dataset from {image_dataset_path}...")
            print(
                f"  WARNING: Full loading mode - may cause memory issues with large datasets!"
            )

            # Get common parameters from the lowdim config
            lowdim_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)

            # Build a fresh config for the image dataset with only compatible parameters
            # The image dataset class has different arguments than lowdim
            image_cfg = {
                "_target_": "diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset",
                "dataset_path": image_dataset_path,
                "horizon": lowdim_cfg.get("horizon", 16),
                "pad_before": lowdim_cfg.get("pad_before", 0),
                "pad_after": lowdim_cfg.get("pad_after", 0),
                "abs_action": lowdim_cfg.get("abs_action", False),
                "rotation_rep": lowdim_cfg.get("rotation_rep", "rotation_6d"),
                "use_legacy_normalizer": lowdim_cfg.get("use_legacy_normalizer", False),
                "seed": lowdim_cfg.get("seed", 42),
                "val_ratio": lowdim_cfg.get("val_ratio", 0.0),
            }

            # Copy dataset_mask_kwargs if present
            if "dataset_mask_kwargs" in lowdim_cfg:
                image_cfg["dataset_mask_kwargs"] = lowdim_cfg["dataset_mask_kwargs"]
            if "sample_curation_config" in lowdim_cfg:
                image_cfg["sample_curation_config"] = lowdim_cfg["sample_curation_config"]
            if "holdout_selection_config" in lowdim_cfg:
                image_cfg["holdout_selection_config"] = lowdim_cfg["holdout_selection_config"]

            # Build shape_meta from the HDF5 file
            with h5py.File(image_dataset_path, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                if len(demo_keys) > 0:
                    first_demo = demo_keys[0]
                    obs_keys = list(f[f"data/{first_demo}/obs"].keys())
                    image_keys = [k for k in obs_keys if "image" in k.lower()]
                    print(f"  Found obs keys in HDF5: {obs_keys}")
                    print(f"  Found image keys: {image_keys}")

                    # Build shape_meta
                    shape_meta = {"obs": {}, "action": {}}

                    for key in image_keys:
                        shape = f[f"data/{first_demo}/obs/{key}"].shape[
                            1:
                        ]  # Skip time dimension
                        print(f"  Image {key} shape in HDF5 (H,W,C): {shape}")
                        # Convert (H, W, C) to (C, H, W) for shape_meta
                        if len(shape) == 3:
                            h, w, c = shape
                            shape_meta["obs"][key] = {
                                "shape": [c, h, w],
                                "type": "rgb",
                            }

                    # Get action shape - need to account for rotation transformation
                    raw_action_shape = f[f"data/{first_demo}/actions"].shape[1:]
                    # If abs_action is True, actions are transformed:
                    # - Original 7D (pos3 + axis_angle3 + gripper1) -> 10D (pos3 + rotation_6d + gripper1)
                    # - Original 14D (dual arm) -> 20D
                    if image_cfg.get("abs_action", False):
                        raw_dim = (
                            raw_action_shape[0] if len(raw_action_shape) > 0 else 7
                        )
                        if raw_dim == 7:
                            action_dim = 10  # 3 + 6 + 1
                        elif raw_dim == 14:
                            action_dim = 20  # 2 * (3 + 6 + 1)
                        else:
                            action_dim = raw_dim
                        shape_meta["action"] = {"shape": [action_dim]}
                    else:
                        shape_meta["action"] = {"shape": list(raw_action_shape)}

                    image_cfg["shape_meta"] = shape_meta
                    print(f"  Built shape_meta: {shape_meta}")

            try:
                print(f"  Instantiating image dataset...")
                image_dataset = hydra.utils.instantiate(OmegaConf.create(image_cfg))
                print(
                    f"  Image dataset loaded successfully, size: {len(image_dataset)}"
                )
                print(
                    f"  Replay buffer keys: {list(image_dataset.replay_buffer.keys())}"
                )
            except Exception as e:
                print(f"  Could not load image dataset: {e}")
                import traceback

                traceback.print_exc()
                image_dataset = None
    else:
        # Try to create from lowdim config
        image_dataset = create_image_dataset_from_config(cfg)
        if image_dataset is not None:
            print("  Created image dataset from lowdim config")

    # If still no image dataset, use demo dataset directly (might have images)
    if image_dataset is None:
        print("  Using demo dataset for images (may not have images)")

    # Load demo quality labels (for MH datasets)
    demo_quality_labels = None
    dataset_path = cfg.task.dataset.get("dataset_path", None)
    if (
        dataset_path is not None
        and quality_labels is not None
        and len(quality_labels) > 0
    ):
        print("Loading demo quality labels (MH dataset tiers)...")
        demo_quality_labels = load_demo_quality_labels(dataset_path, quality_labels)
        if demo_quality_labels is None:
            print("  No quality labels found (missing mask groups in dataset)")
    elif quality_labels is None or len(quality_labels) == 0:
        print("  Quality labels not configured (skipping)")

    return InfluenceData(
        influence_matrix=influence_matrix,
        cfg=cfg,
        demo_dataset=demo_dataset,
        demo_episodes=demo_episodes,
        demo_sample_infos=demo_sample_infos,
        holdout_dataset=holdout_dataset,
        holdout_episodes=holdout_episodes,
        holdout_sample_infos=holdout_sample_infos,
        rollout_episodes=rollout_episodes,
        rollout_sample_infos=rollout_sample_infos,
        image_dataset=image_dataset,
        eval_dir=eval_dir,
        train_dir=train_dir,
        trak_exp_name=exp_name,
        demo_quality_labels=demo_quality_labels,
    )


# =============================================================================
# New unified data loader interface
# =============================================================================


@dataclass
class SampleData:
    """Data for a single sample (observation + action + frame).

    This provides a unified interface to access all data associated with
    a single sample in the influence matrix.
    """

    sample_idx: int
    observation: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    sample_info: Optional[SampleInfo] = None


@dataclass
class TrajectoryData:
    """Data for a trajectory (episode).

    Provides access to episode-level information and the range of samples
    belonging to this trajectory in the influence matrix.
    """

    episode_idx: int
    episode_info: EpisodeInfo
    sample_indices: np.ndarray


class InfluenceDataLoader:
    """Unified interface for accessing influence data at different levels.

    This class wraps InfluenceData and provides a cleaner API for:
    - Sample-level access: get individual samples with state/action/frame
    - Trajectory-level access: get episodes and their sample ranges
    - Labels: get semantic labels for state and action dimensions

    Example:
        config = load_config("robomimic_lift")
        config.eval_dir = "/path/to/eval"
        config.train_dir = "/path/to/train"

        loader = InfluenceDataLoader(config)
        loader.load()

        # Sample-level access
        sample = loader.get_demo_sample(0)
        print(sample.observation, sample.action)

        # Trajectory-level access
        traj = loader.get_rollout_trajectory(0)
        print(traj.episode_info, traj.sample_indices)

        # Get influence matrix for a rollout-demo pair
        matrix = loader.get_pairwise_influence_matrix(rollout_idx=0, demo_idx=5)
    """

    def __init__(self, config: "VisualizerConfig"):
        """Initialize the data loader.

        Args:
            config: VisualizerConfig with data paths and settings
        """
        from influence_visualizer.config import VisualizerConfig

        self.config = config
        self._data: Optional[InfluenceData] = None
        self._loaded = False

    def load(self) -> None:
        """Load data based on config.

        Raises:
            RuntimeError: If data paths are not set and use_mock is False
        """
        from influence_visualizer.profiling import profile

        if self.config.use_mock:
            self._data = create_mock_influence_data()
        else:
            if not self.config.eval_dir or not self.config.train_dir:
                raise RuntimeError(
                    "eval_dir and train_dir must be set in config (or use_mock=True)"
                )
            with profile("load_influence_data"):
                self._data = load_influence_data(
                    eval_dir=self.config.eval_dir,
                    train_dir=self.config.train_dir,
                    train_ckpt=self.config.train_ckpt,
                    exp_date=self.config.exp_date,
                    include_holdout=True,
                    image_dataset_path=self.config.image_dataset_path,
                    lazy_load_images=self.config.lazy_load_images,
                    quality_labels=self.config.quality_labels,
                )
        self._loaded = True

    @property
    def data(self) -> InfluenceData:
        """Access underlying InfluenceData (for backwards compatibility).

        Raises:
            RuntimeError: If data has not been loaded yet
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._data

    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded

    # =========================================================================
    # Sample-level access
    # =========================================================================

    def get_demo_sample(self, sample_idx: int) -> SampleData:
        """Get demo sample data.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)

        Returns:
            SampleData with observation, action, frame, and sample_info
        """
        return SampleData(
            sample_idx=sample_idx,
            observation=self._data.get_demo_obs(sample_idx),
            action=self._data.get_demo_action_chunk(sample_idx),
            frame=self._data.get_demo_frame(sample_idx, self.config.obs_key),
            sample_info=self._data.get_demo_sample_info(sample_idx),
        )

    def get_rollout_sample(self, sample_idx: int) -> SampleData:
        """Get rollout sample data.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)

        Returns:
            SampleData with observation, action, frame, and sample_info
        """
        return SampleData(
            sample_idx=sample_idx,
            observation=self._data.get_rollout_obs(sample_idx),
            action=self._data.get_rollout_action(sample_idx),
            frame=self._data.get_rollout_frame(sample_idx, self.config.obs_key),
            sample_info=self._data.get_rollout_sample_info(sample_idx),
        )

    def get_influence(self, rollout_idx: int, demo_idx: int) -> float:
        """Get influence score for a specific rollout-demo sample pair.

        Args:
            rollout_idx: Rollout sample index
            demo_idx: Demo sample index

        Returns:
            Influence score (float)
        """
        return float(self._data.influence_matrix[rollout_idx, demo_idx])

    # =========================================================================
    # Trajectory-level access
    # =========================================================================

    def get_demo_trajectory(self, episode_idx: int) -> TrajectoryData:
        """Get demo trajectory data.

        Args:
            episode_idx: Episode index (into all_demo_episodes)

        Returns:
            TrajectoryData with episode info and sample indices
        """
        ep = self._data.all_demo_episodes[episode_idx]
        return TrajectoryData(
            episode_idx=episode_idx,
            episode_info=ep,
            sample_indices=np.arange(ep.sample_start_idx, ep.sample_end_idx),
        )

    def get_rollout_trajectory(self, episode_idx: int) -> TrajectoryData:
        """Get rollout trajectory data.

        Args:
            episode_idx: Episode index (into rollout_episodes)

        Returns:
            TrajectoryData with episode info and sample indices
        """
        ep = self._data.rollout_episodes[episode_idx]
        return TrajectoryData(
            episode_idx=episode_idx,
            episode_info=ep,
            sample_indices=np.arange(ep.sample_start_idx, ep.sample_end_idx),
        )

    def get_pairwise_influence_matrix(
        self,
        rollout_idx: int,
        demo_idx: int,
    ) -> np.ndarray:
        """Get sample-level influence matrix for a rollout-demo episode pair.

        Args:
            rollout_idx: Rollout episode index
            demo_idx: Demo episode index

        Returns:
            2D array of shape (num_rollout_samples, num_demo_samples)
        """
        rollout_traj = self.get_rollout_trajectory(rollout_idx)
        demo_traj = self.get_demo_trajectory(demo_idx)
        return self._data.influence_matrix[
            np.ix_(rollout_traj.sample_indices, demo_traj.sample_indices)
        ]

    # =========================================================================
    # Labels
    # =========================================================================

    def get_state_labels(self, dim: Optional[int] = None) -> List[str]:
        """Get semantic labels for state dimensions.

        Args:
            dim: Number of dimensions (if None, tries to infer from data)

        Returns:
            List of labels. Uses config labels if available, otherwise generic.
        """
        from influence_visualizer.config import get_generic_labels

        if self.config.state_labels:
            return self.config.state_labels

        # Try to infer dimension from data
        if dim is None:
            try:
                sample = self._data.get_demo_obs(0)
                dim = len(sample) if sample is not None else 0
            except Exception:
                dim = 0

        return get_generic_labels(dim, "state")

    def get_action_labels(self, dim: Optional[int] = None) -> List[str]:
        """Get semantic labels for action dimensions.

        Args:
            dim: Number of dimensions (if None, tries to infer from data)

        Returns:
            List of labels. Uses config labels if available, otherwise generic.
        """
        from influence_visualizer.config import get_generic_labels

        if self.config.action_labels:
            return self.config.action_labels

        # Try to infer dimension from data
        if dim is None:
            try:
                action = self._data.get_demo_action_chunk(0)
                dim = action.shape[-1] if action is not None else 0
            except Exception:
                dim = 0

        return get_generic_labels(dim, "action")

    # =========================================================================
    # Delegated properties (for backwards compatibility)
    # =========================================================================

    @property
    def num_rollout_episodes(self) -> int:
        """Number of rollout episodes."""
        return self._data.num_rollout_episodes

    @property
    def num_demo_episodes(self) -> int:
        """Number of demo episodes (train + holdout)."""
        return self._data.num_demo_episodes

    @property
    def num_rollout_samples(self) -> int:
        """Total number of rollout samples."""
        return self._data.num_rollout_samples

    @property
    def num_demo_samples(self) -> int:
        """Total number of demo samples."""
        return self._data.num_demo_samples

    @property
    def rollout_episodes(self) -> List[EpisodeInfo]:
        """List of rollout episode info."""
        return self._data.rollout_episodes

    @property
    def demo_episodes(self) -> List[EpisodeInfo]:
        """List of train demo episode info."""
        return self._data.demo_episodes

    @property
    def holdout_episodes(self) -> List[EpisodeInfo]:
        """List of holdout demo episode info."""
        return self._data.holdout_episodes

    @property
    def all_demo_episodes(self) -> List[EpisodeInfo]:
        """List of all demo episodes (train + holdout)."""
        return self._data.all_demo_episodes

    @property
    def influence_matrix(self) -> np.ndarray:
        """The influence matrix (num_rollout_samples, num_demo_samples)."""
        return self._data.influence_matrix

    @property
    def horizon(self) -> int:
        """Action prediction horizon."""
        return self._data.horizon

    @property
    def demo_quality_labels(self) -> Optional[Dict[int, str]]:
        """Demo quality labels (for MH datasets)."""
        return self._data.demo_quality_labels

    def get_rollout_frame(
        self, sample_idx: int, obs_key: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get an image frame for a rollout sample.

        Args:
            sample_idx: Global sample index in the influence matrix (rollout side)
            obs_key: Observation key for the image (defaults to config.obs_key)

        Returns:
            Image as numpy array, or None if not available
        """
        key = obs_key or self.config.obs_key
        return self._data.get_rollout_frame(sample_idx, key)

    def get_demo_frame(
        self, sample_idx: int, obs_key: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get an image frame for a demo sample.

        Args:
            sample_idx: Global sample index in the influence matrix (demo side)
            obs_key: Observation key for the image (defaults to config.obs_key)

        Returns:
            Image as numpy array, or None if not available
        """
        key = obs_key or self.config.obs_key
        return self._data.get_demo_frame(sample_idx, key)

    def get_available_image_keys(self) -> List[str]:
        """Get available image observation keys from the datasets.

        Returns:
            List of available image observation keys
        """
        return self._data.get_available_image_keys()


def create_mock_influence_data(
    num_rollout_episodes: int = 5,
    num_demo_episodes: int = 10,
    samples_per_rollout: int = 20,
    samples_per_demo: int = 50,
) -> InfluenceData:
    """Create mock influence data for testing without real data files.

    Args:
        num_rollout_episodes: Number of rollout episodes.
        num_demo_episodes: Number of demonstration episodes.
        samples_per_rollout: Samples per rollout episode.
        samples_per_demo: Samples per demo episode.

    Returns:
        InfluenceData with random influence scores.
    """
    total_rollout_samples = num_rollout_episodes * samples_per_rollout
    total_demo_samples = num_demo_episodes * samples_per_demo

    # Random influence matrix
    influence_matrix = np.random.randn(
        total_rollout_samples, total_demo_samples
    ).astype(np.float32)

    # Build rollout episodes
    rollout_episodes = []
    rollout_sample_infos = []

    for ep_idx in range(num_rollout_episodes):
        start_idx = ep_idx * samples_per_rollout
        end_idx = start_idx + samples_per_rollout

        rollout_episodes.append(
            EpisodeInfo(
                index=ep_idx,
                num_samples=samples_per_rollout,
                sample_start_idx=start_idx,
                sample_end_idx=end_idx,
                success=np.random.random() > 0.3,
                raw_length=samples_per_rollout,
            )
        )

        for t in range(samples_per_rollout):
            rollout_sample_infos.append(
                SampleInfo(
                    global_idx=start_idx + t,
                    episode_idx=ep_idx,
                    timestep=t,
                    buffer_start_idx=t,
                    buffer_end_idx=t + 1,
                )
            )

    # Build demo episodes
    demo_episodes = []
    demo_sample_infos = []

    for ep_idx in range(num_demo_episodes):
        start_idx = ep_idx * samples_per_demo
        end_idx = start_idx + samples_per_demo

        demo_episodes.append(
            EpisodeInfo(
                index=ep_idx,
                num_samples=samples_per_demo,
                sample_start_idx=start_idx,
                sample_end_idx=end_idx,
                raw_length=samples_per_demo + 15,  # Simulating raw length > num_samples
            )
        )

        for t in range(samples_per_demo):
            demo_sample_infos.append(
                SampleInfo(
                    global_idx=start_idx + t,
                    episode_idx=ep_idx,
                    timestep=t,
                    buffer_start_idx=t,
                    buffer_end_idx=t + 16,  # Simulating horizon
                )
            )

    # Create a minimal mock config
    mock_cfg = OmegaConf.create(
        {
            "task": {
                "dataset": {
                    "horizon": 16,
                    "pad_before": 1,
                    "pad_after": 7,
                    "n_obs_steps": 2,
                }
            }
        }
    )

    # Create mock datasets with replay buffers for actions
    class MockReplayBuffer(dict):
        def __init__(self, length, dim):
            super().__init__()
            self["action"] = np.random.randn(length, dim).astype(np.float32)

    class MockDataset:
        def __init__(self, length, dim):
            self.replay_buffer = MockReplayBuffer(length, dim)

    action_dim = 10  # Default for mock
    demo_dataset = MockDataset(total_demo_samples + 100, action_dim)

    return InfluenceData(
        influence_matrix=influence_matrix,
        cfg=mock_cfg,
        demo_dataset=demo_dataset,
        demo_episodes=demo_episodes,
        demo_sample_infos=demo_sample_infos,
        holdout_dataset=None,
        holdout_episodes=[],
        holdout_sample_infos=[],
        rollout_episodes=rollout_episodes,
        rollout_sample_infos=rollout_sample_infos,
        image_dataset=None,
        eval_dir=None,
        train_dir=None,
    )
