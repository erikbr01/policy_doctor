"""Lazy HDF5 dataset loader for large image datasets.

This module provides a lazy-loading wrapper around HDF5 files that only
reads individual frames on demand, avoiding loading the entire 31GB+ dataset
into memory.
"""

from typing import List, Optional

import h5py
import numpy as np


class LazyHDF5ImageDataset:
    """Lazy-loading wrapper for robomimic HDF5 image datasets.

    This class keeps the HDF5 file open and only reads individual frames
    when requested, making it suitable for very large datasets (e.g., 31GB)
    that would cause memory issues if fully loaded.

    Usage:
        dataset = LazyHDF5ImageDataset("path/to/image_abs.hdf5")
        frame = dataset.get_frame(demo_idx=0, timestep=10, obs_key="agentview_image")
    """

    def __init__(self, dataset_path: str):
        """Initialize the lazy loader.

        Args:
            dataset_path: Path to the HDF5 file
        """
        self.dataset_path = dataset_path
        self._file = None
        self._episode_ends = None
        self._demo_keys = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self):
        """Open the HDF5 file."""
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")

            # Cache demo keys
            self._demo_keys = sorted(
                self._file["data"].keys(), key=lambda x: int(x.split("_")[1])
            )

            # Build episode_ends array
            ends = []
            cumsum = 0
            for demo_key in self._demo_keys:
                demo_len = self._file[f"data/{demo_key}/actions"].shape[0]
                cumsum += demo_len
                ends.append(cumsum)
            self._episode_ends = np.array(ends)

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            try:
                self._file.close()
            except (TypeError, AttributeError):
                # File may already be closed or in invalid state during cleanup
                pass
            finally:
                self._file = None

    def __del__(self):
        """Destructor to ensure file is closed."""
        try:
            self.close()
        except Exception:
            # Suppress exceptions during cleanup
            pass

    @property
    def episode_ends(self) -> np.ndarray:
        """Get episode end indices (same as ReplayBuffer.episode_ends)."""
        if self._file is None:
            self.open()
        return self._episode_ends

    @property
    def num_episodes(self) -> int:
        """Get number of episodes in the dataset."""
        if self._file is None:
            self.open()
        return len(self._demo_keys)

    def get_available_obs_keys(self, demo_idx: int = 0) -> List[str]:
        """Get list of available observation keys.

        Args:
            demo_idx: Demo episode index to check (default: 0)

        Returns:
            List of observation keys available in this dataset
        """
        if self._file is None:
            self.open()

        demo_key = self._demo_keys[demo_idx]
        return list(self._file[f"data/{demo_key}/obs"].keys())

    def get_frame(
        self, demo_idx: int, timestep: int, obs_key: str = "agentview_image"
    ) -> Optional[np.ndarray]:
        """Get a single image frame from the dataset.

        Args:
            demo_idx: Demo episode index
            timestep: Timestep within the episode
            obs_key: Observation key (e.g., "agentview_image")

        Returns:
            RGB image as numpy array (H, W, 3) uint8, or None if not available
        """
        if self._file is None:
            self.open()

        try:
            demo_key = self._demo_keys[demo_idx]
            obs_path = f"data/{demo_key}/obs/{obs_key}"

            if obs_path not in self._file:
                # Try to find any image key
                available_keys = self.get_available_obs_keys(demo_idx)
                image_keys = [k for k in available_keys if "image" in k.lower()]
                if not image_keys:
                    return None
                obs_key = image_keys[0]
                obs_path = f"data/{demo_key}/obs/{obs_key}"

            img = self._file[obs_path][timestep]
            img = np.array(img)

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            return img

        except Exception as e:
            print(f"Error loading frame from HDF5: {e}")
            return None

    def get_action(self, demo_idx: int, timestep: int) -> Optional[np.ndarray]:
        """Get an action from the dataset.

        Args:
            demo_idx: Demo episode index
            timestep: Timestep within the episode

        Returns:
            Action as numpy array, or None if not available
        """
        if self._file is None:
            self.open()

        try:
            demo_key = self._demo_keys[demo_idx]
            action = self._file[f"data/{demo_key}/actions"][timestep]
            return np.array(action)
        except Exception as e:
            print(f"Error loading action from HDF5: {e}")
            return None


class LazyReplayBuffer:
    """Wrapper to make LazyHDF5ImageDataset compatible with ReplayBuffer interface.

    This provides a minimal ReplayBuffer-like interface so it can be used as a
    drop-in replacement in the InfluenceData class.
    """

    def __init__(self, lazy_dataset: LazyHDF5ImageDataset):
        """Initialize the wrapper.

        Args:
            lazy_dataset: LazyHDF5ImageDataset instance
        """
        self.lazy_dataset = lazy_dataset
        self._episode_ends = None

    @property
    def episode_ends(self) -> np.ndarray:
        """Get episode end indices."""
        if self._episode_ends is None:
            self._episode_ends = self.lazy_dataset.episode_ends
        return self._episode_ends

    def __getitem__(self, key: str):
        """Simulate replay_buffer[key] access.

        For lazy loading, this returns a LazyFrameAccessor that
        will load frames on demand when indexed.
        """
        if "image" in key.lower():
            return LazyFrameAccessor(self.lazy_dataset, key)
        else:
            raise KeyError(f"LazyReplayBuffer only supports image keys, got: {key}")

    def keys(self):
        """Get available keys (observation keys from first episode)."""
        return self.lazy_dataset.get_available_obs_keys(0)


class LazyFrameAccessor:
    """Accessor for lazy frame loading.

    This class is returned by LazyReplayBuffer[key] and supports indexing
    to load individual frames.
    """

    def __init__(self, lazy_dataset: LazyHDF5ImageDataset, obs_key: str):
        """Initialize the accessor.

        Args:
            lazy_dataset: LazyHDF5ImageDataset instance
            obs_key: Observation key (e.g., "agentview_image")
        """
        self.lazy_dataset = lazy_dataset
        self.obs_key = obs_key

    def __getitem__(self, buffer_idx: int) -> np.ndarray:
        """Get frame at buffer index.

        Args:
            buffer_idx: Global buffer index (across all episodes)

        Returns:
            RGB image as numpy array
        """
        # Convert global buffer index to (demo_idx, timestep)
        episode_ends = self.lazy_dataset.episode_ends
        demo_idx = np.searchsorted(episode_ends, buffer_idx, side="right")

        # Calculate timestep within episode
        if demo_idx == 0:
            timestep = buffer_idx
        else:
            timestep = buffer_idx - episode_ends[demo_idx - 1]

        return self.lazy_dataset.get_frame(demo_idx, timestep, self.obs_key)
