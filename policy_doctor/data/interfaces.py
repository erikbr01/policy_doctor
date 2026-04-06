"""Abstract interfaces for dataset and influence store."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

from policy_doctor.data.structures import EpisodeInfo, GlobalInfluenceMatrix


class InfluenceStoreInterface(ABC):
    """Interface for read-only access to the influence matrix (slice, cell, shape)."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @abstractmethod
    def read_slice(
        self,
        r_lo: int,
        r_hi: int,
        d_lo: int,
        d_hi: int,
    ) -> Any:
        pass

    @abstractmethod
    def read_cell(self, r: int, d: int) -> float:
        pass


class DatasetAdapter(ABC):
    """Unified interface for dataset + influence source (e.g. Robomimic)."""

    @abstractmethod
    def get_rollout_episodes(self) -> List[EpisodeInfo]:
        pass

    @abstractmethod
    def get_demo_episodes(self) -> List[EpisodeInfo]:
        pass

    @abstractmethod
    def get_global_influence_matrix(self) -> GlobalInfluenceMatrix:
        pass

    def get_influence_backing_path(self) -> None:
        """Optional: path to the backing store (e.g. mmap file). Return None if in-memory."""
        return None

    def get_config(self) -> Any:
        """Optional: training/config dict. Return None if not available."""
        return None
