"""Abstract interfaces for runtime stream scorers and graph assigners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AssignmentResult:
    """Result of assigning an embedding to a behavior graph node."""

    cluster_id: int
    node_id: int     # node in BehaviorGraph; equals cluster_id unless graph was pruned
    distance: float  # L2 distance to the nearest centroid
    node_name: str


@dataclass
class MonitorResult:
    """Full result of processing one streaming sample."""

    embedding: np.ndarray          # (proj_dim,) influence embedding
    influence_scores: np.ndarray   # (N_train,) per-train-sample influence scores
    assignment: Optional[AssignmentResult]
    timing_ms: dict                # keys: "gradient_ms", "project_ms", "score_ms", "assign_ms", "total_ms"


class StreamScorer(ABC):
    """Computes influence embeddings and scores for single test-time samples.

    Requires ``diffusion_policy`` on the Python path (cupid conda env).
    Initialization is expensive (loads model + cached artifacts); keep one
    instance alive for the lifetime of the monitor.
    """

    @abstractmethod
    def embed(self, batch: dict) -> np.ndarray:
        """Project a single-sample batch into the influence embedding space.

        Args:
            batch: dict with keys ``obs`` ``(1,To,Do)``, ``action`` ``(1,Ta,Da)``,
                   ``timesteps`` ``(1,num_ts)`` on the scorer's device.

        Returns:
            ``(proj_dim,)`` float32 numpy array.
        """

    @abstractmethod
    def score(self, batch: dict) -> np.ndarray:
        """Compute influence scores of all training samples on ``batch``.

        Returns:
            ``(N_train,)`` float32 numpy array; higher = more influential.
        """


class GraphAssigner(ABC):
    """Maps an influence embedding to the nearest behavior graph node."""

    @abstractmethod
    def assign(self, embedding: np.ndarray) -> AssignmentResult:
        """Assign an embedding to the nearest cluster centroid.

        Args:
            embedding: ``(proj_dim,)`` float32 array (same space as training embeddings).

        Returns:
            :class:`AssignmentResult`.
        """
