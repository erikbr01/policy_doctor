# Influence Embeddings Pipeline
#
# This module implements the influence embeddings pipeline for robot failure mode
# discovery as specified in INFLUENCE_EMBEDDINGS_PLAN.md.
#
# Components:
# - gradient_projector: Computes projected gradients (influence embeddings)
# - trajectory_aggregator: Aggregates per-timestep embeddings into trajectory embeddings
# - embedding_computer: Processes datasets to generate embeddings
# - failure_clusterer: K-Means clustering of failure embeddings
# - training_attributor: Links failure clusters to training demonstrations
# - pipeline: Main orchestration script

from .embedding_computer import EmbeddingComputer
from .failure_clusterer import FailureClusterer
from .gradient_projector import GradientProjector
from .training_attributor import TrainingAttributor
from .trajectory_aggregator import TrajectoryAggregator

__all__ = [
    "GradientProjector",
    "TrajectoryAggregator",
    "EmbeddingComputer",
    "FailureClusterer",
    "TrainingAttributor",
]
