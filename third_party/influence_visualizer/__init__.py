"""Influence Visualizer: Interactive exploration of influence matrices.

This package provides tools for visualizing the relationship between
policy rollout behaviors and training demonstrations through TRAK
influence scores.
"""

from influence_visualizer.config import (
    VisualizerConfig,
    get_generic_labels,
    list_configs,
    load_config,
)
from influence_visualizer.data_loader import (
    EpisodeInfo,
    InfluenceData,
    InfluenceDataLoader,
    SampleData,
    SampleInfo,
    TrajectoryData,
    create_mock_influence_data,
    load_influence_data,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "VisualizerConfig",
    "load_config",
    "list_configs",
    "get_generic_labels",
    # Data loader
    "EpisodeInfo",
    "SampleInfo",
    "InfluenceData",
    "InfluenceDataLoader",
    "SampleData",
    "TrajectoryData",
    "load_influence_data",
    "create_mock_influence_data",
]
