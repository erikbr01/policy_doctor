"""Load influence data (TRAK matrix, episodes, sample infos).

Uses influence_visualizer when available; otherwise raises. For a fully standalone
policy_doctor, implement a native loader here (same disk layout: checkpoint + TRAK results).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from policy_doctor.data.structures import EpisodeInfo, SampleInfo


@dataclass
class InfluenceDataContainer:
    """Container for influence data. Same interface as needed by run_pipeline."""

    influence_matrix: np.ndarray
    rollout_episodes: List[EpisodeInfo] = field(default_factory=list)
    demo_episodes: List[EpisodeInfo] = field(default_factory=list)
    holdout_episodes: List[EpisodeInfo] = field(default_factory=list)
    demo_sample_infos: List[Any] = field(default_factory=list)  # List[SampleInfo]-like
    holdout_sample_infos: List[Any] = field(default_factory=list)
    #: Full ``influence_visualizer`` ``InfluenceData`` (for frames/actions in Streamlit only).
    _iv_source: Any = field(default=None, repr=False, compare=False)

    @property
    def all_demo_episodes(self) -> List[EpisodeInfo]:
        return self.demo_episodes + self.holdout_episodes

    @property
    def demo_quality_labels(self) -> Any:
        if self._iv_source is None:
            return None
        return getattr(self._iv_source, "demo_quality_labels", None)

    def get_rollout_frame(self, abs_idx: int) -> Any:
        if self._iv_source is None:
            return None
        return self._iv_source.get_rollout_frame(abs_idx)

    def get_demo_frame(self, abs_idx: int) -> Any:
        if self._iv_source is None:
            return None
        return self._iv_source.get_demo_frame(abs_idx)

    def get_rollout_action(self, abs_idx: int) -> Any:
        if self._iv_source is None:
            return None
        return self._iv_source.get_rollout_action(abs_idx)

    def get_demo_action(self, abs_idx: int) -> Any:
        if self._iv_source is None:
            return None
        return self._iv_source.get_demo_action(abs_idx)


def _convert_ep(ep: Any) -> EpisodeInfo:
    return EpisodeInfo(
        index=ep.index,
        num_samples=ep.num_samples,
        sample_start_idx=ep.sample_start_idx,
        sample_end_idx=ep.sample_end_idx,
        success=getattr(ep, "success", None),
        raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
    )


def _convert_si(si: Any) -> Any:
    """Return SampleInfo if we have it; else pass through for compatibility."""
    if hasattr(si, "global_idx") and hasattr(si, "episode_idx"):
        return SampleInfo(
            global_idx=si.global_idx,
            episode_idx=si.episode_idx,
            timestep=getattr(si, "timestep", 0),
            buffer_start_idx=getattr(si, "buffer_start_idx", 0),
            buffer_end_idx=getattr(si, "buffer_end_idx", 0),
            sample_start_idx=getattr(si, "sample_start_idx", 0),
            sample_end_idx=getattr(si, "sample_end_idx", 0),
        )
    return si


def load_influence_data(
    eval_dir: str,
    train_dir: str,
    train_ckpt: str = "latest",
    exp_date: str = "default",
    include_holdout: bool = True,
    image_dataset_path: Optional[str] = None,
    lazy_load_images: bool = True,
    quality_labels: Optional[List[str]] = None,
) -> InfluenceDataContainer:
    """Load influence data from eval_dir and train_dir.

    When influence_visualizer is installed, delegates to it and converts the result.
    Otherwise raises RuntimeError (for a standalone repo, implement a native loader here).
    """
    try:
        from influence_visualizer.data_loader import load_influence_data as iv_load
    except ImportError as e:
        raise RuntimeError(
            "Influence data loading requires influence_visualizer. "
            "Install it or implement a native loader in policy_doctor.data.influence_loader."
        ) from e

    data = iv_load(
        eval_dir=eval_dir,
        train_dir=train_dir,
        train_ckpt=train_ckpt,
        exp_date=exp_date,
        include_holdout=include_holdout,
        image_dataset_path=image_dataset_path,
        lazy_load_images=lazy_load_images,
        quality_labels=quality_labels,
    )

    return InfluenceDataContainer(
        influence_matrix=data.influence_matrix,
        rollout_episodes=[_convert_ep(ep) for ep in data.rollout_episodes],
        demo_episodes=[_convert_ep(ep) for ep in data.demo_episodes],
        holdout_episodes=[_convert_ep(ep) for ep in data.holdout_episodes],
        demo_sample_infos=[_convert_si(si) for si in data.demo_sample_infos],
        holdout_sample_infos=[_convert_si(si) for si in data.holdout_sample_infos],
        _iv_source=data,
    )
