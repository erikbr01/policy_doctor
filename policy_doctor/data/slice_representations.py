"""Slice representations for clustering rollout episodes.

Only PolicyEmbeddingRepresentation is implemented here; the legacy infembed/trak
paths are still handled by clustering_embeddings.py.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from policy_doctor.data.clustering_embeddings import (
    build_windows_from_rollout_timestep_embeddings,
)


@dataclass
class SliceWindowParams:
    window_width: int = 5
    stride: int = 2
    aggregation: str = "sum"


class PolicyEmbeddingRepresentation:
    """Pre-computed policy embeddings loaded from disk, sliding-windowed.

    Embeddings are produced by ``compute_policy_embeddings.py`` (mimicgen_torch2
    env, GPU) and saved as ``<eval_dir>/policy_embeddings/<layer>.npz`` with
    key ``rollout_embeddings``, shape ``(N_total_timesteps, D)``.

    Layers:
      - ``plan_bottleneck``: UNet mid-block activation pooled over time at
        denoising step t=0 (shape D = down_dims[-1]).
      - ``obs_encoder``: normalized observation history flattened
        (shape D = obs_dim * n_obs_steps, very fast to compute).
    """

    name = "policy_emb"

    def _load(
        self, eval_dir: pathlib.Path, *, layer: str = "plan_bottleneck"
    ) -> Tuple[np.ndarray, List[int], List]:
        emb_path = eval_dir / "policy_embeddings" / f"{layer}.npz"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Policy embeddings not found: {emb_path}\n"
                f"Run compute_policy_embeddings.py --layer {layer} first."
            )
        with np.load(emb_path) as f:
            embeddings = np.asarray(f["rollout_embeddings"], dtype=np.float32)
        meta_path = eval_dir / "episodes" / "metadata.yaml"
        with open(meta_path) as fh:
            meta = yaml.safe_load(fh)
        ep_lens = meta["episode_lengths"]
        ep_succ = meta.get("episode_successes", [None] * len(ep_lens))
        return embeddings, ep_lens, ep_succ

    def extract(
        self,
        eval_dir: pathlib.Path,
        params: SliceWindowParams,
        *,
        layer: str = "plan_bottleneck",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        embeddings, ep_lens, ep_succ = self._load(eval_dir, layer=layer)
        return build_windows_from_rollout_timestep_embeddings(
            embeddings,
            ep_lens,
            ep_succ,
            params.window_width,
            params.stride,
            params.aggregation,
        )

    def extract_per_timestep(
        self,
        eval_dir: pathlib.Path,
        *,
        layer: str = "plan_bottleneck",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, List[int], List]:
        return self._load(eval_dir, layer=layer)


__all__ = [
    "SliceWindowParams",
    "PolicyEmbeddingRepresentation",
]
