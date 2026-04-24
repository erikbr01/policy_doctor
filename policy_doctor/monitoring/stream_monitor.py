"""StreamMonitor: real-time runtime monitor that ties a scorer and graph assigner together.

Usage (cupid conda env required for the scorer):

    scorer = InfEmbedStreamScorer(checkpoint=..., infembed_fit_path=..., ...)
    assigner = NearestCentroidAssigner.from_paths(
        rollout_embeddings=scorer.rollout_embeddings,
        clustering_dir=...,
        graph=behavior_graph,
    )
    monitor = StreamMonitor(scorer=scorer, assigner=assigner)

    # In the control loop:
    result = monitor.process_sample(obs=obs, action=action)
    print(result.assignment.node_name, result.timing_ms)
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import torch

from policy_doctor.monitoring.base import (
    AssignmentResult,
    GraphAssigner,
    MonitorResult,
    StreamScorer,
)


class StreamMonitor:
    """Real-time runtime monitor wrapping a scorer and an optional graph assigner.

    Parameters
    ----------
    scorer:
        A :class:`~policy_doctor.monitoring.base.StreamScorer` instance
        (``TRAKStreamScorer`` or ``InfEmbedStreamScorer``).
    assigner:
        An optional :class:`~policy_doctor.monitoring.base.GraphAssigner` instance.
        When ``None``, ``MonitorResult.assignment`` is always ``None``.
    num_diffusion_timesteps:
        Number of random diffusion timesteps to sample when building the batch.
        Overrides the scorer's own default; defaults to 8.
    """

    def __init__(
        self,
        scorer: StreamScorer,
        assigner: Optional[GraphAssigner] = None,
        num_diffusion_timesteps: int = 8,
    ) -> None:
        self.scorer = scorer
        self.assigner = assigner
        self.num_diffusion_timesteps = num_diffusion_timesteps

    def process_sample(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> MonitorResult:
        """Run the full monitor pipeline on one obs-action pair.

        Args:
            obs:    Raw environment observation, shape ``(To, Do)`` or ``(Do,)``.
                    Normalisation is performed internally by the policy model.
            action: Action (or predicted action), shape ``(Ta, Da)`` or ``(Da,)``.

        Returns:
            :class:`~policy_doctor.monitoring.base.MonitorResult` with per-stage
            timing in milliseconds.
        """
        device = self.scorer.device

        # Convert to tensors and add batch dimension
        obs_t = _to_tensor(obs, device).unsqueeze(0)       # (1, To, Do)
        action_t = _to_tensor(action, device).unsqueeze(0) # (1, Ta, Da)

        # Sample random diffusion timesteps
        num_ts = self.num_diffusion_timesteps
        max_ts = self.scorer._num_train_timesteps
        timesteps_t = torch.randint(max_ts, (1, num_ts), device=device).long()

        batch = {"obs": obs_t, "action": action_t, "timesteps": timesteps_t}

        timing: dict = {}

        # --- Gradient + projection (embed) ---
        t0 = time.perf_counter()
        embedding = self.scorer.embed(batch)  # (proj_dim,)
        t1 = time.perf_counter()
        timing["gradient_project_ms"] = (t1 - t0) * 1e3

        # --- Influence scoring ---
        t0 = time.perf_counter()
        scores = self.scorer.score(batch)    # (N_train,)
        t1 = time.perf_counter()
        timing["score_ms"] = (t1 - t0) * 1e3

        # --- Graph assignment ---
        assignment: Optional[AssignmentResult] = None
        if self.assigner is not None:
            t0 = time.perf_counter()
            assignment = self.assigner.assign(embedding)
            t1 = time.perf_counter()
            timing["assign_ms"] = (t1 - t0) * 1e3
        else:
            timing["assign_ms"] = 0.0

        timing["total_ms"] = timing["gradient_project_ms"] + timing["score_ms"] + timing["assign_ms"]

        return MonitorResult(
            embedding=embedding,
            influence_scores=scores,
            assignment=assignment,
            timing_ms=timing,
        )

    def process_sample_embed_only(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> MonitorResult:
        """Compute embedding and graph assignment without full influence scoring.

        Cheaper than :meth:`process_sample` when only the behavior graph node
        assignment is needed at runtime (influence scores can be retrieved offline).
        ``MonitorResult.influence_scores`` is ``None`` in the returned result.

        Returns:
            :class:`~policy_doctor.monitoring.base.MonitorResult` with
            ``influence_scores=None`` and per-stage timing.
        """
        device = self.scorer.device
        obs_t = _to_tensor(obs, device).unsqueeze(0)
        action_t = _to_tensor(action, device).unsqueeze(0)
        num_ts = self.num_diffusion_timesteps
        max_ts = self.scorer._num_train_timesteps
        timesteps_t = torch.randint(max_ts, (1, num_ts), device=device).long()
        batch = {"obs": obs_t, "action": action_t, "timesteps": timesteps_t}

        timing: dict = {}

        t0 = time.perf_counter()
        embedding = self.scorer.embed(batch)
        t1 = time.perf_counter()
        timing["gradient_project_ms"] = (t1 - t0) * 1e3
        timing["score_ms"] = 0.0

        assignment: Optional[AssignmentResult] = None
        if self.assigner is not None:
            t0 = time.perf_counter()
            assignment = self.assigner.assign(embedding)
            t1 = time.perf_counter()
            timing["assign_ms"] = (t1 - t0) * 1e3
        else:
            timing["assign_ms"] = 0.0

        timing["total_ms"] = timing["gradient_project_ms"] + timing["assign_ms"]
        return MonitorResult(
            embedding=embedding,
            influence_scores=None,
            assignment=assignment,
            timing_ms=timing,
        )


def _to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return x.float().to(device)
