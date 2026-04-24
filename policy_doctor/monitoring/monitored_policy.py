"""MonitoredPolicy: wraps any BaseLowdimPolicy with per-timestep behavior classification.

The wrapped policy's predict_action() is intercepted: after each call the obs +
predicted action are classified by the TrajectoryClassifier and the result is stored
in episode_results.

Compatible with any env runner that calls policy.predict_action(obs_dict) and
policy.reset() between episodes (e.g. RobomimicLowdimRunner).

Environment: requires the cupid conda env (diffusion_policy + infembed).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
from policy_doctor.monitoring.base import MonitorResult
from policy_doctor.monitoring.intervention import InterventionDecision, InterventionRule


class MonitoredPolicy:
    """Wraps any BaseLowdimPolicy to add per-timestep behavior monitoring.

    After each ``predict_action()`` call, the obs + predicted action are classified
    and stored in :attr:`episode_results`.  Call :meth:`reset` between episodes (the
    env runner does this automatically).

    An optional :class:`~policy_doctor.monitoring.intervention.InterventionRule` is
    evaluated at each timestep; the resulting
    :class:`~policy_doctor.monitoring.intervention.InterventionDecision` is stored in
    the episode result entry under the ``"intervention"`` key.

    A rolling buffer of the last ``max_influence_window`` embeddings is maintained.
    Call :meth:`get_slice_influence` to rank training demonstrations by their aggregate
    influence over the current window — scores are computed lazily from the buffered
    embeddings (one matmul per step, no gradient pass), so it is cheap to call only
    when an intervention is triggered.

    Parameters
    ----------
    policy:
        Any ``BaseLowdimPolicy`` or ``BaseImagePolicy``.
    classifier:
        A loaded :class:`~policy_doctor.monitoring.TrajectoryClassifier`
        (use ``mode="rollout"`` for live env data).
    intervention_rule:
        Optional :class:`~policy_doctor.monitoring.intervention.InterventionRule`.
        When set, evaluated at every timestep.
    max_influence_window:
        Maximum number of recent influence-score vectors to keep in the rolling
        buffer used by :meth:`get_slice_influence`.
    verbose:
        If True, print the node assignment at each timestep.
    """

    def __init__(
        self,
        policy: Any,
        classifier: TrajectoryClassifier,
        intervention_rule: Optional[InterventionRule] = None,
        max_influence_window: int = 5,
        verbose: bool = False,
    ) -> None:
        self._policy = policy
        self._classifier = classifier
        self._intervention_rule = intervention_rule
        self._max_influence_window = max_influence_window
        self.verbose = verbose

        self.episode_results: List[Dict] = []
        self._episode_idx: int = 0
        self._timestep: int = 0

        self._embedding_buffer: Deque[np.ndarray] = deque(maxlen=max_influence_window)
        self._monitor_history: Deque[MonitorResult] = deque(maxlen=max_influence_window)

    def reset(self) -> None:
        self._policy.reset()
        if self._timestep > 0:
            self._episode_idx += 1
        self._timestep = 0
        self._embedding_buffer.clear()
        self._monitor_history.clear()
        if self._intervention_rule is not None:
            self._intervention_rule.reset()

    def predict_action(self, obs_dict: Dict) -> Dict:
        import torch

        action_dict = self._policy.predict_action(obs_dict)

        obs = obs_dict["obs"]
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        action_key = "action_pred" if "action_pred" in action_dict else "action"
        action = action_dict[action_key]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # obs: (B, To, Do), action: (B, Ta, Da)
        for i in range(obs.shape[0]):
            result = self._classifier.classify_sample_embed_only(obs[i], action[i])

            # Update rolling buffers
            self._monitor_history.append(result)
            self._embedding_buffer.append(result.embedding)

            # Evaluate intervention rule
            decision: Optional[InterventionDecision] = None
            if self._intervention_rule is not None:
                decision = self._intervention_rule.check(
                    result, list(self._monitor_history)
                )

            entry: Dict = {
                "episode": self._episode_idx,
                "timestep": self._timestep,
                "env_idx": i,
                "cluster_id": result.assignment.cluster_id if result.assignment else None,
                "node_id": result.assignment.node_id if result.assignment else None,
                "node_name": result.assignment.node_name if result.assignment else None,
                "distance": result.assignment.distance if result.assignment else None,
                "total_ms": result.timing_ms.get("total_ms"),
                "result": result,
                "intervention": decision,
            }
            self.episode_results.append(entry)

            if self.verbose:
                name = result.assignment.node_name if result.assignment else "N/A"
                ms = result.timing_ms.get("total_ms", 0.0)
                intv = (
                    f" [INTERVENE: {decision.reason}]"
                    if (decision and decision.triggered)
                    else ""
                )
                print(
                    f"  [monitor] ep={self._episode_idx} t={self._timestep:3d}"
                    f" env={i} → {name}  ({ms:.1f} ms){intv}"
                )

        self._timestep += 1
        return action_dict

    def get_slice_influence(
        self,
        top_k: int = 20,
        aggregation_method: str = "sum",
        window_width_demo: int = 1,
        ascending: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rank training demonstrations by aggregate influence over the current rollout window.

        Stacks the rolling buffer of influence-score vectors into a
        ``(rollout_window, N_demo)`` block and calls
        :func:`~policy_doctor.computations.slice_influence.rank_demo_indices_by_slice_influence`.

        Args:
            top_k: Return only the top-k ranked demo indices.
            aggregation_method: ``"sum"`` or ``"mean"`` over the rollout window (rows).
            window_width_demo: If > 1, apply a left-aligned sliding window of this width
                over the demo axis (columns) before ranking.  Aggregates consecutive
                demo timesteps together — matches the window used during offline
                attribution.  Default ``1`` (no windowing).
            ascending: If True, return lowest-influence demos first (e.g. to find
                unrepresented training data).

        Returns:
            ``(sorted_demo_indices, sorted_scores)`` — each of length
            ``min(top_k, N_demo)``.  Empty arrays if the buffer is empty.
        """
        from policy_doctor.computations.slice_influence import rank_demo_indices_by_slice_influence

        if not self._embedding_buffer:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Score each buffered embedding lazily — no gradient pass, just matmul per step.
        scores_per_step = [
            self._classifier.score_embedding(e) for e in self._embedding_buffer
        ]
        block = np.stack(scores_per_step, axis=0)  # (rollout_window, N_demo)
        sorted_indices, sorted_scores, _ = rank_demo_indices_by_slice_influence(
            block,
            window_width_demo=window_width_demo,
            aggregation_method=aggregation_method,
            ascending=ascending,
        )
        return sorted_indices[:top_k], sorted_scores[:top_k]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)
