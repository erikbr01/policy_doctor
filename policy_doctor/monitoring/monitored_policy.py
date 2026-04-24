"""MonitoredPolicy: wraps any BaseLowdimPolicy with per-timestep behavior classification.

The wrapped policy's predict_action() is intercepted: after each call the obs +
predicted action are classified by the TrajectoryClassifier and the result is stored
in episode_results.

Compatible with any env runner that calls policy.predict_action(obs_dict) and
policy.reset() between episodes (e.g. RobomimicLowdimRunner).

Environment: requires the cupid conda env (diffusion_policy + infembed).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier


class MonitoredPolicy:
    """Wraps any BaseLowdimPolicy to add per-timestep behavior monitoring.

    After each ``predict_action()`` call, the obs + predicted action are classified
    and stored in :attr:`episode_results`.  Call :meth:`reset` between episodes (the
    env runner does this automatically).

    Parameters
    ----------
    policy:
        Any ``BaseLowdimPolicy`` or ``BaseImagePolicy``.
    classifier:
        A loaded :class:`~policy_doctor.monitoring.TrajectoryClassifier`
        (use ``mode="rollout"`` for live env data).
    verbose:
        If True, print the node assignment at each timestep.
    """

    def __init__(
        self,
        policy: Any,
        classifier: TrajectoryClassifier,
        verbose: bool = False,
    ) -> None:
        self._policy = policy
        self._classifier = classifier
        self.verbose = verbose

        self.episode_results: List[Dict] = []
        self._episode_idx: int = 0
        self._timestep: int = 0

    def reset(self) -> None:
        self._policy.reset()
        if self._timestep > 0:
            self._episode_idx += 1
        self._timestep = 0

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
            result = self._classifier.classify_sample(obs[i], action[i])
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
            }
            self.episode_results.append(entry)
            if self.verbose:
                name = result.assignment.node_name if result.assignment else "N/A"
                ms = result.timing_ms.get("total_ms", 0.0)
                print(f"  [monitor] ep={self._episode_idx} t={self._timestep:3d} env={i} → {name}  ({ms:.1f} ms)")

        self._timestep += 1
        return action_dict

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)
