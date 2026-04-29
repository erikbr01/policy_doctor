"""Lightweight policy wrappers with the MonitoredPolicy-compatible interface.

All classes expose the three attributes that RobomimicDAggerRunner requires:
  - episode_results: List[Dict]
  - reset() -> None
  - predict_action(obs_dict: Dict) -> Dict
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np



class BarePolicy:
    """Wraps a raw BaseLowdimPolicy with the MonitoredPolicy-compatible interface.

    Use when you have a checkpoint but want to skip InfEmbed monitoring (--no_monitor).
    """

    def __init__(self, raw_policy: Any) -> None:
        self._policy = raw_policy
        self.episode_results: List[Dict] = []

    def reset(self) -> None:
        self._policy.reset()
        self.episode_results = []

    def predict_action(self, obs_dict: Dict) -> Dict:
        return self._policy.predict_action(obs_dict)
