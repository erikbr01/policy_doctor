"""Intervention rules for DAgger-style data collection in runtime monitoring.

An InterventionRule inspects each timestep's MonitorResult (plus a history window)
and returns an InterventionDecision indicating whether the policy should yield to an
expert (or trigger data collection).

Usage::

    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.monitoring.intervention import NodeValueThresholdRule

    node_values = graph.compute_values()  # Dict[int, float]
    rule = NodeValueThresholdRule(node_values=node_values, threshold=0.0)

    # In the control loop (via MonitoredPolicy):
    decision = rule.check(result, history)
    if decision.triggered:
        ...  # query expert / collect demonstration
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from policy_doctor.monitoring.base import MonitorResult


@dataclass
class InterventionDecision:
    """Result of applying an intervention rule at one timestep."""

    triggered: bool
    node_id: Optional[int] = None
    node_value: Optional[float] = None
    reason: str = ""


class InterventionRule(ABC):
    """Abstract base class for intervention rules.

    Subclasses implement :meth:`check`, which receives the current
    :class:`~policy_doctor.monitoring.base.MonitorResult` plus a list of
    recent results (history window) and returns an
    :class:`InterventionDecision`.

    The ``history`` list contains results from the current episode, most
    recent last.  Its length is bounded by ``MonitoredPolicy.max_influence_window``.
    Rules that don't need history can simply ignore it.
    """

    @abstractmethod
    def check(
        self,
        result: MonitorResult,
        history: List[MonitorResult],
    ) -> InterventionDecision:
        """Decide whether to intervene given the current monitor result.

        Args:
            result: Monitor result for the current timestep.
            history: Recent monitor results from the current episode
                (most recent last, bounded by ``max_influence_window``).

        Returns:
            :class:`InterventionDecision`.
        """

    def reset(self) -> None:
        """Reset any episode-level internal state.  Called by ``MonitoredPolicy.reset()``."""


class NodeValueThresholdRule(InterventionRule):
    """Trigger an intervention when the current node's behavior-graph value is below a threshold.

    Node values are Bellman state-values computed from the behavior graph's
    transition structure and terminal rewards (via
    :meth:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph.compute_values`).
    A low value indicates that the policy is in a state from which it is unlikely
    to reach a successful terminal state.

    Parameters
    ----------
    node_values:
        Dict mapping behavior graph node IDs to their Bellman values.
        Obtain via ``graph.compute_values()``.
    threshold:
        Trigger intervention when ``node_values[node_id] < threshold``.
        Typical range is ``[-1, 1]``; ``0.0`` is a reasonable default for
        policies where success reward=1 and failure reward=-1.
    """

    def __init__(self, node_values: Dict[int, float], threshold: float) -> None:
        self.node_values = node_values
        self.threshold = threshold

    def check(
        self,
        result: MonitorResult,
        history: List[MonitorResult],
    ) -> InterventionDecision:
        if result.assignment is None:
            return InterventionDecision(
                triggered=False,
                node_id=None,
                node_value=None,
                reason="no_assignment",
            )

        node_id = result.assignment.node_id
        value = self.node_values.get(node_id)

        if value is None:
            return InterventionDecision(
                triggered=False,
                node_id=node_id,
                node_value=None,
                reason="node_not_in_values",
            )

        triggered = value < self.threshold
        op = "<" if triggered else ">="
        return InterventionDecision(
            triggered=triggered,
            node_id=node_id,
            node_value=value,
            reason=f"value={value:.4f} {op} threshold={self.threshold:.4f}",
        )
