"""Experiment condition enum and metadata.

Spec §1 defines four conditions; the codebase treats each as an opaque string
(used as ``source_condition`` on requests and as a queue label on the server).
We expose the canonical names here so configs, server, runner, and tests
agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Condition(str, Enum):
    """Pre-registered experiment conditions."""

    A_G = "A_G"          # Agent with full graph tool surface (Layers 1-4)
    A_NG = "A_NG"        # Agent with parallel no-graph tool surface
    H_NG = "H_NG"        # Human operator without graph access
    H_G = "H_G"          # Human operator with graph access (optional)


AGENT_CONDITIONS: Tuple[Condition, ...] = (Condition.A_G, Condition.A_NG)
HUMAN_CONDITIONS: Tuple[Condition, ...] = (Condition.H_NG, Condition.H_G)
GRAPH_CONDITIONS: Tuple[Condition, ...] = (Condition.A_G, Condition.H_G)
NO_GRAPH_CONDITIONS: Tuple[Condition, ...] = (Condition.A_NG, Condition.H_NG)


@dataclass(frozen=True)
class ConditionInfo:
    name: str
    is_agent: bool
    has_graph: bool
    description: str


CONDITION_INFO = {
    Condition.A_G: ConditionInfo(
        name="A_G",
        is_agent=True,
        has_graph=True,
        description="Agent with full graph tool surface (Layers 1-4).",
    ),
    Condition.A_NG: ConditionInfo(
        name="A_NG",
        is_agent=True,
        has_graph=False,
        description="Agent with parallel no-graph tool surface (rollouts + outcomes only).",
    ),
    Condition.H_NG: ConditionInfo(
        name="H_NG",
        is_agent=False,
        has_graph=False,
        description="Human operator with rollouts + outcomes only.",
    ),
    Condition.H_G: ConditionInfo(
        name="H_G",
        is_agent=False,
        has_graph=True,
        description="Human operator with full graph access (optional 4th condition).",
    ),
}


def parse_condition(raw: object) -> Condition:
    """Coerce strings / Condition values to a :class:`Condition`."""
    if isinstance(raw, Condition):
        return raw
    return Condition(str(raw).strip())
