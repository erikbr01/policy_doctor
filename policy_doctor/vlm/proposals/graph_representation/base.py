"""Abstract base for graph-representation plugins (E2).

A :class:`GraphRepresentation` turns a :class:`BehaviorGraph` plus a
:class:`RolloutPool` into a :class:`VLMArtefact` (images + text blocks) that the
:mod:`vlm_input` builders splice into the prompt. Heavy deps (matplotlib /
networkx) are deferred to ``render`` to keep this module import-cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.vlm.proposals.pool import RolloutPool


@dataclass
class VLMArtefact:
    """What the VLM sees on top of the rollout pool: graph image(s) + text blocks."""

    images: List[Path] = field(default_factory=list)
    text_blocks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphRepresentation(ABC):
    @abstractmethod
    def render(
        self,
        graph: "BehaviorGraph",
        pool: "RolloutPool",
        output_dir: Path,
    ) -> VLMArtefact:
        """Render the graph + pool as a VLMArtefact, writing any images under ``output_dir``."""
