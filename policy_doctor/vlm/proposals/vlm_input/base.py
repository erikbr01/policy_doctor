"""Abstract base for E2 VLM input builders.

A :class:`VLMInputBuilder` turns the rollout pool, the graph artefact, and a
request distribution into an ordered list of :class:`Message` objects ready for a
multimodal backend. Heavy deps (PIL etc.) are deferred to ``build_messages``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from policy_doctor.vlm.proposals.graph_representation.base import VLMArtefact
    from policy_doctor.vlm.proposals.pool import RolloutPool


@dataclass
class Message:
    role: str                       # "system" | "user" | "assistant"
    text_blocks: List[str] = field(default_factory=list)
    images: List[Path] = field(default_factory=list)


class VLMInputBuilder(ABC):
    @abstractmethod
    def build_messages(
        self,
        *,
        graph_artefact: "VLMArtefact",
        pool: "RolloutPool",
        condition: str,
        n_requests_per_type: Dict[str, int],
        json_schema: Dict[str, Any],
        history: Optional[List[Message]] = None,
        task_hint: str = "",
    ) -> List[Message]:
        """Build the full prompt/message sequence for one VLM call."""
