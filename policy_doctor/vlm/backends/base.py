"""Abstract VLM backend — no heavy imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING, Tuple

from PIL import Image

if TYPE_CHECKING:
    from policy_doctor.vlm.proposals.vlm_input.base import Message


# ---------------------------------------------------------------------------
# Tool-use primitives (used by the agentic proposal loop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """Provider-neutral tool invocation requested by the model.

    Attributes
    ----------
    id:
        Provider-assigned tool_use id; the loop attaches the corresponding
        ``tool_result`` content block referencing this id on the next turn.
    name:
        Tool name (matches the registered :class:`ToolSpec.name`).
    arguments:
        Parsed argument dict; backends are responsible for JSON parsing.
    provider_metadata:
        Opaque per-provider data the loop must echo back on the next turn
        (e.g. Gemini-3's ``thought_signature``). Not interpreted by the
        loop; the producing backend embeds it on parse and the same backend
        consumes it when re-encoding the turn into the next request.
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    provider_metadata: Optional[Dict[str, Any]] = None


@dataclass
class TokenUsage:
    """Token accounting for one assistant turn. Kept loose to match providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class AssistantTurn:
    """One model turn: optional text, optional tool calls, stop reason, usage."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | stop_sequence
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any = None  # opaque provider response, kept for debugging / traces

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class VLMBackend(ABC):
    """Pluggable vision-language model for slice captions."""

    name: str = "base"

    @abstractmethod
    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Return a short semantic description for the given frames."""

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Summarize many per-slice captions for one behavior cluster (text-only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement behavior-level text summarization."
        )

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Judge whether per-slice captions in one cluster are mutually coherent (text-only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement slice-caption coherency judging."
        )

    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
    ) -> str:
        """Classify *query_images* into one of the K example groups.

        *example_sets*: ordered list of ``(opaque_label, images)`` pairs where
        *opaque_label* is a display string like "Group A" and *images* are
        representative frames for that group (typically storyboard composites,
        one per example slice).

        The backend builds a single multimodal prompt::

            [system_prompt]
            [user_preamble]
            Group A: [img1] [img2] ...
            Group B: [img1] ...
            ...
            Query:
            [query_img1] ...
            [user_prompt]

        Returns the raw text response (parsed by the caller).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement slice classification."
        )

    def generate_structured(
        self,
        *,
        messages: Sequence["Message"],
        json_schema: Dict[str, Any],
        max_retries: int = 3,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> Tuple[str, Any, int]:
        """Generate JSON matching ``json_schema`` from a multi-turn ``messages`` list.

        Returns ``(raw_response_text, parsed_json, n_retries_used)``.

        Default implementation raises NotImplementedError; backends used for the
        Experiment-E2 proposal generator override this. The propose pipeline also
        provides a generic shim that wraps any backend exposing a plain
        ``generate(messages, *, temperature, seed) -> str`` method, so most local
        backends only need to implement ``generate``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_structured."
        )

    # ------------------------------------------------------------------
    # Tool-use primitive (used by the agentic proposal loop)
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> AssistantTurn:
        """One model turn over a multi-turn message list with tool declarations.

        Provider-neutral message format follows Anthropic's content-block schema
        (text / image / tool_use / tool_result blocks); backends translate to
        their native API. ``tools`` are
        ``{"name", "description", "input_schema"}`` dicts (Anthropic shape).

        Returns an :class:`AssistantTurn`. The session loop is responsible for
        running the tools and constructing the next message.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement chat_with_tools."
        )
