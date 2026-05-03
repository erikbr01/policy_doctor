"""Provider-neutral data types shared by every tool.

The session loop (:mod:`policy_doctor.vlm.proposals.agents.session`) is the
only place that knows about provider-specific message formats. Tools speak
:class:`ToolResult` with content blocks shaped after Anthropic's
``tool_result`` content (text + image), since that's the most expressive of
the major APIs and translates trivially to Gemini / OpenAI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from PIL import Image


# Content block kinds. Kept narrow on purpose; if a tool needs to return raw
# bytes, write them to disk and return a text block with the path.
ContentKind = Literal["text", "image"]


@dataclass
class TextBlock:
    """Plain UTF-8 string. Always cheap."""

    text: str
    kind: ContentKind = "text"


@dataclass
class ImageBlock:
    """In-memory PIL image. Counts against the visual budget on first emission."""

    image: Image.Image
    # Optional caption rendered inline by the session loop (e.g. "storyboard for r0023").
    caption: Optional[str] = None
    kind: ContentKind = "image"


ContentBlock = Any  # TextBlock | ImageBlock — kept loose to avoid Union noise.


@dataclass
class ToolResult:
    """The uniform return type of every tool.

    Attributes
    ----------
    name:
        Tool name (matches the JSON-schema ``name``).
    ok:
        ``True`` when the tool ran to completion. ``False`` for validation
        errors, budget exhaustion, missing rollouts, etc.; the agent sees the
        error in the ``content`` text block and can recover on the next turn.
    content:
        Ordered list of :class:`TextBlock` / :class:`ImageBlock`. The session
        loop concatenates these into the backend-specific tool_result format.
    metadata:
        Diagnostic info (cumulative budget, cache_hit flag, latency, …) that is
        **not** shown to the model. Persisted to the trace.
    """

    name: str
    ok: bool
    content: List[ContentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience constructors -------------------------------------------------

    @classmethod
    def text(cls, name: str, text: str, **metadata: Any) -> "ToolResult":
        return cls(name=name, ok=True, content=[TextBlock(text=text)], metadata=metadata)

    @classmethod
    def error(cls, name: str, message: str, *, code: str = "tool_error", **metadata: Any) -> "ToolResult":
        meta = dict(metadata)
        meta["error_code"] = code
        return cls(
            name=name,
            ok=False,
            content=[TextBlock(text=f"[error:{code}] {message}")],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Tool spec (the registry's value type)
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """One callable tool, paired with its JSON-schema declaration.

    The session loop builds the ``tools=`` list it sends to the backend by
    flattening ``[spec.declaration() for spec in registry.values()]``. The
    ``func`` is invoked with the parsed JSON arguments dict, after schema
    validation done in the loop, and must return a :class:`ToolResult`.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], ToolResult]

    # Cost classification — drives BudgetTracker accounting.
    # "cheap"  : Layer 1, Layer 3, textual Layer 2, all of Layer 4 except submission noise.
    # "visual" : Layer 2 storyboard calls.
    # "video"  : Layer 2 full-video calls.
    cost: Literal["cheap", "visual", "video"] = "cheap"

    # If True, this tool ends the session when called successfully.
    is_terminal: bool = False

    def declaration(self) -> Dict[str, Any]:
        """Anthropic-shaped tool declaration: ``{name, description, input_schema}``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
