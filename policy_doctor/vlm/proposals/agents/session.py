"""Backend-agnostic tool-use loop for one agent session.

The loop is the only piece in the agentic stack that knows about message
construction. Everything else (tools, backends, conditions) plugs in via
narrow interfaces, so adding another backend (Gemini, OpenAI, …) is a single
``chat_with_tools`` implementation.

Invariants:

* The loop never inspects a tool's ``content`` or ``metadata`` — it only
  trusts the budget tracker and the trace.
* Budget is checked *before* each tool dispatch. Exhaustion returns a
  structured error to the agent on the next turn; it does not raise.
* Cached visual results return immediately without charging budget.
* Termination conditions (in order): ``finalize_strategy`` was called,
  ``max_turns`` reached, session timeout, model returned ``end_turn`` with
  no tool calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from policy_doctor.vlm.backends.base import (
    AssistantTurn,
    ToolCall,
    VLMBackend,
)
from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools.types import (
    ImageBlock,
    TextBlock,
    ToolResult,
    ToolSpec,
)
from policy_doctor.vlm.proposals.agents.trace import SessionTrace


# Anthropic-shaped image source format. The session loop converts in-memory
# PIL images to base64 JPEG once (here) so backends don't each re-encode.
def _image_block_for_message(img) -> Dict[str, Any]:
    import base64
    import io as _io

    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("ascii"),
        },
    }


def _content_to_message_blocks(result: ToolResult) -> List[Dict[str, Any]]:
    """Translate ToolResult.content into Anthropic-shaped tool_result content blocks."""
    blocks: List[Dict[str, Any]] = []
    for blk in result.content or []:
        if isinstance(blk, TextBlock):
            blocks.append({"type": "text", "text": blk.text})
        elif isinstance(blk, ImageBlock):
            if blk.caption:
                blocks.append({"type": "text", "text": blk.caption})
            blocks.append(_image_block_for_message(blk.image))
    if not blocks:
        # Some providers refuse empty tool_result; emit a sentinel.
        blocks.append({"type": "text", "text": ""})
    return blocks


# ---------------------------------------------------------------------------
# Result of one session
# ---------------------------------------------------------------------------


@dataclass
class SessionResult:
    """What ``AgentSession.run`` returns when the loop terminates."""

    condition: str
    seed: int
    submitted_requests: List[Dict[str, Any]] = field(default_factory=list)
    rationale: Optional[str] = None
    n_turns: int = 0
    n_tool_calls: int = 0
    n_failed_tool_calls: int = 0
    stop_reason: str = "end_turn"  # finalize | budget | turn_limit | model_end | error
    budget_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "stop_reason": self.stop_reason,
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "n_failed_tool_calls": self.n_failed_tool_calls,
            "rationale": self.rationale,
            "n_submitted": len(self.submitted_requests),
            "submitted_requests": self.submitted_requests,
            "budget_summary": self.budget_summary,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# AgentSession.run — the loop
# ---------------------------------------------------------------------------


@dataclass
class AgentSession:
    """One agent session: backend + context + tool registry + trace."""

    backend: VLMBackend
    ctx: SessionContext
    tools: Dict[str, ToolSpec]
    system_prompt: str
    user_message: str
    seed: int = 0
    temperature: float = 0.3
    max_tokens: int = 4096
    max_turns: int = 100
    trace: Optional[SessionTrace] = None
    out_dir: Optional[Path] = None  # Where to dump conversation.json + submitted_requests.json

    # Internals -- not user-facing.
    _messages: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------

    def run(self) -> SessionResult:
        """Drive the tool-use loop until a terminal condition fires.

        Always returns a :class:`SessionResult` — even on backend exceptions,
        which are recorded in ``error`` so an outer aggregator can keep going.
        """
        self._messages = [{"role": "user", "content": [{"type": "text", "text": self.user_message}]}]
        result = SessionResult(condition=self.ctx.condition, seed=self.seed)

        if self.trace is not None:
            self.trace.emit(
                "session_start",
                condition=self.ctx.condition,
                seed=self.seed,
                tool_names=list(self.tools.keys()),
                budget_config=self._budget_config_dict(),
            )

        try:
            while result.n_turns < self.max_turns:
                turn = self.backend.chat_with_tools(
                    messages=self._messages,
                    tools=[spec.declaration() for spec in self.tools.values()],
                    system=self.system_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=self.seed,
                )
                result.n_turns += 1
                self._record_assistant_turn(turn)

                if not turn.has_tool_calls:
                    result.stop_reason = "model_end"
                    break

                terminal_called = False
                for call in turn.tool_calls:
                    tool_result, charged_kind, cache_hit, latency_ms = self._dispatch(call)
                    result.n_tool_calls += 1
                    if not tool_result.ok:
                        result.n_failed_tool_calls += 1
                    self._record_tool_result(call, tool_result, charged_kind, cache_hit, latency_ms)
                    if self._is_terminal(call.name) and tool_result.ok:
                        terminal_called = True

                if terminal_called or self.ctx.finalized:
                    result.stop_reason = "finalize"
                    break

                # Wall-clock check after every turn.
                if self.ctx.budget.session_time_remaining() <= 0:
                    result.stop_reason = "session_timeout"
                    break

            else:
                result.stop_reason = "turn_limit"

        except Exception as e:  # backend / dispatch crash
            result.stop_reason = "error"
            result.error = f"{type(e).__name__}: {e}"
            if self.trace is not None:
                self.trace.emit("error", message=result.error)

        # Final state.
        result.submitted_requests = [sr.to_dict() for sr in self.ctx.submitted]
        result.rationale = self.ctx.rationale
        result.budget_summary = self.ctx.budget.snapshot()

        if self.trace is not None:
            self.trace.emit("session_end", **result.to_dict())

        if self.out_dir is not None:
            self._dump_session(result)

        return result

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, call: ToolCall
    ) -> Tuple[ToolResult, str, bool, float]:
        spec = self.tools.get(call.name)
        if spec is None:
            return (
                ToolResult.error(call.name, f"unknown tool {call.name!r}", code="unknown_tool"),
                "cheap",
                False,
                0.0,
            )

        # Cache hit?
        cached = self.ctx.cache.get(call.name, call.arguments)
        if cached is not None:
            self.ctx.budget.note_cache_hit()
            return cached, spec.cost, True, 0.0

        # Cost class can escalate (e.g. get_*_video with format='video').
        kind = self._effective_cost(spec, call.arguments)

        # Pre-flight budget check.
        err = self.ctx.budget.check(call.name, kind, is_terminal=spec.is_terminal)
        if err is not None:
            return err, kind, False, 0.0

        # Run.
        t0 = time.time()
        try:
            result = spec.func(call.arguments)
        except Exception as e:
            result = ToolResult.error(
                call.name,
                f"tool raised: {type(e).__name__}: {e}",
                code="tool_exception",
            )
        latency_ms = (time.time() - t0) * 1000.0

        # Charge if successful.
        if result.ok:
            budget_snap = self.ctx.budget.charge(kind, is_terminal=spec.is_terminal)
            warn = self.ctx.budget.warning_for(kind)
            if warn:
                result.metadata.setdefault("budget_warning", warn)
            result.metadata.setdefault("budget", budget_snap)
            # Cache successful results so re-fetches are free.
            self.ctx.cache.put(call.name, call.arguments, result)
        return result, kind, False, latency_ms

    def _effective_cost(self, spec: ToolSpec, args: Dict[str, Any]) -> str:
        """A few tools escalate cost based on args (storyboard vs video)."""
        if spec.cost == "visual" and (args or {}).get("format") == "video":
            return "video"
        return spec.cost

    def _is_terminal(self, name: str) -> bool:
        spec = self.tools.get(name)
        return bool(spec and spec.is_terminal)

    # ------------------------------------------------------------------
    # Message bookkeeping
    # ------------------------------------------------------------------

    def _record_assistant_turn(self, turn: AssistantTurn) -> None:
        """Append the assistant's turn to the message list and trace."""
        content: List[Dict[str, Any]] = []
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        for tc in turn.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        self._messages.append({"role": "assistant", "content": content})

        if self.trace is not None:
            self.trace.assistant_turn(
                text=turn.text,
                tool_calls=[
                    {"id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in turn.tool_calls
                ],
                stop_reason=turn.stop_reason,
                usage={
                    "input_tokens": turn.usage.input_tokens,
                    "output_tokens": turn.usage.output_tokens,
                    "cache_read_input_tokens": turn.usage.cache_read_input_tokens,
                    "cache_creation_input_tokens": turn.usage.cache_creation_input_tokens,
                },
            )

    def _record_tool_result(
        self,
        call: ToolCall,
        result: ToolResult,
        charged_kind: str,
        cache_hit: bool,
        latency_ms: float,
    ) -> None:
        """Append the tool_result to the message list and trace."""
        # Append the tool_result block to the most recent user message — or
        # start a fresh user message if the prior was assistant.
        block = {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": _content_to_message_blocks(result),
            "is_error": not result.ok,
        }
        # Anthropic expects tool_result blocks inside a user message, batched
        # alongside any other tool_results for the same assistant turn.
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages[-1]["content"].append(block)
        else:
            self._messages.append({"role": "user", "content": [block]})

        if self.trace is not None:
            self.trace.tool_invocation(call.name, call.arguments, tool_use_id=call.id)
            self.trace.tool_outcome(
                result,
                tool_use_id=call.id,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                budget_after=self.ctx.budget.snapshot(),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _budget_config_dict(self) -> Dict[str, Any]:
        cfg = self.ctx.budget.config
        return {
            "max_tool_calls": cfg.max_tool_calls,
            "max_visual_calls": cfg.max_visual_calls,
            "max_video_calls": cfg.max_video_calls,
            "max_session_duration_s": cfg.max_session_duration_s,
        }

    def _dump_session(self, result: SessionResult) -> None:
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # conversation.json — message list with images as captions only (no base64 blobs).
        conv = _strip_image_payloads(self._messages)
        (out / "conversation.json").write_text(json.dumps(conv, indent=2, default=str))
        (out / "submitted_requests.json").write_text(
            json.dumps(result.submitted_requests, indent=2, default=str)
        )
        if result.rationale:
            (out / "rationale.txt").write_text(result.rationale)
        (out / "budget_summary.json").write_text(
            json.dumps(result.budget_summary, indent=2, default=str)
        )
        (out / "session_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )


def _strip_image_payloads(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of ``messages`` with base64 image data replaced by a stub."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "image":
                    new_content.append({"type": "image", "source": {"type": "stripped"}})
                elif isinstance(blk, dict) and blk.get("type") == "tool_result":
                    inner = blk.get("content") or []
                    new_inner = []
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("type") == "image":
                            new_inner.append({"type": "image", "source": {"type": "stripped"}})
                        else:
                            new_inner.append(sub)
                    new_content.append({**blk, "content": new_inner})
                else:
                    new_content.append(blk)
            out.append({**m, "content": new_content})
        else:
            out.append(m)
    return out
