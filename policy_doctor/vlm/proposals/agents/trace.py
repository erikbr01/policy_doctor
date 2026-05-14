"""Structured per-call trace logging for one agent session.

Every tool call (and every assistant turn) is emitted as one JSONL row, so the
trace can be replayed, indexed, or rendered into figures with no extra parsing.

Trace artefact layout (set by ``run_e2_agent.py``):

    experiment_runs/<run_id>/agent_sessions/<condition>/seed_<n>/
        trace.jsonl              # per-event records
        conversation.json        # full provider-neutral message list
        submitted_requests.json  # final strategy
        rationale.txt            # finalize_strategy summary
        budget_summary.json
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from policy_doctor.vlm.proposals.agents.tools.types import (
    ImageBlock,
    TextBlock,
    ToolResult,
)


@dataclass
class TraceEvent:
    """One record in the trace JSONL."""

    seq: int
    timestamp: float
    kind: str                     # "assistant_turn" | "tool_call" | "tool_result" | "session_start" | "session_end"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.timestamp,
            "kind": self.kind,
            **self.payload,
        }


@dataclass
class SessionTrace:
    """Append-only event log for one session."""

    out_path: Optional[Path] = None
    events: List[TraceEvent] = field(default_factory=list)
    _seq: int = 0
    _fp: Optional[io.TextIOBase] = None

    # ---- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "SessionTrace":
        if self.out_path is not None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.out_path, "w", encoding="utf-8")  # noqa: SIM115
        return self

    def __exit__(self, *exc) -> None:
        with contextlib.suppress(Exception):
            if self._fp is not None:
                self._fp.flush()
                self._fp.close()

    # ---- emit ----------------------------------------------------------------

    def emit(self, kind: str, **payload: Any) -> TraceEvent:
        self._seq += 1
        ev = TraceEvent(seq=self._seq, timestamp=time.time(), kind=kind, payload=payload)
        self.events.append(ev)
        if self._fp is not None:
            self._fp.write(json.dumps(ev.to_json(), default=str) + "\n")
            self._fp.flush()
        return ev

    # ---- convenience -------------------------------------------------------

    def assistant_turn(self, *, text: Optional[str], tool_calls: List[Dict[str, Any]],
                        stop_reason: str, usage: Dict[str, int]) -> None:
        self.emit(
            "assistant_turn",
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    def tool_invocation(self, name: str, args: Dict[str, Any], *, tool_use_id: str) -> None:
        self.emit("tool_call", name=name, args=args, tool_use_id=tool_use_id)

    def tool_outcome(
        self,
        result: ToolResult,
        *,
        tool_use_id: str,
        cache_hit: bool = False,
        latency_ms: float = 0.0,
        budget_after: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.emit(
            "tool_result",
            name=result.name,
            ok=result.ok,
            tool_use_id=tool_use_id,
            cache_hit=cache_hit,
            latency_ms=round(latency_ms, 3),
            content=_summarize_content(result.content),
            metadata=result.metadata,
            budget_after=budget_after or {},
        )


def _summarize_content(content: List[Any]) -> List[Dict[str, Any]]:
    """Trace-friendly summary of result content (text length + image dims)."""
    out: List[Dict[str, Any]] = []
    for blk in content or []:
        if isinstance(blk, TextBlock):
            text = blk.text or ""
            out.append({
                "type": "text",
                "len": len(text),
                "preview": text[:240],
            })
        elif isinstance(blk, ImageBlock):
            try:
                w, h = blk.image.size
            except Exception:
                w, h = -1, -1
            out.append({
                "type": "image",
                "size": [w, h],
                "caption": blk.caption,
            })
        else:
            out.append({"type": "unknown"})
    return out
