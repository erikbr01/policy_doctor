"""Budget tracking and visual-result caching for one agent session.

Two concerns, intentionally co-located:

1. :class:`BudgetTracker` enforces per-session limits on tool calls
   (cheap / visual / video). When a limit would be exceeded, the tool returns
   a structured error rather than silently failing — the agent sees it and can
   adapt. Also emits warnings on the final 5 calls before exhaustion.

2. :class:`ResultCache` memoizes tool results by (tool_name, args). Visual
   calls cost zero on the second hit, which encourages the agent to inspect a
   slice once and refer back to it instead of re-fetching.

Both are owned by :class:`SessionContext` and consulted by every tool handler.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

from policy_doctor.vlm.proposals.agents.tools.types import ToolResult


CostKind = Literal["cheap", "visual", "video"]


@dataclass
class BudgetConfig:
    """Per-session budget. Defaults match Section 5.1 of the spec."""

    max_tool_calls: int = 80
    max_visual_calls: int = 30
    max_video_calls: int = 5
    max_session_duration_s: float = 20 * 60.0
    # Calls remaining at which to start emitting warnings (per pool).
    warning_remaining_threshold: int = 5

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "BudgetConfig":
        d = dict(d or {})
        return cls(
            max_tool_calls=int(d.get("max_tool_calls", 80)),
            max_visual_calls=int(d.get("max_visual_calls", 30)),
            max_video_calls=int(d.get("max_video_calls", 5)),
            max_session_duration_s=float(
                d.get("max_session_duration_s", d.get("max_session_duration_min", 20) * 60.0)
            ),
            warning_remaining_threshold=int(d.get("warning_remaining_threshold", 5)),
        )


@dataclass
class BudgetState:
    """Mutable counters. Persisted to ``budget_summary.json`` at session end."""

    n_tool_calls: int = 0
    n_visual_calls: int = 0
    n_video_calls: int = 0
    n_cache_hits: int = 0
    started_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Errors emitted as structured ToolResults so the agent can read them.
# ---------------------------------------------------------------------------


_EXHAUSTED_INSTRUCTION = (
    "Submission tools (propose_collection_request, revise_request, "
    "delete_request, list_submitted_requests) and finalize_strategy are still "
    "callable — submit any remaining strategy now and then call finalize_strategy "
    "with a brief rationale. Do not attempt further exploration."
)


def _budget_exhausted(tool_name: str, kind: CostKind) -> ToolResult:
    return ToolResult.error(
        tool_name,
        f"Budget for {kind} tool calls exhausted. {_EXHAUSTED_INSTRUCTION}",
        code="budget_exhausted",
        budget_kind=kind,
    )


def _session_timeout(tool_name: str) -> ToolResult:
    return ToolResult.error(
        tool_name,
        f"Session wall-clock budget exhausted. {_EXHAUSTED_INSTRUCTION}",
        code="session_timeout",
    )


@dataclass
class BudgetTracker:
    """Owns counts; enforces limits; emits structured errors and warnings."""

    config: BudgetConfig = field(default_factory=BudgetConfig)
    state: BudgetState = field(default_factory=BudgetState)

    # ---- accounting ----------------------------------------------------------

    def remaining(self, kind: CostKind) -> int:
        if kind == "cheap":
            return max(0, self.config.max_tool_calls - self.state.n_tool_calls)
        if kind == "visual":
            return max(0, self.config.max_visual_calls - self.state.n_visual_calls)
        if kind == "video":
            return max(0, self.config.max_video_calls - self.state.n_video_calls)
        raise ValueError(f"unknown cost kind {kind!r}")

    def session_time_remaining(self) -> float:
        return max(0.0, self.config.max_session_duration_s - (time.time() - self.state.started_at))

    # ---- guard ---------------------------------------------------------------

    def check(self, tool_name: str, kind: CostKind, *, bypass: bool = False) -> Optional[ToolResult]:
        """Return a :class:`ToolResult` error if the call cannot proceed, else ``None``.

        ``bypass=True`` (set by ``ToolSpec.is_terminal`` or ``ToolSpec.bypass_budget``)
        skips every gate so the agent can always submit and finalize even after
        exploration budget is exhausted.
        """
        if bypass:
            return None

        # Wall clock first — affects every subsequent call.
        if self.session_time_remaining() <= 0:
            return _session_timeout(tool_name)

        # Total tool-call budget — applies to ALL non-bypass calls.
        if self.state.n_tool_calls >= self.config.max_tool_calls:
            return _budget_exhausted(tool_name, "cheap")

        # Per-kind sub-budgets.
        if kind == "visual" and self.state.n_visual_calls >= self.config.max_visual_calls:
            return _budget_exhausted(tool_name, "visual")
        if kind == "video" and self.state.n_video_calls >= self.config.max_video_calls:
            return _budget_exhausted(tool_name, "video")

        return None

    # ---- charge --------------------------------------------------------------

    def charge(self, kind: CostKind, *, bypass: bool = False) -> Dict[str, Any]:
        """Increment counters for a successful, non-cached call. Return budget snapshot."""
        if not bypass:
            self.state.n_tool_calls += 1
            if kind == "visual":
                self.state.n_visual_calls += 1
            elif kind == "video":
                # Video calls also count against the visual pool, since a video
                # is strictly more expensive than a storyboard.
                self.state.n_video_calls += 1
                self.state.n_visual_calls += 1
        return self.snapshot()

    def note_cache_hit(self) -> None:
        """A cached visual call was returned. Charge nothing; tally for traces."""
        self.state.n_cache_hits += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_tool_calls": self.state.n_tool_calls,
            "n_visual_calls": self.state.n_visual_calls,
            "n_video_calls": self.state.n_video_calls,
            "n_cache_hits": self.state.n_cache_hits,
            "remaining_tool_calls": self.remaining("cheap"),
            "remaining_visual_calls": self.remaining("visual"),
            "remaining_video_calls": self.remaining("video"),
            "session_time_remaining_s": round(self.session_time_remaining(), 2),
        }

    # ---- warnings ------------------------------------------------------------

    def warning_for(self, kind: CostKind) -> Optional[Dict[str, Any]]:
        """Return a non-fatal warning dict when remaining ≤ ``warning_remaining_threshold``.

        Attached to ``ToolResult.metadata`` so the session loop can surface it
        in a system-injected reminder turn.
        """
        thr = self.config.warning_remaining_threshold
        rem = self.remaining(kind)
        if 0 < rem <= thr:
            return {
                "warning": f"approaching_{kind}_budget",
                "remaining": rem,
            }
        return None


# ---------------------------------------------------------------------------
# Result cache — content-addressed by (tool_name, normalized args).
# ---------------------------------------------------------------------------


def _normalize(args: Dict[str, Any]) -> str:
    """Stable JSON encoding for cache keying (sorted keys, no whitespace)."""
    return json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)


def _hash_key(tool_name: str, args: Dict[str, Any]) -> str:
    payload = f"{tool_name}|{_normalize(args)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Status-line formatter — shared between the in-process AgentSession and the
# MCP server. Both paths append this to every tool_result so the model sees
# how much budget is left and how many requests have been submitted.
# ---------------------------------------------------------------------------


def format_status_line(
    *,
    n_submitted: int,
    target_n_submissions: int,
    n_tool_calls: int,
    max_tool_calls: int,
    n_visual: int,
    max_visual: int,
    warning: Optional[Dict[str, Any]] = None,
) -> str:
    """One-line session status appended to every tool_result the agent sees.

    Without this, the agent has no way to know how much budget it has left
    or how many requests it has already submitted — leading to the
    "tool-call drift" failure mode where it explores forever.

    When a budget warning fires, the line is prefixed with REMINDER so the
    agent can adapt before being forced to terminate.
    """
    base = (
        f"[session: {n_submitted}/{target_n_submissions} requests submitted, "
        f"{n_tool_calls}/{max_tool_calls} tool calls used, "
        f"{n_visual}/{max_visual} visual calls used. "
        "Call finalize_strategy when your strategy is complete.]"
    )
    if warning:
        kind = warning.get("warning", "").replace("approaching_", "").replace("_budget", "")
        return (
            f"REMINDER — {warning.get('warning', 'budget low')}: "
            f"{warning.get('remaining', '?')} {kind} calls remain. "
            "Submit any pending requests and call finalize_strategy now. " + base
        )
    return base


@dataclass
class ResultCache:
    """In-memory memo of (tool, args) -> ToolResult.

    Per the spec: cached visual calls do **not** charge the budget on
    subsequent hits. Cheap textual calls are also cached for parity (no token
    cost on repeat). The session loop calls :meth:`get` before invoking the
    tool and :meth:`put` after a successful charge.
    """

    enabled: bool = True
    _store: Dict[str, ToolResult] = field(default_factory=dict)

    def key(self, tool_name: str, args: Dict[str, Any]) -> str:
        return _hash_key(tool_name, args)

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        if not self.enabled:
            return None
        return self._store.get(self.key(tool_name, args))

    def put(self, tool_name: str, args: Dict[str, Any], result: ToolResult) -> None:
        if not self.enabled or not result.ok:
            return
        self._store[self.key(tool_name, args)] = result

    def __len__(self) -> int:
        return len(self._store)
