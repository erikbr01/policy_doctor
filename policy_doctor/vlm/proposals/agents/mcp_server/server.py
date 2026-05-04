"""Stdio MCP server exposing the agent tool surface.

Lifecycle:

* On startup, read configuration from environment variables (see
  ``_load_config_from_env``).
* Build the SessionContext (graph + pool + classifier optional + budget) and
  the per-condition tool registry once.
* Expose every tool via the MCP ``call_tool`` handler. The MCP client (Cursor,
  Claude Desktop, etc.) is responsible for the model side — we only handle
  the server side.
* Submitted requests are written to ``$POLICY_DOCTOR_OUT_DIR/cursor_session/``
  incrementally so the user can see them in the file system as the session
  progresses.

Because Cursor launches MCP servers per-project and passes config through
``mcp.json``, all configuration here goes through env vars rather than CLI
flags. This is the conventional pattern across MCP servers.

Required env vars
-----------------

* ``POLICY_DOCTOR_CLUSTERING_DIR``     path to the saved clustering result
* ``POLICY_DOCTOR_POOL_EPISODES_DIR``  path to eval_save_episodes output

Optional env vars
-----------------

* ``POLICY_DOCTOR_CONDITION``          ``A_G`` (default) | ``A_NG``
* ``POLICY_DOCTOR_OUT_DIR``            where to persist submissions
                                       (default ``./policy_doctor_mcp_session``)
* ``POLICY_DOCTOR_TASK_HINT``          free-text task description
* ``POLICY_DOCTOR_STORYBOARD_DIR``     storyboard sidecar dir
* ``POLICY_DOCTOR_VIDEO_DIR``          video sidecar dir
* ``POLICY_DOCTOR_RAW_STATES_DIR``     raw .npz / .pkl state arrays
* ``POLICY_DOCTOR_MAX_VISUAL_CALLS``   cap visual calls (default unlimited)
* ``POLICY_DOCTOR_MAX_TOOL_CALLS``     cap total tool calls (default unlimited)
* ``POLICY_DOCTOR_KIN_STRATEGY``       ``raw_states`` (default) | ``cluster_stats``

The MCP SDK is an optional dependency — install with ``pip install mcp``
before running this server.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Module logger sends to stderr; stdout is the MCP transport.
logger = logging.getLogger("policy_doctor.mcp_server")


_INSTALL_HINT = (
    "The 'mcp' package is required for this server.\n"
    "Install with:  pip install mcp\n"
    "Or:            pip install 'mcp[cli]'  (also installs mcp-inspector)"
)


def _require_mcp():
    try:
        import mcp  # noqa: F401
        from mcp.server import Server  # noqa: F401
        from mcp.server.stdio import stdio_server  # noqa: F401
        from mcp.types import (
            ImageContent,  # noqa: F401
            TextContent,  # noqa: F401
            Tool,  # noqa: F401
        )
        return True
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_path(key: str) -> Optional[Path]:
    val = os.environ.get(key)
    return Path(val) if val else None


def _env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _load_config_from_env() -> Dict[str, Any]:
    clustering_dir = _env_path("POLICY_DOCTOR_CLUSTERING_DIR")
    pool_dir = _env_path("POLICY_DOCTOR_POOL_EPISODES_DIR")
    if clustering_dir is None or pool_dir is None:
        raise RuntimeError(
            "POLICY_DOCTOR_CLUSTERING_DIR and POLICY_DOCTOR_POOL_EPISODES_DIR "
            "are required env vars for the MCP server."
        )
    return {
        "clustering_dir": clustering_dir,
        "pool_episodes_dir": pool_dir,
        "condition": os.environ.get("POLICY_DOCTOR_CONDITION", "A_G"),
        "out_dir": _env_path("POLICY_DOCTOR_OUT_DIR") or Path.cwd() / "policy_doctor_mcp_session",
        "task_hint": os.environ.get("POLICY_DOCTOR_TASK_HINT", ""),
        "storyboard_dir": _env_path("POLICY_DOCTOR_STORYBOARD_DIR"),
        "video_dir": _env_path("POLICY_DOCTOR_VIDEO_DIR"),
        "raw_states_dir": _env_path("POLICY_DOCTOR_RAW_STATES_DIR"),
        # None means unlimited (the BudgetTracker treats huge values as
        # effectively unlimited for the purposes of an interactive session).
        "max_tool_calls": _env_int("POLICY_DOCTOR_MAX_TOOL_CALLS", default=10_000),
        "max_visual_calls": _env_int("POLICY_DOCTOR_MAX_VISUAL_CALLS", default=10_000),
        "max_video_calls": _env_int("POLICY_DOCTOR_MAX_VIDEO_CALLS", default=10_000),
        "kin_strategy": os.environ.get("POLICY_DOCTOR_KIN_STRATEGY", "raw_states"),
    }


# ---------------------------------------------------------------------------
# Build the SessionContext + tool registry
# ---------------------------------------------------------------------------


def build_context_and_tools(cfg: Dict[str, Any]):
    """Load graph + pool + tool registry from env-derived config."""
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path
    from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
    from policy_doctor.vlm.proposals.agents.context import SessionContext
    from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
    from policy_doctor.vlm.proposals.pool import RolloutPool

    pool = RolloutPool.from_episodes_dir(
        cfg["pool_episodes_dir"],
        storyboard_dir=cfg.get("storyboard_dir"),
        video_dir=cfg.get("video_dir"),
    )
    labels, metadata, manifest = load_clustering_result_from_path(cfg["clustering_dir"])
    graph = BehaviorGraph.from_cluster_assignments(
        labels, metadata, level=manifest.get("level", "rollout")
    )

    budget_cfg = BudgetConfig(
        max_tool_calls=cfg["max_tool_calls"],
        max_visual_calls=cfg["max_visual_calls"],
        max_video_calls=cfg["max_video_calls"],
        # Effectively unlimited wall-clock for an interactive session.
        max_session_duration_s=24 * 60 * 60,
    )
    ctx = SessionContext.build(
        condition=cfg["condition"],
        graph=graph,
        pool=pool,
        cluster_labels=labels,
        cluster_metadata=metadata,
        raw_states_dir=cfg.get("raw_states_dir"),
        storyboards_dir=cfg.get("storyboard_dir"),
        videos_dir=cfg.get("video_dir"),
        budget_config=budget_cfg,
        task_hint=cfg.get("task_hint", ""),
        config={"kinematic_summary_strategy": cfg.get("kin_strategy", "raw_states")},
    )
    tools = build_tool_registry(cfg["condition"], ctx)
    return ctx, tools


# ---------------------------------------------------------------------------
# Tool result translation (ours → MCP types)
# ---------------------------------------------------------------------------


def _decorate_with_status(
    items: List[Any],
    *,
    ctx,
    target_n_submissions: int,
    warning: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Append a ``[session: …]`` text block to MCP content items.

    Mirrors the in-process AgentSession behavior so the MCP client sees the
    same budget signals — without this, an interactive Cursor / Claude Code
    session has no visible deadline and the model may explore indefinitely.

    When the budget tracker reports a warning (low remaining or exhausted),
    the line is prefixed with ``REMINDER —`` and an explicit instruction to
    submit pending requests and finalize.
    """
    from mcp.types import TextContent

    from policy_doctor.vlm.proposals.agents.budget import format_status_line

    cfg = ctx.budget.config
    status_text = format_status_line(
        n_submitted=len(ctx.submitted),
        target_n_submissions=target_n_submissions,
        n_tool_calls=ctx.budget.state.n_tool_calls,
        max_tool_calls=cfg.max_tool_calls,
        n_visual=ctx.budget.state.n_visual_calls,
        max_visual=cfg.max_visual_calls,
        warning=warning,
    )
    return list(items) + [TextContent(type="text", text=status_text)]


def tool_result_to_mcp_content(result):
    """Translate a :class:`ToolResult` into a list of MCP content items.

    Returns ``List[TextContent | ImageContent]``. Image data is base64
    encoded to match MCP's wire format. Errors are returned as TextContent
    with the structured error preamble preserved.
    """
    from mcp.types import ImageContent, TextContent

    from policy_doctor.vlm.proposals.agents.tools.types import ImageBlock, TextBlock

    out = []
    for blk in result.content or []:
        if isinstance(blk, TextBlock):
            out.append(TextContent(type="text", text=blk.text or ""))
        elif isinstance(blk, ImageBlock):
            if blk.caption:
                out.append(TextContent(type="text", text=blk.caption))
            buf = io.BytesIO()
            blk.image.convert("RGB").save(buf, format="JPEG", quality=90)
            data_b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            out.append(ImageContent(type="image", data=data_b64, mimeType="image/jpeg"))
    if not out:
        # MCP refuses fully-empty tool results; emit a sentinel so the client
        # at least sees that the call landed.
        out.append(TextContent(type="text", text="(empty result)"))
    return out


# ---------------------------------------------------------------------------
# Submission persistence — write incrementally so the file system shows progress
# ---------------------------------------------------------------------------


def persist_session(ctx, out_dir: Path) -> None:
    """Write submitted requests + rationale + budget snapshot to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submitted = [sr.to_dict() for sr in ctx.submitted]
    (out_dir / "submitted_requests.json").write_text(
        json.dumps(submitted, indent=2, default=str)
    )
    if ctx.rationale:
        (out_dir / "rationale.txt").write_text(ctx.rationale)
    (out_dir / "budget_summary.json").write_text(
        json.dumps(ctx.budget.snapshot(), indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------


async def run_async() -> None:
    """Async entrypoint — wires the MCP server and runs over stdio."""
    _require_mcp()
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool

    cfg = _load_config_from_env()
    ctx, tools = build_context_and_tools(cfg)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = out_dir / "cursor_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting MCP server: condition=%s pool=%d nodes=%d out=%s",
        cfg["condition"], len(ctx.pool), len(ctx.graph.nodes), session_dir,
    )

    server = Server("policy-doctor-e2")

    # Mirror the in-process AgentSession's contract: append a session-status
    # line to every tool result so the MCP client sees how much budget remains
    # and how many requests have been submitted. Without this the model in
    # Cursor / Claude Code has no signal of the deadline.
    target_n_submissions = int(cfg.get("target_n_submissions", 5))

    @server.list_tools()
    async def _list_tools() -> List[Any]:
        return [
            Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.input_schema,
            )
            for spec in tools.values()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[Any]:
        from policy_doctor.vlm.proposals.agents.budget import format_status_line

        spec = tools.get(name)
        if spec is None:
            from mcp.types import TextContent

            return [TextContent(type="text", text=f"[error:unknown_tool] {name!r}")]

        args = arguments or {}

        # Cost class can escalate: get_*_video with format='video' is more
        # expensive than the storyboard variant.
        kind = spec.cost
        if kind == "visual" and args.get("format") == "video":
            kind = "video"

        # 1. Cache hit — return immediately, no charge. Mirrors AgentSession.
        cached = ctx.cache.get(name, args)
        if cached is not None:
            ctx.budget.note_cache_hit()
            return _decorate_with_status(
                tool_result_to_mcp_content(cached),
                ctx=ctx, target_n_submissions=target_n_submissions,
                warning=None,  # cache hit doesn't change budget; no fresh warning
            )

        # 2. Pre-flight budget check — return a structured error WITHOUT
        # running the tool when the budget is exhausted. finalize_strategy
        # (is_terminal=True) and the submission tools (bypass_budget=True)
        # always pass this gate, so the agent can commit a partial strategy
        # even after exploration runs out.
        bypass = spec.is_terminal or spec.bypass_budget
        budget_err = ctx.budget.check(name, kind, bypass=bypass)
        # Recovery affordance: bypass_when_exhausted tools run even after the
        # budget is gone (they were charged normally before exhaustion).
        if budget_err is not None and spec.bypass_when_exhausted:
            budget_err = None
        if budget_err is not None:
            logger.warning(
                "budget exhausted for %s (kind=%s); returning structured error", name, kind,
            )
            return _decorate_with_status(
                tool_result_to_mcp_content(budget_err),
                ctx=ctx, target_n_submissions=target_n_submissions,
                warning={
                    "warning": f"{kind}_budget_exhausted",
                    "remaining": 0,
                },
            )

        # 3. Run the synchronous tool body in a thread so the async server
        # isn't blocked by pkl-reads or storyboard rendering.
        result = await asyncio.to_thread(spec.func, args)

        # 4. Charge if successful. Note: failed calls don't charge — same
        # as the in-process loop.
        warning: Optional[Dict[str, Any]] = None
        if result.ok:
            ctx.budget.charge(kind, bypass=bypass)
            ctx.cache.put(name, args, result)
            warning = ctx.budget.warning_for(kind)

        # Persist after every submission-affecting tool so the user can see
        # progress in the file system as it happens.
        if result.ok and name in {
            "propose_collection_request",
            "revise_request",
            "delete_request",
            "finalize_strategy",
        }:
            persist_session(ctx, session_dir)

        return _decorate_with_status(
            tool_result_to_mcp_content(result),
            ctx=ctx, target_n_submissions=target_n_submissions, warning=warning,
        )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

    # Final flush at session end.
    persist_session(ctx, session_dir)


def main() -> None:
    """Sync wrapper for ``python -m`` entry."""
    logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
    try:
        asyncio.run(run_async())
    except ImportError as e:
        # Lazy-import failure (no mcp installed): exit with an actionable message.
        print(f"[policy-doctor mcp_server] {e}", file=__import__("sys").stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
