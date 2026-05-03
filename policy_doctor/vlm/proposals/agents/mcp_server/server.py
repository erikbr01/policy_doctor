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
        spec = tools.get(name)
        if spec is None:
            from mcp.types import TextContent

            return [TextContent(type="text", text=f"[error:unknown_tool] {name!r}")]

        # Run the synchronous tool body in a thread so async server isn't blocked
        # by occasional pkl-reads or storyboard rendering.
        result = await asyncio.to_thread(spec.func, arguments or {})

        # Charge budget *after* a successful call (mirrors AgentSession.dispatch).
        if result.ok:
            kind = spec.cost
            if kind == "visual" and (arguments or {}).get("format") == "video":
                kind = "video"
            err = ctx.budget.check(name, kind, is_terminal=spec.is_terminal)
            if err is not None:
                # Budget would be exceeded — return the structured error
                # rather than the result. (Surfaces same way the agent loop does.)
                return tool_result_to_mcp_content(err)
            ctx.budget.charge(kind, is_terminal=spec.is_terminal)

        # Persist after every submission-affecting tool so the user can see
        # progress in the file system as it happens.
        if result.ok and name in {
            "propose_collection_request",
            "revise_request",
            "delete_request",
            "finalize_strategy",
        }:
            persist_session(ctx, session_dir)

        return tool_result_to_mcp_content(result)

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
