"""Parallel no-graph tool surface (A_NG / H_NG conditions).

Same underlying rollouts and videos as the A_G surface, but cluster-level
information is **never exposed**. Each tool builds its response dict from
scratch with a hand-picked whitelist of fields so leaks are impossible by
construction (rather than relying on a denylist scrubber).

The denylist test in :mod:`tests.vlm.proposals.test_leak_audit` already
guards user-facing string fields on the request schema. The parallel test
:mod:`tests.vlm.proposals.agents.test_no_graph_isolation` will additionally
assert that no field of any A_NG tool's output references a cluster, node,
graph, or embedding term.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools import schema as S
from policy_doctor.vlm.proposals.agents.tools.access import (
    _load_storyboard,
    _render_slice_storyboard,
)
from policy_doctor.vlm.proposals.agents.tools.types import (
    ImageBlock,
    TextBlock,
    ToolResult,
    ToolSpec,
)


# Same denylist as :mod:`policy_doctor.vlm.proposals.request`. Re-imported
# locally so this module stays the single source of truth for what gets
# stripped on the no-graph surface.
_DENY_RE = re.compile(
    r"\bcluster(s|ing)?\b|\bnode(s)?\b|\bgraph\b|\bbehavior\s+graph\b|"
    r"\bumap\b|\bk[- ]?means\b|\bcentroid(s)?\b|\bembedding(s)?\b",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    """Defense-in-depth scrubbing for any free-text field we relay through."""
    return _DENY_RE.sub("[redacted]", text or "")


# ---------------------------------------------------------------------------
# ng_list_rollouts  /  ng_list_failure_rollouts  /  ng_list_success_rollouts
# ---------------------------------------------------------------------------


def _ng_list(ctx: SessionContext, *, outcome: str | None, n: int) -> ToolResult:
    rows: List[Dict[str, Any]] = []
    for entry in ctx.pool.entries:
        if outcome == "success" and entry.success is not True:
            continue
        if outcome == "failure" and entry.success is not False:
            continue
        rows.append({
            "rollout_id": entry.rollout_id,
            "outcome": (
                "success" if entry.success is True
                else "failure" if entry.success is False else "unknown"
            ),
            "length": entry.length,
        })
        if len(rows) >= n:
            break
    return ToolResult.text(
        "list_rollouts",
        json.dumps({"n_rollouts": len(rows), "rollouts": rows}),
    )


def _make_list_rollouts(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        outcome = args.get("outcome")
        n = int(args.get("n", 50))
        return _ng_list(ctx, outcome=outcome, n=n)

    return ToolSpec(
        name="list_rollouts",
        description=(
            "Filter the rollout pool by outcome (success / failure / null=any). "
            "Returns rollout_id, outcome, length per rollout."
        ),
        input_schema=S.NG_LIST_ROLLOUTS,
        func=_run,
        cost="cheap",
    )


def _make_list_failure_rollouts(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        n = int(args.get("n", 50))
        return _ng_list(ctx, outcome="failure", n=n)

    return ToolSpec(
        name="list_failure_rollouts",
        description="Convenience: list failure rollouts in the pool.",
        input_schema=S.NG_LIST_FAILURE_ROLLOUTS,
        func=_run,
        cost="cheap",
    )


def _make_list_success_rollouts(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        n = int(args.get("n", 50))
        return _ng_list(ctx, outcome="success", n=n)

    return ToolSpec(
        name="list_success_rollouts",
        description="Convenience: list success rollouts in the pool.",
        input_schema=S.NG_LIST_SUCCESS_ROLLOUTS,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# get_rollout_summary (no cluster_path)
# ---------------------------------------------------------------------------


def _make_get_rollout_summary(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        rid = str(args.get("rollout_id", ""))
        try:
            entry = ctx.pool.by_id(rid)
        except KeyError:
            return ToolResult.error(
                "get_rollout_summary",
                f"rollout_id={rid!r} not in pool",
                code="not_found",
            )
        # Whitelist: rollout_id, outcome, length, has_storyboard, has_video.
        # Notably absent: cluster_path, cluster_path_ids.
        payload = {
            "rollout_id": entry.rollout_id,
            "outcome": (
                "success" if entry.success is True
                else "failure" if entry.success is False else "unknown"
            ),
            "length": entry.length,
            "has_storyboard": entry.storyboard_path is not None and entry.storyboard_path.exists(),
            "has_video": entry.video_path is not None and entry.video_path.exists(),
        }
        return ToolResult.text("get_rollout_summary", json.dumps(payload))

    return ToolSpec(
        name="get_rollout_summary",
        description=(
            "Outcome and metadata for one rollout. Returns rollout_id, outcome, length, "
            "and whether storyboard / video are available. No internal-representation "
            "details are exposed."
        ),
        input_schema=S.NG_GET_ROLLOUT_SUMMARY,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# get_rollout_video (same body as A_G; no cluster info anyway)
# ---------------------------------------------------------------------------


def _make_get_rollout_video(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        rid = str(args.get("rollout_id", ""))
        fmt = str(args.get("format", "storyboard"))
        try:
            entry = ctx.pool.by_id(rid)
        except KeyError:
            return ToolResult.error(
                "get_rollout_video",
                f"rollout_id={rid!r} not in pool",
                code="not_found",
            )

        if fmt == "storyboard":
            img = _load_storyboard(entry.storyboard_path)
            if img is None:
                img = _render_slice_storyboard(entry.episode_pkl, 0, max(entry.length - 1, 0))
            if img is None:
                return ToolResult.error(
                    "get_rollout_video",
                    f"no storyboard available for {rid!r}",
                    code="no_frames",
                )
            ctx.inspected_rollouts.add(rid)
            return ToolResult(
                name="get_rollout_video",
                ok=True,
                content=[
                    TextBlock(text=f"rollout {rid} storyboard"),
                    ImageBlock(image=img, caption=f"rollout {rid} storyboard"),
                ],
                metadata={"rollout_id": rid, "format": "storyboard"},
            )
        # video
        if entry.video_path is None or not entry.video_path.exists():
            return ToolResult.error(
                "get_rollout_video",
                f"no video available for {rid!r}",
                code="no_video",
            )
        ctx.inspected_rollouts.add(rid)
        return ToolResult(
            name="get_rollout_video",
            ok=True,
            content=[TextBlock(text=f"rollout {rid} video at {entry.video_path}")],
            metadata={"rollout_id": rid, "format": "video", "video_path": str(entry.video_path)},
        )

    return ToolSpec(
        name="get_rollout_video",
        description=(
            "Visual content for one rollout. format='storyboard' (cheap, default) or "
            "format='video' (counts against video budget). Same data as the A_G "
            "surface — only cluster-level annotations are withheld."
        ),
        input_schema=S.NG_GET_ROLLOUT_VIDEO,
        func=_run,
        cost="visual",
    )


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def build(ctx: SessionContext) -> List[ToolSpec]:
    return [
        _make_list_rollouts(ctx),
        _make_list_failure_rollouts(ctx),
        _make_list_success_rollouts(ctx),
        _make_get_rollout_summary(ctx),
        _make_get_rollout_video(ctx),
    ]
