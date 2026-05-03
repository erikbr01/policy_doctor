"""Layer 4 — strategy submission tools (the agent's output channel).

Free-text output from the agent that is *not* submitted via these tools is
discarded by the experiment runner. The tools here are thin wrappers over the
existing :class:`policy_doctor.vlm.proposals.request.DemonstrationRequest`
schema and validation, so adherence scoring + operator interface continue to
work unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from policy_doctor.vlm.proposals.agents.context import (
    SessionContext,
    SubmittedRequest,
)
from policy_doctor.vlm.proposals.agents.tools import schema as S
from policy_doctor.vlm.proposals.agents.tools.types import ToolResult, ToolSpec
from policy_doctor.vlm.proposals.request import (
    DemonstrationRequest,
    InitialConditions,
    RequestValidationError,
    validate_request,
)


# ---------------------------------------------------------------------------
# propose_collection_request
# ---------------------------------------------------------------------------


def _make_propose_collection_request(
    ctx: SessionContext, *, with_target_cluster: bool
) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        if ctx.finalized:
            return ToolResult.error(
                "propose_collection_request",
                "session already finalized; further submissions are ignored",
                code="finalized",
            )

        ic_args = dict(args.get("initial_conditions") or {})
        try:
            ic = InitialConditions(
                reference_rollout_id=str(ic_args.get("reference_rollout_id", "")),
                reference_frame=int(ic_args.get("reference_frame", 0)),
            )
        except (TypeError, ValueError) as e:
            return ToolResult.error(
                "propose_collection_request",
                f"bad initial_conditions: {e}",
                code="bad_arg",
            )

        request = DemonstrationRequest(
            request_id=DemonstrationRequest.new_id(),
            request_type=str(args.get("request_type", "")),
            initial_conditions=ic,
            target_behavior=str(args.get("target_behavior", "")),
            prohibitions=list(args.get("prohibitions") or []),
            success_criterion=str(args.get("success_criterion", "task_success")),
            target_cluster=int(args["target_cluster"]) if with_target_cluster and "target_cluster" in args else None,
            source_condition=ctx.condition,
        )

        allowed = set(ctx.pool.rollout_ids)
        try:
            validate_request(request, allowed_rollout_ids=allowed)
        except RequestValidationError as e:
            # Feed the validation error back; the agent can retry on the next turn.
            return ToolResult.error(
                "propose_collection_request",
                f"validation failed: {e}",
                code="validation_failed",
            )

        reasoning = str(args.get("reasoning") or "").strip()
        if not reasoning:
            return ToolResult.error(
                "propose_collection_request",
                "'reasoning' is required and may not be empty",
                code="bad_arg",
            )

        ctx.submitted.append(SubmittedRequest(request=request, reasoning=reasoning))
        return ToolResult.text(
            "propose_collection_request",
            json.dumps(
                {
                    "ok": True,
                    "request_id": request.request_id,
                    "n_submitted": len(ctx.submitted),
                },
                default=str,
            ),
            request_id=request.request_id,
        )

    schema = S.propose_collection_request_schema(with_target_cluster=with_target_cluster)
    desc = (
        "Submit a single DemonstrationRequest. The 'reasoning' field is REQUIRED "
        "and is logged to the trace. The 'target_behavior' / 'success_criterion' "
        "MUST NOT mention clusters, nodes, the graph, embeddings, etc. — those "
        "terms leak the experimental condition and validation will reject."
    )
    if with_target_cluster:
        desc += " 'target_cluster' is required and must reference a behavior cluster id."
    else:
        desc += " 'target_cluster' is computed post-hoc from the reference rollout."
    return ToolSpec(
        name="propose_collection_request",
        description=desc,
        input_schema=schema,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# list_submitted_requests
# ---------------------------------------------------------------------------


def _make_list_submitted_requests(ctx: SessionContext) -> ToolSpec:
    def _run(_args: Dict[str, Any]) -> ToolResult:
        rows: List[Dict[str, Any]] = []
        for sr in ctx.submitted:
            r = sr.request
            rows.append({
                "request_id": r.request_id,
                "request_type": r.request_type,
                "reference_rollout_id": r.initial_conditions.reference_rollout_id,
                "reference_frame": r.initial_conditions.reference_frame,
                "target_behavior": r.target_behavior,
                "target_cluster": r.target_cluster,
                "reasoning": sr.reasoning,
                "n_revisions": len(sr.revision_history),
            })
        return ToolResult.text(
            "list_submitted_requests",
            json.dumps({"n_submitted": len(rows), "requests": rows}, default=str),
        )

    return ToolSpec(
        name="list_submitted_requests",
        description=(
            "Return every request submitted in this session, including the agent's "
            "own reasoning. Useful for self-review before finalizing."
        ),
        input_schema=S.LIST_SUBMITTED_REQUESTS,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# revise_request
# ---------------------------------------------------------------------------


def _make_revise_request(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        rid = str(args.get("request_id", ""))
        sr = _find(ctx, rid)
        if sr is None:
            return ToolResult.error(
                "revise_request",
                f"request_id={rid!r} not in submitted set",
                code="not_found",
            )

        before = sr.request.to_dict()
        # Apply field-by-field updates.
        if "target_behavior" in args:
            sr.request.target_behavior = str(args["target_behavior"])
        if "prohibitions" in args:
            sr.request.prohibitions = list(args["prohibitions"] or [])
        if "success_criterion" in args:
            sr.request.success_criterion = str(args["success_criterion"])

        try:
            validate_request(sr.request, allowed_rollout_ids=set(ctx.pool.rollout_ids))
        except RequestValidationError as e:
            # Roll back.
            sr.request.target_behavior = before["target_behavior"]
            sr.request.prohibitions = list(before.get("prohibitions") or [])
            sr.request.success_criterion = before.get("success_criterion", "task_success")
            return ToolResult.error(
                "revise_request",
                f"validation failed; rolled back: {e}",
                code="validation_failed",
            )

        new_reasoning = str(args.get("reasoning") or "").strip()
        if not new_reasoning:
            # Should be caught by schema, but be defensive.
            return ToolResult.error(
                "revise_request",
                "'reasoning' is required and may not be empty",
                code="bad_arg",
            )

        sr.revision_history.append({"prev": before, "new_reasoning": new_reasoning})
        sr.reasoning = new_reasoning
        return ToolResult.text(
            "revise_request",
            json.dumps({"ok": True, "request_id": rid, "n_revisions": len(sr.revision_history)}),
        )

    return ToolSpec(
        name="revise_request",
        description=(
            "Modify a previously-submitted request (target_behavior, prohibitions, or "
            "success_criterion). The 'reasoning' field is required to document why "
            "the revision was made."
        ),
        input_schema=S.REVISE_REQUEST,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# delete_request
# ---------------------------------------------------------------------------


def _make_delete_request(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        rid = str(args.get("request_id", ""))
        before = len(ctx.submitted)
        ctx.submitted = [sr for sr in ctx.submitted if sr.request_id != rid]
        if len(ctx.submitted) == before:
            return ToolResult.error(
                "delete_request",
                f"request_id={rid!r} not in submitted set",
                code="not_found",
            )
        return ToolResult.text(
            "delete_request",
            json.dumps({"ok": True, "request_id": rid, "n_submitted": len(ctx.submitted)}),
        )

    return ToolSpec(
        name="delete_request",
        description="Remove a submitted request from the strategy. Idempotent on missing ids.",
        input_schema=S.DELETE_REQUEST,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# finalize_strategy  (terminal)
# ---------------------------------------------------------------------------


def _make_finalize_strategy(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        rationale = str(args.get("rationale") or "").strip()
        if not rationale:
            return ToolResult.error(
                "finalize_strategy",
                "'rationale' is required and may not be empty",
                code="bad_arg",
            )
        ctx.rationale = rationale
        ctx.finalized = True
        return ToolResult.text(
            "finalize_strategy",
            json.dumps(
                {
                    "ok": True,
                    "n_submitted": len(ctx.submitted),
                    "rationale_chars": len(rationale),
                }
            ),
        )

    return ToolSpec(
        name="finalize_strategy",
        description=(
            "End the session. The rationale field is required and is logged as the "
            "agent's final summary of its strategy. After this call no further tool "
            "calls are accepted; the session terminates."
        ),
        input_schema=S.FINALIZE_STRATEGY,
        func=_run,
        cost="cheap",
        is_terminal=True,
    )


def _find(ctx: SessionContext, request_id: str):
    for sr in ctx.submitted:
        if sr.request_id == request_id:
            return sr
    return None


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def build(ctx: SessionContext, *, with_target_cluster: bool) -> List[ToolSpec]:
    return [
        _make_propose_collection_request(ctx, with_target_cluster=with_target_cluster),
        _make_list_submitted_requests(ctx),
        _make_revise_request(ctx),
        _make_delete_request(ctx),
        _make_finalize_strategy(ctx),
    ]
