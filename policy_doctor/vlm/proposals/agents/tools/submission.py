"""Layer 4 — strategy submission tools (the agent's output channel).

Free-text output from the agent that is *not* submitted via these tools is
discarded by the experiment runner. The tools here are thin wrappers over the
existing :class:`policy_doctor.vlm.proposals.request.DemonstrationRequest`
schema and validation, so adherence scoring + operator interface continue to
work unchanged.

In addition to the schema/denylist checks inherited from
``request.validate_request``, this module enforces three agentic-experiment
gates that catch failure modes observed during integration testing:

* ``recovery`` requests must have ``reference_frame > 0`` — otherwise they're
  ``full_trajectory`` requests mislabeled, which contaminates the
  cluster-adherence axis.
* ``target_behavior`` text must be unique across submissions — duplicate
  prose means the operator would do the same thing twice, providing little
  experimental signal.
* For graph-condition submissions, ``target_cluster`` must reference a
  cluster the agent has actually inspected (via ``get_node`` or
  ``list_slices_in_node`` or ``get_slice_video``). A stricter
  visual-inspection requirement is opt-in via
  ``ctx.config['require_visual_inspection_for_target_cluster']``.

All gates produce structured errors the agent can read and recover from,
matching the existing validation feedback pattern.
"""

from __future__ import annotations

import json
import re
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


# Minimum reference_frame for recovery requests. 0 means "start of rollout"
# which would make a recovery request indistinguishable from full_trajectory.
_RECOVERY_MIN_FRAME = 1

# Minimum number of evidence items the agent must reference. 3 forces the
# agent to inspect more than one example, so the operator-facing prose is
# grounded in a pattern rather than a single (possibly atypical) frame.
_MIN_EVIDENCE_ITEMS = 3


def _normalize_behavior_text(s: str) -> str:
    """For dedup purposes — case-insensitive, whitespace-collapsed."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _slice_belongs_to_cluster(ctx: SessionContext, slice_id: str, cluster_id: int) -> bool:
    """Check whether ``slice_id`` (as emitted by list_slices_in_node) is in ``cluster_id``."""
    from policy_doctor.vlm.proposals.agents.tools.access import (
        _slice_bounds,
        parse_slice_id,
    )
    from policy_doctor.vlm.proposals.pool import rollout_id_to_episode_idx

    if ctx.cluster_labels is None or ctx.cluster_metadata is None:
        return False
    parsed = parse_slice_id(slice_id)
    if parsed is None:
        return False
    rid_str, start, end = parsed
    try:
        ep_idx = rollout_id_to_episode_idx(rid_str)
    except ValueError:
        return False
    for i, meta in enumerate(ctx.cluster_metadata):
        if int(meta.get("rollout_idx", -1)) != ep_idx:
            continue
        m_start, m_end = _slice_bounds(meta)
        if (m_start, m_end) == (start, end):
            return int(ctx.cluster_labels[i]) == int(cluster_id)
    return False


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
            evidence_slice_ids=list(args.get("evidence_slice_ids") or []),
            evidence_rollout_ids=list(args.get("evidence_rollout_ids") or []),
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

        # ---- Agentic-experiment validation gates ------------------------

        # Gate 0: A_G / H_G submissions must set target_cluster.
        # The JSON-schema "required" constraint is enforced by the model
        # provider's tool parser, but loose tool-call formats (Qwen, some
        # Gemini variants) can drop the field. We re-check here so the
        # experimental signal — A_G submissions carry an explicit cluster
        # annotation — is honored regardless of provider strictness.
        if with_target_cluster and request.target_cluster is None:
            return ToolResult.error(
                "propose_collection_request",
                "target_cluster is required in this condition. Pick a cluster id you "
                "have inspected via get_node, list_slices_in_node, or get_slice_video, "
                "and include it as an integer in the request.",
                code="missing_target_cluster",
            )

        # Gate 1: recovery requires reference_frame > 0.
        if request.request_type == "recovery" and request.initial_conditions.reference_frame < _RECOVERY_MIN_FRAME:
            return ToolResult.error(
                "propose_collection_request",
                f"recovery requests require reference_frame >= {_RECOVERY_MIN_FRAME} "
                f"(starting at frame 0 would make this indistinguishable from full_trajectory). "
                "Pick a frame just before the failure point — read the reference rollout's "
                "cluster_path via get_rollout_summary to find a good split.",
                code="recovery_frame_zero",
            )

        # Gate 2: target_behavior text must be unique across submissions.
        # Compares case-insensitively after whitespace normalization.
        normalized_new = _normalize_behavior_text(request.target_behavior)
        for existing in ctx.submitted:
            if _normalize_behavior_text(existing.request.target_behavior) == normalized_new:
                return ToolResult.error(
                    "propose_collection_request",
                    f"target_behavior duplicates submitted request {existing.request_id}. "
                    "Two requests with the same operator instruction provide little "
                    "additional experimental signal. Describe what differs operationally — "
                    "different approach angle, different grasp, different recovery strategy.",
                    code="duplicate_target_behavior",
                )

        # Gate 3a: target_cluster (when supplied) must have been textually
        # inspected. This is the "you should know what you're targeting"
        # check; visual evidence is gated separately below.
        if request.target_cluster is not None:
            tc = int(request.target_cluster)
            if tc not in ctx.inspected_nodes:
                return ToolResult.error(
                    "propose_collection_request",
                    f"target_cluster={tc} has not been inspected. Call get_node({tc}) "
                    "(or list_slices_in_node) before targeting it.",
                    code="cluster_not_inspected",
                )

        # Gate 3b: visual evidence requirement. Each submission must cite at
        # least N specific examples the agent has actually looked at via
        # get_slice_video / get_rollout_video — so the operator-facing prose
        # is grounded in observed failures rather than templated from priors.
        if with_target_cluster:
            evidence = list(args.get("evidence_slice_ids") or [])
            if len(evidence) < _MIN_EVIDENCE_ITEMS:
                return ToolResult.error(
                    "propose_collection_request",
                    f"evidence_slice_ids requires at least {_MIN_EVIDENCE_ITEMS} entries — "
                    "concrete slice_ids you have inspected via get_slice_video that show the "
                    "failure mode this request corrects. Use list_slices_in_node to find "
                    f"slices in cluster {request.target_cluster}, then get_slice_video on each.",
                    code="insufficient_evidence",
                )
            tc = int(request.target_cluster)
            for sid in evidence:
                if sid not in ctx.inspected_slices:
                    return ToolResult.error(
                        "propose_collection_request",
                        f"evidence slice {sid!r} was not visually inspected. Call "
                        "get_slice_video on every slice you cite as evidence. The operator "
                        "needs prose grounded in what you actually saw.",
                        code="evidence_not_inspected",
                    )
                if not _slice_belongs_to_cluster(ctx, sid, tc):
                    return ToolResult.error(
                        "propose_collection_request",
                        f"evidence slice {sid!r} does not belong to target_cluster={tc}. "
                        "Each evidence slice must be a member of the cluster this request "
                        "intends to correct, so the operator's intervention addresses the "
                        "same failure pattern the agent observed.",
                        code="evidence_wrong_cluster",
                    )
        else:
            # A_NG path: evidence_rollout_ids must be 3+ rollouts the agent
            # watched via get_rollout_video.
            evidence = list(args.get("evidence_rollout_ids") or [])
            if len(evidence) < _MIN_EVIDENCE_ITEMS:
                return ToolResult.error(
                    "propose_collection_request",
                    f"evidence_rollout_ids requires at least {_MIN_EVIDENCE_ITEMS} entries — "
                    "rollout_ids you have inspected via get_rollout_video that show the "
                    "failure mode this request corrects.",
                    code="insufficient_evidence",
                )
            for rid in evidence:
                if rid not in ctx.inspected_rollouts:
                    return ToolResult.error(
                        "propose_collection_request",
                        f"evidence rollout {rid!r} was not visually inspected. Call "
                        "get_rollout_video on every rollout you cite as evidence.",
                        code="evidence_not_inspected",
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
        # Submissions bypass the budget so the agent can always commit a
        # partial strategy after exploration runs out.
        bypass_budget=True,
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
        bypass_budget=True,
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

        def _rollback() -> None:
            sr.request.target_behavior = before["target_behavior"]
            sr.request.prohibitions = list(before.get("prohibitions") or [])
            sr.request.success_criterion = before.get("success_criterion", "task_success")

        try:
            validate_request(sr.request, allowed_rollout_ids=set(ctx.pool.rollout_ids))
        except RequestValidationError as e:
            _rollback()
            return ToolResult.error(
                "revise_request",
                f"validation failed; rolled back: {e}",
                code="validation_failed",
            )

        # The agentic duplicate-target gate (mirrors propose_collection_request
        # gate 2). Without this check, the agent could revise a submission to
        # have the same target_behavior as another submission. Compare against
        # ``ctx.submitted`` skipping the request being revised.
        if "target_behavior" in args:
            normalized_new = _normalize_behavior_text(sr.request.target_behavior)
            for existing in ctx.submitted:
                if existing.request_id == rid:
                    continue
                if _normalize_behavior_text(existing.request.target_behavior) == normalized_new:
                    _rollback()
                    return ToolResult.error(
                        "revise_request",
                        f"revised target_behavior duplicates submitted request "
                        f"{existing.request_id}. Two requests with the same operator "
                        "instruction provide little additional experimental signal. "
                        "Describe what differs operationally — different approach angle, "
                        "different grasp, different recovery strategy.",
                        code="duplicate_target_behavior",
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
        bypass_budget=True,
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
        bypass_budget=True,
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
