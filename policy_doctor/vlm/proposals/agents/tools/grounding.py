"""Grounding check tool — independent visual verification of cited evidence.

The agent calls this after collecting evidence slices but before submitting,
to get a second-pass opinion on whether the cited storyboards actually depict
the claimed failure mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools.types import (
    ImageBlock,
    TextBlock,
    ToolResult,
    ToolSpec,
)

if TYPE_CHECKING:
    pass


_GROUNDING_PROMPT_TEMPLATE = (
    'You are an independent verifier reviewing storyboard images from a robot manipulation rollout.\n\n'
    'The agent claims the following failure mode is visible in these storyboards:\n'
    '"{claim}"\n\n'
    'For each storyboard panel you see, describe in one sentence what the robot is actually doing.\n'
    'Then answer: does the observed robot behavior match the claimed failure mode?\n'
    'Answer with one of: MATCH / PARTIAL_MATCH / NO_MATCH\n'
    'Then explain briefly why.'
)


def _make_verify_evidence_grounding(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        slice_ids: List[str] = list(args.get("slice_ids") or [])
        claim: str = str(args.get("claim", ""))

        if ctx.backend is None:
            return ToolResult.error(
                "verify_evidence_grounding",
                "verify_evidence_grounding requires a backend; set ctx.backend before running this tool.",
                code="no_backend",
            )

        images = []
        skipped = []
        for sid in slice_ids:
            cached = ctx.cache.get("get_slice_video", {"slice_id": sid, "format": "storyboard"})
            if cached is None:
                skipped.append(sid)
                continue
            for blk in cached.content:
                if isinstance(blk, ImageBlock):
                    images.append(blk.image)

        if not images:
            return ToolResult.error(
                "verify_evidence_grounding",
                (
                    "No cached storyboard images found for the cited slices. "
                    "Call get_slice_video on each slice before calling verify_evidence_grounding."
                ),
                code="no_cached_images",
            )

        user_prompt = _GROUNDING_PROMPT_TEMPLATE.format(claim=claim)
        response = ctx.backend.describe_slice(images, system_prompt=None, user_prompt=user_prompt)

        content: list = [TextBlock(text=response)]
        if skipped:
            content.append(
                TextBlock(
                    text=f"[Note: the following slice_ids were not in the cache and were skipped: {skipped}. "
                    "Call get_slice_video on them first if you want them included.]"
                )
            )

        return ToolResult(
            name="verify_evidence_grounding",
            ok=True,
            content=content,
            metadata={"slice_ids": slice_ids, "skipped": skipped, "claim": claim},
        )

    return ToolSpec(
        name="verify_evidence_grounding",
        description=(
            "Independent visual verification: given the slice_ids you plan to cite as evidence "
            "and a one-sentence claim describing the failure mode you observed, this tool "
            "retrieves the storyboard images already in your cache and asks an independent "
            "model to judge whether those frames actually depict the claimed failure. "
            "Returns MATCH / PARTIAL_MATCH / NO_MATCH with an explanation. "
            "You must have called get_slice_video on each slice_id first. "
            "COUNTS AGAINST THE VISUAL BUDGET."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "slice_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The slice_ids you plan to cite as evidence_slice_ids in your submission.",
                },
                "claim": {
                    "type": "string",
                    "description": (
                        "One-sentence description of the failure mode you claim is visible in the storyboards, "
                        "e.g. 'gripper closes while misaligned over the hammer head'."
                    ),
                },
            },
            "required": ["slice_ids", "claim"],
            "additionalProperties": False,
        },
        func=_run,
        cost="visual",
    )


def build(ctx: SessionContext) -> List[ToolSpec]:
    return [_make_verify_evidence_grounding(ctx)]
