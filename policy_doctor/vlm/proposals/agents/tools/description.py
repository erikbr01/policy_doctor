"""Terminal tool for the Stage 1 visual description session.

The description session produces literal, interpretation-free observations
of video clips. Its only output is ``finalize_descriptions``, which writes
``visual_descriptions.json`` and ends the session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools.types import ToolResult, ToolSpec


_CLUSTER_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster_id": {
            "type": "integer",
            "description": "Cluster node id (e.g. 5 for c5).",
        },
        "slices_observed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "slice_ids you actually called get_slice_video on.",
        },
        "literal_description": {
            "type": "string",
            "description": (
                "What you literally saw: arm positions, gripper states, object location, "
                "contact or lack thereof. No interpretation, no failure-mode labels."
            ),
        },
        "gripper_states": {
            "type": "string",
            "description": "Open/closed state of each arm's gripper across the clips.",
        },
        "robot_object_contact": {
            "type": "boolean",
            "description": "True only if a gripper was in physical contact with the hammer.",
        },
        "contact_location": {
            "type": "string",
            "description": "Where on the hammer: 'handle', 'head', or 'unknown'. Omit if no contact.",
        },
        "object_location": {
            "type": "string",
            "description": "Where is the hammer? e.g. 'resting in starting bin', 'held in right gripper 20cm above goal bin'.",
        },
        "sequence_of_events": {
            "type": "string",
            "description": "What happens across the clip duration, as a temporal sequence.",
        },
        "informative": {
            "type": "boolean",
            "description": (
                "False if the clips show no robot-object interaction and cannot "
                "support any grounded claim about a failure mode."
            ),
        },
    },
    "required": [
        "cluster_id",
        "slices_observed",
        "literal_description",
        "robot_object_contact",
        "informative",
    ],
}

_FINALIZE_DESCRIPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster_descriptions": {
            "type": "array",
            "items": _CLUSTER_DESCRIPTION_SCHEMA,
            "description": "One entry per cluster you observed.",
        },
    },
    "required": ["cluster_descriptions"],
}


def _make_finalize_descriptions(
    ctx: SessionContext,
    out_dir: Optional[Path] = None,
) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        descs = args.get("cluster_descriptions")
        if not isinstance(descs, list) or not descs:
            return ToolResult.error(
                "finalize_descriptions",
                "'cluster_descriptions' must be a non-empty list",
                code="bad_arg",
            )

        payload: Dict[str, Any] = {
            "cluster_descriptions": descs,
            "inspected_slices": list(ctx.inspected_slices),
            "inspected_nodes": list(ctx.inspected_nodes),
        }

        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "visual_descriptions.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )

        ctx.finalized = True

        n_informative = sum(
            1 for d in descs if isinstance(d, dict) and d.get("informative", True)
        )
        n_with_contact = sum(
            1 for d in descs if isinstance(d, dict) and d.get("robot_object_contact", False)
        )
        return ToolResult.text(
            "finalize_descriptions",
            json.dumps({
                "ok": True,
                "n_clusters_described": len(descs),
                "n_informative": n_informative,
                "n_with_robot_object_contact": n_with_contact,
            }),
        )

    return ToolSpec(
        name="finalize_descriptions",
        description=(
            "Submit the completed visual descriptions. REQUIRED as the final call. "
            "Writes visual_descriptions.json and ends the description session."
        ),
        input_schema=_FINALIZE_DESCRIPTIONS_SCHEMA,
        func=_run,
        cost="cheap",
        is_terminal=True,
    )


def build(
    ctx: SessionContext,
    out_dir: Optional[Path] = None,
) -> List[ToolSpec]:
    return [_make_finalize_descriptions(ctx, out_dir=out_dir)]
