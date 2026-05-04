"""Terminal tool for the pre-stage exploration session.

The exploration session does not submit demonstration requests. Its only
output channel is ``finalize_exploration``, which writes the cluster taxonomy
to disk and sets ``ctx.finalized``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools.types import ToolResult, ToolSpec


_TAXONOMY_SCHEMA = {
    "type": "array",
    "description": "One entry per surveyed cluster.",
    "items": {
        "type": "object",
        "properties": {
            "cluster_id": {"type": "integer"},
            "failure_likelihood": {"type": "number"},
            "trajectory_phase": {
                "type": "string",
                "enum": ["early", "mid", "late", "unknown"],
            },
            "shows_robot_object_engagement": {"type": "boolean"},
            "failure_mode_category": {"type": "string"},
            "notes": {"type": "string"},
            "recommended_for_submission": {"type": "boolean"},
        },
        "required": [
            "cluster_id",
            "trajectory_phase",
            "shows_robot_object_engagement",
            "recommended_for_submission",
        ],
    },
}

_FINALIZE_EXPLORATION_SCHEMA = {
    "type": "object",
    "properties": {
        "taxonomy": _TAXONOMY_SCHEMA,
        "summary": {
            "type": "string",
            "description": "One paragraph overall summary of the failure landscape.",
        },
    },
    "required": ["taxonomy", "summary"],
}


def _make_finalize_exploration(
    ctx: SessionContext,
    out_dir: Optional[Path] = None,
) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        taxonomy = args.get("taxonomy")
        summary = args.get("summary", "").strip()

        if not isinstance(taxonomy, list):
            return ToolResult.error(
                "finalize_exploration",
                "'taxonomy' must be a list",
                code="bad_arg",
            )
        if not summary:
            return ToolResult.error(
                "finalize_exploration",
                "'summary' is required and may not be empty",
                code="bad_arg",
            )

        payload: Dict[str, Any] = {"taxonomy": taxonomy, "summary": summary}

        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "cluster_taxonomy.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )

        ctx.finalized = True

        n_recommended = sum(
            1
            for entry in taxonomy
            if isinstance(entry, dict) and entry.get("recommended_for_submission")
        )
        return ToolResult.text(
            "finalize_exploration",
            json.dumps(
                {
                    "ok": True,
                    "n_clusters_surveyed": len(taxonomy),
                    "n_recommended_for_submission": n_recommended,
                    "summary_chars": len(summary),
                }
            ),
        )

    return ToolSpec(
        name="finalize_exploration",
        description=(
            "Submit the completed cluster taxonomy. REQUIRED as the final call. "
            "Writes taxonomy to disk and ends the exploration session."
        ),
        input_schema=_FINALIZE_EXPLORATION_SCHEMA,
        func=_run,
        cost="cheap",
        is_terminal=True,
    )


def build(
    ctx: SessionContext,
    out_dir: Optional[Path] = None,
) -> List[ToolSpec]:
    return [_make_finalize_exploration(ctx, out_dir=out_dir)]
