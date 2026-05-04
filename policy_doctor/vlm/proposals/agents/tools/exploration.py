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


def _render_failure_mode_summary(taxonomy: List[Dict[str, Any]], summary: str) -> str:
    """Render a human-readable markdown summary of failure modes from the taxonomy."""
    from collections import defaultdict

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in taxonomy:
        if not isinstance(entry, dict):
            continue
        cat = entry.get("failure_mode_category") or "unknown"
        by_category[cat].append(entry)

    recommended = [
        e for e in taxonomy
        if isinstance(e, dict) and e.get("recommended_for_submission")
    ]

    lines: List[str] = ["# Exploration: Failure Mode Summary", ""]
    lines += [summary, ""]

    lines += ["## Clusters by failure mode", ""]
    # Sort categories: put unknown last
    cats = sorted(by_category, key=lambda c: (c == "unknown", c))
    for cat in cats:
        entries = sorted(by_category[cat], key=lambda e: -e.get("failure_likelihood", 0.0))
        cluster_names = ", ".join(f"c{e['cluster_id']}" for e in entries)
        lines.append(f"### {cat}  ({cluster_names})")
        for e in entries:
            cid = e.get("cluster_id", "?")
            fl = e.get("failure_likelihood", 0.0)
            phase = e.get("trajectory_phase", "unknown")
            engaged = e.get("shows_robot_object_engagement", False)
            rec = "✓ recommended" if e.get("recommended_for_submission") else "✗ skip"
            notes = e.get("notes", "")
            lines.append(
                f"- **c{cid}** — fl={fl:.2f}, phase={phase}, "
                f"engaged={engaged}, {rec}"
                + (f"\n  _{notes}_" if notes else "")
            )
        lines.append("")

    lines += ["## Recommended for submission", ""]
    if recommended:
        for e in sorted(recommended, key=lambda e: -e.get("failure_likelihood", 0.0)):
            lines.append(
                f"- **c{e['cluster_id']}** ({e.get('failure_mode_category','?')}, "
                f"fl={e.get('failure_likelihood',0.0):.2f}, "
                f"phase={e.get('trajectory_phase','?')})"
            )
    else:
        lines.append("_None flagged._")

    return "\n".join(lines) + "\n"


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
            (out_path / "failure_mode_summary.md").write_text(
                _render_failure_mode_summary(taxonomy, summary)
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
