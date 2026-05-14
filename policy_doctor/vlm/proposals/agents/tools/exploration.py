"""Terminal tool for the pre-stage exploration session.

The exploration session does not submit demonstration requests. Its only
output channel is ``finalize_exploration``, which writes the cluster taxonomy
and a self-contained HTML report to disk and sets ``ctx.finalized``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from html import escape
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

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Exploration report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1rem;
         color: #1a1a1a; line-height: 1.5; }
  header { border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; padding-bottom: 0.5rem; }
  h1 { margin: 0; font-size: 1.6rem; }
  h2 { font-size: 1.2rem; margin: 1.5rem 0 0.5rem; }
  h3 { font-size: 1rem; margin: 1rem 0 0.25rem; color: #444; }
  .meta { color: #666; font-size: 0.9rem; margin-top: 0.25rem; }
  .summary-box { background: #f0f6ff; border-left: 4px solid #5599ff;
                 padding: 0.75rem 1rem; margin: 1rem 0 1.5rem;
                 border-radius: 0 4px 4px 0; font-style: italic; }
  /* Failure-mode group cards */
  .mode-group { border: 1px solid #d0d7de; border-radius: 6px;
                margin-bottom: 1.25rem; overflow: hidden; }
  .mode-header { background: #f6f8fa; padding: 0.5rem 1rem;
                 border-bottom: 1px solid #d0d7de;
                 display: flex; justify-content: space-between; align-items: center; }
  .mode-header .mode-name { font-weight: 600; font-size: 1rem; }
  .mode-header .mode-clusters { font-size: 0.85rem; color: #666;
                                 font-family: ui-monospace, SFMono-Regular, monospace; }
  /* Per-cluster row */
  .cluster-row { display: grid; grid-template-columns: 80px 1fr;
                 border-top: 1px solid #eaecef; }
  .cluster-row:first-of-type { border-top: none; }
  .cluster-id { padding: 0.6rem 0.75rem; font-weight: 700;
                font-family: ui-monospace, SFMono-Regular, monospace;
                font-size: 1rem; border-right: 1px solid #eaecef;
                display: flex; align-items: flex-start; }
  .cluster-body { padding: 0.5rem 0.75rem; }
  .pills { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.35rem; }
  .pill { font-size: 0.75rem; padding: 0.1rem 0.45rem; border-radius: 2rem;
          border: 1px solid transparent; }
  .pill-phase-early  { background: #ddf4ff; border-color: #54aeff; color: #0550ae; }
  .pill-phase-mid    { background: #fff3cd; border-color: #d09010; color: #7a5200; }
  .pill-phase-late   { background: #dafbe1; border-color: #2da44e; color: #1a7f37; }
  .pill-phase-unknown{ background: #f6f8fa; border-color: #d0d7de; color: #57606a; }
  .pill-engaged-yes  { background: #dafbe1; border-color: #2da44e; color: #1a7f37; }
  .pill-engaged-no   { background: #ffeef0; border-color: #f85149; color: #cf222e; }
  .pill-fl-high  { background: #ffeef0; border-color: #f85149; color: #cf222e; }
  .pill-fl-med   { background: #fff3cd; border-color: #d09010; color: #7a5200; }
  .pill-fl-low   { background: #f6f8fa; border-color: #d0d7de; color: #57606a; }
  .pill-rec-yes  { background: #dafbe1; border-color: #2da44e; color: #1a7f37;
                   font-weight: 600; }
  .pill-rec-no   { background: #f6f8fa; border-color: #d0d7de; color: #57606a; }
  .cluster-notes { font-size: 0.85rem; color: #57606a; margin-top: 0.2rem; }
  /* Recommended summary strip */
  .rec-strip { background: #dafbe1; border: 1px solid #2da44e;
               border-radius: 6px; padding: 0.75rem 1rem; margin-top: 1.5rem; }
  .rec-strip h2 { margin: 0 0 0.5rem; color: #1a7f37; }
  .rec-strip ul { margin: 0; padding-left: 1.2rem; }
  .rec-strip li { font-family: ui-monospace, SFMono-Regular, monospace;
                  font-size: 0.9rem; }
  .none-rec { color: #86181d; font-style: italic; }
</style>
</head>
<body>
"""


def _fl_pill_class(fl: float) -> str:
    if fl >= 0.7:
        return "pill-fl-high"
    if fl >= 0.3:
        return "pill-fl-med"
    return "pill-fl-low"


def render_exploration_report(
    taxonomy: List[Dict[str, Any]],
    summary: str,
    *,
    task_hint: str = "",
    n_rollouts: int = 0,
) -> str:
    """Return a self-contained HTML report for one exploration session."""
    parts: List[str] = [_HTML_HEAD]

    # Header
    parts.append("<header>")
    parts.append("<h1>Exploration report — failure mode taxonomy</h1>")
    meta_bits = []
    if task_hint:
        meta_bits.append(f"task: {escape(task_hint)}")
    meta_bits.append(f"{len(taxonomy)} clusters surveyed")
    n_rec = sum(1 for e in taxonomy if isinstance(e, dict) and e.get("recommended_for_submission"))
    meta_bits.append(f"{n_rec} recommended for submission")
    if n_rollouts:
        meta_bits.append(f"{n_rollouts} rollouts in pool")
    parts.append(f"<div class='meta'>{escape(' · '.join(meta_bits))}</div>")
    parts.append("</header>")

    # Agent's overall summary
    parts.append(f"<div class='summary-box'>{escape(summary)}</div>")

    # Group by failure mode category, unknown last
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in taxonomy:
        if not isinstance(entry, dict):
            continue
        cat = entry.get("failure_mode_category") or "unknown"
        by_cat[cat].append(entry)

    parts.append("<h2>Clusters by failure mode</h2>")
    cats = sorted(by_cat, key=lambda c: (c == "unknown", c))
    for cat in cats:
        entries = sorted(by_cat[cat], key=lambda e: -e.get("failure_likelihood", 0.0))
        cluster_names = " ".join(f"c{e['cluster_id']}" for e in entries)
        parts.append("<div class='mode-group'>")
        parts.append(
            f"<div class='mode-header'>"
            f"<span class='mode-name'>{escape(cat)}</span>"
            f"<span class='mode-clusters'>{escape(cluster_names)}</span>"
            f"</div>"
        )
        for entry in entries:
            cid = entry.get("cluster_id", "?")
            fl = float(entry.get("failure_likelihood", 0.0))
            phase = entry.get("trajectory_phase", "unknown")
            engaged = entry.get("shows_robot_object_engagement", False)
            rec = entry.get("recommended_for_submission", False)
            notes = entry.get("notes", "")

            parts.append("<div class='cluster-row'>")
            parts.append(f"<div class='cluster-id'>c{cid}</div>")
            parts.append("<div class='cluster-body'>")
            parts.append("<div class='pills'>")
            parts.append(
                f"<span class='pill {_fl_pill_class(fl)}'>fl={fl:.2f}</span>"
            )
            parts.append(
                f"<span class='pill pill-phase-{phase}'>phase:{phase}</span>"
            )
            engaged_cls = "pill-engaged-yes" if engaged else "pill-engaged-no"
            engaged_label = "engaged" if engaged else "no engagement"
            parts.append(f"<span class='pill {engaged_cls}'>{engaged_label}</span>")
            rec_cls = "pill-rec-yes" if rec else "pill-rec-no"
            rec_label = "✓ submit" if rec else "skip"
            parts.append(f"<span class='pill {rec_cls}'>{rec_label}</span>")
            parts.append("</div>")  # pills
            if notes:
                parts.append(f"<div class='cluster-notes'>{escape(notes)}</div>")
            parts.append("</div>")  # cluster-body
            parts.append("</div>")  # cluster-row

        parts.append("</div>")  # mode-group

    # Recommended strip
    recommended = [
        e for e in taxonomy
        if isinstance(e, dict) and e.get("recommended_for_submission")
    ]
    recommended.sort(key=lambda e: -e.get("failure_likelihood", 0.0))
    parts.append("<div class='rec-strip'>")
    parts.append("<h2>Recommended for submission</h2>")
    if recommended:
        parts.append("<ul>")
        for e in recommended:
            cid = e.get("cluster_id", "?")
            fl = float(e.get("failure_likelihood", 0.0))
            cat = e.get("failure_mode_category", "?")
            phase = e.get("trajectory_phase", "?")
            parts.append(
                f"<li>c{cid} &mdash; {escape(cat)}, fl={fl:.2f}, phase={escape(phase)}</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<div class='none-rec'>None flagged.</div>")
    parts.append("</div>")  # rec-strip

    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


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
            html = render_exploration_report(
                taxonomy,
                summary,
                task_hint=ctx.task_hint,
                n_rollouts=len(ctx.pool),
            )
            (out_path / "exploration_report.html").write_text(html)

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
            "Writes cluster_taxonomy.json and exploration_report.html to disk "
            "and ends the exploration session."
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
