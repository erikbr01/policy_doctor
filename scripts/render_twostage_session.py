"""Render a two-stage agent session into a single self-contained HTML report.

The two-stage session writes artefacts under:
  <session_dir>/stage1_description/   — visual_descriptions.json + trace
  <session_dir>/stage2_proposals/     — submitted_requests.json + trace

This script combines both into one HTML report:
  • Stage 1: per-cluster description cards with re-rendered storyboards
  • Stage 2: proposals with evidence storyboards (same format as session_report.html)

Usage::

    python scripts/render_twostage_session.py \\
        --session-dir /mnt/ssdB/.../gemini31pro_A_G_seed0_twostage_v3 \\
        --episodes-dir /mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/.../episodes \\
        --out report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "third_party" / "cupid"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from policy_doctor.vlm.proposals.agents.tools.access import (  # noqa: E402
    _render_slice_storyboard,
    parse_slice_id,
)
from policy_doctor.vlm.proposals.pool import RolloutPool  # noqa: E402
from scripts.render_agent_session import _img_to_data_uri  # noqa: E402


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1200px; margin: 2rem auto; padding: 0 1rem;
       color: #1a1a1a; line-height: 1.5; }
header { border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; padding-bottom: 0.5rem; }
h1 { margin: 0; font-size: 1.6rem; }
h2 { font-size: 1.25rem; margin: 1.8rem 0 0.5rem; border-bottom: 1px solid #eee; padding-bottom: 0.3rem; }
h3 { font-size: 1rem; margin: 0.5rem 0; color: #333; }
.meta { color: #666; font-size: 0.9rem; margin-top: 0.25rem; }
.stage-header { font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.08em; color: #fff; padding: 0.2rem 0.6rem;
                border-radius: 3px; display: inline-block; margin-bottom: 0.5rem; }
.s1-header { background: #6f42c1; }
.s2-header { background: #0969da; }

/* Stage 1 cluster cards */
.cluster-card { border: 1px solid #d0d7de; border-radius: 6px;
                margin-bottom: 1.25rem; overflow: hidden; }
.cluster-card.uninformative { border-color: #f85149; opacity: 0.85; }
.cluster-card .card-header { padding: 0.5rem 1rem; display: flex;
                              justify-content: space-between; align-items: center;
                              background: #f6f8fa; border-bottom: 1px solid #d0d7de; }
.cluster-card.uninformative .card-header { background: #fff1f0; border-color: #f85149; }
.cluster-title { font-weight: 700; font-family: ui-monospace, SFMono-Regular, monospace; }
.badge { font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 2rem; border: 1px solid; }
.badge-ok { background: #dafbe1; border-color: #2da44e; color: #1a7f37; }
.badge-bad { background: #ffeef0; border-color: #f85149; color: #cf222e; }
.badge-contact { background: #ddf4ff; border-color: #54aeff; color: #0550ae; }
.cluster-body { padding: 0.75rem 1rem; }
.desc-text { color: #24292f; margin-bottom: 0.5rem; }
.desc-meta { font-size: 0.85rem; color: #57606a; }
.storyboard-grid { display: grid; grid-template-columns: repeat(2, 1fr);
                   gap: 0.6rem; margin-top: 0.6rem; }
.storyboard-grid figure { margin: 0; }
.storyboard-grid img { width: 100%; height: auto; border: 1px solid #d0d7de;
                       border-radius: 4px; display: block; }
.storyboard-grid figcaption { font-size: 0.72rem; color: #777; margin-top: 0.2rem;
                               text-align: center; font-family: ui-monospace, SFMono-Regular, monospace; }

/* Stage 2 submissions (reuse session_report style) */
.submission { border: 1px solid #d0d7de; border-radius: 6px;
              padding: 1rem 1.25rem; margin-bottom: 1.5rem; background: #fafbfc; }
.submission h3 { margin: 0 0 0.4rem; font-size: 1.05rem; display: flex; gap: 0.5rem; }
.submission .num { color: #999; font-weight: 400; }
.field { margin: 0.5rem 0; }
.field-label { font-weight: 600; color: #444; font-size: 0.82rem;
               text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.15rem; }
.field-body { padding-left: 0.5rem; border-left: 3px solid #d0d7de; }
.reasoning { color: #586069; font-style: italic; }
.evidence-grid { display: grid; grid-template-columns: repeat(3, 1fr);
                 gap: 0.75rem; margin-top: 0.5rem; }
.evidence-grid figure { margin: 0; }
.evidence-grid img { width: 100%; height: auto; border: 1px solid #d0d7de;
                     border-radius: 4px; }
.evidence-grid figcaption { font-size: 0.72rem; color: #777; margin-top: 0.2rem;
                             text-align: center; font-family: ui-monospace, SFMono-Regular, monospace; }
.missing { padding: 1.2rem; background: #ffeef0; color: #86181d;
           border-radius: 4px; font-size: 0.9rem; }
.rationale-box { background: #f0f6ff; border-left: 4px solid #5599ff;
                 padding: 0.75rem 1rem; margin-top: 1.5rem;
                 border-radius: 0 4px 4px 0; }
"""


def _materialize(pool: RolloutPool, slice_id: str):
    parsed = parse_slice_id(slice_id)
    if not parsed:
        return None
    rid, start, end = parsed
    try:
        entry = pool.by_id(rid)
    except KeyError:
        return None
    return _render_slice_storyboard(entry.episode_pkl, start, end)


def render_twostage(
    session_dir: Path,
    pool: RolloutPool,
    *,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    session_dir = Path(session_dir)
    desc_dir = session_dir / "stage1_description"
    prop_dir = session_dir / "stage2_proposals"

    # Load artefacts
    desc_data: Dict[str, Any] = {}
    if (desc_dir / "visual_descriptions.json").exists():
        desc_data = json.loads((desc_dir / "visual_descriptions.json").read_text())

    s1_summary: Dict[str, Any] = {}
    if (desc_dir / "session_summary.json").exists():
        s1_summary = json.loads((desc_dir / "session_summary.json").read_text())

    submitted: List[Dict[str, Any]] = []
    if (prop_dir / "submitted_requests.json").exists():
        submitted = json.loads((prop_dir / "submitted_requests.json").read_text())

    s2_summary: Dict[str, Any] = {}
    if (prop_dir / "session_summary.json").exists():
        s2_summary = json.loads((prop_dir / "session_summary.json").read_text())

    rationale = ""
    if (prop_dir / "rationale.txt").exists():
        rationale = (prop_dir / "rationale.txt").read_text()

    parts: List[str] = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{escape(title or 'Two-stage session report')}</title>"
        f"<style>{_CSS}</style></head><body>"
    ]

    # Header
    title_str = title or f"Two-stage session: {session_dir.name}"
    parts.append(f"<header><h1>{escape(title_str)}</h1>")
    meta_parts = []
    if s1_summary:
        b1 = s1_summary.get("budget_summary", {})
        meta_parts.append(
            f"Stage 1: {b1.get('n_tool_calls','?')} tool calls · "
            f"{b1.get('n_visual_calls','?')} visual · stop={s1_summary.get('stop_reason','?')}"
        )
    if s2_summary:
        b2 = s2_summary.get("budget_summary", {})
        meta_parts.append(
            f"Stage 2: {b2.get('n_tool_calls','?')} tool calls · "
            f"{s2_summary.get('n_submitted','?')} submitted · stop={s2_summary.get('stop_reason','?')}"
        )
    meta_parts.append(f"Pool: {len(pool)} rollouts")
    parts.append(f"<div class='meta'>{escape(' · '.join(meta_parts))}</div></header>")

    # ---- Stage 1: descriptions ----
    parts.append("<span class='stage-header s1-header'>Stage 1 — Visual Descriptions</span>")
    cluster_descs: List[Dict[str, Any]] = desc_data.get("cluster_descriptions", [])
    if not cluster_descs:
        parts.append("<p><em>No descriptions found (Stage 1 may have failed).</em></p>")
    else:
        for d in cluster_descs:
            cid = d.get("cluster_id", "?")
            informative = d.get("informative", True)
            contact = d.get("robot_object_contact", False)
            slices = d.get("slices_observed", [])

            card_cls = "cluster-card" + ("" if informative else " uninformative")
            parts.append(f"<div class='{card_cls}'>")
            parts.append("<div class='card-header'>")
            parts.append(f"<span class='cluster-title'>c{cid}</span>")
            badges = []
            if informative:
                badges.append("<span class='badge badge-ok'>informative</span>")
            else:
                badges.append("<span class='badge badge-bad'>uninformative</span>")
            if contact:
                loc = d.get("contact_location", "")
                badges.append(f"<span class='badge badge-contact'>contact: {escape(loc or 'yes')}</span>")
            parts.append("  <div style='display:flex;gap:0.4rem'>" + "".join(badges) + "</div>")
            parts.append("</div>")  # card-header

            parts.append("<div class='cluster-body'>")
            parts.append(f"<div class='desc-text'>{escape(d.get('literal_description',''))}</div>")
            meta_bits = []
            if d.get("gripper_states"):
                meta_bits.append(f"Gripper: {d['gripper_states']}")
            if d.get("object_location"):
                meta_bits.append(f"Object: {d['object_location']}")
            if d.get("sequence_of_events"):
                meta_bits.append(f"Sequence: {d['sequence_of_events']}")
            if meta_bits:
                parts.append("<div class='desc-meta'>" + escape(" · ".join(meta_bits)) + "</div>")

            # Re-render storyboards for the observed slices
            if slices:
                parts.append("<div class='storyboard-grid'>")
                for sid in slices:
                    img = _materialize(pool, sid)
                    if img is not None:
                        uri = _img_to_data_uri(img, max_dim=512)
                        parts.append(
                            f"<figure><img src='{uri}' alt='{escape(sid)}'>"
                            f"<figcaption>{escape(sid)}</figcaption></figure>"
                        )
                    else:
                        parts.append(f"<figure><div class='missing'>{escape(sid)}: not renderable</div></figure>")
                parts.append("</div>")  # storyboard-grid

            parts.append("</div>")  # cluster-body
            parts.append("</div>")  # cluster-card

    # ---- Stage 2: proposals ----
    parts.append("<h2><span class='stage-header s2-header'>Stage 2 — Demonstration Proposals</span></h2>")
    if not submitted:
        parts.append("<p><em>No submissions found.</em></p>")
    else:
        for i, sr in enumerate(submitted, 1):
            r = sr["request"]
            ic = r["initial_conditions"]
            parts.append("<div class='submission'>")
            cluster_tag = f" · cluster c{r['target_cluster']}" if r.get("target_cluster") is not None else ""
            parts.append(
                f"<h3><span class='num'>#{i}</span>"
                f"<span>{escape(r['request_type'])}</span>"
                f"<span style='color:#888;font-weight:400;font-size:0.9em'>"
                f"→ {escape(ic['reference_rollout_id'])} @ frame {ic['reference_frame']}"
                f"{escape(cluster_tag)}</span></h3>"
            )
            parts.append("<div class='field'><div class='field-label'>Operator instruction</div>")
            parts.append(f"<div class='field-body'>{escape(r['target_behavior'])}</div></div>")
            if r.get("prohibitions"):
                parts.append("<div class='field'><div class='field-label'>Prohibitions</div><div class='field-body'><ul>")
                for p in r["prohibitions"]:
                    parts.append(f"<li style='color:#b15a00'>{escape(p)}</li>")
                parts.append("</ul></div></div>")
            if sr.get("reasoning"):
                parts.append("<div class='field'><div class='field-label'>Agent reasoning</div>")
                parts.append(f"<div class='field-body reasoning'>{escape(sr['reasoning'])}</div></div>")

            # Evidence storyboards
            evidence_ids = r.get("evidence_slice_ids") or r.get("evidence_rollout_ids") or []
            seen = set()
            unique_ids = [x for x in evidence_ids if not (x in seen or seen.add(x))]
            parts.append("<div class='field'><div class='field-label'>Evidence</div>")
            if not unique_ids:
                parts.append("<div class='missing'>No evidence cited.</div>")
            else:
                parts.append("<div class='evidence-grid'>")
                for eid in unique_ids:
                    img = _materialize(pool, eid)
                    if img:
                        uri = _img_to_data_uri(img, max_dim=384)
                        parts.append(
                            f"<figure><img src='{uri}' alt='{escape(eid)}'>"
                            f"<figcaption>{escape(eid)}</figcaption></figure>"
                        )
                    else:
                        parts.append(f"<figure><div class='missing'>{escape(eid)}</div></figure>")
                parts.append("</div>")
            parts.append("</div>")
            parts.append("</div>")  # submission

    if rationale:
        parts.append("<div class='rationale-box'>")
        parts.append("<div class='field-label'>Finalize rationale</div>")
        parts.append(f"<div>{escape(rationale)}</div></div>")

    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    out_path = args.out or (args.session_dir / "twostage_report.html")
    pool = RolloutPool.from_episodes_dir(args.episodes_dir)
    render_twostage(args.session_dir, pool, out_path=out_path, title=args.title)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
