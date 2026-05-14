"""Render an agent session into a single HTML report with evidence storyboards.

Given a session directory (the one ``AgentSession.run`` writes — containing
``submitted_requests.json``, ``rationale.txt``, etc.) and the rollout pool
the agent saw, this script materialises every submission's
``evidence_slice_ids`` (or ``evidence_rollout_ids`` for A_NG) as embedded
base64 storyboards and lays them out alongside the agent's prose.

Usage::

    python scripts/render_agent_session.py \\
        --session-dir /tmp/qwen32b_4bit_plwiogm_/session \\
        --episodes-dir /mnt/ssdB/erik/.../latest/episodes \\
        --out report.html

Produces a self-contained HTML file (images are inlined as base64) that you
can open in any browser. Useful for:

* Reviewing what the agent actually based its target_behavior prose on.
* Comparing submissions side-by-side (the same evidence storyboards appear
  with the agent's reasoning, so you can judge whether the prose tracks the
  imagery).
* Posting in a doc / paper appendix without managing a separate image dir.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "third_party" / "cupid"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from policy_doctor.vlm.proposals.agents.tools.access import (  # noqa: E402
    _render_slice_storyboard,
    parse_slice_id,
)
from policy_doctor.vlm.proposals.pool import RolloutPool  # noqa: E402


def _img_to_data_uri(img: Image.Image, *, max_dim: int = 384) -> str:
    """Inline a PIL image as a data URI so the HTML is self-contained.

    Resizes large images to keep the report file size manageable; agents see
    the original-resolution storyboards, but the HTML report doesn't need to.
    """
    img = img.convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _materialize_slice(pool: RolloutPool, slice_id: str) -> Optional[Image.Image]:
    parsed = parse_slice_id(slice_id)
    if parsed is None:
        return None
    rid, start, end = parsed
    try:
        entry = pool.by_id(rid)
    except KeyError:
        return None
    return _render_slice_storyboard(entry.episode_pkl, start, end)


def _materialize_rollout(pool: RolloutPool, rollout_id: str) -> Optional[Image.Image]:
    try:
        entry = pool.by_id(rollout_id)
    except KeyError:
        return None
    return _render_slice_storyboard(entry.episode_pkl, 0, max(entry.length - 1, 0))


_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agent session report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1rem;
         color: #1a1a1a; line-height: 1.5; }
  header { border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; padding-bottom: 0.5rem; }
  h1 { margin: 0; font-size: 1.6rem; }
  .meta { color: #666; font-size: 0.9rem; margin-top: 0.25rem; }
  .submission { border: 1px solid #d0d7de; border-radius: 6px;
                padding: 1rem 1.25rem; margin-bottom: 1.5rem;
                background: #fafbfc; }
  .submission h2 { margin: 0 0 0.5rem 0; font-size: 1.15rem;
                   display: flex; gap: 0.6rem; align-items: baseline; }
  .submission h2 .num { color: #999; font-weight: 400; }
  .submission .header-meta { display: flex; flex-wrap: wrap; gap: 0.5rem;
                              margin-bottom: 0.75rem; font-size: 0.85rem;
                              color: #555; }
  .submission .header-meta span { background: #eef1f4; padding: 0.1rem 0.5rem;
                                   border-radius: 3px; }
  .field { margin: 0.5rem 0; }
  .field-label { font-weight: 600; color: #444; font-size: 0.85rem;
                  text-transform: uppercase; letter-spacing: 0.04em;
                  margin-bottom: 0.2rem; }
  .field-body { padding-left: 0.5rem; border-left: 3px solid #d0d7de; }
  .reasoning { color: #586069; font-style: italic; }
  .prohibitions li { color: #b15a00; }
  .evidence-grid { display: grid; grid-template-columns: repeat(3, 1fr);
                    gap: 0.75rem; margin-top: 0.5rem; }
  .evidence-grid figure { margin: 0; }
  .evidence-grid img { width: 100%; height: auto; border: 1px solid #d0d7de;
                        border-radius: 4px; display: block; }
  .evidence-grid figcaption { font-size: 0.75rem; color: #777;
                                margin-top: 0.25rem; text-align: center;
                                font-family: ui-monospace, SFMono-Regular, monospace; }
  .missing { padding: 1.5rem; background: #ffeef0; color: #86181d;
             border-radius: 4px; font-size: 0.9rem; }
  .rationale-box { background: #f0f6ff; border-left: 4px solid #5599ff;
                    padding: 0.75rem 1rem; margin-top: 1.5rem;
                    border-radius: 0 4px 4px 0; }
  .rationale-box .field-label { color: #1a4f99; }
</style>
</head>
<body>
"""


def render(
    session_dir: Path,
    pool: RolloutPool,
    *,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    submitted_path = session_dir / "submitted_requests.json"
    if not submitted_path.exists():
        raise FileNotFoundError(f"no submitted_requests.json in {session_dir}")
    submitted = json.loads(submitted_path.read_text())

    summary_path = session_dir / "session_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    rationale_path = session_dir / "rationale.txt"
    rationale = rationale_path.read_text() if rationale_path.exists() else ""

    parts: List[str] = [_HTML_HEAD]

    title = title or f"Agent session: {session_dir.name}"
    parts.append(f"<header><h1>{escape(title)}</h1>")
    meta_lines = []
    if summary:
        meta_lines.append(
            f"condition={escape(str(summary.get('condition', '?')))} "
            f"seed={summary.get('seed', '?')} "
            f"stop={escape(str(summary.get('stop_reason', '?')))} "
            f"turns={summary.get('n_turns', '?')} "
            f"submitted={summary.get('n_submitted', len(submitted))} "
            f"failed_calls={summary.get('n_failed_tool_calls', '?')}"
        )
    meta_lines.append(f"pool: {len(pool)} rollouts ({len(pool.successes())}S/{len(pool.failures())}F)")
    parts.append(f"<div class='meta'>{' &middot; '.join(escape(line) for line in meta_lines)}</div>")
    parts.append("</header>")

    for i, sr in enumerate(submitted, 1):
        r = sr["request"]
        ic = r["initial_conditions"]
        parts.append("<div class='submission'>")
        cluster_tag = (
            f" &middot; cluster c{r['target_cluster']}"
            if r.get("target_cluster") is not None
            else ""
        )
        parts.append(
            f"<h2><span class='num'>#{i}</span>"
            f"<span>{escape(r['request_type'])}</span>"
            f"<span style='color:#888;font-weight:400;font-size:0.9em'>"
            f"→ ref {escape(ic['reference_rollout_id'])} @ frame {ic['reference_frame']}"
            f"{cluster_tag}"
            f"</span></h2>"
        )

        # target_behavior — what the operator reads.
        parts.append("<div class='field'>")
        parts.append("<div class='field-label'>operator instruction (target_behavior)</div>")
        parts.append(f"<div class='field-body'>{escape(r['target_behavior'])}</div>")
        parts.append("</div>")

        # prohibitions, if any.
        if r.get("prohibitions"):
            parts.append("<div class='field'>")
            parts.append("<div class='field-label'>prohibitions</div>")
            parts.append("<div class='field-body'><ul class='prohibitions'>")
            for p in r["prohibitions"]:
                parts.append(f"<li>{escape(p)}</li>")
            parts.append("</ul></div>")
            parts.append("</div>")

        # reasoning — the agent's hypothesis (server-side, never shown to operator).
        if sr.get("reasoning"):
            parts.append("<div class='field'>")
            parts.append("<div class='field-label'>agent reasoning (server-side)</div>")
            parts.append(f"<div class='field-body reasoning'>{escape(sr['reasoning'])}</div>")
            parts.append("</div>")

        # Evidence storyboards.
        evidence_kind, evidence_ids = _evidence_for_request(r)
        parts.append("<div class='field'>")
        parts.append(
            f"<div class='field-label'>evidence ({evidence_kind})</div>"
        )
        if not evidence_ids:
            parts.append("<div class='missing'>No evidence cited.</div>")
        else:
            parts.append("<div class='evidence-grid'>")
            for eid in evidence_ids:
                if evidence_kind == "slice":
                    img = _materialize_slice(pool, eid)
                else:
                    img = _materialize_rollout(pool, eid)
                if img is None:
                    parts.append(
                        f"<figure><div class='missing'>could not render {escape(eid)}</div></figure>"
                    )
                else:
                    data_uri = _img_to_data_uri(img)
                    parts.append(
                        f"<figure><img src='{data_uri}' alt='{escape(eid)}'>"
                        f"<figcaption>{escape(eid)}</figcaption></figure>"
                    )
            parts.append("</div>")
        parts.append("</div>")
        parts.append("</div>")

    if rationale:
        parts.append("<div class='rationale-box'>")
        parts.append("<div class='field-label'>finalize_strategy rationale</div>")
        parts.append(f"<div>{escape(rationale)}</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    out_path.write_text("".join(parts))


def _evidence_for_request(req: Dict[str, Any]) -> tuple[str, List[str]]:
    """Pull (kind, ids) out of a request dict. A_G uses slice_ids; A_NG rollout_ids."""
    if req.get("evidence_slice_ids"):
        return "slice", list(req["evidence_slice_ids"])
    if req.get("evidence_rollout_ids"):
        return "rollout", list(req["evidence_rollout_ids"])
    # Backwards-compatibility: some old submissions don't have evidence yet.
    return ("slice" if req.get("target_cluster") is not None else "rollout"), []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True,
                        help="Path to the agent session output directory.")
    parser.add_argument("--episodes-dir", type=Path, required=True,
                        help="Path to the rollout pool's episodes/ directory.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (defaults to <session_dir>/session_report.html)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    out_path = args.out or (args.session_dir / "session_report.html")
    pool = RolloutPool.from_episodes_dir(args.episodes_dir)
    render(args.session_dir, pool, out_path=out_path, title=args.title)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
