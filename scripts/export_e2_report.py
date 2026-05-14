"""Export a self-contained HTML report of E2 agentic graph exploration sessions.

Shows the full turn-by-turn conversation: model reasoning, tool calls,
tool results, and final submitted requests.

Usage:
    python scripts/export_e2_report.py --out /tmp/e2_report.html
    python scripts/export_e2_report.py --sessions /tmp/qwen32b_4bit_gjofj90m/session /tmp/qwen32b_4bit_plwiogm_/session --out /tmp/e2_report.html
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional, Tuple

# ── Session discovery ─────────────────────────────────────────────────────────

_DEFAULT_SESSION_DIRS = [
    "/tmp/qwen32b_4bit_gjofj90m/session",
    "/tmp/qwen32b_4bit_plwiogm_/session",
]

TOOL_COLORS = {
    "get_graph_summary":          "#4363d8",
    "find_failure_nodes":         "#e6194b",
    "find_recovery_paths":        "#f58231",
    "find_underrepresented_modes":"#911eb4",
    "get_node":                   "#3cb44b",
    "get_edge":                   "#42d4f4",
    "list_nodes":                 "#469990",
    "list_paths":                 "#9a6324",
    "list_slices_in_node":        "#bfef45",
    "get_rollout_summary":        "#fabebe",
    "list_rollouts":              "#e6beff",
    "get_slice_video":            "#fffac8",
    "get_rollout_video":          "#aaffc3",
    "compare_paths":              "#ffd8b1",
    "propose_collection_request": "#000075",
    "revise_request":             "#008080",
    "delete_request":             "#800000",
    "list_submitted_requests":    "#808000",
    "finalize_strategy":          "#000000",
}
DEFAULT_TOOL_COLOR = "#888"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_session(session_dir: pathlib.Path) -> Optional[Dict]:
    trace_f   = session_dir / "trace.jsonl"
    summary_f = session_dir / "session_summary.json"
    budget_f  = session_dir / "budget_summary.json"
    conv_f    = session_dir / "conversation.json"
    rat_f     = session_dir / "rationale.txt"

    if not trace_f.exists():
        return None
    events = [json.loads(l) for l in open(trace_f)]
    summary = json.load(open(summary_f)) if summary_f.exists() else {}
    budget  = json.load(open(budget_f))  if budget_f.exists()  else {}
    rationale = open(rat_f).read().strip() if rat_f.exists() else None

    # Infer model from tool call IDs
    model = "Qwen3-VL (unknown size)"
    for ev in events:
        if ev["kind"] == "assistant_turn":
            for tc in ev.get("tool_calls", []):
                if "qwen" in tc["id"].lower():
                    if "32b" in str(session_dir).lower() or "32b" in str(tc["id"]).lower():
                        model = "Qwen3-VL-32B-Instruct (NF4)"
                    else:
                        model = "Qwen3-VL-8B-Instruct"
                    break

    # System prompt — first user message in conversation
    system_prompt = None
    if conv_f.exists():
        conv = json.load(open(conv_f))
        if conv and conv[0]["role"] == "user":
            content = conv[0]["content"]
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        system_prompt = c["text"]
                        break
            elif isinstance(content, str):
                system_prompt = content

    # Build tool_use_id -> full text from conversation.json (trace only stores previews)
    full_results: Dict[str, str] = {}
    if conv_f.exists():
        try:
            conv = json.load(open(conv_f))
            for msg in conv:
                if msg["role"] != "user":
                    continue
                for c in (msg["content"] if isinstance(msg["content"], list) else []):
                    if not isinstance(c, dict) or c.get("type") != "tool_result":
                        continue
                    tid = c.get("tool_use_id", "")
                    texts = []
                    for ic in (c.get("content", []) if isinstance(c.get("content"), list) else []):
                        if isinstance(ic, dict) and ic.get("type") == "text":
                            texts.append(ic["text"])
                    if texts:
                        full_results[tid] = "\n".join(texts)
        except Exception:
            pass

    return {
        "dir":           session_dir,
        "name":          session_dir.parent.name,
        "events":        events,
        "summary":       summary,
        "budget":        budget,
        "rationale":     rationale,
        "model":         model,
        "system_prompt": system_prompt,
        "full_results":  full_results,
    }


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _e(s: Any) -> str:
    return escape(str(s) if s is not None else "")


def _tool_badge(name: str) -> str:
    col = TOOL_COLORS.get(name, DEFAULT_TOOL_COLOR)
    text_col = "white" if name in TOOL_COLORS else "#333"
    return (f'<span style="background:{col};color:{text_col};padding:2px 8px;'
            f'border-radius:10px;font-size:12px;font-weight:bold">{_e(name)}</span>')


def _json_block(obj: Any, max_chars: int = 800) -> str:
    s = json.dumps(obj, indent=2) if not isinstance(obj, str) else obj
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n… ({len(s)-max_chars} more chars)"
    return f'<pre style="margin:0;font-size:12px;white-space:pre-wrap;word-break:break-word">{_e(s)}</pre>'


def _ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ── Event renderers ───────────────────────────────────────────────────────────

def _render_session_start(ev: Dict) -> str:
    return (f'<div style="background:#f0f4ff;border-left:4px solid #4363d8;'
            f'padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0">'
            f'<b>Session start</b> · condition={_e(ev.get("condition"))} · '
            f'seed={_e(ev.get("seed"))} · '
            f'{len(ev.get("tool_names",[]))} tools available'
            f'</div>')


def _render_assistant_turn(ev: Dict) -> str:
    parts = []
    text = ev.get("text") or ""
    if text.strip():
        parts.append(
            f'<div style="background:#fff8e1;border-radius:6px;padding:10px 14px;'
            f'margin-bottom:6px;font-size:13px;line-height:1.5">'
            f'<b>💬 Model reasoning</b><br>{_e(text)}</div>'
        )
    for tc in ev.get("tool_calls", []):
        name    = tc.get("name", "?")
        inp     = tc.get("input", {})
        tc_id   = tc.get("id", "")
        inp_str = json.dumps(inp, indent=2) if inp else "(no args)"
        preview = json.dumps(inp) if inp else "(no args)"
        preview = preview[:120] + "…" if len(preview) > 120 else preview
        args_html = (
            f'<pre style="font-size:12px;white-space:pre-wrap;word-break:break-word;'
            f'background:#f0f0f0;padding:6px;border-radius:4px;margin:0;'
            f'max-height:200px;overflow-y:auto">{_e(inp_str)}</pre>'
        )
        parts.append(
            f'<div style="display:flex;align-items:flex-start;gap:10px;'
            f'margin:4px 0;padding:8px 12px;background:#f9f9f9;border-radius:6px">'
            f'<div style="min-width:200px">{_tool_badge(name)}'
            f'<br><span style="font-size:10px;color:#aaa">{_e(tc_id)}</span></div>'
            f'<div style="flex:1">{args_html}</div>'
            f'</div>'
        )
    usage = ev.get("usage", {})
    tok_str = (f'in={usage.get("input_tokens","?")} out={usage.get("output_tokens","?")} '
               f'cache_read={usage.get("cache_read_input_tokens",0)}')
    return (f'<div style="border-left:3px solid #f58231;padding:8px 14px;margin:10px 0">'
            f'<div style="font-size:11px;color:#888;margin-bottom:6px">'
            f'<b>Model turn</b> · {_ts(ev["ts"])} · tokens: {tok_str}'
            f'</div>'
            f'{"".join(parts)}'
            f'</div>')


def _render_tool_result(ev: Dict, full_results: Dict[str, str] = None) -> str:
    name      = ev.get("name", "?")
    ok        = ev.get("ok", True)
    latency   = ev.get("latency_ms")
    content   = ev.get("content", [])
    tc_id     = ev.get("tool_use_id", "")
    cache_hit = ev.get("cache_hit", False)

    status_col = "#3cb44b" if ok else "#e6194b"
    status_txt = "✓" if ok else "✗ error"

    body_parts = []
    # Prefer full text from conversation.json over truncated trace preview
    full_text = (full_results or {}).get(tc_id)
    if full_text:
        body_parts.append(
            f'<pre style="font-size:12px;white-space:pre-wrap;word-break:break-word;'
            f'background:#f5f5f5;padding:8px;border-radius:4px;margin:4px 0;'
            f'max-height:400px;overflow-y:auto">{_e(full_text)}</pre>'
        )
    else:
        for c in content:
            ctype = c.get("type", "text")
            if ctype == "text":
                txt = c.get("text") or c.get("preview") or ""
                if txt:
                    body_parts.append(
                        f'<pre style="font-size:12px;white-space:pre-wrap;word-break:break-word;'
                        f'background:#f5f5f5;padding:8px;border-radius:4px;margin:4px 0;'
                        f'max-height:400px;overflow-y:auto">{_e(txt)}</pre>'
                    )

    for c in content:
        if c.get("type") == "image":
            sz  = c.get("size", [])
            cap = c.get("caption", "")
            body_parts.append(
                f'<div style="font-size:11px;color:#888;font-style:italic;padding:4px 8px">'
                f'📷 Image {sz[0] if sz else "?"}×{sz[1] if len(sz)>1 else "?"}'
                f' — {_e(cap)} (stripped from log)</div>'
            )

    latency_str = f" · {latency:.0f}ms" if latency else ""
    cache_str   = " · 🗃 cache hit" if cache_hit else ""

    return (f'<div style="border-left:3px solid {status_col};'
            f'padding:8px 14px;margin:6px 0 10px 20px">'
            f'<div style="font-size:11px;color:#888;margin-bottom:4px">'
            f'{_tool_badge(name)} result · '
            f'<span style="color:{status_col}">{status_txt}</span>'
            f'{latency_str}{cache_str} · {_ts(ev["ts"])}'
            f'</div>'
            f'{"".join(body_parts)}'
            f'</div>')


def _render_session_end(ev: Dict, rationale: Optional[str]) -> str:
    stop   = ev.get("stop_reason", "?")
    n_sub  = ev.get("n_submitted", 0)
    n_turn = ev.get("n_turns", 0)
    n_tc   = ev.get("n_tool_calls", 0)
    col    = "#3cb44b" if stop == "finalize" else ("#e6194b" if "error" in stop else "#f58231")
    rat_html = (f'<div style="margin-top:8px;font-size:13px"><b>Strategy rationale:</b> {_e(rationale)}</div>'
                if rationale else "")
    return (f'<div style="background:#f5f5f5;border-left:4px solid {col};'
            f'padding:10px 14px;margin:16px 0;border-radius:0 6px 6px 0">'
            f'<b>Session end</b> · '
            f'<span style="color:{col};font-weight:bold">{_e(stop)}</span> · '
            f'{n_turn} turns · {n_tc} tool calls · {n_sub} requests submitted'
            f'{rat_html}'
            f'</div>')


def _render_submitted_requests(summary: Dict) -> str:
    reqs = summary.get("submitted_requests", [])
    if not reqs:
        return '<p style="color:#888;font-style:italic">No requests submitted.</p>'

    cards = []
    for i, item in enumerate(reqs):
        req = item if "request_type" in item else item.get("request", item)
        reasoning = item.get("reasoning", "")
        req_type  = req.get("request_type", "?")
        target_c  = req.get("target_cluster")
        ic        = req.get("initial_conditions", {})
        ref_roll  = ic.get("reference_rollout_id", "?")
        ref_frame = ic.get("reference_frame", 0)
        target_beh = req.get("target_behavior", "")
        prohibs    = req.get("prohibitions", [])
        evidence   = req.get("evidence_slice_ids") or req.get("evidence_rollout_ids") or []

        prohib_html = ""
        if prohibs:
            items = "".join(f"<li>{_e(p)}</li>" for p in prohibs)
            prohib_html = f'<div style="margin-top:6px"><b>Prohibitions:</b><ul style="margin:4px 0">{items}</ul></div>'

        reasoning_html = ""
        if reasoning:
            reasoning_html = (f'<div style="margin-top:8px;padding:8px;background:#fff8e1;'
                              f'border-radius:4px;font-size:12px">'
                              f'<b>Agent reasoning:</b> {_e(reasoning)}</div>')

        ev_html = ""
        if evidence:
            ev_html = (f'<div style="margin-top:6px;font-size:12px">'
                       f'<b>Evidence:</b> {", ".join(_e(e) for e in evidence[:10])}'
                       f'{"…" if len(evidence) > 10 else ""}</div>')

        tc_html = (f'<span style="background:#000075;color:white;padding:2px 7px;'
                   f'border-radius:10px;font-size:12px">{_e(req_type)}</span>')
        if target_c is not None:
            tc_html += (f' <span style="background:#3cb44b;color:white;padding:2px 7px;'
                        f'border-radius:10px;font-size:12px">→ cluster {target_c}</span>')

        cards.append(f"""
        <div style="border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:10px 0">
          <div style="margin-bottom:8px">{tc_html}
            <span style="font-size:12px;color:#888;margin-left:8px">
              ref rollout {_e(ref_roll)}, frame {_e(ref_frame)}
            </span>
          </div>
          <div style="font-size:14px"><b>Target behaviour:</b> {_e(target_beh)}</div>
          {prohib_html}{ev_html}{reasoning_html}
        </div>""")

    return "".join(cards)


# ── Session renderer ──────────────────────────────────────────────────────────

def render_session(sess: Dict) -> str:
    summary = sess["summary"]
    events  = sess["events"]
    name    = sess["name"]
    model   = sess["model"]
    rationale = sess["rationale"]

    stop    = summary.get("stop_reason", "?")
    cond    = summary.get("condition", "?")
    n_turns = summary.get("n_turns", 0)
    n_sub   = summary.get("n_submitted", 0)
    budget  = sess["budget"]
    n_tc    = budget.get("n_tool_calls", "?")
    n_vis   = budget.get("n_visual_calls", "?")
    elapsed = budget.get("elapsed", 0)
    stop_col = "#3cb44b" if stop == "finalize" else ("#e6194b" if "error" in stop else "#f58231")

    # System prompt
    sp_html = ""
    if sess.get("system_prompt"):
        sp_html = (f'<details style="margin:10px 0"><summary style="cursor:pointer;'
                   f'font-weight:bold;color:#4363d8">System / user prompt</summary>'
                   f'<pre style="background:#f5f5f5;padding:12px;font-size:12px;'
                   f'white-space:pre-wrap;border-radius:6px;margin-top:6px">'
                   f'{_e(sess["system_prompt"])}</pre></details>')

    # Timeline
    timeline_parts = []
    for ev in events:
        k = ev["kind"]
        if k == "session_start":
            timeline_parts.append(_render_session_start(ev))
        elif k == "assistant_turn":
            timeline_parts.append(_render_assistant_turn(ev))
        elif k == "tool_result":
            timeline_parts.append(_render_tool_result(ev, sess.get("full_results", {})))
        elif k == "session_end":
            timeline_parts.append(_render_session_end(ev, rationale))
        elif k == "error":
            timeline_parts.append(
                f'<div style="background:#fff0f0;border-left:4px solid #e6194b;'
                f'padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0">'
                f'<b>Error</b><br><pre style="font-size:11px;white-space:pre-wrap">'
                f'{_e(ev.get("message","")[:600])}</pre></div>'
            )

    # Submitted requests section
    req_html = _render_submitted_requests(summary)

    return f"""
    <div style="border:2px solid #ddd;border-radius:10px;margin:24px 0;overflow:hidden">
      <!-- Header -->
      <div style="background:#333;color:white;padding:14px 20px">
        <div style="font-size:18px;font-weight:bold">{_e(name)}</div>
        <div style="font-size:13px;margin-top:4px;opacity:0.8">
          {_e(model)} · condition={_e(cond)} ·
          <span style="color:{'#7fff7f' if stop=='finalize' else '#ff7f7f'}">{_e(stop)}</span> ·
          {n_turns} turns · {n_tc} tool calls · {n_vis} visual calls ·
          {n_sub} submitted · {elapsed:.0f}s
        </div>
      </div>
      <div style="padding:16px 20px">
        {sp_html}
        <h3 style="margin:16px 0 8px">Conversation trace</h3>
        {"".join(timeline_parts)}
        <h3 style="margin:24px 0 8px">Submitted requests ({n_sub})</h3>
        {req_html}
      </div>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e2_report.html")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Session dirs to include (default: the two 32B finalized sessions)")
    args = ap.parse_args()

    session_paths = [pathlib.Path(s) for s in (args.sessions or _DEFAULT_SESSION_DIRS)]
    sessions = []
    for p in session_paths:
        if not p.exists():
            print(f"Skip (not found): {p}")
            continue
        s = load_session(p)
        if s is None:
            print(f"Skip (no trace): {p}")
            continue
        sessions.append(s)
        print(f"Loaded: {s['name']} — {s['summary'].get('n_turns',0)} turns, "
              f"stop={s['summary'].get('stop_reason','?')}")

    sessions = [s for s in sessions if s["budget"].get("n_visual_calls", 0) > 0]
    if not sessions:
        print("No sessions with visual calls found.")
        return

    bodies = "".join(render_session(s) for s in sessions)

    # Tool legend
    legend_items = "".join(
        f'<span style="background:{col};color:white;padding:2px 7px;'
        f'border-radius:10px;font-size:11px;margin:2px">{_e(name)}</span>'
        for name, col in TOOL_COLORS.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E2 Agent Exploration Report</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          max-width:1100px;margin:0 auto;padding:24px;color:#222;line-height:1.5 }}
  h1   {{ border-bottom:2px solid #333;padding-bottom:8px }}
  h2   {{ margin-top:32px;border-bottom:1px solid #ccc }}
  pre  {{ font-family:"SFMono-Regular",Consolas,monospace }}
  details > summary {{ list-style:none }}
  details > summary::-webkit-details-marker {{ display:none }}
</style>
</head>
<body>
<h1>E2 Agentic Graph Exploration — Session Report</h1>
<p style="color:#666;font-size:14px">
  Qwen3-VL-32B-Instruct (NF4) · Condition A_G (graph-grounded) ·
  Task: two-arm transport (transport_mh) ·
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>
<details style="margin:12px 0">
  <summary style="cursor:pointer;color:#4363d8;font-size:13px">Tool colour legend</summary>
  <div style="margin-top:8px;line-height:2">{legend_items}</div>
</details>
{bodies}
</body>
</html>"""

    out = pathlib.Path(args.out)
    out.write_text(html)
    print(f"\nDone → {out}  ({out.stat().st_size // 1024} KB)")
    print(f"Download:  scp triton:{out} ~/Desktop/e2_report.html")


if __name__ == "__main__":
    main()
