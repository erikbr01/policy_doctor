"""Export a self-contained HTML report of E1 cluster coherence results.

Embeds all storyboard images as base64, requires no external dependencies
to view. Run on Triton, then scp the HTML to your Mac.

Usage:
    conda run -n policy_doctor python scripts/export_e1_report.py \
        --out /tmp/e1_report.html

Optional flags:
    --exp_dirs  Specific experiment dirs (default: head_to_head + policy_emb_sweep top)
    --max_clusters  Max clusters to show storyboards for (default: 5)
    --frame_seed    RNG seed for storyboard frame sampling (default: 42)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
from typing import List, Optional

import numpy as np
from PIL import Image
from scipy.stats import binomtest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.vlm.frames import (
    extract_window_frames,
    list_rollout_episode_pkls,
    resolve_window_indices,
)
from policy_doctor.vlm.storyboard import make_storyboard

# ── Model display name ────────────────────────────────────────────────────────

def _model_label(backend: str, exp_name: str,
                 model_id: str = None,
                 load_in_4bit: bool = None,
                 load_in_8bit: bool = None) -> str:
    """Human-readable model string from stored fields or name heuristics."""
    if model_id:
        # e.g. "Qwen/Qwen3-VL-8B-Instruct" → "Qwen3-VL-8B-Instruct"
        short = model_id.split("/")[-1]
        if load_in_4bit:
            return f"{short} (NF4)"
        if load_in_8bit:
            return f"{short} (int8)"
        return f"{short} (bf16)"
    # Fallback: infer from experiment directory name
    name = exp_name.lower()
    if "32b" in name and "nf4" in name:
        return "Qwen3-VL-32B-Instruct (NF4)"
    if "32b" in name and "8bit" in name:
        return "Qwen3-VL-32B-Instruct (int8)"
    if "32b" in name:
        return "Qwen3-VL-32B-Instruct"
    if backend in ("qwen2_vl", "qwen3_vl"):
        return "Qwen3-VL-8B-Instruct (bf16)"
    return backend

# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#469990","#9a6324",
]
def _col(cid): return _PALETTE[int(cid) % len(_PALETTE)]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_b64(img: Image.Image, size: int = 380) -> str:
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def _make_storyboard(eval_dir: str, slice_idx: int, metadata: list,
                     max_frames: int, comp_size: int, seed: int) -> Optional[Image.Image]:
    try:
        rng = np.random.default_rng(seed)
        meta = metadata[slice_idx]
        r_idx, w0, w1 = resolve_window_indices(meta)
        frames = extract_window_frames(
            pathlib.Path(eval_dir), r_idx, w0, w1, max_frames=max_frames, rng=rng
        )
        return make_storyboard(frames, target_size=(comp_size, comp_size))
    except Exception as e:
        print(f"  storyboard failed for slice {slice_idx}: {e}")
        return None


def _get_clean(exp_dir: pathlib.Path, K: int):
    try:
        sp    = json.load(open(exp_dir / "sample_plan.json"))
        preds = [json.loads(l) for l in open(exp_dir / "predictions.jsonl")]
    except FileNotFoundError:
        return None
    om = {}
    for cid, cdata in sp["clusters"].items():
        origins = cdata.get("query_origins", [])
        for qi, qidx in enumerate(cdata["query_indices"]):
            om[qidx] = origins[qi] if qi < len(origins) else "tier1_global"
    clean = [p for p in preds if om.get(p["query_idx"]) in ("tier1_global", None)]
    c = sum(1 for p in clean if p["is_correct"])
    n = len(clean)
    if n == 0: return None
    p = binomtest(c, n, 1/K, alternative="greater").pvalue
    return c/n, c, n, p


# ── HTML builders ─────────────────────────────────────────────────────────────

def _acc_bar(acc: float, chance: float, K: int, c: int, n: int, p: float) -> str:
    pct = acc * 100
    chance_pct = chance * 100
    ratio = acc / chance
    sig = "★" if p < 0.001 else ("✓" if p < 0.05 else "NS")
    col = "#3cb44b" if p < 0.05 else "#aaaaaa"
    return f"""
    <div style="margin:4px 0">
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:220px;background:#eee;border-radius:4px;position:relative;height:22px">
          <div style="width:{pct:.1f}%;background:{col};height:100%;border-radius:4px"></div>
          <div style="position:absolute;top:0;left:{chance_pct:.1f}%;width:2px;height:100%;background:#c00"></div>
        </div>
        <span style="font-size:13px">{c}/{n}={acc:.3f} &nbsp; {ratio:.1f}× &nbsp; p={p:.2e} {sig}</span>
      </div>
    </div>"""


def _summary_section(exp_dirs: List[pathlib.Path]) -> str:
    rows = []
    for exp_dir in sorted(exp_dirs):
        try:
            metrics = json.load(open(exp_dir / "metrics.json"))
            sp = json.load(open(exp_dir / "sample_plan.json"))
        except Exception:
            continue
        K = len(sp.get("cluster_ids", []))
        if K == 0: continue
        r = _get_clean(exp_dir, K)
        if r is None: continue
        acc, c, n, p = r
        chance = 1/K
        ratio = acc/chance
        sig = "★" if p < 0.001 else ("✓" if p < 0.05 else "NS")
        col = "#3cb44b" if p < 0.05 else ("#f58231" if p < 0.1 else "#aaaaaa")
        label = exp_dir.name
        model = _model_label(sp.get('backend','?'), label,
                             sp.get('model_id'), sp.get('load_in_4bit'), sp.get('load_in_8bit'))
        bar_w = f"{acc*100:.1f}%"
        chance_w = f"{chance*100:.1f}%"
        rows.append(f"""
        <tr>
          <td style="font-family:monospace;font-size:12px">{label}</td>
          <td>{K}</td>
          <td style="font-size:12px">{model}</td>
          <td>
            <div style="width:180px;background:#eee;border-radius:3px;position:relative;height:18px;display:inline-block">
              <div style="width:{bar_w};background:{col};height:100%;border-radius:3px"></div>
              <div style="position:absolute;top:0;left:{chance_w};width:2px;height:100%;background:#c00"></div>
            </div>
            <span style="font-size:12px;margin-left:6px">{c}/{n}={acc:.3f} · {ratio:.1f}× · {sig}</span>
          </td>
          <td style="font-size:12px">{p:.2e}</td>
        </tr>""")

    return f"""
    <h2>All Experiments — Clean-bucket Accuracy</h2>
    <p style="font-size:13px;color:#666">Red line = chance level. ★ p&lt;0.001 &nbsp; ✓ p&lt;0.05 &nbsp; NS not significant</p>
    <table style="border-collapse:collapse;width:100%">
      <thead><tr style="background:#f0f0f0">
        <th style="text-align:left;padding:6px">Experiment</th>
        <th>K</th><th>Backend</th><th>Clean accuracy</th><th>p-value</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _cluster_section(exp_dir: pathlib.Path, max_clusters: int, frame_seed: int) -> str:
    try:
        sp    = json.load(open(exp_dir / "sample_plan.json"))
        preds = [json.loads(l) for l in open(exp_dir / "predictions.jsonl")]
        metrics = json.load(open(exp_dir / "metrics.json"))
    except Exception as e:
        return f"<p>Could not load {exp_dir.name}: {e}</p>"

    clustering_dir = pathlib.Path(sp["clustering_dir"])
    eval_dir = sp["eval_dir"]
    if not clustering_dir.exists():
        return f"<p>Clustering dir missing: {clustering_dir}</p>"

    try:
        metadata = json.load(open(clustering_dir / "metadata.json"))
    except Exception as e:
        return f"<p>metadata.json missing: {e}</p>"

    pred_by_query = {p["query_idx"]: p for p in preds}
    cluster_ids = sp["cluster_ids"][:max_clusters]
    K = len(sp["cluster_ids"])
    max_frames = sp.get("max_frames_per_storyboard", 4)
    comp_size = min(sp.get("composite_target_size", 512), 400)

    # Overall stats
    r = _get_clean(exp_dir, K)
    acc_html = ""
    if r:
        acc, c, n, p = r
        acc_html = _acc_bar(acc, 1/K, K, c, n, p)

    per_cluster_acc = metrics.get("per_cluster_accuracy", {})

    clusters_html = []
    for cid in cluster_ids:
        cdata = sp["clusters"][str(cid)]
        cid_acc = per_cluster_acc.get(str(cid), per_cluster_acc.get(cid, None))
        col = _col(cid)

        # Example storyboards
        ex_imgs = []
        print(f"  cluster {cid}: generating {len(cdata['example_indices'])} examples...")
        for s_idx in cdata["example_indices"]:
            img = _make_storyboard(eval_dir, s_idx, metadata, max_frames, comp_size, frame_seed)
            if img:
                ex_imgs.append(f'<img src="data:image/jpeg;base64,{_img_b64(img)}" '
                               f'style="width:200px;border-radius:4px;margin:4px">')

        # Query storyboards (first 5, correct/incorrect)
        q_items = []
        print(f"  cluster {cid}: generating {min(5,len(cdata['query_indices']))} queries...")
        for q_idx in cdata["query_indices"][:5]:
            pred = pred_by_query.get(q_idx)
            if not pred: continue
            img = _make_storyboard(eval_dir, q_idx, metadata, max_frames, comp_size, frame_seed)
            if not img: continue
            ok = pred["is_correct"]
            border = "#3cb44b" if ok else "#e6194b"
            icon = "✓" if ok else f"✗→C{pred['majority_predicted_cluster_id']}"
            all_reps = pred.get("repetitions", [])
            responses = "<br>".join(
                f"<b>Rep {r['rep']}:</b> {r.get('raw_response','')}"
                for r in all_reps
            )
            q_items.append(f"""
            <div style="display:inline-block;vertical-align:top;margin:4px;
                        border:2px solid {border};border-radius:6px;padding:4px;width:240px">
              <img src="data:image/jpeg;base64,{_img_b64(img)}"
                   style="width:230px;border-radius:3px;display:block">
              <div style="font-size:11px;font-weight:bold;color:{border};margin-top:4px">{icon}</div>
              <div style="font-size:10px;color:#444;margin-top:4px;max-height:120px;
                          overflow-y:auto;line-height:1.4">{responses}</div>
            </div>""")

        acc_badge = (f'<span style="font-size:12px;margin-left:8px">'
                     f'acc={cid_acc:.2f}</span>' if cid_acc is not None else "")

        clusters_html.append(f"""
        <div style="margin:16px 0;border:1px solid #ddd;border-radius:8px;overflow:hidden">
          <div style="background:{col};color:white;padding:8px 14px;font-weight:bold">
            Cluster {cid}{acc_badge}
          </div>
          <div style="padding:12px">
            <div style="font-size:13px;font-weight:bold;margin-bottom:6px">
              Examples ({len(cdata['example_indices'])})
            </div>
            <div>{"".join(ex_imgs) or "<i>none</i>"}</div>
            <div style="font-size:13px;font-weight:bold;margin:10px 0 6px">
              Queries — first 5
            </div>
            <div>{"".join(q_items) or "<i>none</i>"}</div>
          </div>
        </div>""")

    model = _model_label(sp.get('backend','?'), exp_dir.name,
                         sp.get('model_id'), sp.get('load_in_4bit'), sp.get('load_in_8bit'))
    return f"""
    <h2>{exp_dir.name}</h2>
    <p style="font-size:13px;color:#555">
      K={K} · {model} · n_example={sp.get('n_example')}
      · n_query={sp.get('n_query')} · n_reps={sp.get('n_repetitions')} · seed={sp.get('random_seed')}
    </p>
    {acc_html}
    <p style="font-size:12px;color:#888">Showing first {max_clusters} clusters</p>
    {"".join(clusters_html)}"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e1_report.html")
    ap.add_argument("--max_clusters", type=int, default=999)
    ap.add_argument("--frame_seed", type=int, default=42)
    ap.add_argument("--detail_exp", nargs="*", default=None,
                    help="Experiment dir names to show cluster detail for "
                         "(default: head_to_head subdirs)")
    args = ap.parse_args()

    exp_root = _REPO_ROOT / "experiments"

    # All experiments for summary table
    all_exp_dirs = sorted([
        d.parent for d in exp_root.rglob("metrics.json")
        if (d.parent / "sample_plan.json").exists()
    ])

    # Detail experiments: head_to_head by default
    h2h_root = exp_root / "head_to_head"
    if args.detail_exp:
        detail_dirs = [exp_root / n for n in args.detail_exp]
    else:
        detail_dirs = sorted(h2h_root.iterdir()) if h2h_root.exists() else []
    detail_dirs = [d for d in detail_dirs
                   if d.is_dir() and (d / "metrics.json").exists()
                   and (d / "sample_plan.json").exists()]

    print(f"Found {len(all_exp_dirs)} experiments total")
    print(f"Generating cluster detail for {len(detail_dirs)} experiments")

    # Build HTML
    summary = _summary_section(all_exp_dirs)

    detail_sections = []
    for i, d in enumerate(detail_dirs):
        print(f"[{i+1}/{len(detail_dirs)}] {d.name}")
        detail_sections.append(_cluster_section(d, args.max_clusters, args.frame_seed))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E1 Cluster Coherence Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 24px; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>E1 Cluster Coherence Validation — Results Report</h1>
<p style="color:#666;font-size:14px">
  Transport MH · Qwen3-VL-8B · mar27 r512 rollouts ·
  Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>

{summary}

<h2 style="margin-top:48px">Cluster Detail — Head-to-Head (K=10, n_query=5, n_reps=5)</h2>
<p style="font-size:13px;color:#666">
  Green border = correct classification · Red border = misclassified · Response shown is rep 0.
</p>

{"<hr>".join(detail_sections)}

</body>
</html>"""

    out = pathlib.Path(args.out)
    out.write_text(html)
    print(f"\nDone → {out}  ({out.stat().st_size // 1024} KB)")
    print(f"Download with:  scp triton:{out} ~/Desktop/e1_report.html")


if __name__ == "__main__":
    main()
