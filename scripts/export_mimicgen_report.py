"""Export a self-contained HTML report of MimicGen generation sweep results.

Dimensions:
  - base_demos   : number of base policy training demos (60, 100, 300)
  - policy_seed  : policy training seed (0, 1, 2)
  - heuristic    : seed selection heuristic (random, behavior_graph, diversity)
  - gen_budget   : number of generation trials (20, 100, 500, 1000)

Metrics reported:
  - generation success rate  : % of trials that produced a valid demo
  - num_success              : total demos successfully generated
  - policy eval success rate : mean / best success rate after training on combined data

Usage:
    python scripts/export_mimicgen_report.py --out /tmp/mimicgen_report.html
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from typing import Dict, List, Optional, Tuple

PIPELINE_ROOT = pathlib.Path(
    "/mnt/ssdB/erik/cupid_data/pipeline_runs"
)

HEURISTIC_COLORS = {
    "random":         "#4363d8",
    "behavior_graph": "#3cb44b",
    "diversity":      "#f58231",
}
HEURISTIC_LABELS = {
    "random":         "Random",
    "behavior_graph": "Few Modes",
    "diversity":      "Diverse Modes",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_run_name(name: str) -> Optional[Dict]:
    """Extract (base_demos, policy_seed) from pipeline run directory name."""
    m = re.search(r"seed(\d+)_demos(\d+)", name)
    if m:
        return {"policy_seed": int(m.group(1)), "base_demos": int(m.group(2))}
    return None


def _load_sweep_result(result_path: pathlib.Path, heuristic: str, gen_budget: int) -> Optional[Dict]:
    try:
        r = json.load(open(result_path))
    except Exception:
        return None

    gen = r.get("generate_mimicgen_demos", {})
    stats = gen.get("stats", {})
    ev = r.get("eval_mimicgen_combined", {})

    gen_success_rate = stats.get("success_rate")
    num_success = stats.get("num_success")
    num_attempts = stats.get("num_attempts")

    # per-seed generation stats
    per_seed = stats.get("per_seed_stats", [])

    eval_mean = ev.get("mean_success_rate")
    eval_best = ev.get("best_success_rate")

    return {
        "heuristic":        heuristic,
        "gen_budget":       gen_budget,
        "gen_success_rate": gen_success_rate,
        "num_success":      num_success,
        "num_attempts":     num_attempts,
        "eval_mean":        eval_mean,
        "eval_best":        eval_best,
        "per_seed_stats":   per_seed,
    }


def collect_all_results() -> List[Dict]:
    rows = []
    for run_dir in sorted(PIPELINE_ROOT.iterdir()):
        if not run_dir.name.startswith("mimicgen_square_apr2"):
            continue
        meta = _parse_run_name(run_dir.name)
        if meta is None:
            continue
        sweep_dir = run_dir / "mimicgen_budget_sweep"
        if not sweep_dir.exists():
            continue
        for sub in sorted(sweep_dir.iterdir()):
            if not sub.is_dir():
                continue
            m = re.match(r"mimicgen_(\w+)_budget(\d+)$", sub.name)
            if not m:
                continue
            heuristic   = m.group(1)
            gen_budget  = int(m.group(2))
            result_path = sub / "result.json"
            if not result_path.exists():
                continue
            rec = _load_sweep_result(result_path, heuristic, gen_budget)
            if rec:
                rec.update(meta)
                rec["run_name"] = run_dir.name
                rows.append(rec)
    return rows


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    # normalise: some values are 0-100, others 0-1
    if v > 1.5:
        return f"{v:.1f}%"
    return f"{v*100:.1f}%"


def _bar(value: Optional[float], max_val: float = 1.0, color: str = "#4363d8",
         width: int = 100) -> str:
    if value is None:
        return "<span style='color:#aaa'>—</span>"
    v = value if value <= 1.5 else value / 100.0
    pct = min(v / max_val, 1.0) * width
    return (f'<div style="display:inline-flex;align-items:center;gap:6px">'
            f'<div style="width:{width}px;background:#eee;border-radius:3px;height:14px">'
            f'<div style="width:{pct:.1f}px;background:{color};height:100%;border-radius:3px"></div>'
            f'</div>'
            f'<span style="font-size:12px">{_pct(value)}</span>'
            f'</div>')


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


# ── Table builders ────────────────────────────────────────────────────────────

def _big_table(rows: List[Dict]) -> str:
    heuristic_order = ["random", "behavior_graph", "diversity"]
    header = """
    <tr style="background:#f0f0f0;position:sticky;top:0">
      <th>Run</th><th>Base demos</th><th>Policy seed</th>
      <th>Heuristic</th><th>Gen budget</th>
      <th>Gen success rate</th><th># successes</th>
      <th>Eval mean SR</th><th>Eval best SR</th>
    </tr>"""

    def _sort_key(r):
        h_order = {h: i for i, h in enumerate(heuristic_order)}
        return (r["base_demos"], r["policy_seed"],
                h_order.get(r["heuristic"], 99), r["gen_budget"])

    # Best per (base_demos, gen_budget) group for each metric
    from collections import defaultdict
    group_vals: Dict = defaultdict(lambda: {"gen": [], "eval_mean": [], "eval_best": []})
    for r in rows:
        key = (r["base_demos"], r["gen_budget"])
        if r["gen_success_rate"] is not None: group_vals[key]["gen"].append(r["gen_success_rate"])
        if r["eval_mean"]        is not None: group_vals[key]["eval_mean"].append(r["eval_mean"])
        if r["eval_best"]        is not None: group_vals[key]["eval_best"].append(r["eval_best"])
    group_best = {k: {"gen":       max(v["gen"])       if v["gen"]       else None,
                      "eval_mean": max(v["eval_mean"]) if v["eval_mean"] else None,
                      "eval_best": max(v["eval_best"]) if v["eval_best"] else None}
                  for k, v in group_vals.items()}

    trs = []
    for r in sorted(rows, key=_sort_key):
        h      = r["heuristic"]
        col    = HEURISTIC_COLORS.get(h, "#888")
        hlabel = HEURISTIC_LABELS.get(h, h)
        key    = (r["base_demos"], r["gen_budget"])
        best   = group_best[key]

        def _maybe_bold(val, best_val, cell_html):
            if val is not None and best_val is not None and abs(val - best_val) < 1e-9:
                return f"<b>{cell_html}</b>"
            return cell_html

        gen_cell      = _maybe_bold(r["gen_success_rate"], best["gen"],       _bar(r["gen_success_rate"], color=col))
        eval_mean_cell = _maybe_bold(r["eval_mean"],        best["eval_mean"], _bar(r["eval_mean"],        color=col))
        eval_best_cell = _maybe_bold(r["eval_best"],        best["eval_best"], _bar(r["eval_best"],        color="#333"))

        trs.append(f"""
        <tr>
          <td style="font-family:monospace;font-size:11px">{r['run_name']}</td>
          <td style="text-align:center"><b>{r['base_demos']}</b></td>
          <td style="text-align:center">{r['policy_seed']}</td>
          <td><span style="background:{col};color:white;padding:2px 7px;
                   border-radius:10px;font-size:12px">{hlabel}</span></td>
          <td style="text-align:center">{r['gen_budget']}</td>
          <td>{gen_cell}</td>
          <td style="text-align:center">{r['num_success'] or '—'}</td>
          <td>{eval_mean_cell}</td>
          <td>{eval_best_cell}</td>
        </tr>""")

    return f"""
    <h2>All Runs</h2>
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse;width:100%;min-width:900px">
      <thead>{header}</thead>
      <tbody>{"".join(trs)}</tbody>
    </table>
    </div>"""


def _subtable_by_budget(rows: List[Dict], base_demos: int) -> str:
    """One subtable for a fixed base_demos: heuristics × gen_budgets, averaged over seeds."""
    subset = [r for r in rows if r["base_demos"] == base_demos]
    if not subset:
        return ""

    heuristics  = sorted({r["heuristic"] for r in subset},
                          key=lambda h: ["random", "behavior_graph", "diversity"].index(h)
                          if h in ["random", "behavior_graph", "diversity"] else 99)
    gen_budgets = sorted({r["gen_budget"] for r in subset})

    # Pre-compute means per (heuristic, budget) so we can find column winners
    means: Dict[Tuple, Dict] = {}
    for h in heuristics:
        for b in gen_budgets:
            recs      = [r for r in subset if r["heuristic"] == h and r["gen_budget"] == b]
            gen_vals  = [r["gen_success_rate"] for r in recs if r["gen_success_rate"] is not None]
            eval_vals = [r["eval_mean"] for r in recs if r["eval_mean"] is not None]
            gm = statistics.mean(gen_vals)  if gen_vals  else None
            em = statistics.mean(eval_vals) if eval_vals else None
            em_n = (em if em is not None and em <= 1.0 else (em / 100.0 if em is not None else None))
            means[(h, b)] = {"gen": gm, "eval": em_n,
                             "gen_vals": gen_vals, "eval_vals": eval_vals}

    # Best per budget column
    best_gen  = {b: max((means[(h,b)]["gen"]  for h in heuristics if means[(h,b)]["gen"]  is not None), default=None) for b in gen_budgets}
    best_eval = {b: max((means[(h,b)]["eval"] for h in heuristics if means[(h,b)]["eval"] is not None), default=None) for b in gen_budgets}

    # Build header
    budget_headers = "".join(
        f'<th colspan="2" style="text-align:center;border-left:1px solid #ddd">Budget {b}</th>'
        for b in gen_budgets
    )
    sub_headers = "".join(
        '<th style="font-size:11px;border-left:1px solid #ddd">Gen SR</th>'
        '<th style="font-size:11px">Eval SR</th>'
        for _ in gen_budgets
    )

    trs = []
    for h in heuristics:
        col    = HEURISTIC_COLORS.get(h, "#888")
        hlabel = HEURISTIC_LABELS.get(h, h)
        cells  = [f'<td><span style="background:{col};color:white;padding:2px 8px;'
                  f'border-radius:10px;font-size:12px">{hlabel}</span></td>']
        for b in gen_budgets:
            m        = means[(h, b)]
            gm       = m["gen"]
            em_n     = m["eval"]
            gen_vals  = m["gen_vals"]
            eval_vals = m["eval_vals"]

            is_best_gen  = gm  is not None and best_gen[b]  is not None and abs(gm  - best_gen[b])  < 1e-9
            is_best_eval = em_n is not None and best_eval[b] is not None and abs(em_n - best_eval[b]) < 1e-9

            if gm is not None:
                gs = statistics.stdev(gen_vals) if len(gen_vals) > 1 else 0.0
                bold_o = "<b>" if is_best_gen else ""
                bold_c = "</b>" if is_best_gen else ""
                gen_cell = (f'{bold_o}<div>{_bar(gm, color=col, width=80)}</div>'
                            f'<div style="font-size:10px;color:#888">n={len(gen_vals)}'
                            f'{f" ±{gs:.1f}%" if len(gen_vals) > 1 else ""}</div>{bold_c}')
            else:
                gen_cell = "<span style='color:#aaa'>—</span>"

            if em_n is not None:
                es_n = (statistics.stdev(eval_vals) if len(eval_vals) > 1 else 0.0)
                es_n = es_n if es_n <= 1.0 else es_n / 100.0
                bold_o = "<b>" if is_best_eval else ""
                bold_c = "</b>" if is_best_eval else ""
                eval_cell = (f'{bold_o}<div>{_bar(em_n, color=col, width=80)}</div>'
                             f'<div style="font-size:10px;color:#888">n={len(eval_vals)}'
                             f'{f" ±{es_n*100:.1f}%" if len(eval_vals) > 1 else ""}</div>{bold_c}')
            else:
                eval_cell = "<span style='color:#aaa'>—</span>"

            cells.append(
                f'<td style="border-left:1px solid #eee;padding:6px 4px">{gen_cell}</td>'
                f'<td style="padding:6px 4px">{eval_cell}</td>'
            )
        trs.append(f'<tr>{"".join(cells)}</tr>')

    seeds_present = sorted({r["policy_seed"] for r in subset})
    return f"""
    <h3>Base demos = {base_demos}
      <span style="font-size:13px;font-weight:normal;color:#666;margin-left:8px">
        Policy seeds: {seeds_present} — values averaged across seeds (mean ± std)
      </span>
    </h3>
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse">
      <thead>
        <tr style="background:#f5f5f5">
          <th style="text-align:left">Heuristic</th>
          {budget_headers}
        </tr>
        <tr style="background:#fafafa;font-size:11px">
          <th></th>{sub_headers}
        </tr>
      </thead>
      <tbody>{"".join(trs)}</tbody>
    </table>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mimicgen_report.html")
    args = ap.parse_args()

    print("Collecting results...")
    rows = collect_all_results()
    print(f"Found {len(rows)} experiment entries")

    base_demos_values = sorted({r["base_demos"] for r in rows})

    big = _big_table(rows)
    subtables = "".join(_subtable_by_budget(rows, d) for d in base_demos_values)

    n_runs    = len({r["run_name"] for r in rows})
    n_entries = len(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MimicGen Generation Sweep Report</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          max-width:1400px;margin:0 auto;padding:24px;color:#222 }}
  h1   {{ border-bottom:2px solid #333;padding-bottom:8px }}
  h2   {{ margin-top:40px;border-bottom:1px solid #ccc;padding-bottom:6px }}
  h3   {{ margin-top:28px;color:#333 }}
  table{{ width:100%;border-collapse:collapse }}
  th,td{{ padding:6px 10px;text-align:left;border-bottom:1px solid #eee;vertical-align:middle }}
  th   {{ background:#f5f5f5;font-weight:600 }}
  tr:hover {{ background:#fafafa }}
</style>
</head>
<body>
<h1>MimicGen Generation Sweep — Results Report</h1>
<p style="color:#666;font-size:14px">
  Task: Square MH ·
  {n_runs} pipeline runs · {n_entries} (heuristic × budget) entries ·
  Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>
<p style="font-size:13px">
  <b>Gen SR</b> = MimicGen generation success rate (% of trials producing a valid demo) ·
  <b>Eval SR</b> = policy success rate after training on original + generated data (mean over checkpoints)
</p>

<h2>Heuristic comparison by base demo budget</h2>
<p style="font-size:13px;color:#555">
  Each cell shows mean ± std over policy seeds for that (heuristic, generation budget) combination.
</p>
{subtables}

{big}
</body>
</html>"""

    out = pathlib.Path(args.out)
    out.write_text(html)
    print(f"\nDone → {out}  ({out.stat().st_size // 1024} KB)")
    print(f"Download:  scp triton:{out} ~/Desktop/mimicgen_report.html")


if __name__ == "__main__":
    main()
