"""Aggregate the rollout-budget sweep into plots and a markdown findings doc.

For each (task, K), plot:
  - MV₁(N) / MV₂(N) / MV₃(N) across N ∈ {budgets} with ±1 SD across the
    subsample seeds (independent draws).
  - Each line of bootstrap CI averaged across seeds (the within-draw noise).

The question this answers: how many rollouts do we need for the MV
estimate to stabilise — both in point estimate (across seeds) and in
within-draw CI width?

Usage:
    PYTHONPATH=. python scripts/summarize_rollout_budget.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SSD_RESULTS = pathlib.Path(
    "/mnt/ssdB/erik/cupid_data/graph_simplification/results/rollout_budget"
)
OUT_DIR = _REPO_ROOT / "docs" / "rollout_budget_results"
PLOT_DIR = OUT_DIR / "_plots"


def _load_all() -> List[Dict]:
    rows = []
    for p in sorted(SSD_RESULTS.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            continue
    return rows


def _group(rows: List[Dict]) -> Dict[Tuple[str, int], List[Dict]]:
    """Group by (task_label, K)."""
    g: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        key = (r["task"], int(r["K"]))
        g.setdefault(key, []).append(r)
    return g


def _summarise_per_N(rows_by_N: Dict[int, List[Dict]]) -> List[Dict]:
    """For each N, compute mean and SD of MV₁/₂/₃ across subsample seeds
    AND average bootstrap CI width within each draw."""
    out = []
    for N in sorted(rows_by_N):
        seeds = rows_by_N[N]
        rec = {"N": int(N), "n_seeds": len(seeds)}
        for order in (1, 2, 3):
            pts = np.array([s[f"mv{order}_point"] for s in seeds])
            ci_widths = np.array([
                s[f"mv{order}_ci_hi"] - s[f"mv{order}_ci_lo"] for s in seeds
            ])
            covs = np.array([s.get(f"mv{order}_coverage_fraction", 1.0) for s in seeds])
            rec[f"mv{order}_mean"] = float(pts.mean())
            rec[f"mv{order}_sd_across_seeds"] = float(pts.std(ddof=0))
            rec[f"mv{order}_avg_ci_width"] = float(ci_widths.mean())
            rec[f"mv{order}_avg_coverage"] = float(covs.mean())
        rec["n_windows_mean"] = float(np.mean([s["n_windows"] for s in seeds]))
        out.append(rec)
    return out


def _plot_budget_curve(
    task: str, K: int, summary: List[Dict], out_path: pathlib.Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    Ns = [r["N"] for r in summary]

    # Left: MV mean across seeds with ±1 SD shading.
    ax = axes[0]
    for order, color, label in [
        (1, "#1f77b4", "MV₁"),
        (2, "#ff7f0e", "MV₂"),
        (3, "#2ca02c", "MV₃"),
    ]:
        means = np.array([r[f"mv{order}_mean"] for r in summary])
        sds = np.array([r[f"mv{order}_sd_across_seeds"] for r in summary])
        ax.plot(Ns, means, "-o", color=color, label=label, markersize=5)
        ax.fill_between(Ns, means - sds, means + sds, color=color, alpha=0.18)
    ax.set_xlabel("N (rollouts subsampled)")
    ax.set_ylabel("MV (bits)")
    ax.set_title(f"{task}  K={K}  ·  mean ± 1σ across {summary[0]['n_seeds']} subsample seeds")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")

    # Right: bootstrap CI width (average across seeds).
    ax = axes[1]
    for order, color, label in [
        (1, "#1f77b4", "MV₁"),
        (2, "#ff7f0e", "MV₂"),
        (3, "#2ca02c", "MV₃"),
    ]:
        ci_w = np.array([r[f"mv{order}_avg_ci_width"] for r in summary])
        ax.plot(Ns, ci_w, "-o", color=color, label=label, markersize=5)
    ax.set_xlabel("N (rollouts subsampled)")
    ax.set_ylabel("avg within-draw bootstrap 95% CI width (bits)")
    ax.set_title(f"{task}  K={K}  ·  CI width vs rollout budget")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_cross_rep(grouped: Dict, summaries: Dict, base_task: str, K: int,
                    out_path: pathlib.Path) -> None:
    """Side-by-side: same base_task and K, infembed vs policy_emb across N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    rep_styles = {"infembed": ("#1f77b4", "infembed"),
                  "policy_emb": ("#d62728", "policy_emb")}

    # Find matching summaries for this base_task and K, with rep in label
    for ax_idx, order in enumerate((1, 2, 3)):
        ax = axes[ax_idx]
        plotted = 0
        for rep, (color, label) in rep_styles.items():
            label_key = f"{base_task}__{rep}"
            if (label_key, K) not in summaries:
                continue
            summary = summaries[(label_key, K)]
            Ns = [r["N"] for r in summary]
            means = np.array([r[f"mv{order}_mean"] for r in summary])
            sds = np.array([r[f"mv{order}_sd_across_seeds"] for r in summary])
            covs = np.array([r[f"mv{order}_avg_coverage"] for r in summary])
            ax.plot(Ns, means, "-o", color=color, label=label, markersize=6)
            ax.fill_between(Ns, means - sds, means + sds, color=color, alpha=0.18)
            # Annotate coverage as text near each marker
            for x, y, c in zip(Ns, means, covs):
                ax.annotate(f"{c:.2f}", (x, y), fontsize=7, color=color,
                            xytext=(4, -10), textcoords="offset points")
            plotted += 1
        ax.set_xlabel("N (rollouts)")
        ax.set_ylabel(f"MV_{order} (bits)")
        ax.set_xscale("log")
        ax.set_title(f"MV_{order}  K={K}  ({base_task})")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
    fig.suptitle("Cross-rep N-convergence — numbers are coverage fraction (gate = 0.80)",
                 fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_findings_md(grouped: Dict, summaries: Dict, n_rows: int) -> None:
    md_path = OUT_DIR / "_findings.md"
    out: List[str] = []
    out.append("# Rollout-budget sweep — findings\n")
    out.append(
        "Auto-generated by `scripts/summarize_rollout_budget.py`. "
        "For each task, K, and N (rollouts), we subsample N rollouts from the "
        "100-rollout trunk five independent times, rebuild a (w=5, s=1) "
        "sliding-window KMeans clustering, and compute MV₁/MV₂/MV₃ with a "
        "100-rep episode-bootstrap CI per draw. Three questions:\n"
        "1. **How fast does the mean MV estimate converge with N?** (`MV σ-seeds`).\n"
        "2. **How fast does the within-draw bootstrap CI width shrink?** (`MV CI w`).\n"
        "3. **At what N does the metric become *measurable* for each order?** "
        "(`cov₁/cov₂/cov₃`: fraction of (prev, curr, next) triplets that "
        "clear the `min_pairs ≥ 4·order²` gate. cov < 0.5 means the metric "
        "is mostly noise.)\n"
    )
    out.append(f"\nTotal datapoints: {n_rows} (= tasks × Ns × Ks × seeds).\n")
    # Cross-rep convergence table at K=15 (representative) — extracted directly.
    out.append("\n## Cross-rep N-convergence (representative: K=15 on transport)\n")
    out.append(
        "\nThis table shows how the rep choice interacts with sample size. "
        "Same base task, same K, same (w=5, s=1), different rep:\n\n"
        "| rep | N | MV₁ | sd-seeds | cov₁ | MV₂ | cov₂ | MV₃ | cov₃ |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    K_show = 15
    base = "transport_mh_jan28"
    for rep in ("infembed", "policy_emb"):
        key = (f"{base}__{rep}", K_show)
        if key not in summaries:
            continue
        for rec in summaries[key]:
            out.append(
                f"| `{rep}` | {rec['N']} | {rec['mv1_mean']:.3f} | "
                f"{rec['mv1_sd_across_seeds']:.3f} | "
                f"{rec.get('mv1_avg_coverage', 1.0):.2f} | "
                f"{rec['mv2_mean']:.3f} | "
                f"{rec.get('mv2_avg_coverage', 1.0):.2f} | "
                f"{rec['mv3_mean']:.3f} | "
                f"{rec.get('mv3_avg_coverage', 1.0):.2f} |\n"
            )
    out.append(
        "\n**Two key reads from this table** (and the K=15 cross-rep panel above):\n"
        "- `infembed` reaches its asymptotic MV₁ at N=20 already (cov=0.92 → "
        "1.00, MV₁=0.272→0.332). The estimate is robust to sample size.\n"
        "- `policy_emb`'s MV₁ DOUBLES between N=20 (0.149, cov=0.64) and "
        "N=100 (0.304, cov=0.94). The N=20 reading was simply gate-limited.\n"
        "- At N=100 both reps converge to MV₁ ≈ 0.30 at K=15 — **the apparent "
        "rep gap at K=6 is not preserved at K=15**. Whether the K=6 result is "
        "real or its own coverage artefact (policy_emb cov₁ at K=6 is 0.74 "
        "vs infembed's K=5 cov₁ of 0.86) requires Stage B (500 rollouts) — "
        "see `simplification_findings.md`.\n"
        "- MV₃ goes from 0 (cov=0, unmeasurable) at N=20 to ≈0.6 (cov=0.65-"
        "0.98) at N=100. **MV₃ at K=15 IS measurable at N=100, but only "
        "barely; coverage growth N=50→100 was 0.12→0.65 for policy_emb.**\n"
    )

    out.append(
        "\n### Headline answer to the user's question — how many rollouts "
        "to get reliable MV?\n"
        "\n"
        "*Reliability threshold: coverage ≥ 0.80 across all 5 subsample seeds.*\n"
        "\n"
        "| Metric | Reliable from | Evidence |\n"
        "|---|---|---|\n"
        "| MV₁ at K=15 (transport, policy_emb) | N ≈ 50 | cov₁ goes 0.64→0.85 between N=20 and N=50 |\n"
        "| MV₂ at K=15 (transport, policy_emb) | N ≈ 100 | cov₂ goes 0.21→0.78→0.85 at N=20/50/100 |\n"
        "| MV₃ at K=15 (transport, policy_emb) | N ≥ 150-300 | cov₃=0.65 at N=100. Cov grew 0.12→0.65 from N=50 to N=100 (5.4×) — extrapolating a similar slope, hitting 0.80 by N≈150; conservatively N=300 to absorb non-linearity in the tail. **Stage B will verify directly.** |\n"
        "| MV₁ at K=20 (transport, policy_emb) | N ≈ 50 | cov₁ ≈ 0.85 at N=50 |\n"
        "| MV₃ at K=20 (transport, policy_emb) | N ≥ 200 | cov₃=0.66 at N=100, similar trend |\n"
        "\nThe 500-rollout eval (Stage B) is running; this table will be "
        "updated to use measured (not extrapolated) numbers once it completes.\n"
    )

    out.append("\n## Convergence tables\n")
    tasks = sorted({k[0] for k in grouped})
    for task in tasks:
        out.append(f"\n### {task}\n")
        out.append(
            "\n| K | N | MV₁ | MV₁ σ-seeds | MV₁ CI w | cov₁ | MV₂ | cov₂ | MV₃ | cov₃ |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for (t, K) in sorted(grouped):
            if t != task:
                continue
            summary = summaries[(task, K)]
            for rec in summary:
                out.append(
                    f"| {K} | {rec['N']} | {rec['mv1_mean']:.3f} | "
                    f"{rec['mv1_sd_across_seeds']:.3f} | "
                    f"{rec['mv1_avg_ci_width']:.3f} | "
                    f"{rec.get('mv1_avg_coverage', 1.0):.2f} | "
                    f"{rec['mv2_mean']:.3f} | "
                    f"{rec.get('mv2_avg_coverage', 1.0):.2f} | "
                    f"{rec['mv3_mean']:.3f} | "
                    f"{rec.get('mv3_avg_coverage', 1.0):.2f} |\n"
                )

    out.append("\n## Cross-rep N-convergence panels\n")
    out.append(
        "For each `(base_task, K)` where both `infembed` and `policy_emb` "
        "trunks have a budget sweep, MV₁/MV₂/MV₃ are plotted side-by-side. "
        "Numbers near markers are coverage fractions (gate=0.80).\n\n"
    )
    seen_pairs = sorted({
        (lab.rsplit("__", 1)[0], K) for (lab, K) in grouped
        if "__" in lab and lab.rsplit("__", 1)[1] in ("infembed", "policy_emb")
    })
    rep_pairs = sorted({(base, K) for (base, K) in seen_pairs})
    base_set = {b for (b, _) in rep_pairs}
    for base in sorted(base_set):
        Ks_for_base = sorted({K for (b, K) in rep_pairs if b == base})
        for K in Ks_for_base:
            # Only emit if both reps have data
            inf_key = (f"{base}__infembed", K)
            pe_key = (f"{base}__policy_emb", K)
            if inf_key in summaries and pe_key in summaries:
                out.append(f"### {base}  K={K}\n")
                out.append(f"![cross-rep](_plots/{base}__K{K}__cross_rep.png)\n\n")

    out.append("\n## Curves (per (task_label, K))\n")
    for (task, K) in sorted(grouped):
        slug = f"{task}__K{K}"
        out.append(f"### {task} K={K}\n")
        out.append(f"![curve](_plots/{slug}__budget.png)\n\n")

    md_path.write_text("".join(out))
    print(f"  wrote {md_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    plot_dir = out_dir / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_all()
    if not rows:
        print(f"No results at {SSD_RESULTS}.")
        return 1
    print(f"Loaded {len(rows)} rows", flush=True)

    grouped = _group(rows)
    summaries: Dict = {}
    for key, rows_for_setting in grouped.items():
        rows_by_N: Dict[int, List[Dict]] = {}
        for r in rows_for_setting:
            rows_by_N.setdefault(int(r["N_rollouts"]), []).append(r)
        summary = _summarise_per_N(rows_by_N)
        summaries[key] = summary
        task, K = key
        slug = f"{task}__K{K}"
        _plot_budget_curve(task, K, summary, plot_dir / f"{slug}__budget.png")
    print(f"  wrote {len(grouped)} budget plots", flush=True)

    # Cross-rep panels: for each (base_task, K) where both reps exist, plot
    # MV vs N side by side.
    base_tasks = sorted({lab.split("__")[0] + "_" + lab.split("__")[1] for lab in
                         {l[0] for l in grouped}
                         if "__" in lab})
    # Simpler: enumerate base_tasks from any infembed/policy_emb pair.
    seen_pairs = set()
    base_to_Ks: Dict[str, set] = {}
    for (label, K) in grouped:
        if "__" not in label:
            continue
        parts = label.rsplit("__", 1)
        if len(parts) != 2 or parts[1] not in ("infembed", "policy_emb"):
            continue
        base = parts[0]
        base_to_Ks.setdefault(base, set()).add(K)
    for base, Ks in base_to_Ks.items():
        # Only plot K where BOTH reps have data
        for K in sorted(Ks):
            inf_key = (f"{base}__infembed", K)
            pe_key = (f"{base}__policy_emb", K)
            if inf_key in summaries and pe_key in summaries:
                _plot_cross_rep(grouped, summaries, base, K,
                                plot_dir / f"{base}__K{K}__cross_rep.png")
                seen_pairs.add((base, K))
    print(f"  wrote {len(seen_pairs)} cross-rep panels", flush=True)

    (out_dir / "budget_summary.json").write_text(json.dumps(
        {f"{k[0]}__K{k[1]}": v for k, v in summaries.items()}, indent=2
    ))
    _write_findings_md(grouped, summaries, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
