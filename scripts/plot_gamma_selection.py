"""Plot γ-selection: MV-vs-K curves with horizontal γ·MV_asymp lines.

For each task, plots the MV₁-vs-K curve for the auto-picked (rep, w, s)
setting and overlays horizontal lines at γ·MV_asymp for several γ values.
Marks the picked K at each γ as a vertical drop to the curve.

Output: docs/k_sweep_results/_plots/<task>__gamma_selection.png

Usage:
    PYTHONPATH=. python scripts/plot_gamma_selection.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.behaviors.select_K import (
    Cell,
    rank_candidates,
    select_hyperparams,
)


RESULTS_DIR = pathlib.Path(
    "/mnt/ssdB/erik/cupid_data/graph_simplification/results/k_sweep"
)
PLOTS_DIR = _REPO_ROOT / "docs" / "k_sweep_results" / "_plots"

GAMMAS = [0.3, 0.5, 0.7, 0.9]
GAMMA_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]


def _load_cells(task: str) -> List[Cell]:
    cells = []
    for p in sorted(RESULTS_DIR.glob(f"{task}__*__w*_s*__K*.json")):
        row = json.loads(p.read_text())
        cells.append(Cell(
            rep=row["rep"], w=int(row["w"]), s=int(row["s"]), K=int(row["K"]),
            mv1=float(row["mv1_point"]),
            cov1=float(row.get("mv1_coverage_fraction", 1.0)),
        ))
    return cells


def _curve_at(cells: List[Cell], rep: str, w: int, s: int,
              cov_min: float = 0.80) -> List[Cell]:
    sub = [c for c in cells if c.rep == rep and c.w == w and c.s == s]
    return sorted(sub, key=lambda c: c.K)


def _knee_K_at_gamma(curve: List[Cell], gamma: float,
                     cov_min: float = 0.80) -> Optional[Tuple[int, float]]:
    gated = [c for c in curve if c.cov1 >= cov_min and c.K >= 5]
    if not gated:
        return None
    asymp = max(c.mv1 for c in gated)
    target = gamma * asymp
    for c in gated:
        if c.mv1 >= target:
            return c.K, c.mv1
    return gated[-1].K, gated[-1].mv1


def plot_task(task: str) -> Optional[pathlib.Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = _load_cells(task)
    if not cells:
        return None

    pick = select_hyperparams(cells)
    if pick is None:
        return None
    rep, w, s = pick.rep, pick.w, pick.s
    curve = _curve_at(cells, rep, w, s)
    gated = [c for c in curve if c.cov1 >= 0.80 and c.K >= 5]
    if not gated:
        return None
    asymp = max(c.mv1 for c in gated)

    # Two-panel: left = MV vs K with γ lines; right = K* as a function of γ.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # ---- Left panel: MV-vs-K with γ lines ----
    ax = axes[0]
    K_arr = np.array([c.K for c in curve])
    MV_arr = np.array([c.mv1 for c in curve])
    COV_arr = np.array([c.cov1 for c in curve])

    # Plot ungated cells as hollow markers
    ungated_mask = (COV_arr < 0.80) | (K_arr < 5)
    if ungated_mask.any():
        ax.plot(K_arr[ungated_mask], MV_arr[ungated_mask], "o",
                color="#9ca3af", markersize=7, markerfacecolor="white",
                markeredgewidth=1.4, label="ungated (cov<0.80 or K<5)")

    # Plot gated cells as filled
    gated_mask = ~ungated_mask
    ax.plot(K_arr[gated_mask], MV_arr[gated_mask], "-o",
            color="#1f2937", markersize=8, linewidth=1.8, label="MV₁ (gated)")

    # Horizontal γ·asymp lines + vertical drop to knee
    for gamma, color in zip(GAMMAS, GAMMA_COLORS):
        target = gamma * asymp
        ax.axhline(target, color=color, linestyle="--", linewidth=1.2,
                   alpha=0.9, label=f"γ={gamma}: target={target:.3f}")
        knee = _knee_K_at_gamma(curve, gamma)
        if knee is None:
            continue
        K_star, MV_star = knee
        ax.plot([K_star, K_star], [0, MV_star], color=color, linewidth=1.2,
                alpha=0.7)
        ax.plot(K_star, MV_star, "o", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=1.2)
        ax.annotate(f"K*={K_star}", (K_star, MV_star),
                    xytext=(6, -16), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold")

    # MV_asymp horizontal line
    ax.axhline(asymp, color="#7c3aed", linestyle=":", linewidth=1.2,
               alpha=0.5, label=f"MV_asymp = {asymp:.3f}")

    ax.set_xlabel("K (number of clusters)", fontsize=11)
    ax.set_ylabel("MV₁ (bits)", fontsize=11)
    ax.set_title(f"{task}\n(rep={rep}, w={w}, s={s})", fontsize=11)
    ax.set_xticks(K_arr)
    ax.set_xticklabels([str(k) for k in K_arr], fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_ylim(bottom=0)

    # ---- Right panel: K* as a function of γ ----
    ax = axes[1]
    g_fine = np.linspace(0.05, 1.0, 96)
    K_picks = []
    MV_picks = []
    for g in g_fine:
        knee = _knee_K_at_gamma(curve, float(g))
        if knee is None:
            K_picks.append(np.nan); MV_picks.append(np.nan)
        else:
            K_picks.append(knee[0]); MV_picks.append(knee[1])
    K_picks = np.array(K_picks)
    MV_picks = np.array(MV_picks)

    ax.step(g_fine, K_picks, where="post", color="#1f2937", linewidth=1.8,
            label="K*(γ)")
    # Mark the integer γ choices we used in the left panel
    for gamma, color in zip(GAMMAS, GAMMA_COLORS):
        knee = _knee_K_at_gamma(curve, gamma)
        if knee is None:
            continue
        K_star, _ = knee
        ax.plot(gamma, K_star, "o", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=1.2,
                label=f"γ={gamma} → K*={K_star}")

    ax.set_xlabel("γ (fraction of MV_asymp)", fontsize=11)
    ax.set_ylabel("knee K* (selected K)", fontsize=11)
    ax.set_title(f"K* as a function of γ\n(auto-picked: γ=0.5 → K={pick.K})",
                 fontsize=11)
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = PLOTS_DIR / f"{task}__gamma_selection.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    for task in ["lift_mh_jan26", "square_mh_feb5", "transport_mh_jan28"]:
        out = plot_task(task)
        if out is None:
            print(f"  SKIP {task}: no admissible cells")
        else:
            print(f"  wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
