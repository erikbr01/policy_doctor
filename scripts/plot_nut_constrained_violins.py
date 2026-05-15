"""Generate one violin plot per (source_demos, generation_budget, constraint_type) combination
for the nut-constrained MimicGen experiments.

Labels:
  Behavior Graph → Few Modes
  Diversity      → Diverse Modes
  Random         → Random

Usage:
    conda activate policy_doctor
    python scripts/plot_nut_constrained_violins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.paths import REPO_ROOT
from policy_doctor.analysis.policy_comparison import PolicyGroupSpec, run_comparison
from policy_doctor.plotting.policy_comparison import create_policy_comparison_violin

EVAL_BASE = REPO_ROOT / "data" / "outputs" / "eval_save_episodes"
OUT_DIR = REPO_ROOT / "data" / "outputs" / "comparisons" / "nut_constrained_violins"
STEM = "train_diffusion_unet_lowdim_square_mh_mimicgen"

# Renamed labels
HEURISTICS = {
    "behavior_graph": "Few Modes",
    "diversity":      "Diverse Modes",
    "random":         "Random",
}

# (filename_stem, plot_title, train_date_prefix, budget)
COMBINATIONS = [
    (
        "demos60_budget100_loose",
        "60 source demos · budget 100 · loose constraint",
        "apr26_sweep_demos60_nut_constrained",
        "budget100",
    ),
    (
        "demos60_budget100_tight",
        "60 source demos · budget 100 · tight constraint",
        "apr26_sweep_demos60_nut_constrained_tight",
        "budget100",
    ),
    (
        "demos60_budget300_loose",
        "60 source demos · budget 300 · loose constraint",
        "apr26_sweep_demos60_budget300_nut_constrained",
        "budget300",
    ),
    (
        "demos60_budget300_tight",
        "60 source demos · budget 300 · tight constraint",
        "apr26_sweep_demos60_budget300_nut_constrained_tight",
        "budget300",
    ),
    (
        "demos300_budget300_loose",
        "300 source demos · budget 300 · loose constraint",
        "apr26_sweep_demos300_nut_constrained",
        "budget300",
    ),
    (
        "demos300_budget1000_loose",
        "300 source demos · budget 1000 · loose constraint",
        "apr26_sweep_demos300_nut_constrained",
        "budget1000",
    ),
]


def _run_names(train_date: str, heuristic: str, budget: str) -> list[Path]:
    paths = []
    for seed in [0, 1, 2]:
        suffixes = ["", "-rep1", "-rep2"] if seed == 1 else [""]
        for sfx in suffixes:
            name = f"{train_date}_{STEM}_{seed}-mimicgen_combined-{heuristic}-{budget}{sfx}"
            paths.append(EVAL_BASE / name)
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for fname, title, train_date, budget in COMBINATIONS:
        print(f"\n=== {title} ===")

        group_specs = [
            PolicyGroupSpec(label=display_label, dirs=_run_names(train_date, heuristic, budget))
            for heuristic, display_label in HEURISTICS.items()
        ]

        result = run_comparison(group_specs, global_confidence_level=0.95, top_k_checkpoints=5, verbose=True)

        fig = create_policy_comparison_violin(
            result.group_labels,
            result.success_arrays,
            result.cld_letters,
            rng=np.random.default_rng(42),
            show_empirical_means=True,
            title=title,
        )
        out_path = OUT_DIR / f"{fname}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {out_path}")
        for lbl, rate, n, cld in zip(
            result.group_labels, result.success_rates, result.n_episodes, result.cld_letters
        ):
            print(f"    {lbl:<16} CLD={cld}  {rate:.1%}  n={n}")

    print(f"\nAll plots written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
