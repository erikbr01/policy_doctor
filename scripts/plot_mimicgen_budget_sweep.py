"""Plot success rate vs. episode budget for the MimicGen budget sweep.

One subplot per initial demo count (e.g. d60, d100, d300).
Each heuristic is a line; x-axis = episode budget; y-axis = mean success rate.

When multiple seeds are available the plot shows mean ± std (shaded band) across seeds.
With a single seed only the mean line is drawn.

Usage (policy_doctor env):
    python scripts/plot_mimicgen_budget_sweep.py
    python scripts/plot_mimicgen_budget_sweep.py --seeds 0 1 2 --demos 60 100 300
    python scripts/plot_mimicgen_budget_sweep.py --out /tmp/budget_sweep.html
    python scripts/plot_mimicgen_budget_sweep.py --out /tmp/budget_sweep.png  # requires kaleido

Arguments:
    --pipe_base   Path to cupid/data/pipeline_runs   (default: auto-detected from repo layout)
    --seeds       Seed indices to include            (default: 0 1 2)
    --demos       Demo counts to include             (default: 60 100 300)
    --budgets     Episode budgets to include         (default: 20 100 500 1000)
    --heuristics  Heuristics to include              (default: random behavior_graph diversity)
    --out         Output path (.html or image)       (default: /tmp/mimicgen_budget_sweep.html)
    --ci          Show 95% CI band instead of ±1 std (flag)
    --no_band     Never draw error bands             (flag)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_SEEDS = [0, 1, 2]
_DEFAULT_DEMOS = [60, 100, 300]
_DEFAULT_BUDGETS = [20, 100, 500, 1000]
_DEFAULT_HEURISTICS = ["random", "behavior_graph", "diversity"]

_HEURISTIC_LABELS = {
    "random": "Random",
    "behavior_graph": "Behavior Graph",
    "diversity": "Diversity",
}
_HEURISTIC_COLORS = {
    "random": "#9E9E9E",
    "behavior_graph": "#2196F3",
    "diversity": "#FF9800",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _result_path(pipe_base: Path, seed: int, demos: int, heuristic: str, budget: int) -> Path:
    run_dir = pipe_base / f"mimicgen_square_apr26_sweep_seed{seed}_demos{demos}"
    arm_dir = run_dir / "mimicgen_budget_sweep" / f"mimicgen_{heuristic}_budget{budget}"
    return arm_dir / "eval_mimicgen_combined" / "result.json"


def load_data(
    pipe_base: Path,
    seeds: list[int],
    demos: list[int],
    heuristics: list[str],
    budgets: list[int],
) -> dict[tuple[int, str, int], list[float]]:
    """Return {(demos, heuristic, budget): [mean_success_rate per available seed]}."""
    data: dict[tuple[int, str, int], list[float]] = {}

    for d in demos:
        for h in heuristics:
            for b in budgets:
                key = (d, h, b)
                rates: list[float] = []
                for s in seeds:
                    p = _result_path(pipe_base, s, d, h, b)
                    if p.exists():
                        try:
                            with open(p) as f:
                                result: dict[str, Any] = json.load(f)
                            rates.append(float(result["mean_success_rate"]))
                        except (KeyError, json.JSONDecodeError):
                            pass
                if rates:
                    data[key] = rates
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_figure(
    data: dict[tuple[int, str, int], list[float]],
    demos_list: list[int],
    heuristics: list[str],
    budgets: list[int],
    use_ci: bool = False,
    no_band: bool = False,
) -> "go.Figure":
    import math
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Only include demo counts that have at least one data point
    active_demos = [d for d in demos_list if any((d, h, b) in data for h in heuristics for b in budgets)]
    if not active_demos:
        raise ValueError("No data found for any (demos, heuristic, budget) combination.")

    n_cols = min(len(active_demos), 3)
    n_rows = math.ceil(len(active_demos) / n_cols)

    subplot_titles = [f"{d} base demos" for d in active_demos]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    # z-value for 95% CI (normal approximation)
    z95 = 1.96

    for panel_idx, d in enumerate(active_demos):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        show_legend = panel_idx == 0

        for h in heuristics:
            xs, ys, y_lo, y_hi = [], [], [], []
            for b in budgets:
                key = (d, h, b)
                if key not in data:
                    continue
                rates = data[key]
                mean = float(np.mean(rates))
                xs.append(b)
                ys.append(mean)

                if len(rates) > 1 and not no_band:
                    std = float(np.std(rates, ddof=1))
                    n = len(rates)
                    if use_ci:
                        half = z95 * std / math.sqrt(n)
                    else:
                        half = std
                    y_lo.append(max(0.0, mean - half))
                    y_hi.append(min(1.0, mean + half))
                else:
                    y_lo.append(mean)
                    y_hi.append(mean)

            if not xs:
                continue

            color = _HEURISTIC_COLORS.get(h, "#607D8B")
            label = _HEURISTIC_LABELS.get(h, h)

            # Shaded band (only when multiple seeds)
            has_band = not no_band and any(lo != hi for lo, hi in zip(y_lo, y_hi))
            if has_band:
                # Convert #RRGGBB → rgba(..., 0.2) for older plotly compatibility
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                band_color = f"rgba({r},{g},{b},0.2)"
                fig.add_trace(
                    go.Scatter(
                        x=xs + xs[::-1],
                        y=y_hi + y_lo[::-1],
                        fill="toself",
                        fillcolor=band_color,
                        line=dict(width=0),
                        mode="lines",
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=h,
                    ),
                    row=row,
                    col=col,
                )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    showlegend=show_legend,
                    legendgroup=h,
                ),
                row=row,
                col=col,
            )

    # Axis labels
    for panel_idx in range(len(active_demos)):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        axis_suffix = "" if panel_idx == 0 else str(panel_idx + 1)
        fig.update_layout(**{
            f"xaxis{axis_suffix}": dict(
                title="Episode budget",
                type="log",
                tickmode="array",
                tickvals=budgets,
                ticktext=[str(b) for b in budgets],
            ),
            f"yaxis{axis_suffix}": dict(
                title="Mean success rate" if col == 1 else "",
                range=[0, 1],
                tickformat=".0%",
            ),
        })

    band_note = ""
    if any(len(v) > 1 for v in data.values()):
        n_seeds = max(len(v) for v in data.values())
        band_note = f" | Band = {'95% CI' if use_ci else '±1 std'} across {n_seeds} seeds"

    fig.update_layout(
        title=dict(
            text=f"MimicGen budget sweep — success rate vs. episode budget{band_note}",
            x=0.5,
        ),
        legend=dict(
            title="Heuristic",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=400 * n_rows,
        width=420 * n_cols + 100,
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_pipe_base() -> Path:
    """Auto-detect pipeline_runs dir relative to this script's repo root."""
    candidate = _REPO / "third_party" / "cupid" / "data" / "pipeline_runs"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find pipeline_runs at {candidate}. Pass --pipe_base explicitly."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipe_base", default=None, help="Path to cupid/data/pipeline_runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    ap.add_argument("--demos", nargs="+", type=int, default=_DEFAULT_DEMOS)
    ap.add_argument("--budgets", nargs="+", type=int, default=_DEFAULT_BUDGETS)
    ap.add_argument("--heuristics", nargs="+", default=_DEFAULT_HEURISTICS)
    ap.add_argument("--out", default="/tmp/mimicgen_budget_sweep.html")
    ap.add_argument("--ci", action="store_true", help="Show 95% CI instead of ±1 std")
    ap.add_argument("--no_band", action="store_true", help="Never draw error bands")
    args = ap.parse_args()

    pipe_base = Path(args.pipe_base) if args.pipe_base else _find_pipe_base()

    print(f"Loading data from {pipe_base}")
    data = load_data(pipe_base, args.seeds, args.demos, args.heuristics, args.budgets)

    if not data:
        print("No result.json files found. Check that eval_mimicgen_combined has finished.")
        sys.exit(1)

    # Report what was loaded
    print(f"Loaded {len(data)} (demos, heuristic, budget) cells:")
    for (d, h, b), rates in sorted(data.items()):
        seeds_str = f"{len(rates)} seed{'s' if len(rates) != 1 else ''}"
        rates_str = ", ".join(f"{r:.3f}" for r in rates)
        print(f"  d{d} {h:16s} budget={b:5d}  [{seeds_str}]  rates=[{rates_str}]")

    fig = build_figure(
        data,
        demos_list=args.demos,
        heuristics=args.heuristics,
        budgets=args.budgets,
        use_ci=args.ci,
        no_band=args.no_band,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".html":
        fig.write_html(str(out))
    else:
        fig.write_image(str(out))

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
