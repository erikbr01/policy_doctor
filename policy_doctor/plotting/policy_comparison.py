"""Matplotlib figures for multi-policy success-rate comparison.

Follows the notebook: https://gist.github.com/HarukiNishimura-TRI/f4820826e7d93af5a5c9452cc6dd44ce

Pure functions — no Streamlit imports, no data preprocessing.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ── posterior sampling ─────────────────────────────────────────────────────────

def draw_beta_posterior_samples(
    successes: np.ndarray,
    rng: np.random.Generator,
    num_samples: int = 10_000,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> np.ndarray:
    """Beta posterior samples for a Bernoulli success rate (uniform prior by default)."""
    n = len(successes)
    k = int(successes.sum())
    return stats.beta(alpha_prior + k, beta_prior + n - k).rvs(num_samples, random_state=rng)


# ── main violin plot (matches notebook plot_model_comparison) ──────────────────

def create_policy_comparison_violin(
    group_labels: list[str],
    success_arrays: list[np.ndarray],
    cld_letters: list[str],
    rng: np.random.Generator,
    *,
    n_posterior_samples: int = 10_000,
    show_empirical_means: bool = False,
    overlay_bars: bool = False,
    title: Optional[str] = None,
    unit_width: float = 2.5,
    height: float = 4.0,
    dpi: int = 100,
) -> plt.Figure:
    """Violin plot of Beta posterior success-rate distributions with CLD annotations.

    Directly follows the notebook's plot_model_comparison() function.
    Groups sharing the same CLD letter are not significantly different.
    """
    n = len(group_labels)
    posterior_samples = [
        draw_beta_posterior_samples(a, rng, num_samples=n_posterior_samples)
        for a in success_arrays
    ]
    posterior_means = [float(s.mean()) for s in posterior_samples]
    empirical_means = [float(a.mean()) for a in success_arrays]

    fig, ax = plt.subplots(figsize=(max(unit_width * n, 4.0), height), dpi=dpi)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n)]

    parts = ax.violinplot(
        posterior_samples,
        positions=np.arange(n),
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_zorder(2)
    parts["cmeans"].set_zorder(3)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    for i, (y, letter) in enumerate(zip(posterior_means, cld_letters)):
        ax.text(
            i + 0.15, min(y + 0.03, 0.96),
            letter,
            fontsize=12, fontweight="bold", color="black", va="center", zorder=4,
        )

    if show_empirical_means:
        ax.scatter(
            np.arange(n), empirical_means,
            edgecolors="black", facecolors="darkgrey", zorder=4, label="Empirical mean",
        )

    if overlay_bars:
        ax.bar(
            np.arange(n), empirical_means,
            width=0.6, color=colors, alpha=0.25, edgecolor="black", linewidth=0.5, zorder=1,
        )

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig


# ── bar chart with Wilson CI ──────────────────────────────────────────────────

def create_policy_comparison_bar(
    group_labels: list[str],
    success_rates: list[float],
    n_episodes: list[int],
    cld_letters: list[str],
    *,
    confidence: float = 0.95,
    title: Optional[str] = None,
    unit_width: float = 2.5,
    height: float = 4.0,
    dpi: int = 100,
) -> plt.Figure:
    """Bar chart of empirical success rates with Wilson score confidence intervals.

    Wilson intervals are asymmetric around p and stay bounded in [0,1], unlike
    the normal approximation interval.
    """
    from scipy.stats import norm as _norm

    n = len(group_labels)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n)]
    z = _norm.ppf(1 - (1 - confidence) / 2)

    ci_lo, ci_hi = [], []
    for p, m in zip(success_rates, n_episodes):
        if m == 0:
            ci_lo.append(0.0); ci_hi.append(0.0); continue
        centre = (p + z**2 / (2 * m)) / (1 + z**2 / m)
        margin = z * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2)) / (1 + z**2 / m)
        ci_lo.append(max(0.0, centre - margin))
        ci_hi.append(min(1.0, centre + margin))

    err_lo = [r - lo for r, lo in zip(success_rates, ci_lo)]
    err_hi = [hi - r for r, hi in zip(success_rates, ci_hi)]

    fig, ax = plt.subplots(figsize=(max(unit_width * n, 4.0), height), dpi=dpi)
    xs = np.arange(n)
    ax.bar(xs, success_rates, color=colors, alpha=0.75, edgecolor="black", linewidth=0.7, zorder=2)
    ax.errorbar(xs, success_rates, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=5, linewidth=1.2, zorder=3)

    for i, (r, letter, m, hi_err) in enumerate(zip(success_rates, cld_letters, n_episodes, err_hi)):
        ax.text(i, min(r + hi_err + 0.04, 0.97), letter,
                ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.text(i, -0.07, f"n={m}", ha="center", va="top", fontsize=8, color="#555")

    ax.set_xticks(xs)
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    ax.set_ylim(-0.12, 1.1)
    ax.set_ylabel("Success Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ci_pct = int(round(confidence * 100))
    ax.set_xlabel(f"Error bars: {ci_pct}% Wilson CI")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig


# ── per-run breakdown strip plot ──────────────────────────────────────────────

def create_policy_comparison_breakdown(
    group_labels: list[str],
    per_group_run_data: list[list[tuple[str, float, int]]],
    cld_letters: list[str],
    *,
    title: Optional[str] = None,
    unit_width: float = 2.5,
    height: float = 4.0,
    dpi: int = 100,
) -> plt.Figure:
    """Strip plot of individual run success rates within each group.

    Args:
        per_group_run_data: For each group, a list of (run_name, success_rate, n_episodes).
    """
    n = len(group_labels)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(unit_width * n, 4.0), height), dpi=dpi)

    for gi, (runs, color) in enumerate(zip(per_group_run_data, colors)):
        rates = [r for _, r, _ in runs]
        if not rates:
            continue
        # Group mean
        group_mean = np.mean(rates)
        ax.hlines(group_mean, gi - 0.3, gi + 0.3, colors=color, linewidths=2, zorder=3)
        # Individual runs with slight jitter for visibility
        jitter = np.linspace(-0.15, 0.15, len(rates)) if len(rates) > 1 else [0.0]
        for j, (name, rate, n_ep) in enumerate(runs):
            ax.scatter(gi + jitter[j], rate, color=color, edgecolors="black",
                       linewidths=0.7, s=60, zorder=4)

    for gi, (letter, runs) in enumerate(zip(cld_letters, per_group_run_data)):
        if runs:
            rates = [r for _, r, _ in runs]
            ax.text(gi + 0.2, max(rates) + 0.04, letter,
                    ha="left", va="bottom", fontweight="bold", fontsize=11)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Success Rate (per run)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlabel("Horizontal line = group mean")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig
