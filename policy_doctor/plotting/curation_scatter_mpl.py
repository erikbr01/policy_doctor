"""Matplotlib PDF export for curation scatter / multi-experiment boxplots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policy_doctor.data.curation_eval_scan import (
    CurationScatterPoint,
    experiment_key_from_train_name,
)
from policy_doctor.plotting.curation_scatter import (
    _SEED_COLORS,
    _compact_train_experiment_key,
    _experiment_rollout_total,
    _sequence_pct_summary_for_runs,
    _short_experiment_label,
    ScatterXLMode,
    build_mean_score_order_scatter_layout,
    scatter_x_mean_score_order_baseline,
    scatter_x_mean_score_order_curated,
    sort_experiment_keys_mean_score_order,
)


def _strip_simple_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</?b>", "", s, flags=re.I)
    s = re.sub(r"<sup>.*?</sup>", "", s, flags=re.I | re.S)
    return re.sub(r"<[^>]+>", "", s).strip()


def _box_caption_plain(
    legend_name: str,
    pct_caption: str,
    total_roll: int,
    nr: int,
    n_eval_pts: int,
    n_runs: int,
    n_seeds: int,
) -> str:
    lines = [legend_name]
    if pct_caption:
        lines.append(pct_caption)
    lines.append(
        f"{total_roll} rollouts · n_test={nr}×{n_eval_pts} log evals\n({n_runs} runs · {n_seeds} seeds)"
    )
    return "\n".join(lines)


def save_curation_scatter_matplotlib_pdf(
    path: str | Path,
    points: Sequence[CurationScatterPoint],
    *,
    title: str,
    x_title: str = "Training dataset size (sequence count)",
    y_title: str = "test/mean_score (each point = one eval in last K log lines)",
    rollout_window: int = 5,
    baseline_train_date: str = "jan28",
    task_substring: str = "transport_mh",
    scatter_x: ScatterXLMode = "mean_score_order",
    figsize: tuple[float, float] = (9.0, 6.0),
) -> None:
    """Scatter: Y=log eval scores; color by experiment key; write vector PDF.

    Matches :func:`create_curation_data_vs_success_scatter` x semantics (``scatter_x``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42

    pts = list(points)
    fig, ax = plt.subplots(figsize=figsize)

    def pct_label(p: CurationScatterPoint) -> str:
        if p.sequence_pct_of_uncurated_train is None:
            return ""
        return f"{p.sequence_pct_of_uncurated_train:.0f}%"

    baselines = [p for p in pts if p.is_baseline]
    others = [p for p in pts if not p.is_baseline]
    y_label = y_title.replace("K", str(rollout_window)).replace("k", str(rollout_window))

    ms_layout = (
        build_mean_score_order_scatter_layout(others, task_substring)
        if scatter_x == "mean_score_order" and others
        else None
    )
    n_exp = ms_layout.n_exp if ms_layout else 0
    nb = len(baselines)

    def x_curated(p: CurationScatterPoint) -> float:
        if scatter_x == "data_size" or ms_layout is None:
            return float(p.data_size)
        return scatter_x_mean_score_order_curated(p, ms_layout, task_substring=task_substring)

    exp_keys_plot: list[str]
    if ms_layout is not None:
        exp_keys_plot = ms_layout.exp_keys
    else:
        exp_keys_plot = sorted(
            {experiment_key_from_train_name(p.train_name, task_substring) for p in others}
        )

    for i, exp_key in enumerate(exp_keys_plot):
        sub = [
            p
            for p in others
            if experiment_key_from_train_name(p.train_name, task_substring) == exp_key
        ]
        if not sub:
            continue
        c = _SEED_COLORS[i % len(_SEED_COLORS)]
        xs = [x_curated(p) for p in sub]
        ys = [p.mean_rollout_score for p in sub]
        legend_label = _short_experiment_label(_compact_train_experiment_key(exp_key), 52)
        ax.scatter(
            xs,
            ys,
            s=45,
            c=c,
            edgecolors="#333",
            linewidths=0.35,
            zorder=4,
            label=legend_label,
        )
        for p, xp in zip(sub, xs):
            t = pct_label(p)
            if t:
                ax.annotate(
                    t,
                    (xp, p.mean_rollout_score),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=7,
                    color="#333",
                )

    if baselines:
        if scatter_x == "mean_score_order":
            xs_b = [
                scatter_x_mean_score_order_baseline(
                    n_exp,
                    baseline_index=bi,
                    n_baseline=nb,
                )
                for bi in range(nb)
            ]
        else:
            xs_b = [float(p.data_size) for p in baselines]
        ys_b = [p.mean_rollout_score for p in baselines]
        ax.scatter(
            xs_b,
            ys_b,
            s=200,
            marker="*",
            c="#c0392b",
            edgecolors="#333",
            linewidths=0.8,
            zorder=5,
            label=f"baseline ({baseline_train_date}, no curation)",
        )
        for p, xp in zip(baselines, xs_b):
            t = pct_label(p)
            if t:
                ax.annotate(
                    t,
                    (xp, p.mean_rollout_score),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color="#333",
                )

    if scatter_x == "mean_score_order" and (others or baselines):
        ax.set_xlabel(
            "Experiment columns: mean log score ascending (curated); "
            "spread ∝ train seq; baseline on the right",
            fontsize=9,
        )
        tickvals: list[float] = []
        ticklabels: list[str] = []
        if ms_layout is not None:
            tickvals = [float(i) + 0.5 for i in range(ms_layout.n_exp)]
            ticklabels = [
                _short_experiment_label(_compact_train_experiment_key(k), 32)
                for k in ms_layout.exp_keys
            ]
        if baselines:
            tickvals.append(float(n_exp) + 0.5)
            ticklabels.append(f"baseline\n({baseline_train_date})")
        ax.set_xticks(tickvals)
        ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=7)
        x_max = float(n_exp) + (1.0 if baselines else 0.0)
        ax.set_xlim(-0.35, x_max + 0.35)
    else:
        ax.set_xlabel(
            f"{x_title} (marker text: % train seq vs uncurated)",
            fontsize=10,
        )
    ax.set_ylabel(y_label, fontsize=10)
    full_title = _strip_simple_html(title)
    if any(p.sequence_pct_of_uncurated_train is not None for p in pts):
        full_title += (
            "\nMarker labels: % train sequences vs cfg without "
            "sample_curation_config / holdout_selection_config"
        )
    ax.set_title(full_title, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_multi_experiment_boxplots_matplotlib_pdf(
    path: str | Path,
    points: Sequence[CurationScatterPoint],
    *,
    experiment_keys: Sequence[str],
    task_substring: str = "transport_mh",
    rollout_window: int = 5,
    title: str | None = None,
    label_max_chars: int = 56,
    min_width_per_box_inch: float = 1.55,
    fig_height: float = 7.0,
) -> None:
    """One box per experiment key + jittered points; vector PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42

    pts = list(points)
    keys = sort_experiment_keys_mean_score_order(experiment_keys, pts, task_substring)
    data: list[list[float]] = []
    captions: list[str] = []
    short_label_counts: dict[str, int] = {}

    for key in keys:
        sub = [
            p
            for p in pts
            if experiment_key_from_train_name(p.train_name, task_substring) == key
        ]
        if not sub:
            continue
        sub_sorted = sorted(
            sub,
            key=lambda p: (p.train_name, p.log_eval_tail_index, p.mean_rollout_score),
        )
        y = [p.mean_rollout_score for p in sub_sorted]
        data.append(y)
        compact_key = _compact_train_experiment_key(key)
        short = _short_experiment_label(compact_key, label_max_chars)
        n_same = short_label_counts.get(short, 0)
        short_label_counts[short] = n_same + 1
        legend_name = short if n_same == 0 else f"{short} ({n_same + 1})"
        nr, n_eval_pts, n_seeds_m, n_runs_m, total_roll = _experiment_rollout_total(sub_sorted)
        pct_caption = _sequence_pct_summary_for_runs(sub_sorted)
        captions.append(
            _box_caption_plain(
                legend_name,
                pct_caption,
                total_roll,
                nr,
                n_eval_pts,
                n_runs_m,
                n_seeds_m,
            )
        )

    n = len(data)
    if n == 0:
        raise ValueError("No data for multi boxplot PDF")

    fig_w = max(9.0, n * min_width_per_box_inch)
    fig, ax = plt.subplots(figsize=(fig_w, fig_height))
    positions = np.arange(1, n + 1, dtype=float)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe2f3")
        patch.set_alpha(0.88)

    rng = np.random.default_rng(0)
    for i, yi in enumerate(data):
        if not yi:
            continue
        jitter = 0.07 * rng.standard_normal(len(yi))
        ax.scatter(
            positions[i] + jitter,
            yi,
            s=12,
            alpha=0.55,
            c="#222222",
            zorder=3,
            linewidths=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(captions, fontsize=7.5, rotation=22, ha="right")
    y_label = f"test/mean_score (last {rollout_window} log evals · one y per line)"
    ax.set_xlabel(
        "Experiments: mean log score ↑ (curated), baseline right · % train seq · rollouts",
        fontsize=9,
    )
    ax.set_ylabel(y_label, fontsize=10)
    n_boxes = n
    plot_title = title or (
        f"Score distribution by experiment (pooled last-{rollout_window} log evals)\n"
        f"{n_boxes} experiments · % train seq = curated / same cfg without slice curation YAMLs"
    )
    ax.set_title(_strip_simple_html(plot_title), fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
