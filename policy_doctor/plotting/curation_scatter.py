"""Scatter plot: training dataset size vs mean training rollout success (curation runs)."""

from __future__ import annotations

from typing import Literal, NamedTuple, Sequence

import plotly.graph_objects as go

from policy_doctor.data.curation_eval_scan import (
    CurationScatterPoint,
    experiment_key_from_train_name,
)

ScatterXLMode = Literal["data_size", "mean_score_order"]


class MeanScoreOrderScatterLayout(NamedTuple):
    """X-axis layout: experiment columns left→right by ascending mean ``test/mean_score``."""

    exp_keys: list[str]
    exp_index: dict[str, int]
    ds_min: float
    ds_max: float

    @property
    def n_exp(self) -> int:
        return len(self.exp_keys)


def build_mean_score_order_scatter_layout(
    others: Sequence[CurationScatterPoint],
    task_substring: str,
) -> MeanScoreOrderScatterLayout | None:
    if not others:
        return None
    keys = list({experiment_key_from_train_name(p.train_name, task_substring) for p in others})

    def _mean_y(k: str) -> float:
        sub = [p for p in others if experiment_key_from_train_name(p.train_name, task_substring) == k]
        return sum(p.mean_rollout_score for p in sub) / len(sub)

    ordered = sorted(keys, key=_mean_y)
    exp_index = {k: i for i, k in enumerate(ordered)}
    ds = [float(p.data_size) for p in others]
    return MeanScoreOrderScatterLayout(ordered, exp_index, min(ds), max(ds))


def scatter_x_mean_score_order_curated(
    p: CurationScatterPoint,
    layout: MeanScoreOrderScatterLayout,
    *,
    task_substring: str,
) -> float:
    ek = experiment_key_from_train_name(p.train_name, task_substring)
    i = layout.exp_index[ek]
    span = layout.ds_max - layout.ds_min
    t = 0.5 if span <= 0 else (float(p.data_size) - layout.ds_min) / span
    return i + 0.12 + 0.76 * t


def scatter_x_mean_score_order_baseline(
    n_exp: int,
    *,
    baseline_index: int,
    n_baseline: int,
) -> float:
    if n_exp <= 0:
        return 0.5
    center = float(n_exp) + 0.5
    if n_baseline <= 1:
        return center
    step = min(0.08, 0.35 / (n_baseline - 1))
    return center + (baseline_index - 0.5 * (n_baseline - 1)) * step


def sort_experiment_keys_mean_score_order(
    keys: Sequence[str],
    points: Sequence[CurationScatterPoint],
    task_substring: str,
) -> list[str]:
    """Order experiment keys like the scatter ``mean_score_order`` x-axis.

    Non-baseline experiments ascending by mean ``test/mean_score`` (over pooled points);
    baseline-only experiments after that (also ascending by mean among themselves).
    Keys with no matching points are kept at the end in input order.
    """
    keys_list = list(dict.fromkeys(keys))
    pts = list(points)

    def sub_for(k: str) -> list[CurationScatterPoint]:
        return [p for p in pts if experiment_key_from_train_name(p.train_name, task_substring) == k]

    def mean_y(sub: list[CurationScatterPoint]) -> float:
        return sum(p.mean_rollout_score for p in sub) / len(sub)

    def is_baseline_only(sub: list[CurationScatterPoint]) -> bool:
        return bool(sub) and all(p.is_baseline for p in sub)

    curated = [k for k in keys_list if sub_for(k) and not is_baseline_only(sub_for(k))]
    baseline = [k for k in keys_list if sub_for(k) and is_baseline_only(sub_for(k))]
    empty_tail = [k for k in keys_list if not sub_for(k)]

    curated.sort(key=lambda k: mean_y(sub_for(k)))
    baseline.sort(key=lambda k: mean_y(sub_for(k)))
    return curated + baseline + empty_tail


_SEED_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _hover_text(
    p: CurationScatterPoint,
    rollout_window: int,
    *,
    layout_note: str | None = None,
) -> str:
    lines = [
        f"<b>{p.train_name}</b>",
        f"checkpoint: {p.checkpoint_name}",
        f"seed: {p.seed}",
        f"data_size (train sequences): {p.data_size}",
        (
            f"train seq vs no slice-curation: {p.sequence_pct_of_uncurated_train:.1f}% "
            f"({p.data_size}/{p.train_sequences_uncurated})"
            if p.sequence_pct_of_uncurated_train is not None
            and p.train_sequences_uncurated is not None
            else "train seq % (uncurated baseline): n/a"
        ),
        (
            f"log eval {p.log_eval_tail_index + 1}/{p.log_eval_tail_count} in last-{rollout_window} tail · "
            f"test/mean_score: {p.mean_rollout_score:.4f}"
        ),
        f"n_test (episodes per eval, cfg): {p.rollout_episodes_per_eval}",
        f"full last-{p.log_eval_tail_count} tail scores: {[round(x, 4) for x in p.last_k_scores]}",
    ]
    if p.ckpt_epoch is not None:
        lines.append(f"ckpt epoch: {p.ckpt_epoch}")
    if p.ckpt_global_step is not None:
        lines.append(f"ckpt global_step: {p.ckpt_global_step}")
    if p.alignment_warning:
        lines.append(f"note: {p.alignment_warning}")
    if p.offline_eval_score is not None:
        lines.append(f"offline eval_log test/mean_score: {p.offline_eval_score:.4f}")
    if layout_note:
        lines.append(layout_note)
    return "<br>".join(lines)


def _box_point_hover(
    p: CurationScatterPoint,
    rollout_window: int,
    *,
    experiment_key_html: str | None = None,
) -> str:
    pct = (
        f"<br>train seq {p.sequence_pct_of_uncurated_train:.1f}% "
        f"({p.data_size}/{p.train_sequences_uncurated})"
        if p.sequence_pct_of_uncurated_train is not None
        and p.train_sequences_uncurated is not None
        else ""
    )
    head = f"<b>{experiment_key_html}</b><br>" if experiment_key_html else ""
    tail_note = (
        f"<br>log tail eval {p.log_eval_tail_index + 1}/{p.log_eval_tail_count} · "
        f"test/mean_score={p.mean_rollout_score:.4f}"
    )
    return (
        f"{head}{p.train_name}<br>seed {p.seed}<br>{p.checkpoint_name}<br>epoch={p.ckpt_epoch}{tail_note}{pct}"
    )


_VERBOSE_RUN_SEGMENT = "_train_diffusion_unet_lowdim_"


def _compact_train_experiment_key(key: str) -> str:
    """Drop the repeated policy segment so x-axis text fits (matches default run naming)."""
    return key.replace(_VERBOSE_RUN_SEGMENT, "_")


def _short_experiment_label(key: str, max_chars: int = 40) -> str:
    if len(key) <= max_chars:
        return key
    half = max_chars // 2 - 2
    return key[:half] + "…" + key[-half:]


def _experiment_rollout_total(
    sub: Sequence[CurationScatterPoint],
) -> tuple[int, int, int, int, int]:
    """Episodes per eval, number of log-eval points, seeds, runs, total episode budget.

    Each point is one training-log ``test/mean_score`` row (last-``K`` tail expanded).
    ``total`` ≈ ``n_test × len(sub)``.
    """
    if not sub:
        return 50, 0, 0, 0, 0
    nr = int(sub[0].rollout_episodes_per_eval)
    n_eval_pts = len(sub)
    n_seeds = len({p.seed for p in sub})
    n_runs = len({p.train_name for p in sub})
    total = nr * n_eval_pts
    return nr, n_eval_pts, n_seeds, n_runs, total


def _sequence_pct_summary_for_runs(sub: Sequence[CurationScatterPoint]) -> str:
    """Distinct train_name → one pct; caption for boxplots."""
    by_run: dict[str, float] = {}
    for p in sub:
        if p.sequence_pct_of_uncurated_train is not None:
            by_run[p.train_name] = p.sequence_pct_of_uncurated_train
    if not by_run:
        return ""
    vals = list(by_run.values())
    if len(vals) == 1:
        return f"{vals[0]:.1f}% train seq"
    return f"{min(vals):.1f}–{max(vals):.1f}% train seq ({len(vals)} runs)"


def _box_x_category_label(
    experiment_display_name: str,
    *,
    pct_caption: str,
    total_roll: int,
    nr: int,
    n_eval_pts: int,
    n_runs: int,
    n_seeds: int,
) -> str:
    """Multi-line x tick: name, then bold % train seq, then rollout budget (no <sup> — stays readable)."""
    lines = [f"<b>{experiment_display_name}</b>"]
    if pct_caption:
        lines.append(f"<b>{pct_caption}</b>")
    lines.append(
        f"{total_roll} rollouts · n_test={nr}×{n_eval_pts} log evals<br>"
        f"({n_runs} runs · {n_seeds} seeds)"
    )
    return "<br>".join(lines)


def create_curation_data_vs_success_scatter(
    points: Sequence[CurationScatterPoint],
    *,
    title: str = "Training data size vs training log eval scores",
    x_title: str = "Training dataset size (sequence count)",
    y_title: str = "test/mean_score (each point = one eval in last K log lines)",
    rollout_window: int = 5,
    baseline_train_date: str = "jan28",
    task_substring: str = "transport_mh",
    scatter_x: ScatterXLMode = "mean_score_order",
    height: int = 600,
) -> go.Figure:
    """Plotly scatter: Y=each eval in the last-``K`` ``logs.json.txt`` tail.

    Non-baseline points are colored by :func:`experiment_key_from_train_name` (one color per
    experiment). With ``scatter_x="mean_score_order"`` (default), experiment columns run
    left→right by ascending mean log score; within a column, x spreads with dataset size;
    baselines share a column on the far right. With ``scatter_x="data_size"``, x is raw
    training sequence count.
    """
    pts = list(points)
    fig = go.Figure()

    baselines = [p for p in pts if p.is_baseline]
    others = [p for p in pts if not p.is_baseline]

    def _pct_label(p: CurationScatterPoint) -> str:
        if p.sequence_pct_of_uncurated_train is None:
            return ""
        return f"{p.sequence_pct_of_uncurated_train:.0f}%"

    ms_layout = (
        build_mean_score_order_scatter_layout(others, task_substring)
        if scatter_x == "mean_score_order" and others
        else None
    )
    n_exp = ms_layout.n_exp if ms_layout else 0
    nb = len(baselines)

    def _x_curated(p: CurationScatterPoint) -> float:
        if scatter_x == "data_size" or ms_layout is None:
            return float(p.data_size)
        return scatter_x_mean_score_order_curated(p, ms_layout, task_substring=task_substring)

    def _hover_curated(p: CurationScatterPoint) -> str:
        note = None
        if ms_layout is not None:
            note = (
                "<i>x: experiment column by mean log score (asc.); "
                f"offset ∝ train seq in [{ms_layout.ds_min:.0f}, {ms_layout.ds_max:.0f}]</i>"
            )
        return _hover_text(p, rollout_window, layout_note=note)

    def _hover_baseline(p: CurationScatterPoint) -> str:
        note = None
        if ms_layout is not None or (scatter_x == "mean_score_order" and not others):
            note = "<i>x: baseline column (far right)</i>"
        return _hover_text(p, rollout_window, layout_note=note)

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
        legend_label = _short_experiment_label(_compact_train_experiment_key(exp_key), 52)
        fig.add_trace(
            go.Scatter(
                x=[_x_curated(p) for p in sub],
                y=[p.mean_rollout_score for p in sub],
                mode="markers+text",
                name=legend_label,
                marker=dict(size=9, color=c, line=dict(width=0.5, color="#333")),
                text=[_pct_label(p) for p in sub],
                textposition="top center",
                textfont=dict(size=9, color="#333"),
                hovertext=[_hover_curated(p) for p in sub],
                hoverinfo="text",
            )
        )

    if baselines:
        fig.add_trace(
            go.Scatter(
                x=[
                    scatter_x_mean_score_order_baseline(
                        n_exp,
                        baseline_index=bi,
                        n_baseline=nb,
                    )
                    if scatter_x == "mean_score_order"
                    else float(p.data_size)
                    for bi, p in enumerate(baselines)
                ],
                y=[p.mean_rollout_score for p in baselines],
                mode="markers+text",
                name=f"baseline ({baseline_train_date}, no curation)",
                marker=dict(size=14, symbol="star", color="#c0392b", line=dict(width=1, color="#333")),
                text=[_pct_label(p) for p in baselines],
                textposition="top center",
                textfont=dict(size=9, color="#333"),
                hovertext=[_hover_baseline(p) for p in baselines],
                hoverinfo="text",
            )
        )

    y_label = y_title.replace("K", str(rollout_window)).replace("k", str(rollout_window))
    plot_title = title
    if any(p.sequence_pct_of_uncurated_train is not None for p in pts):
        plot_title = (
            f"{title}<br><sup>Labels: % train sequences vs same cfg without "
            f"sample_curation_config / holdout_selection_config</sup>"
        )
    if scatter_x == "mean_score_order" and (others or baselines):
        x_axis_title = (
            "Experiment columns: mean log score ascending (curated); "
            "spread ∝ train seq count; baseline column on the right"
        )
    else:
        x_axis_title = f"{x_title} (marker text: % train seq vs uncurated)"
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_axis_title,
        yaxis_title=y_label,
        height=height,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    fig.update_yaxes(rangemode="tozero")

    if scatter_x == "mean_score_order" and (others or baselines):
        tickvals: list[float] = []
        ticktext: list[str] = []
        if ms_layout is not None:
            tickvals = [float(i) + 0.5 for i in range(ms_layout.n_exp)]
            ticktext = [
                _short_experiment_label(_compact_train_experiment_key(k), 36)
                for k in ms_layout.exp_keys
            ]
        if baselines:
            tickvals.append(float(n_exp) + 0.5)
            ticktext.append(f"baseline<br>({baseline_train_date})")
        x_max = float(n_exp) + (1.0 if baselines else 0.0)
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[-0.35, x_max + 0.35],
        )
    return fig


def create_experiment_checkpoint_score_boxplot(
    points: Sequence[CurationScatterPoint],
    *,
    experiment_key: str,
    task_substring: str = "transport_mh",
    rollout_window: int = 5,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Boxplot of **individual** ``test/mean_score`` values from the last-``K`` log tail.

    Each run contributes up to ``K`` y-values (one per eval line). ``experiment_key``
    groups runs that match :func:`experiment_key_from_train_name`.
    """
    pts = list(points)
    sub = [
        p
        for p in pts
        if experiment_key_from_train_name(p.train_name, task_substring) == experiment_key
    ]
    if not sub:
        raise ValueError(f"No points for experiment_key={experiment_key!r}")

    sub_sorted = sorted(
        sub,
        key=lambda p: (p.train_name, p.log_eval_tail_index, p.mean_rollout_score),
    )
    y = [p.mean_rollout_score for p in sub_sorted]
    text = [_box_point_hover(p, rollout_window) for p in sub_sorted]

    y_label = f"test/mean_score (last {rollout_window} log evals · one y per line)"
    nr, n_eval_pts, n_seeds_m, n_runs_m, total_roll = _experiment_rollout_total(sub)
    compact_key = _compact_train_experiment_key(experiment_key)
    plot_title = title or (
        f"Score distribution (last-{rollout_window} log evals per run)<br><sup>{compact_key}</sup><br>"
        f"<sup>{len(sub)} y-values · {n_runs_m} runs · {n_seeds_m} seeds · "
        f"~{total_roll} rollouts (n_test={nr}×{n_eval_pts} evals)</sup>"
    )

    pct_caption = _sequence_pct_summary_for_runs(sub)
    x_cat = _box_x_category_label(
        _short_experiment_label(compact_key, 56),
        pct_caption=pct_caption,
        total_roll=total_roll,
        nr=nr,
        n_eval_pts=n_eval_pts,
        n_runs=n_runs_m,
        n_seeds=n_seeds_m,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=y,
            x=[x_cat] * len(y),
            name=experiment_key,
            boxpoints="all",
            jitter=0.35,
            pointpos=-1.8,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=5, opacity=0.65),
            line=dict(width=1.5),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=plot_title,
        xaxis_title="",
        yaxis_title=y_label,
        showlegend=False,
        height=height,
        margin=dict(l=60, r=40, t=90, b=160),
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
    return fig


def create_multi_experiment_checkpoint_score_boxplots(
    points: Sequence[CurationScatterPoint],
    *,
    task_substring: str = "transport_mh",
    rollout_window: int = 5,
    experiment_keys: Sequence[str],
    title: str | None = None,
    height: int = 620,
    min_width_per_box: int = 130,
    label_max_chars: int = 56,
) -> go.Figure:
    """One box per experiment key; each box pools all log-eval y-values (last-``K`` tail per run).

    Boxes are ordered like the scatter ``mean_score_order`` x-axis: curated experiments by
    ascending mean score, then baseline-only experiments (see
    :func:`sort_experiment_keys_mean_score_order`).
    """
    pts = list(points)
    keys = sort_experiment_keys_mean_score_order(experiment_keys, pts, task_substring)
    if not keys:
        raise ValueError("experiment_keys is empty")

    fig = go.Figure()
    # Plotly merges box traces that share the same `name`; truncated labels can collide for
    # distinct experiment_key strings (difference only in the middle of the key).
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
        compact_key = _compact_train_experiment_key(key)
        text = [
            _box_point_hover(p, rollout_window, experiment_key_html=compact_key)
            for p in sub_sorted
        ]
        short = _short_experiment_label(compact_key, label_max_chars)
        n_same_short = short_label_counts.get(short, 0)
        short_label_counts[short] = n_same_short + 1
        legend_name = short if n_same_short == 0 else f"{short} ({n_same_short + 1})"
        nr, n_eval_pts, n_seeds_m, n_runs_m, total_roll = _experiment_rollout_total(sub_sorted)
        pct_caption = _sequence_pct_summary_for_runs(sub_sorted)
        x_cat = _box_x_category_label(
            legend_name,
            pct_caption=pct_caption,
            total_roll=total_roll,
            nr=nr,
            n_eval_pts=n_eval_pts,
            n_runs=n_runs_m,
            n_seeds=n_seeds_m,
        )
        fig.add_trace(
            go.Box(
                y=y,
                x=[x_cat] * len(y),
                name=legend_name,
                boxpoints="all",
                jitter=0.25,
                pointpos=-1.6,
                text=text,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=4, opacity=0.55),
                line=dict(width=1.2),
                showlegend=False,
            )
        )

    if not fig.data:
        raise ValueError("No points for any of the given experiment keys")

    y_label = f"test/mean_score (last {rollout_window} log evals · one y per line)"
    n_boxes = len(fig.data)
    plot_title = title or (
        f"Score distribution by experiment (pooled last-{rollout_window} log evals)<br>"
        f"<sup>{n_boxes} experiments · % train seq = curated / same cfg without slice curation YAMLs</sup>"
    )
    width = min(2800, max(840, min_width_per_box * n_boxes))
    fig.update_layout(
        title=plot_title,
        xaxis_title=(
            "Experiment (mean log score ↑ curated, baseline right · "
            "% train seq · rollout count)"
        ),
        yaxis_title=y_label,
        showlegend=False,
        height=height,
        width=width,
        margin=dict(l=60, r=40, t=72, b=300),
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(
        tickangle=-20,
        tickfont=dict(size=12),
        automargin=True,
    )
    return fig
