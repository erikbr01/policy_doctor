"""Comparison tab: plot test/mean_score over time from wandb, averaged across seeds."""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from influence_visualizer.config import load_config
from influence_visualizer.profiling import profile
from influence_visualizer.data_loader import (
    get_checkpoint_path,
    get_train_dir_for_seed,
    load_checkpoint_config,
)


# Metric key logged by env runners (e.g. robomimic_lowdim_runner)
TEST_MEAN_SCORE_KEY = "test/mean_score"


def _get_logging_from_checkpoint(
    train_dir: pathlib.Path,
    train_ckpt: str = "latest",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get (project, entity) from the training config stored in the checkpoint.

    This is the project/entity that was actually used when the run was trained,
    so we use it to identify the wandb run instead of the visualizer config.
    """
    try:
        ckpt_path = get_checkpoint_path(train_dir, train_ckpt)
        if not ckpt_path.exists():
            return (None, None)
        cfg, _ = load_checkpoint_config(ckpt_path)
        logging = getattr(cfg, "logging", None)
        if logging is None:
            return (None, None)
        project = getattr(logging, "project", None) or (logging.get("project") if hasattr(logging, "get") else None)
        entity = getattr(logging, "entity", None) or (logging.get("entity") if hasattr(logging, "get") else None)
        return (project, entity)
    except Exception:
        return (None, None)


def _find_wandb_run_dir(train_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the run directory under train_dir/wandb/ (run-<timestamp>-<run_id> or run-<run_id>)."""
    wandb_dir = train_dir / "wandb"
    if not wandb_dir.is_dir():
        return None
    # Prefer latest-run symlink if present
    latest = wandb_dir / "latest-run"
    if latest.exists():
        try:
            resolved = latest.resolve()
            if resolved.is_dir():
                return resolved
        except OSError:
            pass
    # Otherwise first run-* directory (newer wandb: run-<id>; older: run-<timestamp>-<id>)
    for p in sorted(wandb_dir.iterdir()):
        if p.is_dir() and p.name.startswith("run-"):
            return p
    return None


def _run_id_from_run_dir(run_dir: pathlib.Path) -> Optional[str]:
    """Extract run id from run dir name. Handles run-<id> and run-<timestamp>-<id>."""
    name = run_dir.name
    if not name.startswith("run-"):
        return None
    # run-<id> (8-char id) or run-20250101-120000-<id>
    parts = name.split("-")
    if len(parts) >= 2:
        # Last part is the run id (typically 8 alphanumeric chars)
        run_id = parts[-1]
        if run_id and len(run_id) <= 32:
            return run_id
    return None


def _get_run_id_and_meta_from_train_dir(
    train_dir: pathlib.Path,
) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """
    Get (run_id, entity, project) from a training output directory.

    Reads train_dir/wandb/run-*/ for run id and optionally wandb-metadata.json
    for entity/project. Returns None if no run found.
    """
    run_dir = _find_wandb_run_dir(train_dir)
    if run_dir is None:
        return None
    run_id = _run_id_from_run_dir(run_dir)
    if not run_id:
        return None
    entity, project = None, None
    meta_file = run_dir / "wandb-metadata.json"
    if meta_file.exists():
        try:
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            entity = meta.get("entity") or meta.get("username")
            project = meta.get("project") or meta.get("wandb_project")
        except Exception:
            pass
    return (run_id, entity, project)


def _diagnose_no_wandb_data(
    train_dir: pathlib.Path,
    seed: str,
) -> str:
    """Return a short reason why wandb data was not found for this path."""
    if not train_dir.exists():
        return f"train_dir does not exist: {train_dir}"
    wandb_dir = train_dir / "wandb"
    if not wandb_dir.is_dir():
        return f"no wandb/ in train_dir (run from training output dir or sync wandb)"
    run_dir = _find_wandb_run_dir(train_dir)
    if run_dir is None:
        return f"no run-* dir in {wandb_dir}"
    run_id = _run_id_from_run_dir(run_dir)
    if not run_id:
        return f"could not parse run id from {run_dir.name}"
    return "run found locally; API or metric fetch may have failed"


def _fetch_run_history(
    run_id: str,
    entity: Optional[str],
    project: Optional[str],
    metric_key: str = TEST_MEAN_SCORE_KEY,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[str]]:
    """Fetch metric history for a run. Returns (data, error_message). data is None on failure."""
    try:
        import wandb
        api = wandb.Api()
        path = _api_run_path(entity, project, run_id)
        run = api.run(path)
        hist = run.history(keys=[metric_key, "_step"], pandas=True)
        if hist is None or hist.empty:
            return (None, "run history is empty")
        if metric_key not in hist.columns:
            # Try underscore variant (wandb sometimes normalizes keys)
            alt_key = metric_key.replace("/", "_")
            if alt_key in hist.columns:
                metric_key = alt_key
            else:
                available = [c for c in hist.columns if "score" in c.lower() or "mean" in c.lower()]
                return (None, f"metric '{metric_key}' not in history (available: {available[:8]})")
        steps = hist["_step"].values.astype(np.float64)
        values = hist[metric_key].values.astype(np.float64)
        return ((steps, values), None)
    except Exception as e:
        err = str(e).strip() or type(e).__name__
        return (None, err)


def _api_run_path(entity: Optional[str], project: Optional[str], run_id: str) -> str:
    """Build wandb API run path."""
    if entity and project:
        return f"{entity}/{project}/{run_id}"
    if project:
        return f"{project}/{run_id}"
    return run_id


def _get_n_test_from_run(
    run_id: str,
    entity: Optional[str],
    project: Optional[str],
) -> Optional[int]:
    """Get n_test (number of test rollouts per eval) from a run's config. Returns None if not found."""
    try:
        import wandb
        api = wandb.Api()
        path = _api_run_path(entity, project, run_id)
        run = api.run(path)
        cfg = getattr(run, "config", None) or {}
        if not isinstance(cfg, dict):
            cfg = dict(cfg)
        # Hydra/OmegaConf may log as nested or flat
        if "task" in cfg and isinstance(cfg["task"], dict):
            task = cfg["task"]
            if "env_runner" in task and isinstance(task["env_runner"], dict):
                n_test = task["env_runner"].get("n_test")
                if n_test is not None:
                    return int(n_test)
        # Flat keys (e.g. task.env_runner.n_test)
        for key in ("task/env_runner/n_test", "env_runner/n_test", "n_test"):
            if key in cfg and cfg[key] is not None:
                return int(cfg[key])
        return None
    except Exception:
        return None


def _align_and_aggregate_seeds(
    seed_curves: List[Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align curves from multiple seeds to common steps and compute mean ± std.

    seed_curves: list of (steps, values) per seed.
    Returns (steps_common, mean, std).
    """
    if not seed_curves:
        return (np.array([]), np.array([]), np.array([]))
    all_steps = np.concatenate([s for s, _ in seed_curves])
    min_step = float(np.min(all_steps))
    max_step = float(np.max(all_steps))
    if max_step <= min_step:
        steps_common = np.array([min_step])
    else:
        steps_common = np.linspace(min_step, max_step, n_bins)
    interpolated = []
    for steps, values in seed_curves:
        if len(steps) < 2:
            continue
        # np.interp expects increasing x
        order = np.argsort(steps)
        steps_s = steps[order]
        values_s = values[order]
        vals = np.interp(steps_common, steps_s, values_s)
        interpolated.append(vals)
    if not interpolated:
        return (steps_common, np.full_like(steps_common, np.nan), np.full_like(steps_common, np.nan))
    arr = np.array(interpolated)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return (steps_common, mean, std)


def _get_comparison_configs(
    current_config_name: str,
    current_config: Any,
) -> List[Tuple[str, Any]]:
    """Return list of (config_name, config) to compare: current + comparison list."""
    out = [(current_config_name, current_config)]
    if not getattr(current_config, "comparison", None):
        return out
    for name in current_config.comparison:
        if name == current_config_name:
            continue
        try:
            config = load_config(name)
            out.append((name, config))
        except FileNotFoundError:
            continue
    return out


def _resolve_train_dir(train_dir_str: str) -> pathlib.Path:
    """Resolve train_dir to absolute path; if relative, try from project root (parent of influence_visualizer)."""
    p = pathlib.Path(train_dir_str)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    # Try from project root (parent of influence_visualizer package)
    try:
        from influence_visualizer import config as _cfg
        root = _cfg.get_configs_dir().parent.parent  # repo root
        candidate = root / train_dir_str
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        pass
    return pathlib.Path(train_dir_str)


def _load_curves_for_config(
    config_name: str,
    config: Any,
    metric_key: str,
    project_override: Optional[str] = None,
    entity_override: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[str]]:
    """Load metric curves for all seeds of a config. Returns (curves, diagnostic_if_empty).

    Project/entity to identify the wandb run come from (in order):
    1. Training config in the checkpoint (cfg.logging.project / entity) - what was actually used
    2. Run's wandb-metadata.json
    3. Visualizer config's wandb_project / wandb_entity (fallback only)
    """
    train_dir_base = getattr(config, "train_dir", None) or ""
    seeds = getattr(config, "seeds", None) or ["0"]
    reference_seed = seeds[0]
    train_ckpt = getattr(config, "train_ckpt", "latest")
    fallback_project = project_override or getattr(config, "wandb_project", None)
    fallback_entity = entity_override or getattr(config, "wandb_entity", None)
    curves = []
    first_failure_reason: Optional[str] = None
    for seed in seeds:
        train_dir_str = get_train_dir_for_seed(train_dir_base, seed=seed, reference_seed=reference_seed)
        path = _resolve_train_dir(train_dir_str)
        info = _get_run_id_and_meta_from_train_dir(path)
        if info is None:
            if first_failure_reason is None:
                first_failure_reason = _diagnose_no_wandb_data(path, seed)
            continue
        run_id, run_entity, run_project = info
        # Use training config from checkpoint first (identifies the run that was actually used)
        ckpt_project, ckpt_entity = _get_logging_from_checkpoint(path, train_ckpt)
        use_project = ckpt_project or run_project or fallback_project
        use_entity = ckpt_entity or run_entity or fallback_entity
        if not use_project:
            if first_failure_reason is None:
                first_failure_reason = (
                    "no project in checkpoint (cfg.logging.project), run metadata, or config"
                )
            continue
        result, fetch_err = _fetch_run_history(run_id, use_entity, use_project, metric_key=metric_key)
        if result is not None:
            curves.append(result)
        elif first_failure_reason is None:
            first_failure_reason = (
                f"run {run_id} found but history fetch failed: {fetch_err or 'unknown'}"
            )
    return (curves, first_failure_reason)


def render_comparison_tab(
    current_config_name: str,
    config: Any,
) -> None:
    """
    Render the Comparison tab: plot test/mean_score over time for current and comparison configs.

    Uses wandb run history for each (config, seed), aligns by step, averages across seeds
    with standard deviation, and plots one line per config with shaded std.
    """
    st.header("Comparison: test/mean_score over time")
    st.markdown(
        "Plot **test/mean_score** from wandb logs, averaged across seeds with standard deviation. "
        "Add config names in the task config's **comparison** field and set **wandb_project** (and optionally **wandb_entity**)."
    )
    configs_to_compare = _get_comparison_configs(current_config_name, config)
    if len(configs_to_compare) <= 0:
        st.info("No configs to compare. Add a **comparison** list and **wandb_project** to your task config.")
        return

    project = getattr(config, "wandb_project", None)
    entity = getattr(config, "wandb_entity", None)
    if not project:
        st.warning("Set **wandb_project** in the task config to fetch run history from wandb.")
        return

    show_comparison_key = f"comparison_show_plot_{current_config_name}"
    with st.expander("Load comparison plot", expanded=False):
        st.caption(
            "Fetch test/mean_score from wandb for each config and plot mean ± std across seeds."
        )
        if st.button(
            "Fetch and show comparison",
            key=f"comparison_btn_load_{current_config_name}",
        ):
            st.session_state[show_comparison_key] = True

        if not st.session_state.get(show_comparison_key, False):
            return

    metric_key = TEST_MEAN_SCORE_KEY
    n_test_from_run: Optional[int] = None
    n_seeds_for_caption: Optional[int] = None
    with st.spinner("Fetching wandb run history..."):
        with profile("comparison_fetch_wandb_curves"):
            series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            for config_name, cfg in configs_to_compare:
                curves, diagnostic = _load_curves_for_config(
                    config_name,
                    cfg,
                    metric_key,
                    project_override=project,
                    entity_override=entity,
                )
                if not curves:
                    reason = diagnostic or "check train_dir and seeds"
                    st.warning(f"No wandb data for **{config_name}**: {reason}")
                    continue
                seeds = getattr(cfg, "seeds", None) or ["0"]
                if n_seeds_for_caption is None:
                    n_seeds_for_caption = len(seeds)
                # Get n_test from first run we see (same for all runs of same task)
                if n_test_from_run is None and getattr(cfg, "train_dir", None):
                    train_dir_base = cfg.train_dir or ""
                    ref_seed = seeds[0]
                    train_dir = get_train_dir_for_seed(train_dir_base, seed=ref_seed, reference_seed=ref_seed)
                    info = _get_run_id_and_meta_from_train_dir(pathlib.Path(train_dir))
                    if info:
                        run_id, run_entity, run_project = info
                        n_test_from_run = _get_n_test_from_run(
                            run_id, entity or run_entity, project or run_project
                        )
                steps, mean, std = _align_and_aggregate_seeds(curves)
                display_name = getattr(cfg, "name", config_name)
                series.append((display_name, steps, mean, std))

    if not series:
        st.error("Could not load wandb history for any config. Check wandb_project, train_dir, and that runs exist.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for the comparison plot.")
        return

    fig = go.Figure()
    # (line color, fill rgba)
    colors = [
        ("rgb(31, 119, 180)", "rgba(31, 119, 180, 0.2)"),
        ("rgb(255, 127, 14)", "rgba(255, 127, 14, 0.2)"),
        ("rgb(44, 160, 44)", "rgba(44, 160, 44, 0.2)"),
        ("rgb(214, 39, 40)", "rgba(214, 39, 40, 0.2)"),
        ("rgb(148, 103, 189)", "rgba(148, 103, 189, 0.2)"),
        ("rgb(140, 86, 75)", "rgba(140, 86, 75, 0.2)"),
    ]
    for i, (label, steps, mean, std) in enumerate(series):
        line_c, fill_c = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                name=label,
                line=dict(color=line_c, width=2),
                mode="lines",
            )
        )
        if np.any(np.isfinite(std)):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate([mean + std, (mean - std)[::-1]]),
                    fill="toself",
                    fillcolor=fill_c,
                    line=dict(width=0),
                    showlegend=False,
                )
            )
    fig.update_layout(
        title="test/mean_score over training (mean ± std across seeds)",
        xaxis_title="Step",
        yaxis_title="test/mean_score",
        hovermode="x unified",
        height=500,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explain how many rollouts each point uses
    if n_test_from_run is not None and n_seeds_for_caption is not None:
        st.caption(f"**Rollouts per data point:** {n_test_from_run} × {n_seeds_for_caption}")
    elif n_test_from_run is not None:
        st.caption(f"**Rollouts per data point:** {n_test_from_run} × n_seeds")
