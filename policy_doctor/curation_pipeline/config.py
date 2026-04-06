"""Load pipeline configs from the package ``configs/<env>/`` tree with override support."""

import pathlib
from typing import Any, Dict, Optional

import yaml

# Policy doctor package root (parent of curation_pipeline)
_PD_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONFIGS_ROOT = _PD_ROOT / "configs"


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get_config_path_for_task(env: str, subdir: str, state: str, task: str) -> pathlib.Path:
    """e.g. robomimic, baseline, low_dim, lift_mh -> configs/robomimic/baseline/low_dim/lift_mh.yaml"""
    base = _CONFIGS_ROOT / env / subdir / state
    path = base / f"{task}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def _merge_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return cfg
    out = dict(cfg)
    for k, v in overrides.items():
        if v is None and k in out and out[k] is not None:
            # null/None override means "not set" — keep the YAML default
            continue
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _merge_overrides(out[k], v)
        else:
            out[k] = v
    return out


def load_task_config(env: str, task: str) -> Dict[str, Any]:
    """Load task-level defaults from configs/<env>/tasks/<task>.yaml."""
    path = _CONFIGS_ROOT / env / "tasks" / f"{task}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {path}")
    return _load_yaml(path)


def load_baseline_config(
    env: str,
    state: str,
    task: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load baseline training config for (env, state, task)."""
    path = _get_config_path_for_task(env, "baseline", state, task)
    cfg = _load_yaml(path)
    task_cfg = load_task_config(env, task)
    cfg.setdefault("task", task_cfg.get("task", task))
    for k in ("dataset_path", "obs_dim", "action_dim", "task_name"):
        if k in task_cfg and k not in cfg:
            cfg[k] = task_cfg[k]
    return _merge_overrides(cfg, overrides)


def load_eval_config(
    env: str,
    state: str,
    task: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load evaluation config (eval_save_episodes params)."""
    path = _get_config_path_for_task(env, "evaluation", state, task)
    cfg = _load_yaml(path)
    return _merge_overrides(cfg, overrides)


def load_attribution_config(
    env: str,
    state: str,
    task: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load attribution config (train_trak, finalize_trak, eval_demonstration_scores)."""
    path = _get_config_path_for_task(env, "attribution", state, task)
    cfg = _load_yaml(path)
    return _merge_overrides(cfg, overrides)


def load_curated_training_config(
    env: str,
    state: str,
    task: str,
    mode: str = "curation_selection",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load curated training config (filtering or selection).

    mode: 'curation_filtering' | 'curation_selection'
    """
    if mode not in ("curation_filtering", "curation_selection"):
        raise ValueError(f"mode must be curation_filtering or curation_selection, got {mode}")
    path = _get_config_path_for_task(env, mode, state, task)
    cfg = _load_yaml(path)
    # Resolve baseline_ref into baseline config for path resolution
    ref = cfg.get("baseline_ref")
    if ref:
        base = load_baseline_config(
            env,
            ref.get("state", state),
            ref.get("task", task),
        )
        cfg["_baseline_config"] = base
        cfg.setdefault("train_date", ref.get("train_date"))
        cfg.setdefault("task", ref.get("task", task))
        cfg.setdefault("state", ref.get("state", state))
        cfg.setdefault("method", ref.get("method"))
    return _merge_overrides(cfg, overrides)


def get_pipeline_config_path() -> pathlib.Path:
    """Path to pipeline/config.yaml for run_pipeline (clustering -> curation config)."""
    return _PD_ROOT / "configs" / "pipeline" / "config.yaml"
