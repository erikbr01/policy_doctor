"""Scan training run folders for curation scatter / comparison plots.

Parses ``logs.json.txt`` (JSON lines with ``test/mean_score``), resolves dataset sizes from
Hydra ``.hydra/config.yaml`` and optional curation YAMLs, and filters runs by curation-config
location (IV vs policy_doctor).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from policy_doctor.curation.config import load_curation_config_from_path

TEST_MEAN_SCORE_KEY = "test/mean_score"


@dataclass(frozen=True)
class CurationScatterPoint:
    train_name: str
    checkpoint_name: str
    seed: str
    data_size: int
    sequence_pct_of_uncurated_train: float | None
    train_sequences_uncurated: int | None
    log_eval_tail_index: int
    log_eval_tail_count: int
    mean_rollout_score: float
    rollout_episodes_per_eval: int
    last_k_scores: tuple[float, ...]
    ckpt_epoch: int | None
    ckpt_global_step: int | None
    alignment_warning: str | None
    offline_eval_score: float | None
    is_baseline: bool


def experiment_key_from_train_name(train_name: str, task_substring: str) -> str:
    """Group runs that share the same experiment (strip trailing policy seed before ``-curation``)."""
    if task_substring not in train_name:
        return train_name
    if "-curation" in train_name:
        head, tail = train_name.split("-curation", 1)
        m = re.match(r"^(.+)_(\d+)$", head)
        head_stem = m.group(1) if m else head
        return f"{head_stem}-curation{tail}"
    m = re.match(r"^(.+)_(\d+)$", train_name)
    if m:
        return m.group(1)
    return train_name


def _iter_train_run_dirs(train_root: Path) -> Iterator[tuple[Path, Path]]:
    if not train_root.is_dir():
        return
    for date_dir in sorted(train_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir()):
            if run_dir.is_dir():
                yield date_dir, run_dir


def _load_hydra_dict(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / ".hydra" / "config.yaml"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return None


def _walk_find_str_values(obj: Any, keys: frozenset[str]) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and v:
                out.append(str(v))
            out.extend(_walk_find_str_values(v, keys))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_walk_find_str_values(x, keys))
    return out


_CURATION_KEYS = frozenset({"sample_curation_config", "holdout_selection_config"})


def _resolve_repo_path(repo_root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _curation_config_allowed(path_str: str | None, repo_root: Path, only_iv: bool) -> bool:
    if not path_str or not str(path_str).strip():
        return True
    p = _resolve_repo_path(repo_root, path_str)
    try:
        rel = p.relative_to(repo_root.resolve())
        s = rel.as_posix()
    except ValueError:
        s = p.as_posix()
    if "influence_visualizer/" in s or s.startswith("influence_visualizer"):
        return True
    if "third_party/influence_visualizer/" in s:
        return True
    if "policy_doctor/policy_doctor/configs" in s or "policy_doctor/configs" in s:
        return not only_iv
    return True


def _slice_train_sequence_total(curation_path: Path) -> int | None:
    try:
        cfg = load_curation_config_from_path(curation_path)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    meta = cfg.metadata or {}
    if "total_raw_samples" in meta:
        return int(meta["total_raw_samples"])
    total = 0
    for s in cfg.slices:
        total += max(0, int(s.end) - int(s.start))
    return total if total > 0 else None


def _data_size_from_hydra(
    cfg: dict[str, Any],
    repo_root: Path,
    run_dir: Path,
) -> tuple[int, int | None, float | None]:
    """Return (effective_train_sequences, uncurated_sequences, pct_vs_uncurated)."""
    from policy_doctor.data.dataset_episode_ends import load_dataset_episode_ends

    paths = _walk_find_str_values(cfg, _CURATION_KEYS)
    effective = 0
    for ps in paths:
        cp = _resolve_repo_path(repo_root, ps)
        if cp.is_file():
            n = _slice_train_sequence_total(cp)
            if n is not None:
                effective = n
                break

    train_uncur: int | None = None
    pct: float | None = None

    if not paths:
        try:
            ends = load_dataset_episode_ends(run_dir, "latest", repo_root)
            effective = int(ends[-1]) if len(ends) > 0 else 0
        except Exception:
            effective = 0
        return effective, None, None

    if effective > 0:
        try:
            ends = load_dataset_episode_ends(run_dir, "latest", repo_root)
            train_uncur = int(ends[-1]) if len(ends) > 0 else None
        except Exception:
            train_uncur = None
        if train_uncur is not None and train_uncur > 0:
            pct = 100.0 * float(effective) / float(train_uncur)

    return effective, train_uncur, pct


def _rollout_n_test(cfg: dict[str, Any]) -> int:
    for key_path in (
        ("task", "env_runner", "n_test"),
        ("task", "n_test"),
    ):
        cur: Any = cfg
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if isinstance(cur, (int, float)):
            return int(cur)
    return 50


def _parse_log_tail_scores(log_path: Path, rollout_window: int) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and TEST_MEAN_SCORE_KEY in obj:
                    rows.append(obj)
    except OSError:
        return [], []
    if not rows:
        return [], []
    tail = rows[-rollout_window:] if len(rows) >= rollout_window else rows
    scores = [float(r[TEST_MEAN_SCORE_KEY]) for r in tail]
    return tail, scores


def _offline_eval_score(
    eval_save_root: Path | None,
    train_name: str,
    load_offline: bool,
) -> float | None:
    if not load_offline or eval_save_root is None or not eval_save_root.is_dir():
        return None
    for sub in eval_save_root.rglob("eval_log.json"):
        try:
            if train_name not in str(sub):
                continue
            with open(sub) as f:
                data = json.load(f)
            if isinstance(data, dict) and TEST_MEAN_SCORE_KEY in data:
                return float(data[TEST_MEAN_SCORE_KEY])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def collect_curation_scatter_points(
    repo_root: Path,
    train_root: Path | None = None,
    eval_save_root: Path | None = None,
    task_substring: str = "transport_mh",
    policy_substring: str = "diffusion_unet_lowdim",
    baseline_train_date: str = "jan28",
    rollout_window: int = 5,
    load_offline_eval: bool = True,
    only_influence_visualizer_configs: bool = True,
) -> list[CurationScatterPoint]:
    repo_root = repo_root.resolve()
    tr = (train_root or (repo_root / "data" / "outputs" / "train")).resolve()
    default_eval = repo_root / "data" / "outputs" / "eval_save_episodes"
    ev_base = eval_save_root if eval_save_root is not None else default_eval
    ev = ev_base.resolve() if ev_base.is_dir() else None

    points: list[CurationScatterPoint] = []
    for _date_dir, run_dir in _iter_train_run_dirs(tr):
        name = run_dir.name
        if task_substring not in name or policy_substring not in name:
            continue
        log_path = run_dir / "logs.json.txt"
        if not log_path.is_file():
            continue
        hydra = _load_hydra_dict(run_dir)
        if hydra is None:
            continue
        paths = _walk_find_str_values(hydra, _CURATION_KEYS)
        if paths and not all(
            _curation_config_allowed(p, repo_root, only_influence_visualizer_configs) for p in paths
        ):
            continue

        tail_rows, last_k_scores = _parse_log_tail_scores(log_path, rollout_window)
        if not tail_rows or not last_k_scores:
            continue

        data_size, train_uncur, seq_pct = _data_size_from_hydra(hydra, repo_root, run_dir)
        n_test = _rollout_n_test(hydra)
        is_baseline = baseline_train_date in name and "-curation" not in name and not paths

        m = re.search(r"_(\d+)(?:-curation|$)", name)
        seed = m.group(1) if m else "?"

        offline = _offline_eval_score(ev, name, load_offline_eval)

        for i, row in enumerate(tail_rows):
            epoch = row.get("epoch")
            gstep = row.get("global_step")
            points.append(
                CurationScatterPoint(
                    train_name=name,
                    checkpoint_name="latest",
                    seed=seed,
                    data_size=data_size,
                    sequence_pct_of_uncurated_train=seq_pct,
                    train_sequences_uncurated=train_uncur,
                    log_eval_tail_index=i,
                    log_eval_tail_count=len(tail_rows),
                    mean_rollout_score=float(row[TEST_MEAN_SCORE_KEY]),
                    rollout_episodes_per_eval=n_test,
                    last_k_scores=tuple(last_k_scores),
                    ckpt_epoch=int(epoch) if isinstance(epoch, (int, float)) else None,
                    ckpt_global_step=int(gstep) if isinstance(gstep, (int, float)) else None,
                    alignment_warning=None,
                    offline_eval_score=offline,
                    is_baseline=is_baseline,
                )
            )

    return points


__all__ = [
    "CurationScatterPoint",
    "TEST_MEAN_SCORE_KEY",
    "collect_curation_scatter_points",
    "experiment_key_from_train_name",
]
