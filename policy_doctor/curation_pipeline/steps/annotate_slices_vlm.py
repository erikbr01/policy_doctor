"""VLM slice annotation — optional pipeline step (modular backend)."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds
from policy_doctor.paths import PACKAGE_ROOT, iv_task_configs_base


def _coerce_max_frames_per_slice(raw: Any) -> Optional[int]:
    """``None`` / non-positive / ``null`` → use every timestep in the window."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("null", "none", ""):
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


class AnnotateSlicesVLMStep(PipelineStep[Dict[str, Any]]):
    """Annotate rollout sliding-window slices with a pluggable VLM.

    Writes per-seed ``annotations_seed{N}.jsonl`` under the step directory and a
    small ``result.json`` summary. Depends on eval rollouts (``episodes/*.pkl``
    with ``img``) and a clustering result directory for the same seed when using
    per-seed clustering outputs from ``run_clustering``.
    """

    name = "annotate_slices_vlm"

    def save(self, result: Dict[str, Any]) -> None:
        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        (self.step_dir / "done").touch()

    def load(self) -> Optional[Dict[str, Any]]:
        p = self.step_dir / "result.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def compute(self) -> Dict[str, Any]:
        from influence_visualizer.data_loader import get_eval_dir_for_seed

        from influence_visualizer.clustering_results import get_clustering_dir as iv_clustering_dir
        from policy_doctor.vlm import get_vlm_backend
        from policy_doctor.vlm.annotate import run_slice_annotation_for_eval, write_annotations_jsonl
        from policy_doctor.vlm.prompts import (
            default_task_hint_for_vlm,
            resolve_vlm_prompts_file_for_task,
        )

        cfg = self.cfg
        va = OmegaConf.select(cfg, "vlm_annotation") or {}
        if isinstance(va, DictConfig):
            va_dict = OmegaConf.to_container(va, resolve=True) or {}
        else:
            va_dict = dict(va) if va else {}

        backend_name = va_dict.get("backend", "mock")
        backend_params = va_dict.get("backend_params") or {}
        task_config = OmegaConf.select(cfg, "task_config")
        task_config_str = str(task_config) if task_config else None
        explicit_pf = va_dict.get("prompts_file")
        prompts_file = resolve_vlm_prompts_file_for_task(
            explicit_pf,
            task_config_str,
            repo_root=self.repo_root,
        )
        max_slices = va_dict.get("max_slices")
        max_frames = _coerce_max_frames_per_slice(va_dict.get("max_frames_per_slice"))
        random_seed = int(va_dict.get("random_seed", 42))
        th_raw = va_dict.get("task_hint")
        if th_raw is not None and str(th_raw).strip() and str(th_raw).strip().lower() not in ("null", "none"):
            task_hint = str(th_raw).strip()
        else:
            hint = default_task_hint_for_vlm(explicit_pf, task_config_str, repo_root=self.repo_root)
            task_hint = hint or task_config_str or "robot task"
        config_root = OmegaConf.select(cfg, "config_root") or "iv"
        reference_seed = str(OmegaConf.select(cfg, "reference_seed") or 0)

        if config_root == "iv":
            base = iv_task_configs_base(self.repo_root)
        else:
            base = PACKAGE_ROOT / "configs"

        task_yaml = base / f"{task_config}.yaml"
        with open(task_yaml) as f:
            task_cfg = yaml.safe_load(f)

        eval_dir_base = task_cfg["eval_dir"]

        policy_seeds = (
            OmegaConf.select(cfg, "policy_seeds")
            or OmegaConf.select(cfg, "seeds")
            or [0]
        )
        if isinstance(policy_seeds, (int, float)):
            policy_seeds = [int(policy_seeds)]
        else:
            policy_seeds = list(policy_seeds)
        seeds = expand_seeds(policy_seeds)

        explicit_cdir = va_dict.get("clustering_dir") or OmegaConf.select(cfg, "clustering_dir")
        clustering_name = va_dict.get("clustering_name") or OmegaConf.select(cfg, "clustering_name")

        clustering_dirs_map: Dict[str, str] = {}
        if not explicit_cdir:
            from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

            prior = RunClusteringStep(cfg, self.run_dir).load()
            if prior:
                clustering_dirs_map = dict(prior.get("clustering_dirs", {}))

        if self.dry_run:
            mf_dbg = "all_in_window" if max_frames is None else str(max_frames)
            print(
                f"[dry_run] AnnotateSlicesVLMStep backend={backend_name} seeds={seeds} "
                f"max_slices={max_slices} max_frames_per_slice={mf_dbg} "
                f"reasoning_effort={va_dict.get('reasoning_effort', 'none')}"
            )
            return {
                "backend": backend_name,
                "prompt_version": "dry_run",
                "per_seed": {},
            }

        self.step_dir.mkdir(parents=True, exist_ok=True)
        backend = get_vlm_backend(backend_name, backend_params)
        per_seed: Dict[str, Any] = {}
        first_pver: Optional[str] = None
        save_debug_plots = bool(va_dict.get("save_debug_plots", False))

        for seed in seeds:
            eval_dir_seed = get_eval_dir_for_seed(eval_dir_base, seed, reference_seed)
            eval_abs = self.repo_root / eval_dir_seed

            if explicit_cdir:
                cpath = pathlib.Path(explicit_cdir)
                if not cpath.is_absolute():
                    cpath = self.repo_root / cpath
            elif clustering_dirs_map.get(seed):
                cpath = pathlib.Path(clustering_dirs_map[seed])
                if not cpath.is_absolute():
                    cpath = self.repo_root / cpath
            elif clustering_name:
                cdir_root = iv_clustering_dir(str(task_config))
                cpath = pathlib.Path(cdir_root) / str(clustering_name)
            else:
                raise ValueError(
                    "Set vlm_annotation.clustering_dir, clustering_name, or run run_clustering "
                    "in the same pipeline run so clustering_dirs are available."
                )

            debug_dir: Optional[pathlib.Path] = None
            if save_debug_plots:
                debug_dir = self.step_dir / "debug_plots" / f"seed_{seed}"
                debug_dir.mkdir(parents=True, exist_ok=True)

            records, pver = run_slice_annotation_for_eval(
                eval_dir=eval_abs,
                clustering_dir=cpath,
                backend=backend,
                task_hint=task_hint,
                prompts_file=prompts_file,
                prompts_inline=va_dict,
                repo_root=self.repo_root,
                max_slices=max_slices,
                max_frames_per_slice=max_frames,
                random_seed=random_seed + int(seed),
                debug_plots_dir=debug_dir,
            )
            if first_pver is None:
                first_pver = pver

            out_file = self.step_dir / f"annotations_seed{seed}.jsonl"
            write_annotations_jsonl(out_file, records)
            seed_info: Dict[str, Any] = {
                "annotations_path": str(out_file.relative_to(self.step_dir)),
                "num_annotated": len(records),
                "eval_dir": str(eval_abs),
                "clustering_dir": str(cpath),
                "prompt_version": pver,
            }
            if debug_dir is not None:
                seed_info["debug_plots_dir"] = str(debug_dir.relative_to(self.step_dir))
            per_seed[seed] = seed_info

        return {
            "backend": backend_name,
            "prompt_version": first_pver or "",
            "per_seed": per_seed,
        }
