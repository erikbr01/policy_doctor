"""Experiment E1: VLM-based cluster coherence classification pipeline step."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds


class ValidateClusterCoherenceVLMStep(PipelineStep[Dict[str, Any]]):
    """Run Experiment E1: classify held-out slices into their influence-derived cluster.

    Reads:
      - ``run_clustering/result.json`` (or explicit ``vlm_cluster_classification.clustering_dir``)
      - Eval episode pickles from the task config's ``eval_dir``

    Writes per-seed under ``validate_cluster_coherence_vlm/<seed>/``:
      - ``sample_plan.json``  — pre-committed sampling (fixed before any VLM call)
      - ``predictions.jsonl`` — per-query classification records
      - ``metrics.json``      — accuracy, confusion matrix, binomial test

    Config knobs (all under ``vlm_cluster_classification``):
      ``backend``                   VLM backend name (default ``mock``)
      ``backend_params``            Backend constructor kwargs
      ``n_example``                 Example slices per cluster (default 5)
      ``n_query``                   Query slices per cluster (default 5)
      ``n_repetitions``             Repetitions per query (default 3)
      ``max_frames_per_storyboard`` Frames per composite storyboard (default 4)
      ``random_seed``               Sampling RNG seed (default 42)
      ``max_clusters``              Cap number of clusters evaluated (default null)
      ``clustering_dir``            Explicit path; overrides ``run_clustering`` result
      ``system_prompt``             Override default system prompt
      ``user_preamble_template``    Override default preamble (use ``{n_groups}`` placeholder)
      ``user_prompt_question``      Override default classification question
    """

    name = "validate_cluster_coherence_vlm"

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

        from policy_doctor.vlm import get_vlm_backend
        from policy_doctor.vlm.cluster_classification import run_cluster_coherence_classification
        from policy_doctor.vlm.prompts import (
            default_task_hint_for_vlm,
            resolve_vlm_prompts_file_for_task,
        )
        from policy_doctor.paths import PACKAGE_ROOT, iv_task_configs_base
        import yaml

        cfg = self.cfg
        vcc_raw = OmegaConf.select(cfg, "vlm_cluster_classification") or {}
        vcc: Dict[str, Any] = (
            OmegaConf.to_container(vcc_raw, resolve=True) or {}
            if isinstance(vcc_raw, DictConfig)
            else (dict(vcc_raw) if vcc_raw else {})
        )

        # Also inherit backend from vlm_annotation when not explicitly set
        va_raw = OmegaConf.select(cfg, "vlm_annotation") or {}
        va: Dict[str, Any] = (
            OmegaConf.to_container(va_raw, resolve=True) or {}
            if isinstance(va_raw, DictConfig)
            else (dict(va_raw) if va_raw else {})
        )

        backend_name = vcc.get("backend") or va.get("backend", "mock")
        bp = vcc.get("backend_params") or va.get("backend_params") or {}

        n_example = int(vcc.get("n_example", 5))
        n_query = int(vcc.get("n_query", 5))
        n_repetitions = int(vcc.get("n_repetitions", 3))
        max_frames = int(vcc.get("max_frames_per_storyboard", 4))
        random_seed = int(vcc.get("random_seed", 42))
        max_clusters_raw = vcc.get("max_clusters")
        max_clusters = int(max_clusters_raw) if max_clusters_raw is not None else None
        global_episode_disjoint = bool(vcc.get("global_episode_disjoint", True))
        view_window_extension = int(vcc.get("view_window_extension", 0))
        include_action_text = bool(vcc.get("include_action_text", False))
        include_state_text = bool(vcc.get("include_state_text", False))
        storyboard_mode = str(vcc.get("storyboard_mode", "composite"))
        composite_target_size = int(vcc.get("composite_target_size", 768))
        query_storyboard_mode_raw = vcc.get("query_storyboard_mode")
        query_storyboard_mode = (
            str(query_storyboard_mode_raw)
            if query_storyboard_mode_raw is not None else None
        )

        system_prompt = vcc.get("system_prompt") or None
        user_preamble = vcc.get("user_preamble_template") or None
        user_question = vcc.get("user_prompt_question") or None

        task_config = OmegaConf.select(cfg, "task_config")
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
        seeds = expand_seeds(list(policy_seeds))

        # Clustering dir resolution: explicit > run_clustering result
        explicit_cdir = vcc.get("clustering_dir")
        clustering_dirs_map: Dict[str, str] = {}
        if not explicit_cdir:
            from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

            prior = RunClusteringStep(cfg, self.run_dir).load()
            if prior:
                clustering_dirs_map = dict(prior.get("clustering_dirs", {}))

        if self.dry_run:
            print(
                f"[dry_run] ValidateClusterCoherenceVLMStep backend={backend_name} "
                f"seeds={seeds} n_example={n_example} n_query={n_query} "
                f"n_reps={n_repetitions} max_frames={max_frames}"
            )
            return {"backend": backend_name, "per_seed": {}, "dry_run": True}

        backend = get_vlm_backend(backend_name, bp)
        per_seed: Dict[str, Any] = {}

        for seed in seeds:
            if explicit_cdir:
                cpath = pathlib.Path(explicit_cdir)
                if not cpath.is_absolute():
                    cpath = self.repo_root / cpath
            elif clustering_dirs_map.get(seed):
                cpath = pathlib.Path(clustering_dirs_map[seed])
                if not cpath.is_absolute():
                    cpath = self.repo_root / cpath
            else:
                raise ValueError(
                    f"No clustering dir found for seed {seed}. "
                    "Either run run_clustering first or set vlm_cluster_classification.clustering_dir."
                )

            eval_dir_seed = get_eval_dir_for_seed(eval_dir_base, seed, reference_seed)
            eval_abs = self.repo_root / eval_dir_seed

            seed_dir = self.step_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            summary = run_cluster_coherence_classification(
                clustering_dir=cpath,
                eval_dir=eval_abs,
                backend=backend,
                n_example=n_example,
                n_query=n_query,
                n_repetitions=n_repetitions,
                max_frames_per_storyboard=max_frames,
                random_seed=random_seed + int(seed),
                step_dir=seed_dir,
                system_prompt=system_prompt,
                user_preamble_template=user_preamble,
                user_prompt_question=user_question,
                max_clusters=max_clusters,
                global_episode_disjoint=global_episode_disjoint,
                view_window_extension=view_window_extension,
                include_action_text=include_action_text,
                include_state_text=include_state_text,
                storyboard_mode=storyboard_mode,
                composite_target_size=composite_target_size,
                query_storyboard_mode=query_storyboard_mode,
            )
            per_seed[seed] = summary

        return {
            "backend": backend_name,
            "per_seed": per_seed,
        }
