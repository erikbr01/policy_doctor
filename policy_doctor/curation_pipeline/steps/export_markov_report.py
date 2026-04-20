"""Export Markov property test results for clustering in this pipeline run."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds
from policy_doctor.data.clustering_loader import load_clustering_result_from_path


class ExportMarkovReportStep(PipelineStep[Dict[str, Any]]):
    """Load clustering / graph outputs and write ``markov_report_seed{N}.json``.

    Loads node assignments from the first available source in priority order:

    1. :class:`BuildBehaviorGraphStep` — supports both ``cupid`` and ``enap``
       builders.  When present, its ``node_assignments_*.npy`` files are used
       directly.
    2. :class:`RunClusteringStep` result + clustering directory — backwards-
       compatible path for runs that do not include ``build_behavior_graph``.
    3. Explicit ``cfg.clustering_dir`` — for standalone usage outside the
       pipeline.

    Uses :func:`policy_doctor.behaviors.behavior_graph.test_markov_property` and
    :func:`markov_test_result_to_jsonable` for JSON-safe statistics.
    """

    name = "export_markov_report"

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

    def _resolve_clustering_path(self, raw: str) -> pathlib.Path:
        p = pathlib.Path(raw)
        if p.is_absolute():
            return p
        return (self.repo_root / p).resolve()

    def _load_labels_and_meta_for_seed(
        self,
        seed: str,
        bbg_prior: Optional[Dict[str, Any]],
        clustering_dirs: Dict[str, str],
    ) -> Tuple[np.ndarray, List[Dict], Dict, Optional[str]]:
        """Return ``(labels, metadata, manifest, source_description)`` for *seed*.

        Tries build_behavior_graph first, then falls back to clustering dirs.
        """
        bbg_step_dir = self.run_dir / "build_behavior_graph"

        # --- Priority 1: build_behavior_graph artifacts ---------------------
        if bbg_prior and bbg_prior.get("seeds"):
            seeds_info: Dict[str, Any] = bbg_prior["seeds"]
            # Try exact seed key match, then fall back to "_single" / "_enap"
            seed_info = (
                seeds_info.get(seed)
                or seeds_info.get(str(seed))
                or seeds_info.get("_single")
                or seeds_info.get("_enap")
            )
            if seed_info and isinstance(seed_info, dict):
                na_rel = seed_info.get("node_assignments_path")
                meta_rel = seed_info.get("metadata_path")
                if na_rel and meta_rel:
                    na_path = bbg_step_dir / na_rel
                    meta_path = bbg_step_dir / meta_rel
                    if na_path.exists() and meta_path.exists():
                        labels = np.load(na_path)
                        with open(meta_path) as f:
                            metadata = json.load(f)
                        manifest = {"level": seed_info.get("level", "rollout"), "source": "build_behavior_graph"}
                        return labels, metadata, manifest, f"build_behavior_graph (seed={seed})"

        # --- Priority 2: clustering directories ----------------------------
        raw_path = clustering_dirs.get(seed) or clustering_dirs.get(str(seed))
        if raw_path and raw_path not in (seed, str(seed)):
            cdir = self._resolve_clustering_path(raw_path)
            labels, metadata, manifest = load_clustering_result_from_path(cdir)
            return labels, metadata, manifest, str(cdir)

        raise KeyError(
            f"No node assignments found for seed {seed!r}. "
            "Run build_behavior_graph or run_clustering first, or set clustering_dir."
        )

    def compute(self) -> Dict[str, Any]:
        from policy_doctor.behaviors.behavior_graph import (
            markov_test_result_to_jsonable,
            test_markov_property,
        )
        from policy_doctor.curation_pipeline.config import load_task_config
        from policy_doctor.curation_pipeline.steps.build_behavior_graph import BuildBehaviorGraphStep
        from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

        cfg = self.cfg
        me = OmegaConf.select(cfg, "markov_export") or {}
        me_dict = (
            OmegaConf.to_container(me, resolve=True) or {}
            if isinstance(me, DictConfig)
            else (dict(me) if me else {})
        )

        significance_level = float(me_dict.get("significance_level") or 0.05)
        exclude_terminals = bool(me_dict.get("exclude_terminals") or False)
        method = str(me_dict.get("method") or "chi2")
        n_perm = int(me_dict.get("n_permutations") or 10000)
        random_state = me_dict.get("random_state")
        if random_state is not None:
            random_state = int(random_state)

        policy_seeds = (
            OmegaConf.select(cfg, "policy_seeds")
            or OmegaConf.select(cfg, "seeds")
            or [0]
        )
        seeds = expand_seeds(policy_seeds)

        # Collect data sources in priority order
        bbg_prior = BuildBehaviorGraphStep(cfg, self.run_dir).load()

        clustering_dirs: Dict[str, str] = {}
        if not (bbg_prior and bbg_prior.get("seeds")):
            # Fall back to run_clustering output
            rc_prior = RunClusteringStep(cfg, self.run_dir).load()
            if rc_prior and isinstance(rc_prior.get("clustering_dirs"), dict):
                clustering_dirs = {str(k): str(v) for k, v in rc_prior["clustering_dirs"].items()}

        explicit = OmegaConf.select(cfg, "clustering_dir")
        if explicit and not clustering_dirs and not (bbg_prior and bbg_prior.get("seeds")):
            for s in seeds:
                clustering_dirs[str(s)] = str(explicit)

        if self.dry_run:
            print(
                f"[dry_run] ExportMarkovReportStep seeds={seeds} "
                f"method={method} significance_level={significance_level}"
            )
            return {"per_seed": {}, "dry_run": True}

        has_sources = (bbg_prior and bbg_prior.get("seeds")) or clustering_dirs
        if not has_sources:
            raise ValueError(
                "No node assignments found. Run build_behavior_graph or run_clustering "
                "first, or set clustering_dir."
            )

        self.step_dir.mkdir(parents=True, exist_ok=True)

        env = OmegaConf.select(cfg, "env") or "robomimic"
        task = OmegaConf.select(cfg, "task")
        task_dims: Dict[str, Any] = {}
        if task:
            try:
                task_dims = load_task_config(str(env), str(task))
            except FileNotFoundError:
                task_dims = {}

        per_seed: Dict[str, Any] = {}
        for seed in seeds:
            labels, metadata, manifest, source = self._load_labels_and_meta_for_seed(
                seed=str(seed),
                bbg_prior=bbg_prior,
                clustering_dirs=clustering_dirs,
            )
            level = me_dict.get("level") or manifest.get("level") or "rollout"

            raw_result = test_markov_property(
                labels,
                metadata,
                level=str(level),
                significance_level=significance_level,
                exclude_terminals=exclude_terminals,
                method=method,
                n_permutations=n_perm,
                random_state=random_state,
            )
            payload = markov_test_result_to_jsonable(raw_result)
            payload["source"] = source
            payload["level"] = level
            payload["manifest"] = manifest
            payload["task"] = task
            if task_dims:
                payload["task_dims_ref"] = {
                    k: task_dims[k]
                    for k in ("obs_dim", "action_dim", "dataset_path")
                    if k in task_dims
                }

            out_path = self.step_dir / f"markov_report_seed{seed}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            per_seed[seed] = {
                "report_path": str(out_path.relative_to(self.step_dir)),
                "markov_holds": raw_result.get("markov_holds"),
                "num_states_tested": raw_result.get("num_states_tested"),
                "source": source,
            }

        return {
            "method": method,
            "significance_level": significance_level,
            "exclude_terminals": exclude_terminals,
            "per_seed": per_seed,
        }
