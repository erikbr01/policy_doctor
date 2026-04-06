"""Summarize behavior clusters from slice-level VLM captions (text-only path on backend)."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds


class SummarizeBehaviorsVLMStep(PipelineStep[Dict[str, Any]]):
    """Group slice annotations by ``cluster_id`` and produce one summary per cluster.

    Reads ``annotate_slices_vlm/annotations_seed*.jsonl`` from the same pipeline
    ``run_dir`` (prior :class:`AnnotateSlicesVLMStep`). Writes
    ``behavior_summaries_seed{N}.json`` under this step's directory.
    """

    name = "summarize_behaviors_vlm"

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
        from policy_doctor.curation_pipeline.steps.annotate_slices_vlm import AnnotateSlicesVLMStep
        from policy_doctor.vlm import get_vlm_backend
        from policy_doctor.vlm.behavior_summarize import (
            load_slice_annotations_jsonl,
            run_behavior_summarization,
            write_behavior_summaries_json,
        )
        from policy_doctor.vlm.prompts import (
            default_task_hint_for_vlm,
            resolve_vlm_prompts_file_for_task,
        )

        cfg = self.cfg
        va = OmegaConf.select(cfg, "vlm_annotation") or {}
        va_dict = (
            OmegaConf.to_container(va, resolve=True) or {}
            if isinstance(va, DictConfig)
            else (dict(va) if va else {})
        )
        bs = OmegaConf.select(cfg, "vlm_behavior_summarization") or {}
        bs_dict = (
            OmegaConf.to_container(bs, resolve=True) or {}
            if isinstance(bs, DictConfig)
            else (dict(bs) if bs else {})
        )

        backend_name = bs_dict.get("backend") or va_dict.get("backend", "mock")
        bp = bs_dict.get("backend_params")
        if bp is None:
            backend_params = va_dict.get("backend_params") or {}
        elif isinstance(bp, dict):
            backend_params = bp
        else:
            backend_params = OmegaConf.to_container(bp, resolve=True) or {}

        task_cfg = OmegaConf.select(cfg, "task_config")
        task_cfg_str = str(task_cfg) if task_cfg else None
        explicit_bs_pf = bs_dict.get("prompts_file")
        explicit_va_pf = va_dict.get("prompts_file")
        explicit_pf = (
            explicit_bs_pf
            if explicit_bs_pf is not None
            and str(explicit_bs_pf).strip()
            and str(explicit_bs_pf).strip().lower() not in ("null", "none")
            else explicit_va_pf
        )
        prompts_file = resolve_vlm_prompts_file_for_task(
            explicit_pf,
            task_cfg_str,
            repo_root=self.repo_root,
        )

        th_bs = bs_dict.get("task_hint")
        th_va = va_dict.get("task_hint")
        th_raw = th_bs if th_bs is not None and str(th_bs).strip() and str(th_bs).strip().lower() not in ("null", "none") else th_va
        if th_raw is not None and str(th_raw).strip() and str(th_raw).strip().lower() not in ("null", "none"):
            task_hint = str(th_raw).strip()
        else:
            hint = default_task_hint_for_vlm(explicit_pf, task_cfg_str, repo_root=self.repo_root)
            task_hint = hint or task_cfg_str or "robot task"

        max_labels = bs_dict.get("max_slice_labels_per_cluster")
        max_clusters = bs_dict.get("max_clusters")

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

        if self.dry_run:
            print(
                f"[dry_run] SummarizeBehaviorsVLMStep backend={backend_name} seeds={seeds}"
            )
            return {"backend": backend_name, "behavior_prompt_version": "dry_run", "per_seed": {}}

        annotate_dir = self.run_dir / AnnotateSlicesVLMStep.name
        prior_ann = AnnotateSlicesVLMStep(cfg, self.run_dir).load()

        self.step_dir.mkdir(parents=True, exist_ok=True)
        backend = get_vlm_backend(backend_name, backend_params)
        per_seed: Dict[str, Any] = {}
        first_pver: Optional[str] = None

        merged_prompts: Dict[str, Any] = {}
        if isinstance(va_dict.get("prompts"), dict):
            merged_prompts.update(va_dict["prompts"])
        if isinstance(bs_dict.get("prompts"), dict):
            merged_prompts.update(bs_dict["prompts"])
        # Empty dict is falsy in merge_* helpers; pass None to use file/defaults only.
        prompts_inline_merged: Optional[Dict[str, Any]] = (
            {"prompts": merged_prompts} if merged_prompts else None
        )

        for seed in seeds:
            ann_rel = None
            if prior_ann:
                ann_rel = prior_ann.get("per_seed", {}).get(seed, {}).get("annotations_path")
            ann_path = annotate_dir / f"annotations_seed{seed}.jsonl"
            if ann_rel:
                ann_path = annotate_dir / pathlib.Path(str(ann_rel))
            if not ann_path.is_file():
                raise FileNotFoundError(
                    f"Missing slice annotations for seed {seed}: {ann_path}. "
                    f"Run annotate_slices_vlm first in this run_dir."
                )

            records = load_slice_annotations_jsonl(ann_path)
            summaries, pver = run_behavior_summarization(
                records,
                backend=backend,
                task_hint=task_hint,
                prompts_file=prompts_file,
                prompts_inline=prompts_inline_merged,
                repo_root=self.repo_root,
                max_slice_labels_per_cluster=max_labels,
                max_clusters=max_clusters,
            )
            if first_pver is None:
                first_pver = pver

            out_json = self.step_dir / f"behavior_summaries_seed{seed}.json"
            write_behavior_summaries_json(out_json, summaries)
            per_seed[seed] = {
                "summaries_path": str(out_json.relative_to(self.step_dir)),
                "num_clusters": len(summaries),
                "num_slice_records": len(records),
                "source_annotations": str(ann_path.relative_to(self.run_dir)),
                "behavior_prompt_version": pver,
            }

        return {
            "backend": backend_name,
            "behavior_prompt_version": first_pver or "",
            "per_seed": per_seed,
        }
