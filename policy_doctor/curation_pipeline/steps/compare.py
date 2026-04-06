"""Compare baseline vs curated success rates — pipeline step class."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import numpy as np
import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.paths import iv_task_configs_base


class CompareStep(PipelineStep[Dict[str, Any]]):
    """Load eval logs for baseline and curated runs and print a comparison table.

    Result: ``{"baseline": {seed: score, ...}, "curated": {seed: score, ...}}``
    """

    name = "compare"

    def compute(self) -> Dict[str, Any]:
        from influence_visualizer.data_loader import get_eval_dir_for_seed
        from policy_doctor.curation_pipeline.paths import get_train_name
        from policy_doctor.curation_pipeline.config import load_baseline_config

        cfg = self.cfg
        task_config = OmegaConf.select(cfg, "task_config")
        experiment_name = (
            OmegaConf.select(cfg, "experiment_name")
            or OmegaConf.select(cfg, "train_date")
            or "default"
        )

        base = iv_task_configs_base(self.repo_root)
        with open(base / f"{task_config}.yaml") as f:
            task_cfg = yaml.safe_load(f)

        baseline_eval_dir = task_cfg["eval_dir"]
        reference_seed = str(OmegaConf.select(cfg, "reference_seed") or 0)
        seeds = OmegaConf.select(cfg, "seeds") or OmegaConf.select(cfg, "policy_seeds") or [0, 1, 2]
        seeds = [str(s) for s in seeds]

        try:
            baseline_cfg = load_baseline_config("robomimic", "low_dim", "transport_mh")
            policy = baseline_cfg["policy"]
        except Exception:
            policy = "diffusion_unet_lowdim"

        filter_ratio_val = OmegaConf.select(cfg, "filter_ratio")
        filter_ratio = filter_ratio_val if filter_ratio_val is not None else 0.50
        select_ratio_val = OmegaConf.select(cfg, "select_ratio")
        select_ratio = select_ratio_val if select_ratio_val is not None else 0.00
        curation_method = OmegaConf.select(cfg, "curation_method") or "influence_sum_official"
        run_tag = OmegaConf.select(cfg, "run_tag")
        train_ckpt = OmegaConf.select(cfg, "train_ckpt") or "best"
        task_name = OmegaConf.select(cfg, "task") or "transport_mh"

        if self.dry_run:
            print(f"[dry_run] CompareStep experiment={experiment_name}")
            return {}

        results: Dict[str, Dict[str, Any]] = {"baseline": {}, "curated": {}}

        for seed in seeds:
            eval_dir_seed = get_eval_dir_for_seed(baseline_eval_dir, seed, reference_seed)
            eval_log = self.repo_root / eval_dir_seed / "eval_log.json"
            if eval_log.exists():
                with open(eval_log) as f:
                    results["baseline"][seed] = json.load(f).get("test/mean_score")
            else:
                results["baseline"][seed] = None

            train_name = get_train_name(experiment_name, task_name, policy, seed)
            curated_name = f"{train_name}-curation_{curation_method}-filter_{filter_ratio}-select_{select_ratio}"
            if run_tag:
                curated_name = f"{curated_name}-{run_tag}"
            curated_eval_log = (
                self.repo_root / "data" / "outputs" / "eval_save_episodes"
                / experiment_name / curated_name / train_ckpt / "eval_log.json"
            )
            if curated_eval_log.exists():
                with open(curated_eval_log) as f:
                    results["curated"][seed] = json.load(f).get("test/mean_score")
            else:
                results["curated"][seed] = None

        self._print_table(results, experiment_name, task_config, filter_ratio)
        return results

    def _print_table(
        self,
        results: Dict[str, Any],
        experiment_name: str,
        task_config: str,
        filter_ratio: float,
    ) -> None:
        seeds = list(results["baseline"])
        print("\n" + "=" * 60)
        print(f"RESULTS: {experiment_name}")
        print(f"Task: {task_config}")
        print(f"Mode: curation_{'filtering' if filter_ratio > 0 else 'selection'}")
        print("-" * 60)
        print(f"{'Seed':<8} {'Baseline':>12} {'Curated':>12} {'Delta':>12}")
        print("-" * 60)

        b_vals, c_vals = [], []
        for seed in seeds:
            b = results["baseline"].get(seed)
            c = results["curated"].get(seed)
            b_str = f"{b:.2%}" if b is not None else "N/A"
            c_str = f"{c:.2%}" if c is not None else "N/A"
            d_str = f"{c - b:+.2%}" if (b is not None and c is not None) else "N/A"
            print(f"{seed:<8} {b_str:>12} {c_str:>12} {d_str:>12}")
            if b is not None:
                b_vals.append(b)
            if c is not None:
                c_vals.append(c)

        print("-" * 60)
        if b_vals:
            mb = np.mean(b_vals)
            mc = np.mean(c_vals) if c_vals else None
            mc_str = f"{mc:.2%}" if mc is not None else "N/A"
            delta_str = f"{mc - mb:+.2%}" if mc is not None else "N/A"
            mb_str = f"{mb:.2%}"
            print(f"{'Mean':<8} {mb_str:>12} {mc_str:>12} {delta_str:>12}")
        print("=" * 60)


