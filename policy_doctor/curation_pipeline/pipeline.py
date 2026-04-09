"""CurationPipeline: orchestrates pipeline steps with Hydra configs and run-folder persistence."""

from __future__ import annotations

import datetime
import pathlib
from typing import Any, Dict, List, Optional, Type

from omegaconf import DictConfig, OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.paths import PACKAGE_ROOT, REPO_ROOT

_REPO_ROOT = REPO_ROOT

# Ordered list of all steps — default execution order.
ALL_STEPS: List[str] = [
    "train_baseline",
    "eval_policies",
    "train_attribution",
    "finalize_attribution",
    "compute_demonstration_scores",
    "compute_infembed",
    "run_clustering",
    "export_markov_report",
    "annotate_slices_vlm",
    "summarize_behaviors_vlm",
    "evaluate_cluster_coherency_vlm",
    "run_curation_config",
    "train_curated",
    "eval_curated",
    "compare",
]


def _build_step_registry() -> Dict[str, Type[PipelineStep]]:
    """Import step classes lazily to avoid heavy top-level imports."""
    from policy_doctor.curation_pipeline.steps.train_baseline import TrainBaselineStep
    from policy_doctor.curation_pipeline.steps.eval_policies import EvalPoliciesStep
    from policy_doctor.curation_pipeline.steps.train_attribution import TrainAttributionStep
    from policy_doctor.curation_pipeline.steps.finalize_attribution import FinalizeAttributionStep
    from policy_doctor.curation_pipeline.steps.compute_demonstration_scores import ComputeDemonstrationScoresStep
    from policy_doctor.curation_pipeline.steps.compute_infembed import ComputeInfembedStep
    from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep
    from policy_doctor.curation_pipeline.steps.export_markov_report import ExportMarkovReportStep
    from policy_doctor.curation_pipeline.steps.annotate_slices_vlm import AnnotateSlicesVLMStep
    from policy_doctor.curation_pipeline.steps.summarize_behaviors_vlm import SummarizeBehaviorsVLMStep
    from policy_doctor.curation_pipeline.steps.evaluate_cluster_coherency_vlm import (
        EvaluateClusterCoherencyVLMStep,
    )
    from policy_doctor.curation_pipeline.steps.run_curation_config import RunCurationConfigStep
    from policy_doctor.curation_pipeline.steps.train_curated import TrainCuratedStep
    from policy_doctor.curation_pipeline.steps.eval_curated import EvalCuratedStep
    from policy_doctor.curation_pipeline.steps.compare import CompareStep

    return {
        "train_baseline": TrainBaselineStep,
        "eval_policies": EvalPoliciesStep,
        "train_attribution": TrainAttributionStep,
        "finalize_attribution": FinalizeAttributionStep,
        "compute_demonstration_scores": ComputeDemonstrationScoresStep,
        "compute_infembed": ComputeInfembedStep,
        "run_clustering": RunClusteringStep,
        "export_markov_report": ExportMarkovReportStep,
        "annotate_slices_vlm": AnnotateSlicesVLMStep,
        "summarize_behaviors_vlm": SummarizeBehaviorsVLMStep,
        "evaluate_cluster_coherency_vlm": EvaluateClusterCoherencyVLMStep,
        "run_curation_config": RunCurationConfigStep,
        "train_curated": TrainCuratedStep,
        "eval_curated": EvalCuratedStep,
        "compare": CompareStep,
    }


class CurationPipeline:
    """Orchestrate curation pipeline steps with Hydra configs and run-folder persistence.

    The pipeline creates a *run folder* (``cfg.run_dir``) that persists:

    - ``pipeline_config.yaml``: snapshot of the config at launch time
    - ``<step_name>/done``: sentinel written when a step completes
    - ``<step_name>/result.json``: structured result for steps that return data

    This lets a partially-complete pipeline be resumed by running it again with
    ``skip_if_done=True`` (the default) — completed steps are skipped and their
    cached results are loaded.

    Usage::

        from omegaconf import OmegaConf
        from policy_doctor.curation_pipeline.pipeline import CurationPipeline
        from policy_doctor.paths import PACKAGE_ROOT

        cfg = OmegaConf.load(str(PACKAGE_ROOT / "configs" / "config.yaml"))
        pipeline = CurationPipeline(cfg)
        pipeline.run(steps=["run_clustering", "run_curation_config", "train_curated"])
    """

    def __init__(self, cfg: DictConfig) -> None:
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        # Ensure repo_root is set
        if not OmegaConf.select(cfg, "repo_root"):
            OmegaConf.update(cfg, "repo_root", str(_REPO_ROOT), merge=True)

        # Resolve run_name
        run_name = OmegaConf.select(cfg, "run_name")
        if not run_name:
            run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            OmegaConf.update(cfg, "run_name", run_name, merge=True)

        # Resolve run_dir
        run_dir_str = OmegaConf.select(cfg, "run_dir")
        if not run_dir_str:
            run_dir_str = f"data/pipeline_runs/{run_name}"
            OmegaConf.update(cfg, "run_dir", run_dir_str, merge=True)

        run_dir = pathlib.Path(run_dir_str)
        if not run_dir.is_absolute():
            run_dir = pathlib.Path(OmegaConf.select(cfg, "repo_root")) / run_dir

        self.cfg: DictConfig = cfg
        self.run_dir: pathlib.Path = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        steps: Optional[List[str]] = None,
        skip_if_done: bool = True,
    ) -> Dict[str, Any]:
        """Execute pipeline steps in order, returning a mapping of step → result.

        Args:
            steps: Step names to run (default: ``ALL_STEPS``).
            skip_if_done: Skip steps whose ``done`` sentinel already exists.

        Returns:
            ``{step_name: result}`` for each executed step.
        """
        registry = _build_step_registry()
        steps = steps or ALL_STEPS

        unknown = [s for s in steps if s not in registry]
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}. Valid: {ALL_STEPS}")

        results: Dict[str, Any] = {}
        for step_name in steps:
            print(f"\n[Pipeline] ── {step_name}")
            step = registry[step_name](self.cfg, self.run_dir)
            results[step_name] = step.run(skip_if_done=skip_if_done)

        return results

    def step(self, name: str) -> PipelineStep:
        """Return an instantiated step object for manual interaction."""
        registry = _build_step_registry()
        if name not in registry:
            raise ValueError(f"Unknown step: {name!r}. Valid: {ALL_STEPS}")
        return registry[name](self.cfg, self.run_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_config(self) -> None:
        config_path = self.run_dir / "pipeline_config.yaml"
        if not config_path.exists():
            OmegaConf.save(self.cfg, config_path)

    def __repr__(self) -> str:
        return f"CurationPipeline(run_dir={self.run_dir})"
