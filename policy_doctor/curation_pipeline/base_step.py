"""Abstract base class for pipeline steps with compute / save / load semantics."""

from __future__ import annotations

import json
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from omegaconf import DictConfig, OmegaConf

from policy_doctor.paths import REPO_ROOT

T = TypeVar("T")

_REPO_ROOT = REPO_ROOT


class PipelineStep(ABC, Generic[T]):
    """Base class for all curation pipeline steps.

    Each step owns a ``step_dir`` inside the pipeline run folder where it
    persists its result.  The ``run()`` method handles skip-if-done logic so
    that a partially-completed pipeline can be resumed without re-running
    expensive steps.

    Subclasses must define:
      - ``name: str`` class attribute (used as the sub-directory name)
      - ``compute() -> T`` method (does the actual work)

    Subclasses may override:
      - ``save(result)`` to customise persistence (default: JSON dump)
      - ``load() -> T`` to customise loading (default: JSON load)

    Args:
        cfg:            Hydra config for the pipeline run.
        run_dir:        Directory this step writes its results into.
                        For top-level steps this is the pipeline run root.
                        For steps inside a :class:`CompositeStep` this is the
                        composite's ``step_dir``, so results are namespaced
                        under ``<run_root>/<composite_name>/<step_name>/``.
        parent_run_dir: Top-level pipeline run root.  Used by steps that need
                        to read results from *sibling* top-level steps (e.g.
                        ``SelectMimicgenSeedStep`` reading ``RunClusteringStep``).
                        Defaults to *run_dir* for top-level steps.
    """

    name: str  # must be set by every concrete subclass

    def __init__(
        self,
        cfg: DictConfig,
        run_dir: pathlib.Path,
        parent_run_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.parent_run_dir = parent_run_dir if parent_run_dir is not None else run_dir
        self.step_dir = run_dir / self.name

    # ------------------------------------------------------------------
    # Properties derived from config
    # ------------------------------------------------------------------

    @property
    def repo_root(self) -> pathlib.Path:
        root = OmegaConf.select(self.cfg, "repo_root")
        return pathlib.Path(root) if root else _REPO_ROOT

    @property
    def dry_run(self) -> bool:
        return bool(OmegaConf.select(self.cfg, "dry_run"))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self) -> T:
        """Run the computation and return the result."""

    def save(self, result: T) -> None:
        """Persist *result* to ``step_dir`` and write a ``done`` sentinel.

        The ``done`` sentinel is intentionally **not** written for dry runs so
        that re-running with real data does not skip the step.
        """
        self.step_dir.mkdir(parents=True, exist_ok=True)
        if result is not None:
            with open(self.step_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
        if not self.dry_run:
            (self.step_dir / "done").touch()

    def load(self) -> Optional[T]:
        """Load a previously persisted result, or ``None`` if not found."""
        result_path = self.step_dir / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)
        return None

    def is_done(self) -> bool:
        """Return ``True`` if the step has been completed and saved."""
        return (self.step_dir / "done").exists()

    def run(self, skip_if_done: bool = True) -> T:
        """Execute the step, optionally skipping if already completed.

        If *skip_if_done* is ``True`` and ``is_done()`` returns ``True``, the
        saved result is loaded and returned without re-running ``compute()``.
        """
        if skip_if_done and self.is_done():
            print(f"  [{self.name}] skipped (already done — loading cached result)")
            return self.load()
        result = self.compute()
        self.save(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_in_process(self, target: Any, kwargs: dict) -> None:
        """Run *target*(**kwargs) in an isolated child process via ``multiprocessing``.

        Only use this for Hydra-based training steps where ``hydra.initialize_config_dir``
        cannot be called more than once per process (global singleton state).
        All other steps should call Python functions directly in-process.

        Raises ``RuntimeError`` if the child exits with a non-zero code.
        """
        import multiprocessing as mp

        # Use 'spawn' to avoid fork-after-multithreaded-parent deadlocks (CUDA, WandB).
        p = mp.get_context("spawn").Process(target=target, kwargs=kwargs, daemon=False)
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"[{self.name}] subprocess failed with exit code {p.exitcode}"
            )


class CompositeStep(PipelineStep[dict]):
    """A pipeline step that groups a fixed sequence of sub-steps under one namespace.

    Sub-steps write their results to::

        <run_dir>/<composite_name>/<sub_step_name>/

    so that multiple composite steps sharing the same top-level ``run_dir`` each
    have their own isolated output tree.  Sub-steps can still read results from
    *sibling* top-level steps (e.g. ``RunClusteringStep``) via ``parent_run_dir``,
    which is transparently set to the top-level run root.

    Resumability: each sub-step honours its own ``done`` sentinel, so a partially
    completed arm can be resumed without re-running earlier sub-steps.  The
    composite's own ``done`` sentinel is written only after all sub-steps finish.

    Subclasses must define:
        name               str — directory name for this arm (e.g. ``"mimicgen_random"``).
        sub_step_classes   list[type[PipelineStep]] — ordered sub-step classes to run.

    Subclasses may define:
        cfg_overrides      dict[str, Any] — dotpath→value pairs applied to the cfg
                           before any sub-step runs.  Use this to fix heuristic
                           choices or other arm-specific settings without requiring
                           separate experiment configs.
    """

    sub_step_classes: list = []
    cfg_overrides: dict = {}

    def compute(self) -> dict:
        import copy
        from omegaconf import OmegaConf

        # Build arm-specific config by applying overrides on top of the shared cfg.
        sub_cfg = copy.deepcopy(self.cfg)
        for dotpath, value in (self.cfg_overrides or {}).items():
            OmegaConf.update(sub_cfg, dotpath, value, merge=True)

        # Sub-steps write into this composite's step_dir; cross-boundary lookups
        # (e.g. RunClusteringStep) resolve against the top-level run root.
        # self.parent_run_dir is the top-level root whether this composite is
        # a direct child of the pipeline or nested inside another step.
        sub_run_dir = self.step_dir
        parent_run_dir = self.parent_run_dir

        results: dict = {}
        for cls in self.sub_step_classes:
            step = cls(sub_cfg, sub_run_dir, parent_run_dir=parent_run_dir)
            result = step.run(skip_if_done=True)
            results[cls.name] = result

        return results
