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
    """

    name: str  # must be set by every concrete subclass

    def __init__(self, cfg: DictConfig, run_dir: pathlib.Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
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
        """Persist *result* to ``step_dir`` and write a ``done`` sentinel."""
        self.step_dir.mkdir(parents=True, exist_ok=True)
        if result is not None:
            with open(self.step_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
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
