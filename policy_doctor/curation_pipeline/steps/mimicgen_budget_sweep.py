"""MimicgenBudgetSweepStep — config-driven heuristic × budget matrix with device pool."""

from __future__ import annotations

import copy
import queue
import threading
import concurrent.futures
from typing import Any

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


def _resolve_budgets(sweep_cfg: Any) -> list[int]:
    """Return the list of success_budget values to sweep over.

    Priority:
    1. ``budgets`` — explicit list (e.g. ``[20, 100, 200, 500, 1000]``)
    2. ``start`` / ``stop`` / ``step`` — generates ``range(start, stop+1, step)``
    """
    budgets = OmegaConf.select(sweep_cfg, "budgets")
    if budgets is not None:
        return [int(b) for b in budgets]
    start = int(OmegaConf.select(sweep_cfg, "start") or 100)
    stop = int(OmegaConf.select(sweep_cfg, "stop") or 1000)
    step = int(OmegaConf.select(sweep_cfg, "step") or 100)
    return list(range(start, stop + 1, step))


def _resolve_demo_counts(sweep_cfg: Any) -> list[int] | None:
    """Return the list of baseline demo counts to sweep over, or ``None``.

    Returns ``None`` when no demo-count sweep is configured — callers should
    then use the ``baseline.max_train_episodes`` value from the experiment YAML
    without adding a ``_demos<N>`` suffix to ``run_dir``.

    Priority:
    1. ``demo_counts`` — explicit list (e.g. ``[60, 100, 300]``)
    2. ``start`` / ``stop`` / ``step`` — generates ``range(start, stop+1, step)``
    3. Neither present → returns ``None``
    """
    if sweep_cfg is None:
        return None
    counts = OmegaConf.select(sweep_cfg, "demo_counts")
    if counts is not None:
        return [int(c) for c in counts]
    start = OmegaConf.select(sweep_cfg, "start")
    if start is not None:
        stop = int(OmegaConf.select(sweep_cfg, "stop") or start)
        step = int(OmegaConf.select(sweep_cfg, "step") or 1)
        return list(range(int(start), stop + 1, step))
    return None


class MimicgenBudgetSweepStep(PipelineStep[dict]):
    """Run all heuristic × budget combinations with a concurrent device pool.

    Reads experiment parameters from ``mimicgen_budget_sweep`` in the pipeline
    config:

    .. code-block:: yaml

        mimicgen_budget_sweep:
          heuristics: [random, behavior_graph, diversity]
          budgets: [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
          # OR: start/stop/step for a regular grid
          devices: [cuda:0, cuda:0, cuda:1, cuda:1]   # 2 slots per GPU

    The ``devices`` list defines a pool of device slots.  Each concurrent arm
    claims one slot, runs its full sub-pipeline (select → generate → train →
    eval) with that device, then releases the slot.  The number of concurrent
    arms equals ``len(devices)``.

    Results land under::

        <run_dir>/mimicgen_budget_sweep/
            mimicgen_random_budget20/
            mimicgen_behavior_graph_budget100/
            ...
    """

    name = "mimicgen_budget_sweep"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import _make_budget_arm_class

        sweep_cfg = OmegaConf.select(self.cfg, "mimicgen_budget_sweep") or {}
        budgets = _resolve_budgets(sweep_cfg)
        heuristics: list[str] = list(
            OmegaConf.select(sweep_cfg, "heuristics")
            or ["random", "behavior_graph", "diversity"]
        )
        devices: list[str] = list(
            OmegaConf.select(sweep_cfg, "devices") or ["cuda:0"]
        )

        top_level_run_dir = self.run_dir

        # Build the full (heuristic, budget) work list
        arms = [
            (heuristic, budget)
            for heuristic in heuristics
            for budget in budgets
        ]

        # Device pool: each slot is one device string; workers borrow and return
        device_pool: queue.Queue[str] = queue.Queue()
        for d in devices:
            device_pool.put(d)

        all_results: dict[str, Any] = {}
        lock = threading.Lock()

        def run_arm(heuristic: str, budget: int) -> tuple[str, Any]:
            device = device_pool.get()
            try:
                arm_cls = _make_budget_arm_class(heuristic, budget)
                # Deep-copy cfg and inject the assigned device before the arm sees it.
                # The arm's cfg_overrides (heuristic, budget, run_tag) are applied on
                # top inside CompositeStep.compute(), so they don't clobber device.
                arm_cfg = copy.deepcopy(self.cfg)
                OmegaConf.update(arm_cfg, "device", device)
                arm = arm_cls(arm_cfg, self.step_dir, parent_run_dir=top_level_run_dir)
                print(f"\n[mimicgen_budget_sweep] ── {arm.name}  device={device}")
                result = arm.run(skip_if_done=True)
                return arm.name, result
            finally:
                device_pool.put(device)

        n_workers = len(devices)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_arm, h, b): (h, b)
                for h, b in arms
            }
            for future in concurrent.futures.as_completed(futures):
                h, b = futures[future]
                try:
                    name, result = future.result()
                    with lock:
                        all_results[name] = result
                except Exception as exc:
                    raise RuntimeError(
                        f"[mimicgen_budget_sweep] arm mimicgen_{h}_budget{b} failed: {exc}"
                    ) from exc

        return all_results
