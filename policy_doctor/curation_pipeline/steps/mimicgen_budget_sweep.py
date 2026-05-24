"""MimicgenBudgetSweepStep and MimicgenBudgetRepSweepStep — config-driven sweeps with device pool."""

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
        failures: list[str] = []
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
                    print(
                        f"\n[mimicgen_budget_sweep] ERROR: arm mimicgen_{h}_budget{b} FAILED: {exc}"
                        "\n  Continuing with remaining arms...",
                        flush=True,
                    )
                    failures.append(f"mimicgen_{h}_budget{b}: {exc}")

        if failures:
            print(
                f"\n[mimicgen_budget_sweep] WARNING: {len(failures)} arm(s) failed"
                f" — partial results written:\n"
                + "\n".join(f"  {f}" for f in failures),
                flush=True,
            )

        return all_results


class MimicgenBudgetRepSweepStep(PipelineStep[dict]):
    """Run rep-2 and rep-3 arms for a heuristic × budget matrix, reusing existing clustering.

    This step is the companion to :class:`MimicgenBudgetSweepStep`.  Where that
    step runs rep-1 arms (``random_seed=null``), this step runs additional
    replicates with different ``random_seed`` values (e.g. 1 and 2), varying only
    which specific rollouts are drawn as seeds while keeping the baseline policy,
    clustering, and behavior graph fixed.

    This mirrors the apr23 replicate design — fixing all upstream state and only
    varying the seed draw — which is the controlled comparison needed to detect
    heuristic differences cleanly.

    .. code-block:: yaml

        mimicgen_budget_rep_sweep:
          heuristics: [random, behavior_graph, diversity]
          budgets: [20, 100, 500, 1000]
          rep_seeds: [1, 2]          # random_seed values for rep-2 and rep-3
          devices: [cuda:0, cuda:0, cuda:1, cuda:1]

    Rep-1 arms (e.g. ``mimicgen_random_budget100``) land under the same run dir
    and are produced by :class:`MimicgenBudgetSweepStep`.  Rep-2/3 arms get a
    ``_rep{seed}`` suffix (e.g. ``mimicgen_random_budget100_rep1``) so they write
    to separate sub-dirs without colliding.
    """

    name = "mimicgen_budget_rep_sweep"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import _make_budget_arm_class

        sweep_cfg = OmegaConf.select(self.cfg, "mimicgen_budget_rep_sweep") or {}
        budgets = _resolve_budgets(sweep_cfg)
        heuristics: list[str] = list(
            OmegaConf.select(sweep_cfg, "heuristics")
            or ["random", "behavior_graph", "diversity"]
        )
        rep_seeds_raw = OmegaConf.select(sweep_cfg, "rep_seeds")
        rep_seeds: list[int] = (
            [int(s) for s in rep_seeds_raw] if rep_seeds_raw is not None else [1, 2]
        )
        devices: list[str] = list(
            OmegaConf.select(sweep_cfg, "devices") or ["cuda:0"]
        )

        top_level_run_dir = self.run_dir

        # Build the full (heuristic, budget, rep_seed) work list.
        # Budget OUTER so all budget-N arms (across heuristics, reps) complete
        # before moving to budget-(N+1) — gives early signal on small budgets.
        arms = [
            (heuristic, budget, rep_seed)
            for budget in budgets
            for heuristic in heuristics
            for rep_seed in rep_seeds
        ]

        device_pool: queue.Queue[str] = queue.Queue()
        for d in devices:
            device_pool.put(d)

        all_results: dict[str, Any] = {}
        lock = threading.Lock()

        def run_arm(heuristic: str, budget: int, rep_seed: int) -> tuple[str, Any]:
            device = device_pool.get()
            try:
                arm_cls = _make_budget_arm_class(heuristic, budget, random_seed=rep_seed)
                arm_cfg = copy.deepcopy(self.cfg)
                OmegaConf.update(arm_cfg, "device", device)
                arm = arm_cls(arm_cfg, self.step_dir, parent_run_dir=top_level_run_dir)
                print(f"\n[mimicgen_budget_rep_sweep] ── {arm.name}  device={device}")
                result = arm.run(skip_if_done=True)
                return arm.name, result
            finally:
                device_pool.put(device)

        n_workers = len(devices)
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_arm, h, b, r): (h, b, r)
                for h, b, r in arms
            }
            for future in concurrent.futures.as_completed(futures):
                h, b, r = futures[future]
                try:
                    name, result = future.result()
                    with lock:
                        all_results[name] = result
                except Exception as exc:
                    print(
                        f"\n[mimicgen_budget_rep_sweep] ERROR: arm mimicgen_{h}_budget{b}_rep{r}"
                        f" FAILED: {exc}\n  Continuing with remaining arms...",
                        flush=True,
                    )
                    failures.append(f"mimicgen_{h}_budget{b}_rep{r}: {exc}")

        if failures:
            print(
                f"\n[mimicgen_budget_rep_sweep] WARNING: {len(failures)} arm(s) failed"
                f" — partial results written:\n"
                + "\n".join(f"  {f}" for f in failures),
                flush=True,
            )

        return all_results
