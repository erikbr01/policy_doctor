"""ENAP EM loop orchestration — pipeline step.

Runs the full ENAP **Expectation–Maximisation** loop on fixed rollout data:

- **E-step** (re-run ``extract_enap_graph``): Re-learn the PMM topology from
  the current (fixed) rollout dataset.
- **M-step** (``train_enap_residual``): Freeze the PMM and optimise the
  :class:`~policy_doctor.enap.residual_policy.ResidualMLP` to refine action
  predictions.

On fixed-dataset graph construction the E-step data does not change across
iterations, so convergence is rapid (typically 1–3 rounds).  Multiple EM
rounds become more meaningful when the pipeline is extended to collect new
rollouts using the refined policy.

The step forces re-execution of its inner steps (``skip_if_done=False``) for
every iteration after the first.

Saved outputs (in ``step_dir/``):
- ``em_log.json``   — per-iteration E/M-step result summaries
- ``result.json`` / ``done``
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class TrainENAPEMStep(PipelineStep[Dict[str, Any]]):
    """Orchestrate E-step ↔ M-step EM iterations for ENAP.

    Config keys consumed (all under ``graph_building.enap``):
    - ``em_iterations``: number of full EM rounds (default 1)

    All other ENAP config keys are passed through to the inner steps.

    .. note::
        With fixed rollout data, a single EM round (``em_iterations=1``) is
        equivalent to running ``extract_enap_graph`` then
        ``train_enap_residual`` once.  Increase ``em_iterations`` to 2–3 for a
        mild regularisation effect via repeated hypothesis refinement.
    """

    name = "train_enap_em"

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
        from policy_doctor.curation_pipeline.steps.extract_enap_graph import (
            ExtractENAPGraphStep,
        )
        from policy_doctor.curation_pipeline.steps.train_enap_residual import (
            TrainENAPResidualStep,
        )

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}
        em_iterations = int(OmegaConf.select(enap_cfg, "em_iterations") or 1)

        print(f"  [EM] Running {em_iterations} EM iteration(s).")

        em_log: List[Dict[str, Any]] = []

        for it in range(em_iterations):
            print(f"\n  [EM] Iteration {it + 1}/{em_iterations}")

            # E-step: (re-)learn PMM topology
            # After the first iteration, force re-run so updated policy can
            # affect future data collections (when extended to online rollouts).
            skip = (it == 0) and self.dry_run is False  # skip on first iter if already done
            # Always run on all iterations (skip only if done for iter 0 in resume context)
            skip_if_done = (it == 0)  # only the first iter respects skip_if_done

            e_step = ExtractENAPGraphStep(cfg, self.run_dir)
            print(f"  [EM] E-step (extract_enap_graph)...")
            e_result = e_step.run(skip_if_done=skip_if_done)

            # M-step: train residual policy conditioned on current PMM
            m_step = TrainENAPResidualStep(cfg, self.run_dir)
            print(f"  [EM] M-step (train_enap_residual)...")
            m_result = m_step.run(skip_if_done=skip_if_done)

            em_log.append(
                {
                    "iteration": it + 1,
                    "e_step": e_result,
                    "m_step": m_result,
                }
            )

        # Persist EM log
        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "em_log.json", "w") as f:
            json.dump(em_log, f, indent=2, default=str)

        return {
            "em_iterations": em_iterations,
            "final_e_step": em_log[-1]["e_step"] if em_log else {},
            "final_m_step": em_log[-1]["m_step"] if em_log else {},
        }
