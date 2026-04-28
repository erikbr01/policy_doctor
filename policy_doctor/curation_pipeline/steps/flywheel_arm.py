"""Multi-iteration MimicGen data flywheel pipeline steps.

The flywheel simulates multiple rounds of: seed selection → data generation →
training → evaluation → re-clustering, where each iteration's clustering feeds
the next iteration's seed selection.

Three step classes form a nesting hierarchy:

    FlyWheelMultiArmStep  ("mimicgen_flywheel", registered in pipeline)
      └─ FlyWheelArmStep  (one per strategy sequence, e.g. "bg_div")
           └─ FlyWheelIterationStep  (one per iteration, e.g. "iter_0", "iter_1")

Directory layout::

    <run_dir>/
      run_clustering/                       # shared initial baseline clustering
      mimicgen_flywheel/
        bg_div/                             # arm: [behavior_graph, diversity]
          iter_0/                           # heuristic = behavior_graph
            select_mimicgen_seed/
            generate_mimicgen_demos/
            train_on_combined_data/         # base + gen_iter_0
            eval_mimicgen_combined/
            eval_flywheel_policy/           # full rollout save
            compute_infembed/
            run_clustering/                 # feeds iter_1 seed selection
          iter_1/                           # heuristic = diversity (last → no re-cluster)
            select_mimicgen_seed/
            generate_mimicgen_demos/
            train_on_combined_data/         # base + gen_iter_0 + gen_iter_1
            eval_mimicgen_combined/
        div_bg/                             # arm: [diversity, behavior_graph]
          ...
"""

from __future__ import annotations

import copy
import itertools
import json
import pathlib
from typing import Any

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep

# Mapping from full heuristic name to a compact directory-safe short code
_HEURISTIC_SHORT: dict[str, str] = {
    "behavior_graph": "bg",
    "diversity": "div",
    "random": "rand",
}


def _arm_name(strategy_sequence: list[str]) -> str:
    """Derive a short, filesystem-safe arm name from a strategy sequence."""
    return "_".join(_HEURISTIC_SHORT.get(s, s[:4]) for s in strategy_sequence)


class FlyWheelIterationStep(PipelineStep[dict]):
    """One iteration of the data flywheel.

    Runs in order:
      1. select_mimicgen_seed      — picks seed trajectory from behavior graph
      2. generate_mimicgen_demos   — generates N demos via MimicGen
      3. train_on_combined_data    — trains on base + all accumulated generated data
      4. eval_mimicgen_combined    — checkpoint-sweep success rate eval
      If not the last iteration, also runs:
      5. eval_flywheel_policy      — full eval_save_episodes for infembed
      6. compute_infembed          — infembed embeddings on retrained policy
      7. run_clustering            — re-clusters to produce the next iter's behavior graph

    Not registered in the pipeline registry; created programmatically by FlyWheelArmStep.

    Args:
        iteration_idx:       0-based index of this iteration.
        is_last_iteration:   Skip re-clustering when True (no next iteration to feed).
        heuristic:           Seed-selection heuristic for this iteration.
        all_iter_dirs:       Paths for ALL iterations in this arm (iter_0, iter_1, …),
                             used by TrainFlywheelIterStep to collect accumulated data.
    """

    def __init__(
        self,
        cfg,
        run_dir: pathlib.Path,
        parent_run_dir: pathlib.Path | None = None,
        *,
        iteration_idx: int,
        is_last_iteration: bool,
        heuristic: str,
        all_iter_dirs: list[pathlib.Path],
    ) -> None:
        self.name = f"iter_{iteration_idx}"  # set before super so step_dir is correct
        super().__init__(cfg, run_dir, parent_run_dir)
        self.iteration_idx = iteration_idx
        self.is_last_iteration = is_last_iteration
        self.heuristic = heuristic
        self.all_iter_dirs = all_iter_dirs

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.select_mimicgen_seed import (
            SelectMimicgenSeedStep,
        )
        from policy_doctor.curation_pipeline.steps.generate_mimicgen_demos import (
            GenerateMimicgenDemosStep,
        )
        from policy_doctor.curation_pipeline.steps.eval_mimicgen_combined import (
            EvalMimicgenCombinedStep,
        )
        from policy_doctor.curation_pipeline.steps.flywheel_sub_steps import (
            EvalFlywheelPolicyStep,
            ComputeInfembedFlywheelStep,
            RunClusteringFlywheelStep,
            TrainFlywheelIterStep,
        )

        # Build iteration-specific config: fix heuristic and generation budget
        sub_cfg = copy.deepcopy(self.cfg)
        OmegaConf.update(sub_cfg, "mimicgen_datagen.seed_selection_heuristic", self.heuristic, merge=True)
        generation_budget = OmegaConf.select(self.cfg, "flywheel.generation_budget")
        if generation_budget is not None:
            OmegaConf.update(sub_cfg, "mimicgen_datagen.success_budget", int(generation_budget), merge=True)

        # For N>0: inject rollouts_hdf5_path so SelectMimicgenSeedStep skips task-YAML lookup
        if self.iteration_idx > 0:
            prev_iter_dir = self.all_iter_dirs[self.iteration_idx - 1]
            prev_eval_result_path = prev_iter_dir / "eval_flywheel_policy" / "result.json"
            if prev_eval_result_path.exists():
                with open(prev_eval_result_path) as f:
                    prev_eval = json.load(f)
                rh = prev_eval.get("rollouts_hdf5_path", "")
                if rh:
                    OmegaConf.update(sub_cfg, "mimicgen_datagen.rollouts_hdf5_path", rh, merge=True)

        iter_dir = self.step_dir  # arm_dir/iter_i/
        skip = bool(OmegaConf.select(self.cfg, "skip_if_done") if OmegaConf.select(self.cfg, "skip_if_done") is not None else True)
        results: dict[str, Any] = {}

        # 1. Seed selection
        select_step = SelectMimicgenSeedStep(sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir)
        results["select_mimicgen_seed"] = select_step.run(skip_if_done=skip)

        # 2. Data generation
        gen_step = GenerateMimicgenDemosStep(sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir)
        results["generate_mimicgen_demos"] = gen_step.run(skip_if_done=skip)

        # 3. Train on accumulated data (base + all generated so far)
        prior_iter_dirs = self.all_iter_dirs[:self.iteration_idx]
        train_step = TrainFlywheelIterStep(
            sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir,
            iteration_idx=self.iteration_idx,
            all_prior_iter_dirs=prior_iter_dirs,
        )
        results["train_on_combined_data"] = train_step.run(skip_if_done=skip)

        # 4. Success rate eval (checkpoint sweep)
        eval_combined = EvalMimicgenCombinedStep(sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir)
        results["eval_mimicgen_combined"] = eval_combined.run(skip_if_done=skip)

        if not self.is_last_iteration:
            # 5. Full rollout save for infembed
            eval_policy = EvalFlywheelPolicyStep(sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir)
            results["eval_flywheel_policy"] = eval_policy.run(skip_if_done=skip)

            # 6. Compute infembed embeddings on retrained policy
            infembed = ComputeInfembedFlywheelStep(sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir)
            results["compute_infembed"] = infembed.run(skip_if_done=skip)

            # 7. Re-cluster for next iteration's seed selection
            cluster = RunClusteringFlywheelStep(sub_cfg, iter_dir, parent_run_dir=self.parent_run_dir)
            results["run_clustering"] = cluster.run(skip_if_done=skip)

        return results


class FlyWheelArmStep(PipelineStep[dict]):
    """Run all flywheel iterations for one strategy sequence.

    Sequences the iterations, passing each iteration's clustering result to the
    next via parent_run_dir routing.  Writes to::

        <run_dir>/<arm_name>/

    Not registered in the pipeline registry; created programmatically by FlyWheelMultiArmStep.

    Args:
        arm_name:          Short name derived from strategy_sequence (e.g. ``"bg_div"``).
        strategy_sequence: Ordered list of heuristic names, one per iteration.
    """

    def __init__(
        self,
        cfg,
        run_dir: pathlib.Path,
        parent_run_dir: pathlib.Path | None = None,
        *,
        arm_name: str,
        strategy_sequence: list[str],
    ) -> None:
        self.name = arm_name  # set before super so step_dir is correct
        super().__init__(cfg, run_dir, parent_run_dir)
        self.arm_name = arm_name
        self.strategy_sequence = strategy_sequence

    def compute(self) -> dict[str, Any]:
        flywheel_cfg = OmegaConf.select(self.cfg, "flywheel") or {}
        num_iterations = int(
            OmegaConf.select(flywheel_cfg, "num_iterations") or len(self.strategy_sequence)
        )
        if len(self.strategy_sequence) < num_iterations:
            raise ValueError(
                f"strategy_sequence {self.strategy_sequence!r} has fewer entries than "
                f"num_iterations={num_iterations}."
            )

        arm_dir = self.step_dir  # mimicgen_flywheel/<arm_name>/
        iter_dirs = [arm_dir / f"iter_{i}" for i in range(num_iterations)]
        skip = bool(
            OmegaConf.select(self.cfg, "skip_if_done")
            if OmegaConf.select(self.cfg, "skip_if_done") is not None
            else True
        )

        arm_results: dict[str, Any] = {}
        for i, heuristic in enumerate(self.strategy_sequence[:num_iterations]):
            is_last = (i == num_iterations - 1)
            # Seed selection parent_run_dir:
            #   iter_0 → top-level run_dir (contains initial run_clustering/)
            #   iter_N → iter_{N-1} dir   (contains run_clustering/ from flywheel iter N-1)
            iter_parent = self.parent_run_dir if i == 0 else iter_dirs[i - 1]

            print(f"\n  [flywheel] arm={self.arm_name}  iter={i}  heuristic={heuristic!r}  last={is_last}")
            iter_step = FlyWheelIterationStep(
                self.cfg,
                arm_dir,
                parent_run_dir=iter_parent,
                iteration_idx=i,
                is_last_iteration=is_last,
                heuristic=heuristic,
                all_iter_dirs=iter_dirs,
            )
            arm_results[f"iter_{i}"] = iter_step.run(skip_if_done=skip)

        return arm_results


class FlyWheelMultiArmStep(PipelineStep[dict]):
    """Run all configured flywheel arms (registered as ``"mimicgen_flywheel"``).

    Reads the ``flywheel`` config block to enumerate strategy sequences, then
    creates and runs one :class:`FlyWheelArmStep` per sequence.  All arms share
    the top-level ``run_clustering/`` result as their iter-0 seed-selection source.

    Config (under ``flywheel``):
        num_iterations       Number of flywheel iterations per arm.
        generation_budget    MimicGen ``success_budget`` per iteration.

        Strategy specification — exactly one of:

        strategy_sequence    Single arm: list of heuristic names (length = num_iterations).
        strategy_sequences   Multiple specific arms: list of lists.
        strategy_mode        ``"all_permutations"`` — enumerate automatically.
        strategies           Heuristics to use with strategy_mode (default: all three).
        permutation_type     ``"with_replacement"`` (default, K^N) or
                             ``"without_replacement"`` (K! permutations).
    """

    name = "mimicgen_flywheel"

    def compute(self) -> dict[str, Any]:
        flywheel_cfg = OmegaConf.select(self.cfg, "flywheel") or {}
        num_iterations = int(OmegaConf.select(flywheel_cfg, "num_iterations") or 2)
        sequences = self._enumerate_sequences(flywheel_cfg, num_iterations)

        skip = bool(
            OmegaConf.select(self.cfg, "skip_if_done")
            if OmegaConf.select(self.cfg, "skip_if_done") is not None
            else True
        )

        multi_results: dict[str, Any] = {}
        for seq in sequences:
            arm_name = _arm_name(seq)
            print(f"\n[flywheel] ── arm: {arm_name}  sequence={seq}")
            arm_step = FlyWheelArmStep(
                self.cfg,
                self.step_dir,          # mimicgen_flywheel/ dir
                parent_run_dir=self.run_dir,  # top-level run_dir
                arm_name=arm_name,
                strategy_sequence=seq,
            )
            multi_results[arm_name] = arm_step.run(skip_if_done=skip)

        return multi_results

    @staticmethod
    def _enumerate_sequences(flywheel_cfg: Any, num_iterations: int) -> list[list[str]]:
        """Resolve strategy sequences from the flywheel config block."""
        from omegaconf import DictConfig

        if not isinstance(flywheel_cfg, DictConfig):
            flywheel_cfg = OmegaConf.create(flywheel_cfg)

        # Option A: single explicit sequence
        single = OmegaConf.select(flywheel_cfg, "strategy_sequence")
        if single is not None:
            return [list(single)]

        # Option B: explicit list of sequences
        multi = OmegaConf.select(flywheel_cfg, "strategy_sequences")
        if multi is not None:
            return [list(s) for s in multi]

        # Option C: auto-enumerate permutations
        mode = OmegaConf.select(flywheel_cfg, "strategy_mode")
        if mode == "all_permutations":
            strategies: list[str] = list(
                OmegaConf.select(flywheel_cfg, "strategies")
                or ["behavior_graph", "diversity", "random"]
            )
            perm_type = OmegaConf.select(flywheel_cfg, "permutation_type") or "with_replacement"
            if perm_type == "without_replacement":
                return [list(p) for p in itertools.permutations(strategies, num_iterations)]
            else:
                return [list(p) for p in itertools.product(strategies, repeat=num_iterations)]

        raise ValueError(
            "flywheel config must specify one of: strategy_sequence, strategy_sequences, "
            "or strategy_mode=all_permutations with a strategies list."
        )
