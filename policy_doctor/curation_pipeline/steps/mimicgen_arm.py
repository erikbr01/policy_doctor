"""Composite pipeline steps for the MimicGen trajectory generation experiment.

Each arm runs the same sub-steps (select seed → generate → train → eval) but
with a different seed-selection heuristic and/or generation budget baked in via
``cfg_overrides``.  Arms can be included in the same pipeline run, sharing the
upstream ``run_clustering`` result while writing to separate namespaced dirs::

    <run_dir>/
        run_clustering/               # shared
        mimicgen_random/              # RandomSelectionHeuristic, budget=200
        mimicgen_behavior_graph/      # BehaviorGraphPathHeuristic, budget=200
        mimicgen_diversity/           # DiversitySelectionHeuristic, budget=200
        mimicgen_random_20/           # Random, budget=20 (data-volume ablation)
        mimicgen_behavior_graph_20/   # BG, budget=20
        mimicgen_diversity_20/        # Diversity, budget=20
        mimicgen_random_rep2/         # Random, random_seed=1 (variance rep)
        mimicgen_random_rep3/         # Random, random_seed=2
        mimicgen_behavior_graph_rep2/ # BG, random_seed=1 (shuffles path rollout order)
        mimicgen_behavior_graph_rep3/ # BG, random_seed=2
        mimicgen_diversity_rep2/      # Diversity, random_seed=1
        mimicgen_diversity_rep3/      # Diversity, random_seed=2

Register in ``pipeline.py``'s ``ALL_STEPS`` / ``_build_step_registry()``::

    steps: [run_clustering, mimicgen_random, mimicgen_behavior_graph, mimicgen_diversity]
"""

from __future__ import annotations

from policy_doctor.curation_pipeline.base_step import CompositeStep
from policy_doctor.curation_pipeline.steps.eval_mimicgen_combined import (
    EvalMimicgenCombinedStep,
)
from policy_doctor.curation_pipeline.steps.generate_mimicgen_demos import (
    GenerateMimicgenDemosStep,
)
from policy_doctor.curation_pipeline.steps.select_mimicgen_seed import (
    SelectMimicgenSeedStep,
)
from policy_doctor.curation_pipeline.steps.train_on_combined_data import (
    TrainOnCombinedDataStep,
)

_SUB_STEPS = [SelectMimicgenSeedStep, GenerateMimicgenDemosStep, TrainOnCombinedDataStep, EvalMimicgenCombinedStep]


class MimicgenRandomArmStep(CompositeStep):
    """MimicGen arm: random seed selection (baseline condition).

    Runs ``select_mimicgen_seed → generate_mimicgen_demos → train_on_combined_data``
    with ``seed_selection_heuristic=random``, regardless of the value in the shared
    config.  Results land under ``<run_dir>/mimicgen_random/``.
    """

    name = "mimicgen_random"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {"mimicgen_datagen.seed_selection_heuristic": "random"}


class MimicgenBehaviorGraphArmStep(CompositeStep):
    """MimicGen arm: behavior-graph seed selection (proposed method).

    Runs ``select_mimicgen_seed → generate_mimicgen_demos → train_on_combined_data``
    with ``seed_selection_heuristic=behavior_graph``, regardless of the value in the
    shared config.  Results land under ``<run_dir>/mimicgen_behavior_graph/``.
    """

    name = "mimicgen_behavior_graph"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {"mimicgen_datagen.seed_selection_heuristic": "behavior_graph"}


# ---------------------------------------------------------------------------
# Data-volume ablation: budget=20 (small augmentation)
# ---------------------------------------------------------------------------

class MimicgenRandom20ArmStep(CompositeStep):
    """MimicGen arm: random selection, success_budget=20.

    Tests whether seed selection matters more when the augmentation budget is
    small (fewer generated demos → each seed has more impact).
    """

    name = "mimicgen_random_20"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "random",
        "mimicgen_datagen.success_budget": 20,
        "run_tag": "budget20",
    }


class MimicgenBehaviorGraph20ArmStep(CompositeStep):
    """MimicGen arm: behavior-graph selection, success_budget=20."""

    name = "mimicgen_behavior_graph_20"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "behavior_graph",
        "mimicgen_datagen.success_budget": 20,
        "run_tag": "budget20",
    }


# ---------------------------------------------------------------------------
# Variance replicates: random_seed=1 and random_seed=2
# For the random heuristic this changes which rollouts are sampled.
# For behavior_graph this shuffles the eligible rollout order within each path,
# so different rollouts from the same highest-probability path are selected.
# ---------------------------------------------------------------------------

class MimicgenRandomRep2ArmStep(CompositeStep):
    """MimicGen arm: random selection, random_seed=1 (variance replicate 2)."""

    name = "mimicgen_random_rep2"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "random",
        "mimicgen_datagen.random_seed": 1,
        "run_tag": "rep2",
    }


class MimicgenRandomRep3ArmStep(CompositeStep):
    """MimicGen arm: random selection, random_seed=2 (variance replicate 3)."""

    name = "mimicgen_random_rep3"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "random",
        "mimicgen_datagen.random_seed": 2,
        "run_tag": "rep3",
    }


class MimicgenBehaviorGraphRep2ArmStep(CompositeStep):
    """MimicGen arm: behavior-graph selection, random_seed=1 (variance replicate 2)."""

    name = "mimicgen_behavior_graph_rep2"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "behavior_graph",
        "mimicgen_datagen.random_seed": 1,
        "run_tag": "rep2",
    }


class MimicgenBehaviorGraphRep3ArmStep(CompositeStep):
    """MimicGen arm: behavior-graph selection, random_seed=2 (variance replicate 3)."""

    name = "mimicgen_behavior_graph_rep3"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "behavior_graph",
        "mimicgen_datagen.random_seed": 2,
        "run_tag": "rep3",
    }


# ---------------------------------------------------------------------------
# Diversity arms: one seed per distinct behavior-graph path
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Budget=20 variance replicates (budget=20 + random_seed=1 or 2)
# ---------------------------------------------------------------------------

class MimicgenRandom20Rep2ArmStep(CompositeStep):
    """MimicGen arm: random selection, budget=20, random_seed=1 (variance replicate 2)."""

    name = "mimicgen_random_20_rep2"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "random",
        "mimicgen_datagen.success_budget": 20,
        "mimicgen_datagen.random_seed": 1,
        "run_tag": "budget20-rep2",
    }


class MimicgenRandom20Rep3ArmStep(CompositeStep):
    """MimicGen arm: random selection, budget=20, random_seed=2 (variance replicate 3)."""

    name = "mimicgen_random_20_rep3"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "random",
        "mimicgen_datagen.success_budget": 20,
        "mimicgen_datagen.random_seed": 2,
        "run_tag": "budget20-rep3",
    }


class MimicgenBehaviorGraph20Rep2ArmStep(CompositeStep):
    """MimicGen arm: behavior-graph selection, budget=20, random_seed=1 (variance replicate 2)."""

    name = "mimicgen_behavior_graph_20_rep2"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "behavior_graph",
        "mimicgen_datagen.success_budget": 20,
        "mimicgen_datagen.random_seed": 1,
        "run_tag": "budget20-rep2",
    }


class MimicgenBehaviorGraph20Rep3ArmStep(CompositeStep):
    """MimicGen arm: behavior-graph selection, budget=20, random_seed=2 (variance replicate 3)."""

    name = "mimicgen_behavior_graph_20_rep3"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "behavior_graph",
        "mimicgen_datagen.success_budget": 20,
        "mimicgen_datagen.random_seed": 2,
        "run_tag": "budget20-rep3",
    }


class MimicgenDiversity20Rep2ArmStep(CompositeStep):
    """MimicGen arm: diversity selection, budget=20, random_seed=1 (variance replicate 2)."""

    name = "mimicgen_diversity_20_rep2"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "diversity",
        "mimicgen_datagen.success_budget": 20,
        "mimicgen_datagen.random_seed": 1,
        "run_tag": "budget20-rep2",
    }


class MimicgenDiversity20Rep3ArmStep(CompositeStep):
    """MimicGen arm: diversity selection, budget=20, random_seed=2 (variance replicate 3)."""

    name = "mimicgen_diversity_20_rep3"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "diversity",
        "mimicgen_datagen.success_budget": 20,
        "mimicgen_datagen.random_seed": 2,
        "run_tag": "budget20-rep3",
    }


# ---------------------------------------------------------------------------
# Diversity arms: one seed per distinct behavior-graph path
# ---------------------------------------------------------------------------

class MimicgenDiversityArmStep(CompositeStep):
    """MimicGen arm: diversity selection, budget=200.

    Picks one rollout per distinct success path (ranked by probability), so
    every seed represents a different execution strategy.  Tests whether
    behavioral coverage of seeds matters more than concentration on the
    highest-probability path.
    """

    name = "mimicgen_diversity"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {"mimicgen_datagen.seed_selection_heuristic": "diversity"}


class MimicgenDiversity20ArmStep(CompositeStep):
    """MimicGen arm: diversity selection, budget=20."""

    name = "mimicgen_diversity_20"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "diversity",
        "mimicgen_datagen.success_budget": 20,
        "run_tag": "budget20",
    }


class MimicgenDiversityRep2ArmStep(CompositeStep):
    """MimicGen arm: diversity selection, random_seed=1 (variance replicate 2)."""

    name = "mimicgen_diversity_rep2"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "diversity",
        "mimicgen_datagen.random_seed": 1,
        "run_tag": "rep2",
    }


class MimicgenDiversityRep3ArmStep(CompositeStep):
    """MimicGen arm: diversity selection, random_seed=2 (variance replicate 3)."""

    name = "mimicgen_diversity_rep3"
    sub_step_classes = _SUB_STEPS
    cfg_overrides = {
        "mimicgen_datagen.seed_selection_heuristic": "diversity",
        "mimicgen_datagen.random_seed": 2,
        "run_tag": "rep3",
    }
