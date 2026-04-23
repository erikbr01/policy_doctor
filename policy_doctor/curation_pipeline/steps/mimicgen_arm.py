"""Composite pipeline steps for the MimicGen trajectory generation experiment.

Each arm runs the same three sub-steps (select seed → generate → train on combined
data) but with a different seed-selection heuristic baked in via ``cfg_overrides``.
Both arms can be included in the same pipeline run, sharing the upstream
``run_clustering`` result while writing to separate namespaced directories::

    <run_dir>/
        run_clustering/            # shared
        mimicgen_random/           # RandomSelectionHeuristic arm
            select_mimicgen_seed/
            generate_mimicgen_demos/
            train_on_combined_data/
        mimicgen_behavior_graph/   # BehaviorGraphPathHeuristic arm
            select_mimicgen_seed/
            generate_mimicgen_demos/
            train_on_combined_data/

Register both in ``pipeline.py``'s ``ALL_STEPS`` / ``_build_step_registry()`` to
run them together::

    steps: [run_clustering, mimicgen_random, mimicgen_behavior_graph]
"""

from __future__ import annotations

from policy_doctor.curation_pipeline.base_step import CompositeStep
from policy_doctor.curation_pipeline.steps.generate_mimicgen_demos import (
    GenerateMimicgenDemosStep,
)
from policy_doctor.curation_pipeline.steps.select_mimicgen_seed import (
    SelectMimicgenSeedStep,
)
from policy_doctor.curation_pipeline.steps.train_on_combined_data import (
    TrainOnCombinedDataStep,
)

_SUB_STEPS = [SelectMimicgenSeedStep, GenerateMimicgenDemosStep, TrainOnCombinedDataStep]


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
