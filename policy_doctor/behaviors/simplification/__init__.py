"""Behavior graph simplification methods + Pareto sweep.

Public API:
    SimplificationResult: dataclass returned by every method.
    METHODS: dict of {name: callable}.
    run_method(name, graph, labels, metadata, lever) -> SimplificationResult
    pareto_sweep(name, graph, labels, metadata, lever_grid) -> List[SimplificationResult]

Each method exposes a single scalar `lever`. See `frontier.LEVER_GRIDS` for
recommended sweep ranges per method.
"""

from policy_doctor.behaviors.simplification.api import (
    METHODS,
    METHOD_DESCRIPTIONS,
    METHOD_LEVER_LABELS,
    SimplificationResult,
    run_method,
)
from policy_doctor.behaviors.simplification.frontier import (
    LEVER_GRIDS,
    pareto_sweep,
)
from policy_doctor.behaviors.simplification.metrics import (
    GraphMetrics,
    bootstrap_metric,
    compute_metrics,
    hoeffding_compatible,
    js_bits,
    kl_bits,
    markov_violation_bits,
    trajectory_nll_bits,
)

__all__ = [
    "METHODS",
    "METHOD_DESCRIPTIONS",
    "METHOD_LEVER_LABELS",
    "LEVER_GRIDS",
    "SimplificationResult",
    "GraphMetrics",
    "run_method",
    "pareto_sweep",
    "compute_metrics",
    "bootstrap_metric",
    "kl_bits",
    "js_bits",
    "hoeffding_compatible",
    "trajectory_nll_bits",
    "markov_violation_bits",
]
