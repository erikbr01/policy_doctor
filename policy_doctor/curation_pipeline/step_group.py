"""StepGroup abstraction: config-driven expansion of pipeline step groups.

A :class:`StepGroupDef` maps a config key to a set of concrete step names
for each supported method.  :func:`resolve_steps` expands any group names
found in the requested step list before execution.

Example::

    # In config: graph_building.method = "enap"
    resolve_steps(["compute_infembed", "graph_building", "export_markov_report"], cfg)
    # → ["compute_infembed",
    #    "train_enap_perception", "train_enap_rnn", "extract_enap_graph",
    #    "build_behavior_graph",
    #    "export_markov_report"]

The registry is intentionally simple — only one group exists today, but more
can be added as new multi-method pipeline stages arise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf


@dataclass
class StepGroupDef:
    """Definition of a named pipeline step group.

    Args:
        method_cfg_key: Dotted OmegaConf key that selects the active method
            (e.g. ``"graph_building.method"``).
        steps_by_method: Maps each method string to an ordered list of
            concrete step names.
        default_method: Fallback method when the config key is absent.
    """

    method_cfg_key: str
    steps_by_method: Dict[str, List[str]]
    default_method: str = "cupid"

    def expand(self, cfg: DictConfig) -> List[str]:
        """Return the concrete step list for the method selected by *cfg*."""
        method = OmegaConf.select(cfg, self.method_cfg_key) or self.default_method
        method = str(method)
        if method not in self.steps_by_method:
            valid = list(self.steps_by_method.keys())
            raise ValueError(
                f"Unknown method {method!r} for step group "
                f"(key={self.method_cfg_key!r}). Valid: {valid}"
            )
        return list(self.steps_by_method[method])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STEP_GROUPS: Dict[str, StepGroupDef] = {
    "graph_building": StepGroupDef(
        method_cfg_key="graph_building.method",
        steps_by_method={
            "cupid": ["run_clustering", "build_behavior_graph"],
            "enap": [
                "train_enap_perception",
                "train_enap_rnn",
                "extract_enap_graph",
                "train_enap_residual",
                "build_behavior_graph",
            ],
            "enap_custom": [
                "train_enap_perception",
                "train_enap_rnn_custom",
                "extract_enap_graph_custom",
                "build_behavior_graph",
            ],
        },
        default_method="cupid",
    ),
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_steps(step_names: List[str], cfg: DictConfig) -> List[str]:
    """Expand group names in *step_names* to their constituent concrete steps.

    Any name present in :data:`STEP_GROUPS` is replaced by the method-specific
    list of steps resolved from *cfg*.  All other names are left unchanged.

    Args:
        step_names: Requested step names (may include group names).
        cfg: Hydra/OmegaConf config used to resolve method choices.

    Returns:
        Flat list of concrete step names in execution order.
    """
    resolved: List[str] = []
    for name in step_names:
        if name in STEP_GROUPS:
            resolved.extend(STEP_GROUPS[name].expand(cfg))
        else:
            resolved.append(name)
    return resolved
