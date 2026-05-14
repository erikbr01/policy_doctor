"""Hydra-string registry for E2 plugins (graph representations + VLM input builders).

Mirrors :mod:`policy_doctor.vlm.registry` so prompts/inputs/graph-rendering can
all be swapped via Hydra config strings without touching code paths.

Two registries are exposed:

  - ``get_graph_representation(name, params)``  → :class:`GraphRepresentation`
  - ``get_vlm_input_builder(name, params)``     → :class:`VLMInputBuilder`

Concrete implementations register themselves at import time via
:func:`register_graph_representation` / :func:`register_vlm_input_builder`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# --------------------------------------------------------------------------
# Lightweight params normalization (dict | DictConfig → dict)
# --------------------------------------------------------------------------


def _normalize_params(params: Any) -> Dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(params, DictConfig):
            c = OmegaConf.to_container(params, resolve=True)
            return dict(c) if isinstance(c, dict) else {}
    except ImportError:
        pass
    raise TypeError(f"params must be dict or DictConfig, got {type(params)}")


# --------------------------------------------------------------------------
# GraphRepresentation registry
# --------------------------------------------------------------------------

_GRAPH_REPR_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_graph_representation(name: str, factory: Callable[[Dict[str, Any]], Any]) -> None:
    _GRAPH_REPR_REGISTRY[name] = factory


def list_graph_representations() -> list:
    _ensure_loaded()
    return sorted(_GRAPH_REPR_REGISTRY)


def get_graph_representation(name: str, params: Any = None):
    """Instantiate a :class:`GraphRepresentation` by name."""
    _ensure_loaded()
    key = (name or "").strip().lower()
    if key not in _GRAPH_REPR_REGISTRY:
        raise ValueError(
            f"Unknown graph representation {name!r}. Registered: {list_graph_representations()}"
        )
    return _GRAPH_REPR_REGISTRY[key](_normalize_params(params))


# --------------------------------------------------------------------------
# VLMInputBuilder registry
# --------------------------------------------------------------------------

_INPUT_BUILDER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_vlm_input_builder(name: str, factory: Callable[[Dict[str, Any]], Any]) -> None:
    _INPUT_BUILDER_REGISTRY[name] = factory


def list_vlm_input_builders() -> list:
    _ensure_loaded()
    return sorted(_INPUT_BUILDER_REGISTRY)


def get_vlm_input_builder(name: str, params: Any = None):
    """Instantiate a :class:`VLMInputBuilder` by name."""
    _ensure_loaded()
    key = (name or "").strip().lower()
    if key not in _INPUT_BUILDER_REGISTRY:
        raise ValueError(
            f"Unknown VLM input builder {name!r}. Registered: {list_vlm_input_builders()}"
        )
    return _INPUT_BUILDER_REGISTRY[key](_normalize_params(params))


# --------------------------------------------------------------------------
# Lazy import of bundled plugins
# --------------------------------------------------------------------------


_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # Importing each module triggers register_* calls at module top level.
    from policy_doctor.vlm.proposals.graph_representation import image_only as _g1  # noqa: F401
    from policy_doctor.vlm.proposals.graph_representation import text_table as _g2  # noqa: F401
    from policy_doctor.vlm.proposals.graph_representation import combined as _g3   # noqa: F401
    from policy_doctor.vlm.proposals.vlm_input import graph_condition as _v1       # noqa: F401
    from policy_doctor.vlm.proposals.vlm_input import outcome_condition as _v2     # noqa: F401
    _LOADED = True
