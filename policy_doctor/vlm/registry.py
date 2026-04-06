"""Register and instantiate VLM backends from name + params (Hydra-friendly)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.backends.mock import build_mock_backend

BackendFactory = Callable[[Optional[Dict[str, Any]]], VLMBackend]

_REGISTRY: Dict[str, BackendFactory] = {
    "mock": build_mock_backend,
}


def _build_qwen2(params: Optional[Dict[str, Any]]) -> VLMBackend:
    from policy_doctor.vlm.backends.qwen2_vl import build_qwen2_vl_backend

    return build_qwen2_vl_backend(params)


def _build_qwen3(params: Optional[Dict[str, Any]]) -> VLMBackend:
    """Same implementation as qwen2_vl; ``model_id`` should point at a Qwen3-VL checkpoint."""
    from policy_doctor.vlm.backends.qwen2_vl import build_qwen2_vl_backend

    backend = build_qwen2_vl_backend(params)
    backend.name = "qwen3_vl"
    return backend


_REGISTRY["qwen2_vl"] = _build_qwen2
_REGISTRY["qwen3_vl"] = _build_qwen3


def _build_molmo2(params: Optional[Dict[str, Any]]) -> VLMBackend:
    from policy_doctor.vlm.backends.molmo2 import build_molmo2_backend

    return build_molmo2_backend(params)


def _build_molmo(params: Optional[Dict[str, Any]]) -> VLMBackend:
    from policy_doctor.vlm.backends.molmo2 import build_molmo_backend

    return build_molmo_backend(params)


_REGISTRY["molmo2"] = _build_molmo2
_REGISTRY["molmo"] = _build_molmo


def _build_cosmos_reason2(params: Optional[Dict[str, Any]]) -> VLMBackend:
    """NVIDIA Cosmos-Reason2 (Qwen3-VL stack); gated on Hugging Face — accept the license first."""
    from policy_doctor.vlm.backends.qwen2_vl import build_qwen2_vl_backend

    p = dict(params or {})
    p.setdefault("model_id", "nvidia/Cosmos-Reason2-2B")
    p.setdefault("max_new_tokens", 2048)
    p.setdefault("attn_implementation", "sdpa")
    p.setdefault("torch_dtype", "bfloat16")
    backend = build_qwen2_vl_backend(p)
    backend.name = "cosmos_reason2"
    return backend


_REGISTRY["cosmos_reason2"] = _build_cosmos_reason2


def register_vlm_backend(name: str, factory: BackendFactory) -> None:
    """Register a custom backend (e.g. from another package)."""
    _REGISTRY[name] = factory


def list_vlm_backend_names() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_vlm_backend(name: str, params: Any = None) -> VLMBackend:
    """Instantiate backend by name. *params* may be dict or DictConfig."""
    key = (name or "mock").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown VLM backend {name!r}. Registered: {list_vlm_backend_names()}"
        )
    p: Dict[str, Any] = {}
    if params is not None:
        if isinstance(params, DictConfig):
            c = OmegaConf.to_container(params, resolve=True)
            p = dict(c) if isinstance(c, dict) else {}
        elif isinstance(params, dict):
            p = dict(params)
        else:
            raise TypeError(f"backend_params must be dict or DictConfig, got {type(params)}")
    return _REGISTRY[key](p)
