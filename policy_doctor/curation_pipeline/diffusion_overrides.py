"""Extra Hydra overrides for cupid diffusion compose (baseline + curated train)."""

from __future__ import annotations

from typing import Any, List

from omegaconf import ListConfig, OmegaConf


def baseline_diffusion_extra_overrides(baseline: Any) -> List[str]:
    """Hydra overrides appended after standard training overrides.

    Supports:
    - ``diffusion_compose_overrides``: list of strings (e.g. ``task=square_mimicgen_lowdim``)
    - ``diffusion_dataset_path``: shorthand for ``++task.dataset.dataset_path`` and env_runner
    """
    if baseline is None:
        return []
    if isinstance(baseline, dict):
        baseline = OmegaConf.create(baseline)
    elif not OmegaConf.is_config(baseline):
        return []
    extra: List[str] = []
    raw = OmegaConf.select(baseline, "diffusion_compose_overrides")
    if raw is not None:
        if isinstance(raw, (list, tuple, ListConfig)):
            extra.extend(str(x) for x in raw)
        elif isinstance(raw, str) and raw.strip():
            extra.append(raw.strip())
    ds = OmegaConf.select(baseline, "diffusion_dataset_path")
    if ds:
        p = str(ds).strip()
        if p:
            extra.append(f"++task.dataset.dataset_path={p}")
            extra.append(f"++task.env_runner.dataset_path={p}")
    return extra
