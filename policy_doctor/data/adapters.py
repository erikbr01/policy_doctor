"""Third-party sim / dataset stacks: path helpers (no heavy imports at import time)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from policy_doctor.paths import MIMICGEN_ROOT, ROBOCASA_ROOT


def ensure_mimicgen_on_path() -> Path:
    """Insert vendored MimicGen repo on ``sys.path`` (same idea as tests support)."""
    root = MIMICGEN_ROOT.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def ensure_robocasa_on_path() -> Path:
    """Insert vendored RoboCasa repo on ``sys.path`` for scripts and optional imports."""
    root = ROBOCASA_ROOT.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


# ---------------------------------------------------------------------------
# Attribution dataset path resolution
# ---------------------------------------------------------------------------
# TRAK and InfEmbed load the training dataset via
#   ``hydra.utils.instantiate(cfg.task.dataset)``
# where ``cfg`` comes from the checkpoint saved at training time.  The stored
# ``dataset_path`` can be stale when machines change or data directories are
# renamed.  The functions below resolve a valid path across all three data
# sources (robomimic / mimicgen / robocasa) and patch the config in-place so
# the attribution scripts can instantiate the dataset without manual edits.
#
# Usage (in train_trak_diffusion.py / compute_infembed_embeddings.py):
#
#   from policy_doctor.data.adapters import patch_attribution_dataset_path
#   patch_attribution_dataset_path(cfg, repo_root=repo_root,
#                                   dataset_path_override=args.dataset_path)
#   train_set = hydra.utils.instantiate(cfg.task.dataset)
# ---------------------------------------------------------------------------

# Ordered candidate search paths for each data source (relative to repo_root / cupid root).
# Checked in order; first existing file wins.
_MIMICGEN_CANDIDATE_PATHS = [
    "data/mimicgen/square_merged.hdf5",
    "data/source/mimicgen/square_merged.hdf5",
    "data/mimicgen/square/mh/demo.hdf5",
]

_ROBOCASA_CANDIDATE_PATHS = [
    "data/robocasa/datasets/kitchen_lowdim_merged.hdf5",
    "data/source/robocasa/datasets/kitchen_lowdim_merged.hdf5",
    "data/robocasa/kitchen_lowdim_merged.hdf5",
]


def _detect_data_source(cfg) -> str:
    """Return 'mimicgen', 'robocasa', or 'robomimic' from checkpoint config."""
    try:
        target: str = cfg.task.dataset.get("_target_", "") or ""
    except Exception:
        target = ""
    target_lower = target.lower()
    if "mimicgen" in target_lower:
        return "mimicgen"
    if "robocasa" in target_lower:
        return "robocasa"
    # Fall back to inspecting the stored path.
    try:
        stored = str(cfg.task.dataset.get("dataset_path", "") or "")
    except Exception:
        stored = ""
    stored_lower = stored.lower()
    if "mimicgen" in stored_lower:
        return "mimicgen"
    if "robocasa" in stored_lower:
        return "robocasa"
    return "robomimic"


def _resolve_mimicgen_path(repo_root: Path) -> Optional[Path]:
    """Search known MimicGen HDF5 locations under repo_root."""
    for rel in _MIMICGEN_CANDIDATE_PATHS:
        p = repo_root / rel
        if p.exists():
            return p
    # Broad search: any demo.hdf5 under data/ with mimicgen in its path.
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.rglob("demo.hdf5")):
            if "mimicgen" in str(p).lower() or "core_datasets" in str(p):
                return p
        # Merged-style files named *square*merged*.hdf5 or *mimicgen*.hdf5.
        for p in sorted(data_dir.rglob("*.hdf5")):
            name = p.name.lower()
            if "mimicgen" in name or ("square" in name and "merged" in name):
                return p
    return None


def _resolve_robocasa_path(repo_root: Path) -> Optional[Path]:
    """Search known RoboCasa low-dim HDF5 locations under repo_root."""
    for rel in _ROBOCASA_CANDIDATE_PATHS:
        p = repo_root / rel
        if p.exists():
            return p
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.rglob("*.hdf5")):
            if "robocasa" in str(p).lower() and "lowdim" in p.name.lower():
                return p
    return None


def _resolve_robomimic_path(cfg, repo_root: Path) -> Optional[Path]:
    """Attempt to reconstruct the standard robomimic dataset path from task metadata."""
    try:
        task_name_full: str = str(cfg.task.dataset.get("dataset_path", "") or "")
    except Exception:
        task_name_full = ""
    # Parse parts[2] and parts[3] from a path like data/robomimic/datasets/square/mh/...
    p = Path(task_name_full)
    try:
        # parts: ('data', 'robomimic', 'datasets', task, dtype, file)
        if len(p.parts) >= 5 and p.parts[1] == "robomimic":
            task = p.parts[3]
            dtype = p.parts[4]
            filename = p.parts[-1]
            candidate = repo_root / "data" / "robomimic" / "datasets" / task / dtype / filename
            if candidate.exists():
                return candidate
    except Exception:
        pass
    # Heuristic: known robomimic tasks.
    robomimic_tasks = ["transport", "lift", "can", "square", "tool_hang"]
    try:
        task_name = str(getattr(cfg, "task_name", "") or "").lower()
    except Exception:
        task_name = ""
    matched = next((t for t in robomimic_tasks if t in task_name), None)
    if matched:
        dtype = "ph" if "ph" in task_name else "mh"
        filename = "image_abs.hdf5" if "image" in task_name else "low_dim_abs.hdf5"
        candidate = repo_root / "data" / "robomimic" / "datasets" / matched / dtype / filename
        if candidate.exists():
            return candidate
    return None


def resolve_attribution_dataset_path(
    cfg,
    repo_root: Path,
    dataset_path_override: Optional[str] = None,
) -> Optional[Path]:
    """Return a resolved dataset path for attribution, or None if not found.

    Priority:
    1. Explicit ``dataset_path_override`` argument.
    2. Stored path in ``cfg.task.dataset.dataset_path`` if it exists on disk.
    3. Data-source-specific fallback search under ``repo_root``.

    Args:
        cfg: Hydra DictConfig loaded from checkpoint (or similar namespace).
        repo_root: Cupid / project repo root for resolving relative paths.
        dataset_path_override: Optional explicit path from CLI / pipeline config.

    Returns:
        Resolved Path, or None if no file found.
    """
    repo_root = Path(repo_root).resolve()

    if dataset_path_override is not None and dataset_path_override.strip():
        p = Path(dataset_path_override)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p
        raise FileNotFoundError(
            f"attribution dataset_path_override not found: {p}"
        )

    # Try the stored path.
    stored = ""
    try:
        stored = str(cfg.task.dataset.get("dataset_path", "") or "")
    except Exception:
        pass
    if stored:
        p = Path(stored)
        if p.is_absolute() and p.exists():
            return p
        if not p.is_absolute() and (repo_root / p).exists():
            return (repo_root / p).resolve()

    # Fallback search by data source kind.
    kind = _detect_data_source(cfg)
    if kind == "mimicgen":
        return _resolve_mimicgen_path(repo_root)
    if kind == "robocasa":
        return _resolve_robocasa_path(repo_root)
    return _resolve_robomimic_path(cfg, repo_root)


def patch_attribution_dataset_path(
    cfg,
    repo_root: Path,
    dataset_path_override: Optional[str] = None,
) -> Optional[Path]:
    """Resolve and patch ``cfg.task.dataset.dataset_path`` in-place for attribution.

    Stores the absolute path so that ``hydra.utils.instantiate`` can open the
    file regardless of the current working directory.  Data-source detection
    (``_detect_data_source``) checks for substrings like ``'mimicgen'`` or
    ``'robocasa'``, which are present in the absolute path as well.

    Returns the resolved Path, or None if no file was found (cfg unchanged).
    """
    resolved = resolve_attribution_dataset_path(cfg, repo_root, dataset_path_override)
    if resolved is None:
        return None

    path_str = str(resolved)

    try:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass
    try:
        cfg.task.dataset.dataset_path = path_str
    except Exception:
        pass
    try:
        cfg.task.dataset_path = path_str
    except Exception:
        pass
    return resolved
