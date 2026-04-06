"""Filesystem roots for the ``policy_doctor`` package."""

from __future__ import annotations

from pathlib import Path

# Inner package dir (``policy_doctor/policy_doctor/``): configs, curation_pipeline, …
PACKAGE_ROOT = Path(__file__).resolve().parent

# Standalone project dir (repo root: pyproject.toml, tests/, third_party/, …)
PROJECT_ROOT = PACKAGE_ROOT.parent

# Vendored dependencies (standalone layout)
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
CUPID_ROOT = THIRD_PARTY_ROOT / "cupid"
# Conda env from cupid ``conda_environment.yaml`` (diffusion / transport training stack).
CUPID_CONDA_ENV_NAME = "cupid"
INFLUENCE_VISUALIZER_ROOT = THIRD_PARTY_ROOT / "influence_visualizer"
MIMICGEN_ROOT = THIRD_PARTY_ROOT / "mimicgen"
# Conda env from ``environment_mimicgen.yaml`` (MuJoCo 2.3.2 + pinned robosuite/robomimic; see MimicGen docs).
MIMICGEN_CONDA_ENV_NAME = "mimicgen"
IV_CONFIGS_DIR = INFLUENCE_VISUALIZER_ROOT / "configs"


def _default_repo_root() -> Path:
    """Training / eval repo root: vendored ``cupid`` when present, else monorepo cupid root."""
    if CUPID_ROOT.is_dir():
        return CUPID_ROOT
    parent = PROJECT_ROOT.parent
    if (parent / "influence_visualizer").is_dir():
        return parent
    return PROJECT_ROOT


# Hydra train configs, eval_save_episodes, data/outputs/* live under this tree
REPO_ROOT = _default_repo_root()

CONFIGS_DIR = PACKAGE_ROOT / "configs"


def iv_task_configs_base(cupid_repo_root: Path | None = None) -> Path:
    """Per-task ``*.yaml`` and ``<task>/clustering/`` for IV-backed ``config_root=iv``.

    Standalone: ``third_party/influence_visualizer/configs``. Monorepo: sibling
    ``<cupid>/influence_visualizer/configs`` (when vendored layout is absent).
    """
    if IV_CONFIGS_DIR.is_dir():
        return IV_CONFIGS_DIR
    root = cupid_repo_root if cupid_repo_root is not None else REPO_ROOT
    return root / "influence_visualizer" / "configs"
