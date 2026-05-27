"""Centralized helper for running pipeline steps in a target env.

Replaces the previous ``subprocess.run(["conda", "run", "-n", <env>, ...])``
pattern. Each step's config specifies ``uv_env: <name>`` (or
``data_source.uv_env_train``); this helper translates those names to the
corresponding uv extra and dispatches via ``scripts/uv_env.sh``.

Phase 5 renamed the YAML field from ``conda_env`` to ``uv_env``. Step config
selectors still accept the legacy ``conda_env*`` keys as a fallback so
external configs that haven't been migrated keep resolving.

The stored value may be either:

* A historical conda env name (``policy_doctor``, ``mimicgen_torch2``,
  ``cupid_torch25``, ``robocasa``, ...) — translated via ``_ENV_NAME_MAP``.
* A uv extra name directly (``analysis``, ``cupid``, ``mimicgen``,
  ``robocasa``) — passed through unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# Historical conda env names → uv extra names. Anything not in this map
# falls through unchanged so the existing extras (`analysis`, `cupid`,
# `mimicgen`, `robocasa`) work without translation. The legacy names are
# retained here so YAMLs that still use ``conda_env: <legacy_name>`` keep
# resolving via the backward-compat selector chain in each pipeline step.
_ENV_NAME_MAP: dict[str, str] = {
    "policy_doctor": "analysis",
    "policy_doctor_dagger": "analysis",
    "cupid": "cupid",
    "cupid_torch2": "cupid",
    "cupid_torch25": "cupid",
    "mimicgen": "mimicgen",
    "mimicgen_torch2": "mimicgen",
    "robocasa": "robocasa",
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_uv_extra(env_name: str) -> str:
    """Map a config-level env name to its uv extra name. Unmapped → pass-through."""
    return _ENV_NAME_MAP.get(env_name, env_name)


def run_in_env(
    env_name: str,
    cmd: list[str],
    *,
    cwd: Path | str | None = None,
    check: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` inside the uv env corresponding to ``env_name``.

    ``cmd`` should be a typical argv list (e.g. ``["python", "train.py", ...]``).
    This helper prepends ``scripts/uv_env.sh <extra>`` and dispatches via
    subprocess.

    Args:
        env_name: Historical conda env name OR a uv extra name directly.
        cmd:      Argv list to run inside the env.
        cwd:      Working directory. Defaults to repo root.
        check:    Raise CalledProcessError on non-zero exit (default False —
                  matches the prior conda-dispatch behavior).
        **kwargs: Forwarded to ``subprocess.run`` (env, stdout, stderr, etc.).
    """
    extra = resolve_uv_extra(env_name)
    wrapper = _REPO_ROOT / "scripts" / "uv_env.sh"
    full_cmd: list[str] = [str(wrapper), extra, *cmd]
    return subprocess.run(
        full_cmd,
        cwd=str(cwd) if cwd is not None else str(_REPO_ROOT),
        check=check,
        **kwargs,
    )
