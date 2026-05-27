"""The :class:`Experiment` dataclass — a self-contained on-disk experiment."""

from __future__ import annotations

import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from policy_doctor.experiment.paths import experiment_dir

_SUBDIRS = ("config", "shared", "artifacts", "logs")


def _now_timestamp() -> str:
    """UTC timestamp suitable for filenames (no separators, sortable)."""
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class Experiment:
    """Handle to a self-contained on-disk experiment.

    Construct via :meth:`create` (new) or :meth:`load` (existing).
    """

    name: str
    root: Path

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.yaml"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def shared_dir(self) -> Path:
        return self.root / "shared"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        *,
        baseline_from: Optional[str] = None,
        manifest_extras: Optional[dict[str, Any]] = None,
    ) -> "Experiment":
        """Create a new experiment. Fails if it already exists."""
        root = experiment_dir(name)
        if root.exists():
            raise FileExistsError(
                f"Experiment '{name}' already exists at {root}. "
                f"Use Experiment.load(...) to resume, or pick a new name."
            )
        root.mkdir(parents=True, exist_ok=False)
        for sub in _SUBDIRS:
            (root / sub).mkdir(exist_ok=False)

        manifest: dict[str, Any] = {
            "name": name,
            "created_at": _now_timestamp(),
            "baseline_from": baseline_from,
        }
        if manifest_extras:
            manifest.update(manifest_extras)
        (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True))

        exp = cls(name=name, root=root)
        if baseline_from:
            exp._copy_baseline_from(baseline_from)
        return exp

    @classmethod
    def load(cls, name: str) -> "Experiment":
        """Load an existing experiment by name."""
        root = experiment_dir(name)
        if not (root / "manifest.yaml").is_file():
            raise FileNotFoundError(f"Experiment '{name}' not found at {root}")
        return cls(name=name, root=root)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return the parsed manifest contents."""
        return yaml.safe_load(self.manifest_path.read_text())

    def update_manifest(self, **updates: Any) -> dict[str, Any]:
        """Merge ``updates`` into the manifest and persist."""
        manifest = self.manifest()
        manifest.update(updates)
        self.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
        return manifest

    def append_config_snapshot(self, resolved_config: Any) -> Path:
        """Append a Hydra config snapshot. Returns the new snapshot path.

        The first snapshot is symlinked as ``config/canonical.yaml`` for
        easy reference.
        """
        ts = _now_timestamp()
        path = self.config_dir / f"snapshot_{ts}.yaml"
        # Avoid collision if two snapshots happen in the same second.
        suffix = 1
        while path.exists():
            path = self.config_dir / f"snapshot_{ts}_{suffix}.yaml"
            suffix += 1
        path.write_text(yaml.safe_dump(resolved_config, sort_keys=True))

        canonical = self.config_dir / "canonical.yaml"
        if not canonical.exists() and not canonical.is_symlink():
            canonical.symlink_to(path.name)
        return path

    def step_dir(self, step_name: str, *, version: Optional[str] = None) -> Path:
        """Return (and create) the dir for a pipeline step's artifacts.

        If ``version`` is given, the dir is suffixed (``<step>__<version>``)
        so a re-run can sit alongside the original.
        """
        name = f"{step_name}__{version}" if version else step_name
        d = self.artifacts_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def open_log(self, label: str = "invocation") -> Path:
        """Return a new per-invocation log file path."""
        ts = _now_timestamp()
        return self.logs_dir / f"{label}_{ts}.log"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _copy_baseline_from(self, source_name: str) -> None:
        """Hard-copy the baseline checkpoint dir from another experiment."""
        source = experiment_dir(source_name) / "shared" / "baseline_ckpt"
        if not source.exists():
            raise FileNotFoundError(
                f"baseline_from='{source_name}' but {source} doesn't exist. "
                f"Create the source experiment and populate its baseline first."
            )
        dst = self.shared_dir / "baseline_ckpt"
        shutil.copytree(source, dst, dirs_exist_ok=False)
