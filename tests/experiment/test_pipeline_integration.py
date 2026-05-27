"""Tests for CurationPipeline ↔ Experiment integration.

Verifies that:
  * `cfg.experiment_name` makes the pipeline create-or-resume an Experiment.
  * The pipeline's `run_dir` becomes the experiment's `artifacts_dir`.
  * Config snapshots accumulate under `<experiment>/config/`.
  * Legacy `run_name` / `run_dir` configs keep working unchanged.
  * Steps receive `experiment` as a kwarg and can access it via `self.experiment`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.pipeline import CurationPipeline
from policy_doctor.experiment import Experiment


class _ProbeStep(PipelineStep[dict]):
    """No-op step that records what it received."""

    name = "probe_step"
    observed: dict = {}

    def compute(self) -> dict:
        _ProbeStep.observed = {
            "run_dir": self.run_dir,
            "step_dir": self.step_dir,
            "experiment_present": self.experiment is not None,
            "experiment_name": self.experiment.name if self.experiment else None,
        }
        return {"ok": True}


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("POLICY_DOCTOR_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _build_step_registry so we don't import every heavy step."""
    import policy_doctor.curation_pipeline.pipeline as pipeline_mod

    def fake_registry() -> dict:
        return {"probe_step": _ProbeStep}

    monkeypatch.setattr(pipeline_mod, "_build_step_registry", fake_registry)
    # Also widen ALL_STEPS so the unknown-step check passes.
    monkeypatch.setattr(pipeline_mod, "ALL_STEPS", ["probe_step"])


def test_pipeline_with_experiment_name_creates_experiment(
    tmp_data_root: Path, patch_registry: None
) -> None:
    cfg = OmegaConf.create({"experiment_name": "foo", "steps": ["probe_step"]})
    pipeline = CurationPipeline(cfg)
    assert pipeline.experiment is not None
    assert pipeline.experiment.name == "foo"
    assert pipeline.run_dir == pipeline.experiment.artifacts_dir
    pipeline.run()
    assert _ProbeStep.observed["experiment_present"] is True
    assert _ProbeStep.observed["experiment_name"] == "foo"
    assert _ProbeStep.observed["run_dir"] == pipeline.experiment.artifacts_dir
    assert _ProbeStep.observed["step_dir"] == (
        pipeline.experiment.artifacts_dir / "probe_step"
    )


def test_pipeline_resumes_existing_experiment(
    tmp_data_root: Path, patch_registry: None
) -> None:
    Experiment.create("resumeme")
    cfg = OmegaConf.create({"experiment_name": "resumeme"})
    pipeline = CurationPipeline(cfg)
    assert pipeline.experiment is not None
    assert pipeline.experiment.name == "resumeme"


def test_config_snapshot_appended(tmp_data_root: Path, patch_registry: None) -> None:
    cfg = OmegaConf.create({"experiment_name": "snap"})
    CurationPipeline(cfg)
    snapshot_dir = (Experiment.load("snap").config_dir)
    snapshots = sorted(p.name for p in snapshot_dir.glob("snapshot_*.yaml"))
    assert len(snapshots) == 1
    # A second invocation appends, not overwrites.
    CurationPipeline(cfg)
    snapshots = sorted(p.name for p in snapshot_dir.glob("snapshot_*.yaml"))
    assert len(snapshots) == 2
    canonical = snapshot_dir / "canonical.yaml"
    assert canonical.is_symlink()


def test_legacy_run_dir_still_works(
    tmp_data_root: Path, tmp_path: Path, patch_registry: None
) -> None:
    legacy_run_dir = tmp_path / "legacy_run"
    cfg = OmegaConf.create(
        {
            "run_dir": str(legacy_run_dir),
            "repo_root": str(tmp_path),
            "steps": ["probe_step"],
        }
    )
    pipeline = CurationPipeline(cfg)
    assert pipeline.experiment is None
    assert pipeline.run_dir == legacy_run_dir
    assert (legacy_run_dir / "pipeline_config.yaml").is_file()
    pipeline.run()
    assert _ProbeStep.observed["experiment_present"] is False
    assert _ProbeStep.observed["step_dir"] == legacy_run_dir / "probe_step"


def test_baseline_from_propagates(tmp_data_root: Path, patch_registry: None) -> None:
    src = Experiment.create("src")
    (src.shared_dir / "baseline_ckpt").mkdir()
    (src.shared_dir / "baseline_ckpt" / "model.pt").write_bytes(b"ckpt")

    cfg = OmegaConf.create({"experiment_name": "tgt", "baseline_from": "src"})
    pipeline = CurationPipeline(cfg)
    assert pipeline.experiment is not None
    copied = pipeline.experiment.shared_dir / "baseline_ckpt" / "model.pt"
    assert copied.is_file()
    assert copied.read_bytes() == b"ckpt"
