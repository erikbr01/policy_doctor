"""Tests for policy_doctor.experiment.Experiment."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from policy_doctor.experiment import Experiment, experiments_dir


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin POLICY_DOCTOR_DATA to a fresh tmp dir for each test."""
    monkeypatch.setenv("POLICY_DOCTOR_DATA", str(tmp_path))
    return tmp_path


def test_create_makes_skeleton(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    assert exp.root == experiments_dir() / "foo"
    assert exp.manifest_path.is_file()
    for sub in (exp.config_dir, exp.shared_dir, exp.artifacts_dir, exp.logs_dir):
        assert sub.is_dir()


def test_manifest_has_expected_keys(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    manifest = yaml.safe_load(exp.manifest_path.read_text())
    assert manifest["name"] == "foo"
    assert "created_at" in manifest
    assert manifest["baseline_from"] is None


def test_create_fails_if_exists(tmp_data_root: Path) -> None:
    Experiment.create("foo")
    with pytest.raises(FileExistsError):
        Experiment.create("foo")


def test_load_existing(tmp_data_root: Path) -> None:
    Experiment.create("foo")
    exp = Experiment.load("foo")
    assert exp.name == "foo"


def test_load_nonexistent_raises(tmp_data_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Experiment.load("missing")


def test_append_config_snapshot(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    p1 = exp.append_config_snapshot({"foo": "bar"})
    p2 = exp.append_config_snapshot({"foo": "baz"})
    assert p1.is_file() and p2.is_file() and p1 != p2

    canonical = exp.config_dir / "canonical.yaml"
    assert canonical.is_symlink()
    # Symlink points at the first snapshot.
    assert (canonical.parent / canonical.readlink()).resolve() == p1


def test_step_dir_versioned(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    d1 = exp.step_dir("run_clustering")
    d2 = exp.step_dir("run_clustering", version="v2")
    assert d1 == exp.artifacts_dir / "run_clustering"
    assert d2 == exp.artifacts_dir / "run_clustering__v2"
    assert d1.is_dir() and d2.is_dir()


def test_baseline_from_copies(tmp_data_root: Path) -> None:
    src = Experiment.create("source")
    ckpt = src.shared_dir / "baseline_ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"fake checkpoint")

    tgt = Experiment.create("target", baseline_from="source")
    copied = tgt.shared_dir / "baseline_ckpt" / "model.pt"
    assert copied.is_file()
    assert copied.read_bytes() == b"fake checkpoint"
    # Hard copy, not symlink — portable across machines.
    assert not (tgt.shared_dir / "baseline_ckpt").is_symlink()


def test_baseline_from_missing_source_raises(tmp_data_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="baseline_from"):
        Experiment.create("target", baseline_from="nonexistent")


def test_update_manifest(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    exp.update_manifest(extra_field="hello", baseline_from="changed")
    manifest = exp.manifest()
    assert manifest["extra_field"] == "hello"
    assert manifest["baseline_from"] == "changed"


def test_invalid_name_rejected(tmp_data_root: Path) -> None:
    for bad in ("../escape", "with/slash", ".hidden", ""):
        with pytest.raises(ValueError):
            Experiment.create(bad)


def test_open_log(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    log_path = exp.open_log()
    assert log_path.parent == exp.logs_dir
    assert log_path.name.startswith("invocation_")
    assert log_path.name.endswith(".log")
