"""Tests for the experiment-init CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from policy_doctor.experiment import Experiment
from policy_doctor.scripts.experiment_init import main as experiment_init


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("POLICY_DOCTOR_DATA", str(tmp_path))
    return tmp_path


def test_experiment_init_creates_experiment(
    tmp_data_root: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = experiment_init(["foo"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Created experiment at:" in captured.out
    exp = Experiment.load("foo")
    assert exp.manifest_path.is_file()


def test_experiment_init_baseline_from_copies(tmp_data_root: Path) -> None:
    src = Experiment.create("src")
    (src.shared_dir / "baseline_ckpt").mkdir()
    (src.shared_dir / "baseline_ckpt" / "model.pt").write_bytes(b"ckpt")

    rc = experiment_init(["tgt", "--baseline-from", "src"])
    assert rc == 0
    tgt = Experiment.load("tgt")
    assert (tgt.shared_dir / "baseline_ckpt" / "model.pt").read_bytes() == b"ckpt"


def test_experiment_init_duplicate_name_errors(
    tmp_data_root: Path, capsys: pytest.CaptureFixture
) -> None:
    Experiment.create("foo")
    rc = experiment_init(["foo"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err


def test_experiment_init_missing_baseline_errors(
    tmp_data_root: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = experiment_init(["tgt", "--baseline-from", "nonexistent"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "doesn't exist" in captured.err


def test_seed_dir_helpers(tmp_data_root: Path) -> None:
    exp = Experiment.create("foo")
    sd = exp.seed_dir("train_baseline", 7)
    assert sd == exp.artifacts_dir / "train_baseline" / "seed_7"
    assert sd.is_dir()
    ck = exp.ckpt_dir("eval_policies", "0", "latest")
    assert ck == exp.artifacts_dir / "eval_policies" / "seed_0" / "latest"
    assert ck.is_dir()
    # Versioned arm support carries through
    sd_v = exp.seed_dir("train_baseline", 7, version="v2")
    assert sd_v == exp.artifacts_dir / "train_baseline__v2" / "seed_7"
