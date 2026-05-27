"""Tests for the experiment-bundle CLI."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from policy_doctor.experiment import Experiment
from policy_doctor.scripts.experiment_bundle import main as experiment_bundle


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("POLICY_DOCTOR_DATA", str(tmp_path))
    return tmp_path


def test_bundle_writes_tarball(
    tmp_data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = Experiment.create("foo")
    (exp.artifacts_dir / "run_clustering").mkdir()
    (exp.artifacts_dir / "run_clustering" / "cluster_labels.npy").write_bytes(b"data")

    out = tmp_path / "out.tar.gz"
    monkeypatch.chdir(tmp_path)
    rc = experiment_bundle(["foo", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    assert "foo/manifest.yaml" in names
    assert "foo/artifacts/run_clustering/cluster_labels.npy" in names


def test_bundle_dereferences_symlinks(
    tmp_data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_dataset = tmp_path / "real_dataset.hdf5"
    real_dataset.write_bytes(b"hdf5 contents")

    exp = Experiment.create("foo")
    (exp.shared_dir / "source_dataset.hdf5").symlink_to(real_dataset)

    out = tmp_path / "out.tar.gz"
    monkeypatch.chdir(tmp_path)
    rc = experiment_bundle(["foo", "--out", str(out)])
    assert rc == 0

    with tarfile.open(out, "r:gz") as tar:
        member = tar.getmember("foo/shared/source_dataset.hdf5")
        # Inside the tarball, the symlink has been materialized: it's now a
        # regular file with the source bytes, not a symlink.
        assert member.isfile()
        f = tar.extractfile(member)
        assert f is not None
        assert f.read() == b"hdf5 contents"


def test_bundle_missing_experiment_errors(
    tmp_data_root: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = experiment_bundle(["doesnotexist"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
