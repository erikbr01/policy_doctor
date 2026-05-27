"""Unit tests for policy_doctor._env (env dispatch helper)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from policy_doctor import _env


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("policy_doctor", "analysis"),
        ("cupid", "cupid"),
        ("cupid_torch2", "cupid"),
        ("cupid_torch25", "cupid"),
        ("mimicgen", "mimicgen"),
        ("mimicgen_torch2", "mimicgen"),
        ("robocasa", "robocasa"),
        ("analysis", "analysis"),
        ("unknown_env", "unknown_env"),  # pass-through
    ],
)
def test_resolve_uv_extra(input_name: str, expected: str) -> None:
    assert _env.resolve_uv_extra(input_name) == expected


def test_run_in_env_constructs_wrapper_command() -> None:
    with mock.patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        _env.run_in_env("mimicgen_torch2", ["python", "-c", "print('ok')"], cwd="/tmp")
        called_args = mock_run.call_args.args[0]
        # First two args: wrapper path + resolved extra name
        assert called_args[0].endswith("scripts/uv_env.sh")
        assert called_args[1] == "mimicgen"  # mapped from mimicgen_torch2
        # Then the original cmd
        assert called_args[2:] == ["python", "-c", "print('ok')"]
        # cwd forwarded
        assert mock_run.call_args.kwargs["cwd"] == "/tmp"
        # check defaults to False (preserves prior conda-dispatch behavior)
        assert mock_run.call_args.kwargs["check"] is False


def test_run_in_env_default_cwd_is_repo_root() -> None:
    with mock.patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        _env.run_in_env("analysis", ["python", "--version"])
        cwd = Path(mock_run.call_args.kwargs["cwd"])
        # Repo root should contain the pyproject.toml
        assert (cwd / "pyproject.toml").is_file()
