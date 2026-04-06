#!/usr/bin/env bash
# Install pip-only deps and all three editable packages into conda env `policy_doctor`.
# Prerequisites: conda env created from environment_policy_doctor.yaml
#
# For diffusion / robomimic training (separate stack), use scripts/install_cupid_env.sh.
# For MimicGen (MuJoCo 2.3 / pinned robosuite), use scripts/install_mimicgen_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BASE="$(conda info --base)"
PD_PY="${CONDA_BASE}/envs/policy_doctor/bin/python"
if [[ ! -x "${PD_PY}" ]]; then
  echo "Python not found at ${PD_PY}. Create the env first:" >&2
  echo "  conda env create -f ${REPO_ROOT}/environment_policy_doctor.yaml" >&2
  exit 1
fi

echo "Using: ${PD_PY}"
"${PD_PY}" -m pip install -U pip setuptools wheel
"${PD_PY}" -m pip install -r "${REPO_ROOT}/requirements_policy_doctor.txt"
"${PD_PY}" -m pip install -e "${REPO_ROOT}/third_party/cupid"
"${PD_PY}" -m pip install -e "${REPO_ROOT}/third_party/influence_visualizer"
"${PD_PY}" -m pip install -e "${REPO_ROOT}"

echo "Done. Activate with: conda activate policy_doctor"
