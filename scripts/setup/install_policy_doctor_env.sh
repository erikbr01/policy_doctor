#!/usr/bin/env bash
# Install editable packages into the policy_doctor conda env.
# Prerequisite: conda env created from environment_policy_doctor.yaml
#   (the yaml is a full export with pinned CUDA torch; do not pip-install torch on top).
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
"${PD_PY}" -m pip install -e "${REPO_ROOT}/third_party/cupid"
"${PD_PY}" -m pip install -e "${REPO_ROOT}/third_party/cupid/third_party/infembed"
"${PD_PY}" -m pip install -e "${REPO_ROOT}/third_party/influence_visualizer"
"${PD_PY}" -m pip install -e "${REPO_ROOT}"

# infembed's pyproject.toml has `[tool.setuptools.packages.find] where = ["infembed"]`,
# which makes pip's editable .pth point one level too deep — `import infembed` then fails.
# Repoint the .pth at the package's parent dir.
SITE_PACKAGES="$("${PD_PY}" -c 'import site; print(site.getsitepackages()[0])')"
INFEMBED_PTH="$(ls "${SITE_PACKAGES}"/__editable__.infembed-*.pth 2>/dev/null | head -1)"
if [[ -n "${INFEMBED_PTH}" ]]; then
  echo "${REPO_ROOT}/third_party/cupid/third_party/infembed" > "${INFEMBED_PTH}"
  echo "Patched infembed .pth -> ${INFEMBED_PTH}"
fi

echo "Done. Activate with: conda activate policy_doctor"
