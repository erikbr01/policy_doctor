#!/usr/bin/env bash
# Create or update the main ``cupid`` conda env from ``conda_environment.yaml`` (Py 3.9,
# PyTorch 1.12 / cu11.6, diffusion_policy editable, robosuite fork, robomimic, MuJoCo 3.2.6, …).
#
# Run from the cupid repo root:
#   ./scripts/install_cupid_env.sh
#   ./scripts/install_cupid_env.sh --update
#
# From policy_doctor: use policy_doctor/scripts/install_cupid_env.sh
#
# Prerequisites (system packages for free-mujoco-py — see yaml comment):
#   e.g. Debian/Ubuntu: libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
#
# ``conda_environment.yaml`` pip section uses ``-e .`` and ``-e third_party/trak`` relative
# to this repo root; conda resolves them from the directory containing the yaml file.
#
set -euo pipefail

CUPID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CUPID_ROOT}"

YAML="${CUPID_ROOT}/conda_environment.yaml"
if [[ ! -f "${YAML}" ]]; then
  echo "Missing ${YAML}" >&2
  exit 1
fi

UPDATE=0
for arg in "$@"; do
  case "${arg}" in
    --update|-u) UPDATE=1 ;;
    -h|--help)
      echo "Usage: $0 [--update|-u]"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${UPDATE}" -eq 1 ]]; then
  echo "Updating conda env from ${YAML} ..."
  conda env update -n cupid -f "${YAML}" --prune
else
  if conda env list | grep -qE '^cupid\s'; then
    echo "Env 'cupid' already exists. Re-run with --update to sync from yaml, or:" >&2
    echo "  conda env remove -n cupid" >&2
    exit 1
  fi
  echo "Creating conda env 'cupid' from ${YAML} ..."
  conda env create -f "${YAML}"
fi

CONDA_BASE="$(conda info --base)"
CUPID_PY="${CONDA_BASE}/envs/cupid/bin/python"
echo "Verifying: ${CUPID_PY}"
"${CUPID_PY}" -c "import diffusion_policy; print('diffusion_policy OK:', diffusion_policy.__file__)"

echo "Done. Activate: conda activate cupid"
