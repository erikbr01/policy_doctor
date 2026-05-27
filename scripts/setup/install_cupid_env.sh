#!/usr/bin/env bash
# Create or update the ``cupid`` conda env using ``conda_environment.yaml`` from the cupid
# checkout (vendored ``third_party/cupid`` or sibling ``../cupid``).
#
#   ./scripts/install_cupid_env.sh
#   ./scripts/install_cupid_env.sh --update
#
# Logic matches ``cupid/scripts/install_cupid_env.sh``; this copy is self-contained so a
# vendored cupid tree without ``scripts/`` still works.
#
set -euo pipefail

PD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

discover_cupid_root() {
  if [[ -f "${PD_ROOT}/third_party/cupid/conda_environment.yaml" ]]; then
    echo "${PD_ROOT}/third_party/cupid"
    return 0
  fi
  if [[ -f "${PD_ROOT}/../cupid/conda_environment.yaml" ]]; then
    echo "$(cd "${PD_ROOT}/../cupid" && pwd)"
    return 0
  fi
  return 1
}

if ! CUPID_ROOT="$(discover_cupid_root)"; then
  echo "Could not find cupid/conda_environment.yaml. Expected either:" >&2
  echo "  ${PD_ROOT}/third_party/cupid/conda_environment.yaml" >&2
  echo "  ${PD_ROOT}/../cupid/conda_environment.yaml" >&2
  exit 1
fi

YAML="${CUPID_ROOT}/conda_environment.yaml"
cd "${CUPID_ROOT}"

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
  echo "Updating conda env 'cupid' from ${YAML} ..."
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

echo "Done. Cupid root: ${CUPID_ROOT}"
echo "Activate: conda activate cupid"
