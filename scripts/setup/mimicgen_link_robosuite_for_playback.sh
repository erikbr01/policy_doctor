#!/usr/bin/env bash
# Prefer fixing HDF5 in place (no fake directory tree):
#   python scripts/rewrite_robomimic_model_xml_paths_in_hdf5.py --backup demo.hdf5
#
# If robomimic playback / make_dataset_video fails with:
#   could not open STL file '.../third_party/mimicgen/mimicgen/lib/python3.8/site-packages/robosuite/...'
# the generated HDF5 stores absolute mesh paths from whatever robosuite prefix was used during
# generate_dataset. When that prefix was wrong, symlink the expected directory to this env's
# robosuite package so MuJoCo can open the STLs.
#
# Usage (mimicgen env active):
#   ./scripts/mimicgen_link_robosuite_for_playback.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRONG_PARENT="${REPO_ROOT}/third_party/mimicgen/mimicgen/lib/python3.8/site-packages"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate your mimicgen conda env first (need CONDA_PREFIX)." >&2
  exit 1
fi

MG_PY="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${MG_PY}" ]]; then
  echo "No python at ${MG_PY}" >&2
  exit 1
fi

RS_SRC="$("${MG_PY}" -c "import os, robosuite; print(os.path.dirname(robosuite.__file__))")"
RS_DST="${WRONG_PARENT}/robosuite"

if [[ ! -d "${RS_SRC}/models/assets" ]]; then
  echo "robosuite assets not found under ${RS_SRC}" >&2
  exit 1
fi

mkdir -p "${WRONG_PARENT}"
if [[ -e "${RS_DST}" && ! -L "${RS_DST}" ]]; then
  echo "${RS_DST} exists and is not a symlink; remove or move it, then re-run." >&2
  exit 1
fi

ln -sfn "${RS_SRC}" "${RS_DST}"
echo "Linked:"
echo "  ${RS_DST}"
echo "  -> ${RS_SRC}"
echo "Retry make_dataset_video or robomimic playback."
