#!/usr/bin/env bash
# Create the ``mimicgen`` conda env if missing, then install MimicGen + pinned
# robosuite / robomimic / MuJoCo per NVlabs docs.
# For the diffusion / transport training stack (Py 3.9, MuJoCo 3.x), use install_cupid_env.sh instead.
#
# Pins (see https://mimicgen.github.io/docs/introduction/installation.html ):
#   - mujoco==2.3.2
#   - robosuite @ b9d8d3de5e3dfd1724f4a0e6555246c460407daa  (or v1.4.1-equivalent)
#   - robomimic @ d0b37cf214bd24fb590d182edb6384333f67b661
#   - PyTorch 1.12.1 (MimicGen troubleshooting; CPU wheels by default)
#
# Usage:
#   ./scripts/install_mimicgen_env.sh
#
# If the env already exists but you changed environment_mimicgen.yaml:
#   conda env update -f environment_mimicgen.yaml -n mimicgen
#
# CUDA PyTorch: set TORCH_INDEX before running, e.g.
#   export TORCH_INDEX=https://download.pytorch.org/whl/cu113
#   ./scripts/install_mimicgen_env.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

YAML="${REPO_ROOT}/environment_mimicgen.yaml"
if [[ ! -f "${YAML}" ]]; then
  echo "Missing ${YAML}" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
MG_PY="${CONDA_BASE}/envs/mimicgen/bin/python"
if [[ ! -x "${MG_PY}" ]]; then
  echo "Conda env 'mimicgen' not found; creating from ${YAML} ..."
  conda env create -f "${YAML}" --yes
fi
if [[ ! -x "${MG_PY}" ]]; then
  echo "Expected python at ${MG_PY} after conda env create." >&2
  exit 1
fi

ROBOSUITE_REF="b9d8d3de5e3dfd1724f4a0e6555246c460407daa"
ROBOMIMIC_REF="d0b37cf214bd24fb590d182edb6384333f67b661"
# MimicGen installation.md — Kitchen / Hammer Cleanup; strip deps to avoid mujoco-py
TASK_ZOO_REF="74eab7f88214c21ca1ae8617c2b2f8d19718a9ed"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

echo "Using: ${MG_PY}"
echo "PyTorch extra index: ${TORCH_INDEX}"

"${MG_PY}" -m pip install -U pip setuptools wheel

# MuJoCo — MimicGen docs: 2.3.2 for robosuite compatibility / dataset replay
"${MG_PY}" -m pip install "mujoco==2.3.2"

# PyTorch — robomimic depends on it; MimicGen docs suggest 1.12.1 for stability
"${MG_PY}" -m pip install \
  torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
  --extra-index-url "${TORCH_INDEX}"

# Pinned robosuite (not v1.5+ per MimicGen)
"${MG_PY}" -m pip install \
  "git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_REF}"

# Default robosuite import warns without macros_private.py; copy once (stderr may show 3 lines on this import).
"${MG_PY}" <<'PY'
import os
import shutil
import robosuite

b = robosuite.__path__[0]
src = os.path.join(b, "macros.py")
dst = os.path.join(b, "macros_private.py")
if not os.path.isfile(dst):
    shutil.copyfile(src, dst)
PY

# Task zoo at MimicGen-pinned commit; --no-deps skips deprecated mujoco-py from its setup.py
"${MG_PY}" -m pip install \
  "git+https://github.com/ARISE-Initiative/robosuite-task-zoo.git@${TASK_ZOO_REF}" \
  --no-deps

# Pinned robomimic (EnvBase API MimicGen scripts expect)
if ! "${MG_PY}" -m pip install \
  "git+https://github.com/ARISE-Initiative/robomimic.git@${ROBOMIMIC_REF}"; then
  echo "" >&2
  echo "robomimic install failed. If egl_probe fails, install build tools and retry:" >&2
  echo "  conda install -n mimicgen cmake" >&2
  echo "  # or: pip install cmake" >&2
  exit 1
fi

# Vendored MimicGen (policy_doctor submodule)
MIMICGEN_DIR="${REPO_ROOT}/third_party/mimicgen"
if [[ ! -f "${MIMICGEN_DIR}/setup.py" ]]; then
  echo "Missing ${MIMICGEN_DIR}. Initialize submodule:" >&2
  echo "  git submodule update --init --recursive third_party/mimicgen" >&2
  exit 1
fi
"${MG_PY}" -m pip install -e "${MIMICGEN_DIR}"

echo ""
echo "Done."
echo "  Headless / SSH (no DISPLAY):"
echo "    conda activate mimicgen"
echo "    MUJOCO_GL=egl python \"${REPO_ROOT}/scripts/mimicgen_headless_smoke_test.py\""
echo "    # if EGL fails: MUJOCO_GL=osmesa ...  or: xvfb-run -a python \".../demo_random_action.py\""
echo "  Interactive (needs X11 + DISPLAY):"
echo "    python \"${REPO_ROOT}/third_party/mimicgen/mimicgen/scripts/demo_random_action.py\""
echo "Later, to import policy_doctor from this env: pip install -e \"${REPO_ROOT}\""
