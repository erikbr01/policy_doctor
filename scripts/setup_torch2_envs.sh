#!/usr/bin/env bash
# Bootstrap the canonical torch-2 conda envs on a fresh machine.
#
# Final state:
#   policy_doctor    — analysis / orchestration / clustering / InfEmbed (no robosuite/mimicgen).
#   mimicgen_torch2  — clone of policy_doctor + robosuite 1.4.1 + robomimic 0.3.0 + mimicgen 1.0.0.
#
# Re-runnable: if an env already exists the corresponding step is skipped.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

echo "Step 1/3  policy_doctor (analysis env)"
if env_exists policy_doctor; then
  echo "  policy_doctor already exists — skip."
else
  conda env create -f "${REPO}/environment_policy_doctor.yaml"
  echo "  policy_doctor created"
fi

echo "Step 2/3  mimicgen_torch2 (sim env = clone of policy_doctor + sim deps)"
if env_exists mimicgen_torch2; then
  echo "  mimicgen_torch2 already exists — skip."
else
  conda create --name mimicgen_torch2 --clone policy_doctor --yes
  conda activate mimicgen_torch2
  pip install --upgrade \
    'robosuite==1.4.1' \
    'robomimic==0.3.0' \
    'mimicgen==1.0.0'
  conda deactivate
  echo "  mimicgen_torch2 created and sim deps installed"
fi

echo "Step 3/3  install editable packages into policy_doctor"
"${REPO}/scripts/install_policy_doctor_env.sh"

echo
echo "All envs ready."
echo "  conda activate policy_doctor    # analysis / orchestration"
echo "  conda activate mimicgen_torch2  # training / sim / eval"
