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

echo "Step 3/4  install editable packages into policy_doctor + mimicgen_torch2"
"${REPO}/scripts/install_policy_doctor_env.sh"
# mimicgen_torch2 also needs `import policy_doctor` for our chained-warp
# generator subclass, which scripts/run_mimicgen_generate.py imports at runtime.
conda run -n mimicgen_torch2 pip install -e "${REPO}"

echo "Step 4/4  install LD_LIBRARY_PATH hook for torch's bundled nvidia libs"
# System cuDNN at /usr/lib (e.g. 9.13 for CUDA 13) shadows torch's bundled cuDNN
# (e.g. 9.10.2 for CUDA 12.8). When system cuDNN dlopens libcublasLt.so.13 (CUDA 13)
# and that's not installed → "Invalid handle. Cannot load symbol cublasLtCreate"
# crashes on the first matmul under torch.compile. Force the bundled libs to win.
install_torch_ld_hook() {
  local env_name="$1"
  local prefix; prefix="$(conda info --envs | awk -v e="$env_name" '$1==e {print $NF}')"
  if [ -z "$prefix" ]; then echo "  ! env $env_name not found, skipping hook"; return; fi
  mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
  cat > "$prefix/etc/conda/activate.d/torch_cuda_libs.sh" <<'EOF'
_TORCH_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia"
if [ -d "$_TORCH_NVIDIA_DIR" ]; then
  _TORCH_NV_LIBS=""
  for d in cudnn cublas cuda_cupti cuda_nvrtc cuda_runtime cufft curand cusolver cusparse cusparselt nccl nvtx cufile; do
    [ -d "$_TORCH_NVIDIA_DIR/$d/lib" ] && _TORCH_NV_LIBS="$_TORCH_NV_LIBS:$_TORCH_NVIDIA_DIR/$d/lib"
  done
  export _PRIOR_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${_TORCH_NV_LIBS#:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
EOF
  cat > "$prefix/etc/conda/deactivate.d/torch_cuda_libs.sh" <<'EOF'
if [ -n "${_PRIOR_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$_PRIOR_LD_LIBRARY_PATH"
  unset _PRIOR_LD_LIBRARY_PATH
fi
EOF
  echo "  installed for: $env_name"
}
install_torch_ld_hook policy_doctor
install_torch_ld_hook mimicgen_torch2

echo
echo "All envs ready."
echo "  conda activate policy_doctor    # analysis / orchestration"
echo "  conda activate mimicgen_torch2  # training / sim / eval"
