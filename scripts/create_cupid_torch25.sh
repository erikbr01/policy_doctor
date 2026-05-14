#!/usr/bin/env bash
# Creates the cupid_torch25 conda environment — a torch 2.5.1+cu124 upgrade of
# cupid_torch2 that fixes the TensorAlias AOT-autograd bug, enabling full
# torch.compile on both obs_encoder and UNet (1.19× fwd+bwd speedup).
#
# Prerequisites:
#   - cupid_torch2 must exist (used as the package-version reference)
#   - Run from the policy_doctor project root
#   - Requires internet access and ~20 GB disk (pytorch3d builds from source)
#
# Usage:
#   bash scripts/create_cupid_torch25.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ENV=cupid_torch25

if conda env list | grep -q "^${ENV} "; then
    echo "env ${ENV} already exists — remove it first if you want to recreate:"
    echo "  conda env remove -n ${ENV} -y"
    exit 1
fi

# ── 1. Base env ────────────────────────────────────────────────────────────────
echo "--- [1/8] creating base env (Python 3.9) ---"
conda create -n "${ENV}" python=3.9 -y

# ── 2. PyTorch 2.5.1 ──────────────────────────────────────────────────────────
echo "--- [2/8] installing torch 2.5.1+cu124 ---"
conda run -n "${ENV}" pip install \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. av (needs bundled ffmpeg — use conda-forge, not pip) ───────────────────
echo "--- [3/8] installing av via conda-forge ---"
conda install -n "${ENV}" -c conda-forge "av>=10,<11" -y

# ── 4. Bulk deps from cupid_torch2 ────────────────────────────────────────────
# Export the versioned list, then strip packages that need special handling:
#   - torch/torchvision: already at 2.5.1
#   - av: installed above via conda
#   - dataclasses: Python 3.9 built-in (no PyPI 0.8 wheel)
#   - free-mujoco-py / gym / pybullet-svl / pytorch3d / pytorchvideo: below
#   - pyzed: ZED camera SDK (hardware-specific, not on PyPI)
#   - r3m: no PyPI release
echo "--- [4/8] bulk-installing deps matching cupid_torch2 ---"
REQS=$(mktemp)
conda run -n cupid_torch2 pip list --format=freeze 2>/dev/null \
    | grep -v "^-e" \
    | grep -v -E "^(torch==|torchvision==|torchaudio==|av==|dataclasses==|free-mujoco-py==|gym==|pybullet-svl==|pytorch3d==|pytorchvideo==|pyzed==|r3m==|robomimic==|robosuite==)" \
    > "${REQS}"
# diffusers 0.11.1 uses HfFolder which was removed in huggingface_hub 1.x
sed -i 's/^huggingface-hub==.*/huggingface-hub==0.17.3/' "${REQS}"
conda run -n "${ENV}" pip install --no-deps -r "${REQS}"
rm "${REQS}"

# ── 5. robomimic / robosuite without mujoco-py ────────────────────────────────
# robosuite 1.2.0 requires mujoco-py 2.0.2.9 which needs the legacy MuJoCo
# 2.0 binary.  The benchmark only uses neural-network code, not simulation.
echo "--- [5/8] installing robomimic + robosuite (--no-deps, no mujoco-py) ---"
conda run -n "${ENV}" pip install --no-deps "robomimic==0.2.0" "robosuite==1.2.0"

# ── 6. Packages that need --no-deps or specific versions ──────────────────────
echo "--- [6/8] installing gym / free-mujoco-py / pybullet-svl / pytorchvideo ---"
# gym 0.21.0 won't build on modern setuptools; 0.26.x has the same public API
conda run -n "${ENV}" pip install --no-deps "gym==0.26.2"
conda run -n "${ENV}" pip install --no-deps "free-mujoco-py==2.1.6"
conda run -n "${ENV}" pip install --no-deps "pybullet-svl==3.1.6.4"
conda run -n "${ENV}" pip install --no-deps "pytorchvideo==0.1.5"

# ── 7. pytorch3d (no prebuilt wheel for py39/cu124 — build from source) ───────
echo "--- [7/8] building pytorch3d from source (takes ~15 min) ---"
conda run -n "${ENV}" pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# ── 8. Editable installs ──────────────────────────────────────────────────────
echo "--- [8/8] installing editable packages ---"
conda run -n "${ENV}" pip install -e third_party/cupid
conda run -n "${ENV}" pip install --no-deps -e third_party/cupid/third_party/infembed
conda run -n "${ENV}" pip install --no-deps -e third_party/cupid/third_party/trak
conda run -n "${ENV}" pip install --no-deps -e third_party/mimicgen
conda run -n "${ENV}" pip install --no-deps -e third_party/influence_visualizer

echo ""
conda run -n "${ENV}" python -c "import torch; print(f'=== done: torch {torch.__version__} ===')"
