#!/usr/bin/env bash
# Bootstrap cupid_torch25 directly from third_party/cupid/conda_environment.yaml.
# Unlike create_cupid_torch25.sh, this does NOT require an existing cupid_torch2
# env to copy package versions from — it installs the cupid yaml's pinned deps
# from PyPI, then overrides torch with 2.5.1+cu124 and adds the editable installs.
#
# Server-only flavour: skips trak / mimicgen / influence_visualizer / r3m /
# free-mujoco-py / pybullet — none are needed for policy_server.py + HttpPolicy.
# Skips pytorch3d (very heavy, only used for training-time augmentations).
#
# Usage:
#   bash scripts/create_cupid_torch25_from_yaml.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ENV=cupid_torch25

if conda env list | grep -q "^${ENV} "; then
    echo "env ${ENV} already exists — remove with: conda env remove -n ${ENV} -y"
    exit 1
fi

echo "--- [1/6] create env (python 3.9) ---"
conda create -n "${ENV}" python=3.9 -y

echo "--- [2/6] torch 2.5.1+cu124 ---"
conda run -n "${ENV}" pip install \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "--- [3/6] av via conda-forge (bundled ffmpeg) ---"
conda install -n "${ENV}" -c conda-forge "av>=10,<11" -y

echo "--- [4/6] pinned deps from cupid yaml ---"
conda run -n "${ENV}" pip install \
    "numpy==1.23.3" "numba==0.56.4" "scipy==1.9.1" \
    "opencv-python==4.6.0.66" "h5py==3.7.0" \
    "hydra-core==1.2.0" "einops==0.4.1" "tqdm==4.64.1" \
    "dill==0.3.5.1" "zarr==2.12.0" "numcodecs==0.10.2" \
    "matplotlib==3.6.1" "imageio==2.22.0" "imageio-ffmpeg==0.4.7" \
    "termcolor==2.0.1" "click==8.0.4" "pandas==1.5.3" \
    "diffusers==0.11.1" "accelerate==0.21.0" \
    "transformers==4.34.1" "tokenizers==0.14.1" \
    "huggingface-hub==0.17.3" \
    "protobuf==3.19.6" "urllib3==1.26.19" \
    "flask" "requests" "websockets"

# robomimic + robosuite without mujoco-py (we never simulate; just need imports)
echo "--- [5/6] robomimic / robosuite no-deps ---"
conda run -n "${ENV}" pip install --no-deps "robomimic==0.2.0" "robosuite==1.2.0"
conda run -n "${ENV}" pip install --no-deps "gym==0.26.2" "pytorchvideo==0.1.5"

echo "--- [6/6] editable: diffusion_policy ---"
conda run -n "${ENV}" pip install --no-deps -e third_party/cupid
# policy_doctor itself (so policy_server.py and PolicyClient are importable)
conda run -n "${ENV}" pip install --no-deps -e .

echo ""
conda run -n "${ENV}" python -c "
import torch, diffusion_policy, hydra, flask
print(f'=== cupid_torch25 ready ===')
print(f'  torch:           {torch.__version__}')
print(f'  cuda available:  {torch.cuda.is_available()}')
print(f'  diffusion_policy: {diffusion_policy.__file__}')
"
