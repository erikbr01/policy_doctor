#!/usr/bin/env bash
# Setup the pi05-libero evaluation environment using openpi's uv-based toolchain.
#
# Prerequisites:
#   - uv installed (https://docs.astral.sh/uv/getting-started/installation/)
#   - CUDA 12 driver available
#   - ~40 GB free disk (checkpoint download)
#
# After setup, run evaluations with:
#   cd third_party/openpi
#   uv run examples/libero/eval_save_with_embeddings.py \
#       --task-suite-name libero_spatial \
#       --output-dir /path/to/output/libero_spatial \
#       --num-trials-per-task 20
#
# Then open the streamlit app:
#   conda activate policy_doctor
#   streamlit run policy_doctor/streamlit_app/app.py
# Select a "pi05_libero_*" config, go to the Clustering tab,
# choose "Policy embeddings", layer = "pi05", and run clustering.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="$REPO_ROOT/third_party/openpi"

echo "=== Setting up openpi uv environment ==="
cd "$OPENPI_DIR"

# Install base dependencies (JAX + CUDA, transformers, etc.)
uv sync

# Install the libero simulation environment from the bundled submodule
uv pip install -e third_party/libero

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run evaluation:"
echo "  cd $OPENPI_DIR"
echo "  uv run examples/libero/eval_save_with_embeddings.py \\"
echo "      --task-suite-name libero_spatial \\"
echo "      --output-dir /tmp/pi05_libero_spatial \\"
echo "      --num-trials-per-task 20"
echo ""
echo "The checkpoint (~10 GB) will be downloaded automatically to ~/.cache/openpi on first run."
echo ""
echo "After running eval, edit policy_doctor/configs/pi05_libero_spatial.yaml"
echo "and set:  eval_dir: /tmp/pi05_libero_spatial"
echo "Then launch streamlit and select the pi05_libero_spatial config."
