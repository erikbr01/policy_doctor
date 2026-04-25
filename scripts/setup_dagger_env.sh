#!/bin/bash
# Setup script for policy_doctor_dagger environment on macOS
#
# Usage:
#   bash scripts/setup_dagger_env.sh
#

set -e

echo "==================================="
echo "Creating policy_doctor_dagger env"
echo "==================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Create environment
echo "Creating conda environment..."
conda env create -f conda_environment_dagger_macos.yaml

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate policy_doctor_dagger

# Install local packages in editable mode
echo "Installing policy_doctor packages..."
pip install -e .
pip install -e third_party/cupid
pip install -e third_party/cupid/third_party/infembed
pip install -e third_party/robocasa

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate policy_doctor_dagger"
echo ""
echo "To run DAgger, see:"
echo "  python scripts/run_dagger_robocasa.py --help"
echo ""
echo "Note: On first run, you may be prompted for Accessibility permission"
echo "(for keyboard input). Grant permission in System Preferences > Security & Privacy."
echo ""
