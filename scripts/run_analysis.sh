#!/bin/bash
# Kariba Solar Project - Fixed version

echo "=== Kariba Solar ERA5 Analysis ==="

# Source Conda
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: Conda not found at $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"

# Activate environment
conda activate deploy_env
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate 'deploy_env'"
    exit 1
fi

# Debug: Show Python info
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Run from correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go up one level from scripts/ to charara_01/

# Run the analysis
python era5_analysis_05.py "$@"