#!/usr/bin/env bash
# For script permissions: chmod +x setup.sh
set -euo pipefail

# ── Active-Learning Trace Labeler — one-command setup & run ──
# Prerequisites: conda or miniconda installed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="gui-labeler"

# Create conda env if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}' …"
    conda env create -f environment.yml
else
    echo "Conda environment '${ENV_NAME}' already exists."
fi

echo "To activate the environment and run the GUI labeler, use:
    conda activate ${ENV_NAME}
    python -m gui_labeler
"   
