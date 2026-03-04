#!/usr/bin/env bash
# For script permissions: chmod +x run.sh

set -euo pipefail

# ── Active-Learning Trace Labeler — one-command setup & run ──
# Prerequisites: conda or miniconda installed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME="gui-labeler"


# Check env exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Missing conda environment '${ENV_NAME}'"
    echo "Run setup.sh first to create the environment."
    exit 1
fi

echo "Starting GUI Labeler …"
conda run -n "$ENV_NAME" python -m gui_labeler "$@"