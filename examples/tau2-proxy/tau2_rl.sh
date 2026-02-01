#!/bin/bash
# Tau2 training with proxy server and archon backend
# This script runs in single-controller mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default config file
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

# Run training with slurm scheduler (single-controller mode)
# The scheduler.type=slurm enables single-controller mode
python3 tau2_train.py "$CONFIG_FILE" scheduler.type=slurm "$@"
