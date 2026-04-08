#!/usr/bin/env bash
# Run this script on EACH node to set up the environment.
# Usage: source setup_env.sh
#   or:  . setup_env.sh
set -e

export VERL_TOOL_ROOT="/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool_071"
export AREAL_ROOT="/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL"
export PYTHONPATH="$VERL_TOOL_ROOT:$AREAL_ROOT:${PYTHONPATH:-}"

# Install dependencies
pip install timm fire iopath ftfy tensordict codetiming qwen_omni_utils nvitop httptools colorlog

echo "Environment setup complete on $(hostname)"
echo "  VERL_TOOL_ROOT=$VERL_TOOL_ROOT"
echo "  AREAL_ROOT=$AREAL_ROOT"