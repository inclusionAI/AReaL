#!/usr/bin/env bash
# Run this script on EACH node to set up the environment.
# Usage: source setup_env.sh
#   or:  . setup_env.sh
set -e

pip install vllm==0.17.0
# Install dependencies
pip install timm fire iopath ftfy tensordict codetiming qwen_omni_utils nvitop httptools colorlog
