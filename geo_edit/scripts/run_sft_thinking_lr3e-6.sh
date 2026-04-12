#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# SFT: Qwen3-VL-8B-Thinking  |  lr=3e-6  |  reasonmap dataset
# Reference: /storage/openpsi/models/lcy_image_edit/sft_workspace/configs/qwen3vl8b-thinking-reasonmap.yaml
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/qwen3vl8b-thinking-reasonmap-lr3e-6.yaml"
LOG_DIR="/storage/openpsi/models/lcy_image_edit/sft_workspace/logs"
LOG_FILE="${LOG_DIR}/exp-thinking-reasonmap-lr3e-6.log"

mkdir -p "${LOG_DIR}"

echo "=== [$(date)] Starting SFT: Qwen3-VL-8B-Thinking, lr=3e-6 ==="
echo "Config : ${CONFIG}"
echo "Log    : ${LOG_FILE}"

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
  llamafactory-cli train "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== [$(date)] Training finished ==="
