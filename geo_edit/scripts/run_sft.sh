#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# Generic LLaMA-Factory SFT launcher
# Usage: bash run_sft.sh <config.yaml> [NPROC]
#   config.yaml  - path to llamafactory YAML config
#   NPROC        - number of GPUs (default: 8)
#
# Examples:
#   bash run_sft.sh configs/qwen3vl8b-thinking-235B-lr1e-5.yaml
#   bash run_sft.sh configs/qwen3vl8b-thinking-235B-lr3e-6.yaml 4
# =============================================================

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [NPROC]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$1"

# Resolve relative paths against script dir
if [[ ! "$CONFIG" = /* ]]; then
  CONFIG="${SCRIPT_DIR}/${CONFIG}"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config file not found: ${CONFIG}"
  exit 1
fi

NPROC="${2:-8}"

# Derive experiment name from config filename (strip path + extension)
EXP_NAME="$(basename "${CONFIG}" .yaml)"
LOG_DIR="/storage/openpsi/models/lcy_image_edit/sft_workspace/logs"
LOG_FILE="${LOG_DIR}/exp-${EXP_NAME}.log"

mkdir -p "${LOG_DIR}"

# Force disable PIL decompression bomb check for all subprocesses (including torchrun workers)
# sitecustomize.py is auto-imported by Python at startup, before any user code runs
PIL_PATCH_DIR=$(mktemp -d)
cat > "${PIL_PATCH_DIR}/sitecustomize.py" << 'PYEOF'
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
PYEOF
export PYTHONPATH="${PIL_PATCH_DIR}:${PYTHONPATH:-}"

echo "=== [$(date)] Starting SFT: ${EXP_NAME} ==="
echo "Config : ${CONFIG}"
echo "GPUs   : ${NPROC}"
echo "Log    : ${LOG_FILE}"

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="${NPROC}" \
  llamafactory-cli train "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== [$(date)] Training finished: ${EXP_NAME} ==="
