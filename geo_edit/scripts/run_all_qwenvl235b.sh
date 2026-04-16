#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== [$(date)] Starting 2 qwenvl235b experiments ==="

bash "${SCRIPT_DIR}/run_sft.sh" "${SCRIPT_DIR}/configs/qwen3vl8b-thinking-qwenvl235b-lr1e-5.yaml"
echo "=== [$(date)] Experiment 1/2 (lr=1e-5) done ==="

bash "${SCRIPT_DIR}/run_sft.sh" "${SCRIPT_DIR}/configs/qwen3vl8b-thinking-qwenvl235b-lr3e-6.yaml"
echo "=== [$(date)] Experiment 2/2 (lr=3e-6) done ==="

echo "=== [$(date)] ALL 2 EXPERIMENTS COMPLETED ==="
echo "Results:"
for d in qwen3vl8b-thinking-reasonmapplus_qwenvl235b-lr1e-5 qwen3vl8b-thinking-reasonmapplus_qwenvl235b-lr3e-6; do
  echo "--- ${d} ---"
  cat "/storage/openpsi/models/lcy_image_edit/sft_workspace/${d}/all_results.json" 2>/dev/null || echo "FAILED"
done
