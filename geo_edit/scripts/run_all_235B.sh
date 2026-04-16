#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== [$(date)] Starting all 3 experiments ==="

bash "${SCRIPT_DIR}/run_sft.sh" "${SCRIPT_DIR}/configs/qwen3vl8b-thinking-235B-lr1e-5.yaml"
echo "=== [$(date)] Experiment 1/3 (lr=1e-5) done ==="

bash "${SCRIPT_DIR}/run_sft.sh" "${SCRIPT_DIR}/configs/qwen3vl8b-thinking-235B-lr3e-6.yaml"
echo "=== [$(date)] Experiment 2/3 (lr=3e-6) done ==="

bash "${SCRIPT_DIR}/run_sft.sh" "${SCRIPT_DIR}/configs/qwen3vl8b-thinking-235B-lr1e-6.yaml"
echo "=== [$(date)] Experiment 3/3 (lr=1e-6) done ==="

echo "=== [$(date)] ALL 3 EXPERIMENTS COMPLETED ==="
echo "Results:"
for d in qwen3vl8b-thinking-235B-lr1e-5 qwen3vl8b-thinking-235B-lr3e-6 qwen3vl8b-thinking-235B-lr1e-6; do
  echo "--- ${d} ---"
  cat "/storage/openpsi/models/lcy_image_edit/sft_workspace/${d}/all_results.json" 2>/dev/null || echo "FAILED"
done
