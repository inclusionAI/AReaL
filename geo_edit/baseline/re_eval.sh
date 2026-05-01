#!/usr/bin/env bash
set -uo pipefail
cd /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL

MODEL_NAME="${1:?MODEL_NAME required}"
KIND="${2:?KIND required: vtool_r1 | mini_o3 | direct}"

RESULT_ROOT=/storage/openpsi/data/lcy_image_edit/eval_results
EVAL_ROOT=/storage/openpsi/data/lcy_image_edit/eval_output

declare -A DS_NAME=(
  [visual_probe_easy]=visual_probe
  [visual_probe_medium]=visual_probe
  [visual_probe_hard]=visual_probe
  [map_trace]=map_trace
  [reason_map]=reason_map
  [reason_map_plus]=reason_map_plus
)

echo "===== Re-eval: $MODEL_NAME [$KIND] ====="
for ds_key in "${!DS_NAME[@]}"; do
  result_path="${RESULT_ROOT}/${ds_key}/${MODEL_NAME}_${KIND}"
  out_path="${EVAL_ROOT}/${ds_key}/${MODEL_NAME}_${KIND}"
  if [ ! -d "$result_path" ] || [ $(find "$result_path" -name meta_info.jsonl 2>/dev/null | wc -l) -eq 0 ]; then
    echo "[$ds_key] SKIP: no meta_info at $result_path"
    continue
  fi
  mkdir -p "$out_path"
  judge_args="--use_judge --judge_model gpt-4.1-mini-2025-04-14 --judge_api_key $JUDGE_API_KEY --judge_api_base $JUDGE_API_BASE"
  [ "$ds_key" = "map_trace" ] && judge_args=""
  echo "[$ds_key] eval starting..."
  python -m geo_edit.evaluation.eval_unified \
    --dataset_name "${DS_NAME[$ds_key]}" \
    --result_path "$result_path" \
    --output_path "$out_path" \
    $judge_args 2>&1 | grep -E "^(Dataset|Total|Correct|Accuracy|LLM|NDTW|Per)" | head -12
done
echo "===== DONE: $MODEL_NAME ====="
