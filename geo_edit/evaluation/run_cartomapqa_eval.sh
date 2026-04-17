#!/usr/bin/env bash
set -x

RESULT_BASE=${RESULT_BASE:-"/storage/openpsi/data/lcy_image_edit/cartomapqa_test_0417"}

API_KEY=${API_KEY:-${OPENAI_API_KEY:-""}}
API_BASE=${API_BASE:-${OPENAI_API_BASE:-""}}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-5-mini-2025-08-07"}

SUBSETS=(
    "MFS:cartomapqa_mfs"
    "MML:cartomapqa_mml"
    "MTMF:cartomapqa_mtmf"
    "RLE:cartomapqa_rle"
    "SRN:cartomapqa_srn"
    "STMF_counting:cartomapqa_stmf_counting"
    "STMF_name_listing:cartomapqa_stmf_name_listing"
    "STMF_presence:cartomapqa_stmf_presence"
)

total=${#SUBSETS[@]}
idx=0

for entry in "${SUBSETS[@]}"; do
    idx=$((idx + 1))
    name="${entry%%:*}"
    task="${entry##*:}"
    result_path="${RESULT_BASE}/${name}"
    output_path="${RESULT_BASE}/${name}_eval"

    echo "=== [$idx/$total] ${name} (${task}) ==="

    if [ ! -d "$result_path" ]; then
        echo "  SKIP: $result_path not found"
        continue
    fi

    judge_args=(--use_judge --judge_model "$JUDGE_MODEL")
    [ -n "$API_KEY" ]  && judge_args+=(--api_key "$API_KEY")
    [ -n "$API_BASE" ] && judge_args+=(--api_base "$API_BASE")

    python -m geo_edit.evaluation.cartomapqa.evaluate \
        --task "$task" \
        --result_path "$result_path" \
        --output_path "$output_path" \
        "${judge_args[@]}"

    echo "  DONE: ${name} -> ${output_path}"
done

echo "All $total subsets evaluated."
