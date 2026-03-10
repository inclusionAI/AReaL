#!/usr/bin/env bash
# Full evaluation pipeline for testing tool effectiveness on small models.
# Combines: package_trajectory → run_trajectory_test → openai_as_judge
#
# Supports batch mode: if data_dir contains subdirectories with trajectory data,
# it will process each subdirectory automatically.
#
# Usage:
#   OPENAI_API_KEY=xxx bash run_full_eval_pipeline.sh \
#       --data_dir /path/to/trajectory_data \
#       --model_name /path/to/model
set -euo pipefail

# Default values
API_BASE="http://127.0.0.1:8000"
JUDGE_API_BASE="https://matrixllm.alipay.com/v1"
NUM_WORKERS=32
LEAKAGE_CHECK_MODE="full"
FORCE=false
COMPARE_WITH=""
EVAL_MODE="mapqa"  # mapqa or judge

# Filter flags
FILTER_WRONG_ANSWERS=false
FILTER_ANSWER_LEAKAGE=false
FILTER_TOOL_MISMATCH=false

# Required parameters
DATA_DIR=""
MODEL_NAME=""

usage() {
    cat <<EOF
Usage: OPENAI_API_KEY=xxx $0 [OPTIONS]

Required:
  --data_dir DIR        Trajectory data directory (or parent of multiple experiments)
  --model_name PATH     Model name or path

Optional:
  --api_base URL        vLLM server URL (default: $API_BASE)
  --judge_api_base URL  Judge API URL (default: $JUDGE_API_BASE)
  --num_workers N       Parallel workers (default: $NUM_WORKERS)
  --compare_with PATH   Baseline eval path for comparison
  --eval_mode MODE      Evaluation mode: mapqa|judge (default: $EVAL_MODE)
  --force               Force re-run all steps

Filters (Step 1):
  --filter_wrong_answers
  --filter_answer_leakage
  --filter_tool_mismatch
  --leakage_check_mode MODE   quick|full (default: $LEAKAGE_CHECK_MODE)

Environment:
  OPENAI_API_KEY        Required for package_trajectory and openai_as_judge

Batch Mode:
  If data_dir contains subdirectories (each with trajectory data), the script
  will automatically process each subdirectory.

  Example structure:
    mapqa_batch/
      gpt_crop/
      gpt_zoom/

  Output paths:
    parquet: packaged_trajectory/{subdir}_{parent}_full.parquet
    result:  {model}_{subdir}_{parent}_full/
    eval:    {model}_{subdir}_{parent}_full_eval/
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --api_base) API_BASE="$2"; shift 2 ;;
        --judge_api_base) JUDGE_API_BASE="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --compare_with) COMPARE_WITH="$2"; shift 2 ;;
        --eval_mode) EVAL_MODE="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        --filter_wrong_answers) FILTER_WRONG_ANSWERS=true; shift ;;
        --filter_answer_leakage) FILTER_ANSWER_LEAKAGE=true; shift ;;
        --filter_tool_mismatch) FILTER_TOOL_MISMATCH=true; shift ;;
        --leakage_check_mode) LEAKAGE_CHECK_MODE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required parameters
if [[ -z "$DATA_DIR" ]]; then
    echo "Error: --data_dir is required"
    usage
fi
if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model_name is required"
    usage
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is required"
    exit 1
fi

MODEL_BASENAME=$(basename "$MODEL_NAME")

# Build filter args once
FILTER_ARGS=""
if [[ "$FILTER_WRONG_ANSWERS" == true ]]; then
    FILTER_ARGS="$FILTER_ARGS --filter_wrong_answers"
fi
if [[ "$FILTER_ANSWER_LEAKAGE" == true ]]; then
    FILTER_ARGS="$FILTER_ARGS --filter_answer_leakage"
fi
if [[ "$FILTER_TOOL_MISMATCH" == true ]]; then
    FILTER_ARGS="$FILTER_ARGS --filter_tool_mismatch"
fi

# Function to run pipeline for a single directory
run_single_pipeline() {
    local SUB_DATA_DIR="$1"
    local SUBDIR_NAME="$2"
    local PARENT_NAME="$3"
    local OUTPUT_BASE="$4"

    # Derive paths: insert suffix after "mapeval_visual_"
    # Extract suffix from parent name (e.g., mapqa_batch_gpt -> gpt)
    local SUFFIX="${PARENT_NAME##*_}"
    # Insert suffix: mapeval_visual_auto_segment_1r -> mapeval_visual_gpt_auto_segment_1r
    local NAME_PREFIX="${SUBDIR_NAME/mapeval_visual_/mapeval_visual_${SUFFIX}_}"
    local PARQUET_DIR="${OUTPUT_BASE}/packaged_trajectory"
    local PARQUET_PATH="${PARQUET_DIR}/${NAME_PREFIX}_full.parquet"
    local RESULT_DIR="${OUTPUT_BASE}/${MODEL_BASENAME}_${NAME_PREFIX}_full"
    local EVAL_DIR="${RESULT_DIR}_eval"

    echo ""
    echo "=============================================="
    echo "Processing: $SUBDIR_NAME"
    echo "=============================================="
    echo "Data dir:    $SUB_DATA_DIR"
    echo "Parquet:     $PARQUET_PATH"
    echo "Result dir:  $RESULT_DIR"
    echo "Eval dir:    $EVAL_DIR"
    echo ""

    # Step 1: Package trajectory
    echo "=== Step 1: Package Trajectory ==="
    if [[ -f "$PARQUET_PATH" && "$FORCE" == false ]]; then
        echo "SKIP: Parquet already exists at $PARQUET_PATH"
    else
        mkdir -p "$PARQUET_DIR"

        python -m geo_edit.data_preprocess.package_trajectory \
            --data_dir "$SUB_DATA_DIR" \
            --out_path "$PARQUET_PATH" \
            --api_base "$JUDGE_API_BASE" \
            --api_key "$OPENAI_API_KEY" \
            --max_workers "$NUM_WORKERS" \
            --leakage_check_mode "$LEAKAGE_CHECK_MODE" \
            $FILTER_ARGS

        echo "Step 1 completed: $PARQUET_PATH"
    fi
    echo ""

    # Step 2: Run trajectory test
    echo "=== Step 2: Run Trajectory Test ==="
    if [[ -d "$RESULT_DIR" && "$FORCE" == false ]]; then
        echo "SKIP: Result directory already exists at $RESULT_DIR"
    else
        python -m geo_edit.scripts.run_trajectory_test \
            --parquet_path "$PARQUET_PATH" \
            --output_path "$RESULT_DIR" \
            --model_name "$MODEL_NAME" \
            --api_base "$API_BASE" \
            --num_workers "$NUM_WORKERS"

        echo "Step 2 completed: $RESULT_DIR"
    fi
    echo ""

    # Step 3: Evaluation
    echo "=== Step 3: Evaluation (mode: $EVAL_MODE) ==="
    if [[ -d "$EVAL_DIR" && "$FORCE" == false ]]; then
        echo "SKIP: Eval directory already exists at $EVAL_DIR"
    else
        COMPARE_ARGS=""
        if [[ -n "$COMPARE_WITH" ]]; then
            COMPARE_ARGS="--compare_with $COMPARE_WITH"
        fi

        if [[ "$EVAL_MODE" == "mapqa" ]]; then
            python -m geo_edit.evaluation.eval_mapeval_visual \
                --result_path "$RESULT_DIR" \
                --output_path "$EVAL_DIR" \
                $COMPARE_ARGS
        else
            python -m geo_edit.evaluation.openai_as_judge \
                --api_key "$OPENAI_API_KEY" \
                --api_base "$JUDGE_API_BASE" \
                --result_path "$RESULT_DIR" \
                --output_path "$EVAL_DIR" \
                $COMPARE_ARGS
        fi

        echo "Step 3 completed: $EVAL_DIR"
    fi
    echo ""

    echo "=== Completed: $SUBDIR_NAME ==="
    if [[ -f "${EVAL_DIR}/summary.txt" ]]; then
        echo "Summary:"
        cat "${EVAL_DIR}/summary.txt"
    fi
}

# Main logic: batch mode - process all subdirectories
DATA_DIR_NAME=$(basename "$DATA_DIR")
PARENT_DIR=$(dirname "$DATA_DIR")

echo "=== Full Evaluation Pipeline (Batch Mode) ==="
echo "Model: $MODEL_NAME"
echo "Data dir: $DATA_DIR"
echo ""

# Collect subdirectories
SUBDIRS=()
for subdir in "$DATA_DIR"/*/; do
    if [[ -d "$subdir" ]]; then
        SUBDIRS+=("$subdir")
    fi
done

if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
    echo "Error: No subdirectories found in $DATA_DIR"
    exit 1
fi

echo "Found ${#SUBDIRS[@]} subdirectories to process"
echo ""

for subdir in "${SUBDIRS[@]}"; do
    subdir_name=$(basename "$subdir")
    run_single_pipeline "$subdir" "$subdir_name" "$DATA_DIR_NAME" "$PARENT_DIR"
done

echo ""
echo "=============================================="
echo "=== All ${#SUBDIRS[@]} experiments completed ==="
echo "=============================================="
