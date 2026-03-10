#!/usr/bin/env bash
# Full evaluation pipeline for testing tool effectiveness on small models.
# Combines: package_trajectory → run_trajectory_test → openai_as_judge
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
  --data_dir DIR        Trajectory data directory
  --model_name PATH     Model name or path

Optional:
  --api_base URL        vLLM server URL (default: $API_BASE)
  --judge_api_base URL  Judge API URL (default: $JUDGE_API_BASE)
  --num_workers N       Parallel workers (default: $NUM_WORKERS)
  --compare_with PATH   Baseline eval path for comparison
  --force               Force re-run all steps

Filters (Step 1):
  --filter_wrong_answers
  --filter_answer_leakage
  --filter_tool_mismatch
  --leakage_check_mode MODE   quick|full (default: $LEAKAGE_CHECK_MODE)

Environment:
  OPENAI_API_KEY        Required for package_trajectory and openai_as_judge
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

# Derive paths
DATA_DIR_NAME=$(basename "$DATA_DIR")
PARENT_DIR=$(dirname "$DATA_DIR")
MODEL_BASENAME=$(basename "$MODEL_NAME")

PARQUET_DIR="${PARENT_DIR}/packaged_trajectory"
PARQUET_PATH="${PARQUET_DIR}/${DATA_DIR_NAME}_full.parquet"
RESULT_DIR="${PARENT_DIR}/${MODEL_BASENAME}_${DATA_DIR_NAME}_full"
EVAL_DIR="${RESULT_DIR}_eval"

echo "=== Full Evaluation Pipeline ==="
echo "Data dir:    $DATA_DIR"
echo "Model:       $MODEL_NAME"
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

    python -m geo_edit.data_preprocess.package_trajectory \
        --data_dir "$DATA_DIR" \
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

# Step 3: Evaluate with OpenAI as judge
echo "=== Step 3: OpenAI as Judge Evaluation ==="
if [[ -d "$EVAL_DIR" && "$FORCE" == false ]]; then
    echo "SKIP: Eval directory already exists at $EVAL_DIR"
else
    COMPARE_ARGS=""
    if [[ -n "$COMPARE_WITH" ]]; then
        COMPARE_ARGS="--compare_with $COMPARE_WITH"
    fi

    python -m geo_edit.evaluation.openai_as_judge \
        --api_key "$OPENAI_API_KEY" \
        --api_base "$JUDGE_API_BASE" \
        --result_path "$RESULT_DIR" \
        --output_path "$EVAL_DIR" \
        $COMPARE_ARGS

    echo "Step 3 completed: $EVAL_DIR"
fi
echo ""

echo "=== Pipeline Complete ==="
echo "Results: $RESULT_DIR"
echo "Evaluation: $EVAL_DIR"
if [[ -f "${EVAL_DIR}/summary.txt" ]]; then
    echo ""
    echo "=== Summary ==="
    cat "${EVAL_DIR}/summary.txt"
fi
