#!/bin/bash

# Run evaluation 32 times IN PARALLEL with automatic server launch
# Each run gets its own server on a different port
# This is more resource-intensive but faster

# Get the model path from terminal input
while getopts "m:i:j:" opt; do
  case $opt in
    m)
      MODEL=$OPTARG
      ;;
    i)
      INPUT_FILE=$OPTARG
      ;;
    j)
      PARALLEL_JOBS=$OPTARG
      ;;
    *)
      echo "Usage: $0 -m <model-path> [-i <input-file>] [-j <parallel-jobs>]"
      echo "  -j: Number of parallel jobs (default: 4, max: 32)"
      exit 1
      ;;
  esac
done

# Ensure MODEL path is provided
if [ -z "$MODEL" ]; then
  echo "Error: You must provide a model path with the -m option."
  exit 1
fi

# Default input file
if [ -z "$INPUT_FILE" ]; then
  INPUT_FILE="S1-parallel/AIME2425.jsonl"
fi

# Default parallel jobs
if [ -z "$PARALLEL_JOBS" ]; then
  PARALLEL_JOBS=4
fi

# Create Test_{timestamp} directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR="${MODEL}/Test_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE_DIR"

echo "========================================================================"
echo "Running 32 evaluations in parallel"
echo "========================================================================"
echo "Model: $MODEL"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_BASE_DIR"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "========================================================================"
echo ""

# Function to run a single evaluation
run_single_eval() {
  local run_num=$1
  local port=$((30000 + run_num - 1))
  
  echo "[Run $run_num] Starting on port $port..."
  
  python3 run_inference.py \
    -m "$MODEL" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_BASE_DIR" \
    --port "$port" \
    --tp-size 8 \
    > "$OUTPUT_BASE_DIR/run_${run_num}.log" 2>&1
  
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo "[Run $run_num] ✅ Completed successfully"
  else
    echo "[Run $run_num] ❌ Failed (exit code: $exit_code)"
  fi
  
  return $exit_code
}

export -f run_single_eval
export MODEL INPUT_FILE OUTPUT_BASE_DIR

# Run 32 evaluations in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
  # Use GNU parallel if available (better progress reporting)
  echo "Using GNU parallel..."
  seq 1 32 | parallel -j "$PARALLEL_JOBS" run_single_eval {}
else
  # Fallback to xargs
  echo "Using xargs (install 'parallel' for better control)..."
  seq 1 32 | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'run_single_eval "$@"' _ {}
fi

echo ""
echo "========================================================================"
echo "All 32 inference runs completed"
echo "Results saved to: $OUTPUT_BASE_DIR"
echo "========================================================================"

# Count successes and failures
success_count=$(find "$OUTPUT_BASE_DIR" -name "grading_report.txt" | wc -l)
echo "Successful runs: $success_count/32"

# Run the aggregation script to generate summary report
echo ""
echo "Generating summary report..."
python3 aggregate_accuracy.py --test-dir "$OUTPUT_BASE_DIR"

echo ""
echo "========================================================================"
echo "DONE!"
echo "========================================================================"
echo "Check $OUTPUT_BASE_DIR for:"
echo "  - Individual run directories (with timestamps)"
echo "  - summary_report.txt (aggregated statistics)"
echo "  - run_*.log files (logs for each run)"
echo "========================================================================"
