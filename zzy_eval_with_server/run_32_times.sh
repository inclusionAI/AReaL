#!/bin/bash

# Run evaluation 32 times with automatic server launch
# This mimics the behavior of the original test_32_times_auto.sh

# Get the model path from terminal input
while getopts "m:i:" opt; do
  case $opt in
    m)
      MODEL=$OPTARG
      ;;
    i)
      INPUT_FILE=$OPTARG
      ;;
    *)
      echo "Usage: $0 -m <model-path> [-i <input-file>]"
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

# Create Test_{timestamp} directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR="${MODEL}/Test_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE_DIR"

echo "Created output directory: $OUTPUT_BASE_DIR"
echo "Model: $MODEL"
echo "Input: $INPUT_FILE"
echo "Running 32 evaluations..."
echo ""

# Run 32 times sequentially (each with its own server)
# Note: Sequential because each run launches its own server
# For parallel runs, use different ports
for i in {1..32}; do
  echo "========================================================================"
  echo "Starting run $i/32..."
  echo "========================================================================"
  
  # Each run uses a different port to avoid conflicts if running in parallel
  PORT=$((30000 + i - 1))
  
  # Run inference (server launches automatically)
  python3 run_inference.py \
    -m "$MODEL" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_BASE_DIR" \
    --port "$PORT" \
    --tp-size 8
  
  if [ $? -eq 0 ]; then
    echo "✅ Run $i/32 completed successfully"
  else
    echo "❌ Run $i/32 failed"
  fi
  
  echo ""
done

echo "========================================================================"
echo "All 32 inference runs completed"
echo "Results saved to: $OUTPUT_BASE_DIR"
echo "========================================================================"

# Run the aggregation script to generate summary report
echo "Generating summary report..."
python3 aggregate_accuracy.py --test-dir "$OUTPUT_BASE_DIR"

echo ""
echo "========================================================================"
echo "DONE!"
echo "========================================================================"
echo "Check $OUTPUT_BASE_DIR for:"
echo "  - Individual run directories (with timestamps)"
echo "  - summary_report.txt (aggregated statistics)"
echo "========================================================================"
