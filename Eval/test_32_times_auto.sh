#!/bin/bash

# Get the model path from terminal input
while getopts "m:" opt; do
  case $opt in
    m)
      MODEL=$OPTARG
      ;;
    *)
      echo "Usage: $0 -m <model-path>"
      exit 1
      ;;
  esac
done

# Ensure MODEL path is provided
if [ -z "$MODEL" ]; then
  echo "Error: You must provide a model path with the -m option."
  exit 1
fi

INPUT_FILE="S1-parallel/AIME2425.jsonl" # Update this to your input file path
# INPUT_FILE="/storage/openpsi/users/zzy/RL-clean/gsm8k.jsonl"
# Create Test_{timestamp} directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR="${MODEL}/Test_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE_DIR"

echo "Created output directory: $OUTPUT_BASE_DIR"

# OUTPUT_DIR set to the new Test_{timestamp} directory
OUTPUT_DIR="$OUTPUT_BASE_DIR"

PORT=30007

# Optional: Process subset
START_IDX=0
END_IDX=  # Leave empty to process all

# export CUDA_VISIBLE_DEVICES=0

# Launch servers on different ports
python3 -m sglang.launch_server --model-path $MODEL --tp-size 8 &
sleep 5

sleep 100

echo "START INFERENCE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"

# Start inference on each port
python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5
python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &

sleep 5
python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5
python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
# Wait for all background processes to complete
wait

echo "All inference processes completed"
echo "Results saved to: $OUTPUT_BASE_DIR"

# Run the aggregation script to generate summary report
echo "Generating summary report..."
python aggregate_results.py --test-dir "$OUTPUT_BASE_DIR"

echo "Done! Check $OUTPUT_BASE_DIR for results and summary_report.txt"