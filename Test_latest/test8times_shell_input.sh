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

INPUT_FILE="/storage/openpsi/users/zzy/parallel_thinking/parallel-thinking-main/Test/AIME24.jsonl"

# OUTPUT_DIR set to MODEL path
OUTPUT_DIR=$MODEL

PORT=30007

# Optional: Process subset
START_IDX=0
END_IDX=  # Leave empty to process all

export CUDA_VISIBLE_DEVICES=0

# Launch servers on different ports
python -m sglang.launch_server --model-path $MODEL --port 30001 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=1
python -m sglang.launch_server --model-path $MODEL --port 30002 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=2
python -m sglang.launch_server --model-path $MODEL --port 30003 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=3
python -m sglang.launch_server --model-path $MODEL --port 30004 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=4
python -m sglang.launch_server --model-path $MODEL --port 30005 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=5
python -m sglang.launch_server --model-path $MODEL --port 30006 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=6
python -m sglang.launch_server --model-path $MODEL --port 30007 --mem-fraction-static 0.7 &
sleep 5

export CUDA_VISIBLE_DEVICES=7
python -m sglang.launch_server --model-path $MODEL --port 30008 --mem-fraction-static 0.7 &
sleep 5

sleep 100

echo "START INFERENCE"

# Start inference on each port
python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30001 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30002 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30003 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30004 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30005 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30006 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30007 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_new.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30008 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &

