#cat test8times_shell_input_copy.sh
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
python3 -m sglang.launch_server --model-path $MODEL --tp-size 8
sleep 5

sleep 100

echo "START INFERENCE"

# Start inference on each port
python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python new_batch_inference_copy.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30000 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
