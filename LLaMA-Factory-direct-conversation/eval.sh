MODEL=models/zzy/Qwen3-ins-4b-5-test
OUTPUT_DIR="output"
INPUT_FILE="/home/ligengz/workspace/AReaL/LLaMA-Factory-direct-conversation/s1-parallel/AIME24.jsonl"

python -m sglang.launch_server \
    --model-path $MODEL --port 30006 --mem-fraction-static 0.7 \
    --data-parallel-size 8

python new_batch_inference_new.py \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --port 30006 \
    --model "$MODEL"