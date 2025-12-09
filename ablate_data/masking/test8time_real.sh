INPUT_FILE="path/to/aime/jsonl" # with key: "problem" and "answer"
OUTPUT_DIR="/storage/openpsi/experiments/logs/admin/zzy_test/result/multiverse"
PORT=30007
MODEL="/storage/openpsi/models/Multiverse-clean/Multiverse-32B_eac17a503a6ecb95"

# Optional: Process subset
START_IDX=0
END_IDX=  # Leave empty to process all
export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server --model-path $MODEL --port 30001 --mem-fraction-static 0.7 &
sleep 5
export CUDA_VISIBLE_DEVICES=1

python -m sglang.launch_server --model-path $MODEL --port 30002 --mem-fraction-static 0.7&
sleep 5
export CUDA_VISIBLE_DEVICES=2

python -m sglang.launch_server --model-path $MODEL --port 30003 --mem-fraction-static 0.7&
sleep 5
export CUDA_VISIBLE_DEVICES=3

python -m sglang.launch_server --model-path $MODEL --port 30004 --mem-fraction-static 0.7&
sleep 5
export CUDA_VISIBLE_DEVICES=4

python -m sglang.launch_server --model-path $MODEL --port 30005 --mem-fraction-static 0.7&
sleep 5
export CUDA_VISIBLE_DEVICES=5

python -m sglang.launch_server --model-path $MODEL --port 30006 --mem-fraction-static 0.7&
sleep 5
export CUDA_VISIBLE_DEVICES=6

python -m sglang.launch_server --model-path $MODEL --port 30007 --mem-fraction-static 0.7&
sleep 5
export CUDA_VISIBLE_DEVICES=7

python -m sglang.launch_server --model-path $MODEL --port 30008 --mem-fraction-static 0.7&
sleep 5
sleep 200
echo "START INFERENCE"
python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30001 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30002 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30003 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30004 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30005 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30006 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30007 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &
sleep 5 

python run_batch_inference_hier.py \
        --input "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --port 30008 \
        --model "$MODEL" \
        --start-idx "$START_IDX" &