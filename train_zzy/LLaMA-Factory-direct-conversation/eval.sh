#/bin/bash
set -euo pipefail
MODEL=${1:-"models/zzy/Qwen3-ins-4b-5-test"}
OUTPUT_DIR=${2:-"logs/eval/output"}
INPUT_FILE="/home/ligengz/workspace/AReaL/LLaMA-Factory-direct-conversation/s1-parallel/AIME24.jsonl"
PORT=30006
mkdir -p $OUTPUT_DIR

python -m sglang.launch_server \
    --model-path $MODEL --port $PORT --mem-fraction-static 0.7 \
    --data-parallel-size 8 > $OUTPUT_DIR/sglang.log &

# Wait for port 30006 to be open (sglang server ready), checking every 20 seconds
echo "Waiting for port 30006 to become available..."

while true; do
    if nc -z localhost $PORT; then
        echo "Port 30006 is open!"
        break
    else
        echo "Port 30006 not available yet, sleeping 20s..."
        sleep 20
    fi
done


python new_batch_inference_new.py \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --port $PORT \
    --model "$MODEL"
exit 0

./eval.sh models/zzy/Qwen3-ins-4b-5-test ./logs/eval/output-qwen3-ins-4b-5-test
eai-run -i --pty bash ./eval.sh models/zzy/Qwen3-ins-4b-5-test ./logs/eval/output-qwen3-ins-4b-5-test
eai-run -i --pty bash ./eval.sh Qwen/Qwen3-4b-instruct-2507 ./logs/eval/qwen3-4b-instruct-2507

for i in {1..32}; do
# Limit to maximum 8 concurrent jobs
while [ $(squeue --me | wc -l) -ge 10 ]; do
    sleep 20
done

# eai-run -J qwen3-4b-instruct-$i bash ./eval.sh Qwen/Qwen3-4b-instruct-2507 ./logs/eval/qwen3-4b-instruct-2507-$i &
eai-run -J psner-4b-5-test-$i bash ./eval.sh models/zzy/Qwen3-ins-4b-5-test ./logs/eval/output-qwen3-ins-4b-5-test-$i &

sleep 5
done
wait 
