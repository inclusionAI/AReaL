# MCP GSM8K (MCP calculator + GSM8K)

Minimal notes to run preprocess, eval, and train.

## Data
Generate parquet from GSM8K:
```bash
python examples/data_preprocess/mcp_gsm8k.py --out_dir data/mcp_gsm8k --max_train 500 --max_test 100
python examples/data_preprocess/mcp_gsm8k.py --out_dir examples/data_preprocess/output/mcp_gsm8k --max_train 50 --max_test 30  # small set used by scripts
```

## Eval (validation only)
```bash
source .venv/bin/activate
DATA_DIR=examples/data_preprocess/output/mcp_gsm8k \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_gsm8k/eval.sh
```
Defaults baked into the script: `REWARD_MANAGER=mcp_static`, GRPO with small batch sizes, `action_stop_tokens=['</tool_call>']`; gateway/tool logs go to `/tmp/mcp_gateway.log` and `/tmp/mcp_tool_server.log`.

## Train
Enable training with:
```bash
RUN_TRAINING=1 \
DATA_DIR=data/mcp_gsm8k \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_gsm8k/train.sh
```
Use `TOTAL_EPOCHS` or `TOTAL_TRAINING_STEPS` to override training length if needed; other hyperparams can be tuned via environment variables (see the script).
