# MCP Weather (dynamic MCP reward)

Minimal instructions to generate data and run eval/train with the weather MCP server.

## Data
Generate parquet with embedded `validation_calls`:
```bash
python examples/data_preprocess/mcp_weather.py --out_dir data/mcp_weather --max_train 50 --max_test 20
```

## Eval (validation only)
```bash
source .venv/bin/activate
DATA_DIR=data/mcp_weather \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_weather/eval.sh
```


## Train
Enable PPO training with:
```bash
RUN_TRAINING=1 \
DATA_DIR=data/mcp_weather \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_weather/train.sh
```
