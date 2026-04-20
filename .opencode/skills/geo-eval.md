# geo-eval — Geo/Map Model Benchmark

Run full evaluation pipeline for VLM models on 7 geo/map datasets.

## Trigger Phrases

- "evaluate model on geo datasets"
- "run benchmark"
- "test model on 7 datasets"
- "run geo eval"

## Pipeline Overview

3 stages: **vLLM Launch → Inference → Eval + Judge**

## Datasets (7 total)

| Key | Eval Name | Parquet Path |
|---|---|---|
| visual_probe_easy | visual_probe | /storage/openpsi/data/VisualProbe_Easy/val.parquet |
| visual_probe_medium | visual_probe | /storage/openpsi/data/VisualProbe_Medium/val.parquet |
| visual_probe_hard | visual_probe | /storage/openpsi/data/VisualProbe_Hard/val.parquet |
| map_trace | map_trace | /storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet |
| reason_map | reason_map | /storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet |
| reason_map_plus | reason_map_plus | /storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet |
| mapqa | mm_mapqa | /storage/openpsi/data/lcy_image_edit/MapQA_all/mapqa_test_0418.parquet |

## Stage 1: Kill Old vLLM & Launch New

```bash
# Kill all existing vLLM and GPU processes
ps aux | grep -E "vllm|api_server" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9 2>/dev/null
sleep 3

# Verify GPUs are free
nvidia-smi | grep "0MiB" | wc -l  # should be 8

# Launch vLLM (use GPU_MEM_UTIL=0.7 for RL checkpoints with 8 shards)
GPU_MEM_UTIL=0.8 bash geo_edit/scripts/launch_vllm_generate.sh /path/to/model 8000

# Verify
curl -s http://127.0.0.1:8000/v1/models
```

## Stage 2: Inference

### Tool mode (default for all 7 datasets)

```bash
MODEL_PATH=/path/to/model \
MODE=tool \
MODEL_TYPE=vLLM \
API_BASE=http://127.0.0.1:8000 \
MAX_CONCURRENT=64 \
SAMPLE_RATE=1.0 \
MAX_TOOL_CALLS=10 \
NO_IMAGE_COMPRESSION=1 \
DATASETS="visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus mapqa" \
bash geo_edit/scripts/run_all_eval.sh
```

**Requires**: Ray cluster running with `tool_agent` resources for PaddleOCR actors.

### Direct mode (no tools)

```bash
python -m geo_edit.scripts.direct_generate \
  --api_base http://127.0.0.1:8000 \
  --dataset_path /path/to/dataset.parquet \
  --dataset_name EVAL_NAME \
  --output_dir /storage/openpsi/data/lcy_image_edit/eval_results/DATASET_KEY/MODEL_NAME_direct \
  --model_name_or_path /path/to/model \
  --model_type vLLM \
  --api_mode chat_completions \
  --max_concurrent_requests 64 \
  --sample_rate 1.0 \
  --no_image_compression
```

### Output structure

```
/storage/openpsi/data/lcy_image_edit/eval_results/
  visual_probe_easy/MODEL_NAME_tool/
    visual_probe_easy_0/meta_info.jsonl
    visual_probe_easy_1/meta_info.jsonl
    ...
```

## Stage 3: Evaluation with Judge

### Per-dataset eval command

```bash
python -m geo_edit.evaluation.eval_unified \
  --dataset_name EVAL_NAME \
  --result_path /storage/openpsi/data/lcy_image_edit/eval_results/DATASET_KEY/MODEL_NAME_MODE \
  --output_path /storage/openpsi/data/lcy_image_edit/eval_output/DATASET_KEY/MODEL_NAME_MODE \
  --use_judge \
  --judge_model gpt-4.1-mini-2025-04-14 \
  --judge_api_key "$JUDGE_API_KEY" \
  --judge_api_base "$JUDGE_API_BASE"
```

### Dataset-specific notes

- **map_trace**: NO `--use_judge`. Uses NDTW metric only.
- **reason_map_plus**: Judge rarely overturns (binary True/False + counting).
- **mapqa, visual_probe_*, reason_map**: Judge significantly improves accuracy (~10-20pp over rule-only).

### Judge API timeout

Judge calls ~1400 records for mapqa, can take 10+ minutes. Run in tmux or with long timeout. Pass `--judge_api_key` and `--judge_api_base` explicitly (env vars don't propagate to subshells).

## One-Command Script

```bash
bash geo_edit/scripts/run_model_benchmark.sh \
  --model /path/to/model \
  --no-image-compression \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-api-base "$JUDGE_API_BASE"
```

Options: `--skip-vllm`, `--skip-inference`, `--skip-eval`, `--mode tool|direct`, `--datasets "ds1 ds2"`, `--gpu-mem-util 0.7`

## Key Implementation Files

- `geo_edit/scripts/run_model_benchmark.sh` — Full pipeline script
- `geo_edit/scripts/run_all_eval.sh` — Inference-only batch script
- `geo_edit/scripts/launch_vllm_generate.sh` — vLLM launcher
- `geo_edit/scripts/async_generate_with_tool_call_api.py` — Tool mode inference
- `geo_edit/scripts/direct_generate.py` — Direct mode inference
- `geo_edit/evaluation/eval_unified.py` — Unified evaluation with optional judge
- `geo_edit/evaluation/reason_map_verifier.py` — Route verification for reason_map
- `geo_edit/utils/image_utils.py` — Image compression (`max_base64_bytes`, disabled via `--no_image_compression`)

## Image Compression

- Default: 4MB base64 limit, auto JPEG compression
- `--no_image_compression` / `NO_IMAGE_COMPRESSION=1`: sends original images
- vLLM processor max pixels: ~16M (Qwen3-VL config), effectively no resize
- Training max pixels: 5120 * 28 * 28 ≈ 4M pixels (from `vision_process.py`)

## Known Issues

- RL checkpoints (8 shards) may OOM at `gpu-mem-util=0.8`, use `0.7`
- SFT models on mapqa: tool mode hurts (OCR noise), direct mode better for simple VQA
- reason_map: `unknown_route` failures from strict route name matching (e.g. "1号线" vs "Line 1"), falls back to judge
- reason_map fallback_generic (45%): samples without metro_data use substring matching
