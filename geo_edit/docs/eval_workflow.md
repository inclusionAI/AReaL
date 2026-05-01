# 6 数据集 Direct-Generate 评估流程

对任意 VLM 做 direct-generate（无工具调用）推理并评分，覆盖 6 个地图/视觉推理数据集。

## 数据集

| key | 判分方式 | Parquet 路径 |
|---|---|---|
| visual_probe_easy | rule + LLM judge | `/storage/openpsi/data/VisualProbe_Easy/val.parquet` |
| visual_probe_medium | rule + LLM judge | `/storage/openpsi/data/VisualProbe_Medium/val.parquet` |
| visual_probe_hard | rule + LLM judge | `/storage/openpsi/data/VisualProbe_Hard/val.parquet` |
| map_trace | NDTW（无 judge）| `/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet` |
| reason_map | rule + LLM judge | `/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet` |
| reason_map_plus | rule + LLM judge | `/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet` |

**map_trace 判定规则**（[`eval_unified.py:113`](../geo_edit/evaluation/eval_unified.py#L113)）：`score = 1.0 if parse_success and NDTW < 1.0 else 0.0`

## 前置条件

- 8 张 GPU（每张 ≥ 80GB 显存），默认 `--data-parallel-size 8`
- 环境变量：`JUDGE_API_KEY` 和 `JUDGE_API_BASE`（OpenAI 兼容端点）
- vLLM 0.17+

## 三个关键脚本

| 脚本 | 作用 |
|---|---|
| [`launch_vllm_generate.sh`](../geo_edit/scripts/launch_vllm_generate.sh) | 起 vLLM OpenAI 兼容服务 |
| [`run_all_eval.sh`](../geo_edit/scripts/run_all_eval.sh) | 批量推理（被下面的 orchestrator 调用）|
| [`run_model_benchmark.sh`](../geo_edit/scripts/run_model_benchmark.sh) | 一键 orchestrator：vLLM 启动 → 推理 → judge → 汇总 |
| [`eval_unified.py`](../geo_edit/evaluation/eval_unified.py) | 统一判分（含 LLM judge 兜底）|
| [`openai_as_judge.py`](../geo_edit/evaluation/openai_as_judge.py) | 独立 judge 入口（可选）|

## 一键跑法（推荐）

```bash
cd /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL

bash geo_edit/scripts/run_model_benchmark.sh \
  --model /storage/openpsi/models/YOUR_MODEL \
  --mode direct \
  --datasets "visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus" \
  --no-image-compression \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-api-base "$JUDGE_API_BASE" \
  --gpu-mem-util 0.8
```

### 常用开关

| 参数 | 说明 |
|---|---|
| `--gpu-mem-util 0.7` | RL checkpoint（8 shard）OOM 时降到 0.7 |
| `--skip-vllm` | vLLM 已在跑，跳过启停 |
| `--skip-inference` | 推理结果已存在，只重跑 judge |
| `--skip-eval` | 只推理不 judge |
| `--sample-rate 0.1` | 抽样 10% 快速 smoke test |
| `--max-concurrent 32` | 降低并发（遇 rate limit 时）|

### 模型长度限制 → 必须调整 MAX_MODEL_LEN

部分模型的 `max_position_embeddings` 小于 launch 脚本默认的 65536，会直接拒绝启动。启动前检查模型 `config.json`，前置导出环境变量：

```bash
export MAX_MODEL_LEN=40960   # 或更小，不超过模型的 max_position_embeddings
```

示例：InternVL3.5-8B-HF 是 40960，Qwen3-VL-8B 是 262144，InternVL3-8B-hf 是 65536。

## 分阶段跑法（调试用）

```bash
# 阶段 1: 启 vLLM
MAX_MODEL_LEN=40960 GPU_MEM_UTIL=0.8 \
bash geo_edit/scripts/launch_vllm_generate.sh /path/to/model 8000
curl -s http://127.0.0.1:8000/v1/models   # 健康检查

# 阶段 2: 推理 6 个数据集
MODEL_PATH=/path/to/model MODE=direct \
API_BASE=http://127.0.0.1:8000 MODEL_TYPE=vLLM \
MAX_CONCURRENT=64 SAMPLE_RATE=1.0 NO_IMAGE_COMPRESSION=1 \
DATASETS="visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus" \
bash geo_edit/scripts/run_all_eval.sh

# 阶段 3: 判分（map_trace 用 NDTW，其余用 rule + judge）
MODEL_NAME=$(basename /path/to/model)
RESULT_ROOT=/storage/openpsi/data/lcy_image_edit/eval_results
EVAL_ROOT=/storage/openpsi/data/lcy_image_edit/eval_output

declare -A DS_NAME=(
  [visual_probe_easy]=visual_probe
  [visual_probe_medium]=visual_probe
  [visual_probe_hard]=visual_probe
  [map_trace]=map_trace
  [reason_map]=reason_map
  [reason_map_plus]=reason_map_plus
)
for ds_key in "${!DS_NAME[@]}"; do
  use_judge="--use_judge --judge_api_key $JUDGE_API_KEY --judge_api_base $JUDGE_API_BASE --judge_model gpt-4.1-mini-2025-04-14"
  [ "$ds_key" = "map_trace" ] && use_judge=""
  python -m geo_edit.evaluation.eval_unified \
    --dataset_name "${DS_NAME[$ds_key]}" \
    --result_path "$RESULT_ROOT/$ds_key/${MODEL_NAME}_direct" \
    --output_path "$EVAL_ROOT/$ds_key/${MODEL_NAME}_direct" \
    $use_judge
done
```

## 产物路径

```
/storage/openpsi/data/lcy_image_edit/
  eval_results/<ds_key>/<MODEL_NAME>_direct/<task_id>/meta_info.jsonl   # 推理结果
  eval_output/<ds_key>/<MODEL_NAME>_direct/{eval_result.jsonl, summary.txt}   # 判分结果
/tmp/log/vllm_api.log    # vLLM 日志
```

`summary.txt` 内容示例：
```
Dataset: visual_probe
Total: 141
Correct: 24
Accuracy: 0.1702 (24/141)
LLM Judge called: 36
LLM Judge overturned: 13
```

## 支持的模型

任何 vLLM 0.17 支持的视觉语言模型都能跑（见 [vLLM 支持列表](https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models)）。`/storage/openpsi/models/` 下已知可跑的：

### Qwen 系列（推荐，chat template 完全兼容 OpenAI 多模态）

| 模型 | 路径 | max_model_len 默认 |
|---|---|---|
| Qwen2-VL-2B-Instruct | `/storage/openpsi/models/Qwen2-VL-2B-Instruct` | 32768 |
| Qwen2-VL-7B | `/storage/openpsi/models/Qwen2-VL-7B` | 32768 |
| Qwen2.5-VL-3B-Instruct | `/storage/openpsi/models/Qwen2.5-VL-3B-Instruct` | 128000 |
| Qwen2.5-VL-7B-Instruct | `/storage/openpsi/models/Qwen2.5-VL-7B-Instruct` | 128000 |
| Qwen2.5-VL-32B-Instruct | `/storage/openpsi/models/Qwen2.5-VL-32B-Instruct` | 128000 |
| Qwen2.5-VL-72B-Instruct | `/storage/openpsi/models/Qwen2.5-VL-72B-Instruct` | 128000 |
| Qwen3-VL-2B-Instruct / Thinking | `/storage/openpsi/models/Qwen3-VL-2B-{Instruct,Thinking}` | 262144 |
| Qwen3-VL-4B-Instruct / Thinking | `/storage/openpsi/models/Qwen3-VL-4B-{Instruct,Thinking}` | 262144 |
| Qwen3-VL-8B-Instruct / Thinking | `/storage/openpsi/models/Qwen3-VL-8B-{Instruct,Thinking}` | 262144 |
| Qwen3-VL-30B-A3B-Instruct / Thinking | `/storage/openpsi/models/Qwen3-VL-30B-A3B-{Instruct,Thinking}` | 262144 |
| Qwen3-VL-32B-Instruct / Thinking | `/storage/openpsi/models/Qwen3-VL-32B-{Instruct,Thinking}` | 262144 |
| Qwen3-VL-235B-A22B-Instruct / Thinking | `/storage/openpsi/models/Qwen3-VL-235B-A22B-{Instruct,Thinking}` | 262144 |
| Qwen__QVQ-72B-Preview | `/storage/openpsi/models/Qwen__QVQ-72B-Preview` | 32768 |

### InternVL 系列（⚠️ 必须用 `-hf` / `-HF` 变体）

非 HF 版本的 chat template 不支持 OpenAI content list 格式，会返回 400 `can only concatenate str (not "list") to str`。

| 模型 | 路径 | max_model_len 默认 |
|---|---|---|
| InternVL3-8B-hf | `/storage/openpsi/models/InternVL3-8B-hf` | 65536 |
| InternVL3_5-8B-HF | `/storage/openpsi/models/InternVL3_5-8B-HF` | **40960**（必须设 `MAX_MODEL_LEN=40960`）|
| InternVL2_5-8B | `/storage/openpsi/models/InternVL2_5-8B` | 需验证 chat template |

同系列大/小尺寸（需要自行下载或确认 HF 变体）：
- `InternVL3-{1B,2B,14B,38B,78B}` — 非 HF 版，若 chat template 不兼容，需手动转换或暂不支持
- `InternVL3_5-{1B,38B,241B-A28B}` — 同上

### 其他 VLM

| 模型 | 路径 | 说明 |
|---|---|---|
| GLM-4.1V-9B-Thinking | `/storage/openpsi/models/GLM-4.1V-9B-Thinking` | 需验证 vLLM 支持 |
| GLM-4.1V-9B-Base | `/storage/openpsi/models/GLM-4.1V-9B-Base` | 需验证 |
| gemma-3-{4b,12b,27b}-it | `/storage/openpsi/models/google__gemma-3-*-it` | Gemma3 多模态 |
| Kimi-VL-A3B | `/storage/openpsi/models/Kimi-VL-A3B` | 需验证 |
| Kimi-VL-A3B-Thinking-2506 | `/storage/openpsi/models/Kimi-VL-A3B-Thinking-2506_9261938df674c9a2` | 需验证 |
| PaddleOCR-VL-1.5 | `/storage/openpsi/models/PaddleOCR-VL-1.5` | OCR 专用 |
| SmolVLM-{Instruct,256M,500M,2.2B} | `/storage/openpsi/models/SmolVLM*` | 小模型 smoke test |
| llava-v1.6-mistral-7b-hf | `/storage/openpsi/models/llava-v1.6-mistral-7b-hf` | LLaVA |
| Perception-LM-8B | `/storage/openpsi/models/Perception-LM-8B` | 需验证 |
| MathCoder-VL-8B | `/storage/openpsi/models/MathCoder-VL-8B` | 数学向 VLM |
| G-LLaVA-7B | `/storage/openpsi/models/G-LLaVA-7B` | 几何向 VLM |
| multimath-7b-llava-v1.5 | `/storage/openpsi/models/multimath-7b-llava-v1.5` | 多模态数学 |
| Chart-R1 | `/storage/openpsi/models/Chart-R1` | 图表向 |
| OVR-7B-RL | `/storage/openpsi/models/OVR-7B-RL` | 视觉推理 |

### 项目内部训练产物

| 路径 | 说明 |
|---|---|
| `/storage/openpsi/models/lcy_EGM_models/` | 本项目 EGM 训练 checkpoint |
| `/storage/openpsi/models/lcy_image_edit/` | 图像编辑相关 |
| `/storage/openpsi/models/AReaL-boba-*` | AReaL RL checkpoint（可能是 8-shard，需 `--gpu-mem-util 0.7`）|

### 纯文本模型（无视觉输入，需要文本改写 pipeline，**不推荐直接跑此评估**）

DeepSeek-R1 / V3、Llama-3.x、Qwen3 非 VL 变体、gpt-oss-120b、Ling*、Bailing* 等纯语言模型都不能直接跑这套——6 个数据集都要求读图。

## 新模型接入 checklist

1. 确认 **vLLM 支持该模型架构**（`Architecture: XxxForConditionalGeneration` 在 [vLLM 文档](https://docs.vllm.ai/en/latest/models/supported_models.html) 里存在）
2. 看 `config.json` 的 `max_position_embeddings`，决定 `MAX_MODEL_LEN`
3. 看 `tokenizer_config.json` 的 `chat_template` **是否支持 OpenAI content list 多模态格式**（有 `image_url` / `image` 类型分支就可）
4. 先 smoke test 小规模样本确认 chat template 正常：

   ```bash
   MAX_MODEL_LEN=XXX bash geo_edit/scripts/launch_vllm_generate.sh /path/to/model 8000
   curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "/path/to/model",
     "messages": [{"role":"user","content":[{"type":"text","text":"2+2?"}]}],
     "max_tokens": 30
   }'
   ```

5. 若通过，抽样 10% 跑完整 pipeline 验证：`--sample-rate 0.1`

## 踩坑记录

| 现象 | 原因 | 修法 |
|---|---|---|
| `can only concatenate str (not "list") to str` 400 错误 | 模型 chat template 不支持 content list | 换 `-hf` 变体或自行改写 template |
| `User-specified max_model_len (65536) is greater than derived max_model_len` | 模型 position_embeddings 小于 65536 | 设 `MAX_MODEL_LEN=...`（见模型 config.json）|
| `argument --limit-mm-per-prompt: Value image=5 cannot be converted` | vLLM ≥ 0.10 要求 JSON 格式 | launch 脚本已修为 `'{"image": 5}'`，若是旧版本 vLLM 需改回 `image=5` |
| `Executor failed` / `DP Coordinator receives unexpected message` | 端口冲突 / 残留进程 | `pkill -9 -f vllm; pkill -9 -f EngineCore; sleep 5` 后重启 |
| vLLM 启动后 curl 健康检查超时 | launch 脚本死循环等 endpoint | 上面的杀进程流程 + 检查 `/tmp/log/vllm_api.log` 真正错误 |
| reason_map judge 零翻盘 | judge prompt 对 metro 路线匹配严格 | 正常现象，看 [`reason_map_verifier.py`](../geo_edit/evaluation/reason_map_verifier.py) |
| map_trace 准确率偏低 | NDTW 是累积代价不归一化，复杂路径容易 > 1.0 | 正常，看分布（median ≈ 0.8 是常见水平）|

## 参考

- 详细推理侧说明：[`geo_edit/README.md`](../geo_edit/README.md)
- geo-eval skill 文档：[`.opencode/skills/geo-eval/SKILL.md`](../.opencode/skills/geo-eval/SKILL.md)
- vLLM 多模态模型支持：https://docs.vllm.ai/en/latest/models/supported_models.html
