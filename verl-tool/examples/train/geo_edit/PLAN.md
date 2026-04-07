# ChartQA RL Training Plan

## Overview

- Base models: SFT checkpoints from 1/3 data experiments
  - `qwen3vl8b-instruct-chartqa-1third` (train=0.863, eval=0.638)
  - `qwen3vl8b-thinking-chartqa-1third` (train=0.838, eval=0.629)
- RL data: 1438 removed trajectories (2/3), questions + answers only, no intermediate tool traces
- Algorithm: GRPO
- Framework: verl-tool (local at `/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool`)

## 新环境注意事项（verl 0.6.0 Docker）

### 必须应用的代码修改（2 个文件）

#### 1. `verl_tool/servers/tools/geo_edit_tool.py`

已重写。核心变更：
- 新增 `_load_all_tools_via_router()` 通过 ToolRouter 加载全部 19 个工具（原来只有 6 个函数工具）
- 通过 `GEOEDIT_ENABLE_TOOLS` 环境变量控制工具类型（默认 `general,chart`）
- observation 格式修复：`Observation {N}:\n<image>`（原来缺少换行）
- 失败时自动回退到 function-only 模式

#### 2. `verl_tool/trainer/main_ppo.py`

在 `run_ppo()` 中 ray.init 之前的 runtime_env 构建部分，添加 ~5 行：
```python
env_vars = OmegaConf.to_container(runtime_env.get("env_vars", {}), resolve=True)
env_vars["ROCR_VISIBLE_DEVICES"] = ""              # 防止 ROCR/CUDA 冲突
if os.environ.get("PYTHONPATH"):
    env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]  # 传播到 worker 节点
runtime_env = OmegaConf.merge(runtime_env, {"env_vars": env_vars})
```

### 可能需要的环境修复（视 Docker 情况）

| 问题 | 触发条件 | 修复方式 |
|---|---|---|
| Worker 节点 `ModuleNotFoundError: No module named 'geo_edit'` | tool agent actors 调度到 worker 节点但找不到 geo_edit | 在 worker 节点 site-packages 创建 `areal_paths.pth`，内容为 AReaL 根目录路径 |
| `tokenizer_config.json` 报 `'list' object has no attribute 'keys'` | transformers 新版要求 `extra_special_tokens` 为 dict | 复制 SFT 模型目录，将 `extra_special_tokens` 从 list 转为 `{token: token}` dict |
| `config.json` 报 `'NoneType' has no attribute 'get'` on `rope_scaling` | transformers 新版要求 `rope_scaling` 非 null | 从官方 Qwen3-VL-8B 的 `text_config.rope_scaling` 拷贝到顶层 `rope_scaling` |
| `runtime_env.yaml` 中 `ROCR_VISIBLE_DEVICES: ""` 导致冲突 | worker 节点同时设了 ROCR 和 CUDA | 已在 main_ppo.py 中处理，如仍报错可删除 runtime_env.yaml 中该行 |
| `pip install fire` 缺失导致 serve.py 无法启动 | verl-tool 依赖 fire 但未声明 | `pip install fire` |

### 必须安装的 pip 依赖（HEAD 和 WORKER 都需要）

```bash
pip install "vllm<=0.11.0" fire colorlog iopath ftfy "pyzmq==26.2.0" qwen_omni_utils timeout-decorator
```

| 包 | 用途 | 安装节点 |
|---|---|---|
| `vllm<=0.11.0` | rollout engine + paddleocr/chartr1 agent 推理 | HEAD + WORKER |
| `fire` | verl-tool CLI 依赖 | HEAD + WORKER |
| `colorlog` | geo_edit ToolRouter 日志 | HEAD + WORKER |
| `iopath` | sam3 模型加载 (`g_pathmgr`) | WORKER |
| `ftfy` | sam3 tokenizer 依赖 | WORKER |
| `pyzmq==26.2.0` | 修复 vllm LLM import (pyzmq 27.x 不兼容) | HEAD + WORKER |
| `qwen_omni_utils` | verl-tool audio_utils 依赖 | HEAD |
| `timeout-decorator` | verl-tool 依赖 | HEAD |

### 必须创建的 .pth 文件（两个节点都需要）

```bash
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > $SITE/verl_tool_paths.pth << 'PTH'
/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool
/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool/verl
/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL
PTH
```

### 必须修复的 tokenizer_config.json

两个 SFT 模型都需要将 `extra_special_tokens` 从 list 转为 dict：
```python
python3 -c "
import json
for m in ['qwen3vl8b-instruct-chartqa-1third', 'qwen3vl8b-thinking-chartqa-1third']:
    p = f'/storage/openpsi/models/lcy_image_edit/sft_workspace/{m}/tokenizer_config.json'
    with open(p) as f: c = json.load(f)
    ets = c.get('extra_special_tokens', [])
    if isinstance(ets, list):
        c['extra_special_tokens'] = {t: t for t in ets}
        with open(p, 'w') as f: json.dump(c, f, indent=2, ensure_ascii=False)
        print(f'{m}: fixed ({len(ets)} tokens)')
"
```

### 必须修复的 config.json rope_scaling

两个 SFT 模型的 `rope_scaling` 为 null，transformers 4.57+ 要求非 null：
```python
python3 -c "
import json
ref = json.load(open('/storage/openpsi/models/Qwen3-VL-8B-Instruct/config.json'))
rope = ref.get('text_config', {}).get('rope_scaling')
for m in ['qwen3vl8b-instruct-chartqa-1third', 'qwen3vl8b-thinking-chartqa-1third']:
    p = f'/storage/openpsi/models/lcy_image_edit/sft_workspace/{m}/config.json'
    c = json.load(open(p))
    c['rope_scaling'] = rope
    if 'text_config' in c and isinstance(c['text_config'], dict):
        c['text_config']['rope_scaling'] = rope
    json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
    print(f'{m}: rope_scaling set to {rope}')
"
```

### 必须杀掉两个节点的 idle Ray workers

创建 .pth 文件后，必须杀掉已有的 idle Ray workers 让新 worker 拿到 .pth：
```bash
ps aux | grep 'ray::IDLE' | grep -v grep | awk '{print $2}' | xargs kill -9
```

### paddleocr agent 已知问题

paddleocr (PaddleOCR-VL-1.5) agent 使用 vLLM 加载模型时会遇到 EngineCore distributed 网络错误。
此问题不影响 chart QA 训练（chartr1 是核心 chart 工具）。如需修复可尝试：
- 升级 vllm 到更新版本
- 在 GEOEDIT_ENABLE_TOOLS 中排除 paddleocr 相关工具

### verl 0.6.0 的 vllm 版本

verl-tool 声明 `vllm<=0.11.0`。如果 Docker 自带的 vllm 版本正确（0.8~0.11），则：
- **不需要** `vllm_compat_patch.py`
- **不需要** `launch_ppo.py` wrapper
- **不需要** 任何 vllm shim 文件
- 训练脚本直接用 `python3 -m verl_tool.trainer.main_ppo`

## Cluster Architecture & GPU 调度

```
Head node  (33.180.161.27):  192 CPU, 8 GPU, 0 tool_agent  → 训练
Worker node(33.180.171.201): 192 CPU, 8 GPU, 8 tool_agent  → Tool Agents
```

### 自然调度策略（无 hack）

1. **先启动 Tool Server** → ToolRouter 创建 6 个 Ray Actor agents（各占 1 GPU + 1 tool_agent）
2. Worker 节点 GPU 被占用 6/8 → 只剩 2 GPU
3. **后启动训练** → `n_nodes=1, n_gpus_per_node=8, STRICT_PACK`
4. Worker 不够 8 GPU → 训练自然落到 Head 节点

### Tool Agents 资源占用

| Agent | Model | Replicas | GPU/each |
|---|---|---|---|
| sam3 | sam3.1_multiplex.pt | 1 | 1 |
| paddleocr | PaddleOCR-VL-1.5 | 1 | 1 |
| grounding_dino | grounding-dino-base | 1 | 1 |
| chartr1 | Chart-R1 | 3 | 1 |
| **合计** | | **6 actors** | **6 GPU** |

### Tool Server 启动命令

```bash
export GEOEDIT_ENABLE_TOOLS="general,chart"
python -m verl_tool.servers.serve \
  --host $host --port 30888 \
  --tool_type geo_edit_tool \
  --workers_per_tool 1 --uvi_workers 1 --router_workers 1 \
  --max_concurrent_requests 128 \
  --use_ray True
```

**关键**：`--uvi_workers 1 --max_concurrent_requests 128` 防止多个 backend 重复创建 agents。

## Data

### Source
- Removed task IDs: `/storage/openpsi/data/lcy_image_edit/chartqa_sft_data_1third/split_info.json`
- Raw data: `/storage/openpsi/data/lcy_image_edit/chartqa_augmented_data/{task_id}/`

### Generated
- `data/chartqa_rl_train.parquet` — 1294 records ✅
- `data/chartqa_rl_val.parquet` — 144 records ✅

### Parquet Schema
```python
{
    "data_source": "chartqa_rl",
    "prompt": [
        {"role": "system", "content": "<从SFT数据提取的system_prompt>"},
        {"role": "user", "content": "Observation 0:\n<image>\nQuestion: ..."}
    ],
    "images": [PIL.Image],
    "reward_model": {"style": "rule", "ground_truth": "<answer>"},
    "extra_info": {"task_id": "...", "answer": "...", "question": "..."}
}
```

System prompt 从 SFT 数据 `train.json[0]['system']` 直接提取，保证完全一致。

## Reward

`geo_vision_qa` reward manager：
- 从 `<answer>...</answer>` 提取预测
- Exact match ground_truth
- correct=1.0, wrong=0.0, no answer tag=-0.5

## Training Configuration

### Experiment 1: Instruct
```
model_path = /storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-instruct-chartqa-1third
run_name   = chartqa-rl-instruct
```

### Experiment 2: Thinking
```
model_path = /storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-chartqa-1third
run_name   = chartqa-rl-thinking
```

### Shared Hyperparameters
```
rl_alg                    = grpo
strategy                  = fsdp2
n_gpus_per_node           = 8
n_nodes                   = 1
batch_size                = 128
ppo_mini_batch_size       = 128
n                         = 8
lr                        = 1e-6
temperature               = 1.0
top_p                     = 1.0

max_prompt_length         = 8192
max_response_length       = 8192
max_action_length         = 2048
max_obs_length            = 4096
max_turns                 = 10

tensor_model_parallel_size = 2
gpu_memory_utilization    = 0.6
ppo_micro_batch_size_per_gpu = 1
log_prob_micro_batch_size_per_gpu = 1

enable_agent              = True
mask_observations         = True
enable_mtrl               = True
action_stop_tokens        = </action>
additional_eos_token_ids  = [151645]

reward_manager            = geo_vision_qa

kl_loss_coef              = 0.0
kl_coef                   = 0
entropy_coeff             = 0
kl_loss_type              = low_var_kl

total_epochs              = 3
save_freq                 = 10
test_freq                 = 10
rollout_mode              = async
use_dynamic_bsz           = True
```

## File Structure

```
rl_workspace/
├── PLAN.md                           ← this file
├── data/
│   ├── chartqa_rl_train.parquet      ← 1294 records ✅
│   └── chartqa_rl_val.parquet        ← 144 records ✅
├── scripts/
│   ├── preprocess_chartqa_rl.py      ← 数据预处理（可重新生成）
│   ├── run_smoke_test.sh             ← 小规模测试（2 steps）
│   ├── run_instruct_rl.sh            ← 正式训练 instruct（3 epochs）
│   ├── run_thinking_rl.sh            ← 正式训练 thinking（3 epochs）
│   └── setup_worker_node.sh          ← worker 节点环境初始化（视需要）
├── logs/                             ← 训练日志输出目录
└── checkpoints/                      ← 模型 checkpoint 输出目录
```

## Execution Steps

1. 启动 Tool Server（tmux 中持久运行，等待 agents 加载完毕 ~2min）
2. 验证 GPU 占用（`ray.available_resources()` 确认 worker GPU 被 agents 占用）
3. 运行 smoke test（`bash scripts/run_smoke_test.sh`，2 steps 验证 pipeline）
4. 运行正式训练 instruct（`bash scripts/run_instruct_rl.sh`，3 epochs）
5. 运行正式训练 thinking（`bash scripts/run_thinking_rl.sh`，3 epochs）
