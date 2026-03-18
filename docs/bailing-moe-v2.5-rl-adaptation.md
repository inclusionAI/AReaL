# BailingMoe V2.5 RL 适配设计文档

## 1. 背景

在 `chucai.dzq/bailing-sft-v1.0.1` 分支上，BailingMoeV2.5 的 SFT 训练已完成适配， 包括：

- Lightning Attention 模块（`lightning_attention.py`，基于 fla Triton 内核）
- Config 转换 + 异构 Layer Spec（`bailing_moe.py`，4 种层类型组合）
- 权重加载 Bridge（`bailing_moe_bridge.py`，HF ↔ mcore 格式转换）
- Registry 注册（`registry.py`）
- NCCL 权重转换函数（`convert_bailingmoe_to_hf` in `megatron.py`）
- SGLang >= 0.5.9 + fla 依赖（`pyproject.toml`）

本文档描述从 SFT 到 RL (GRPO) 的适配工作。

______________________________________________________________________

## 2. SFT vs RL 架构差异

SFT 只用 Megatron 训练路径。RL (GRPO) 需要 **训练 + 推理** 双路协同：

```
                          RL (GRPO) 数据流
                          ================

┌──────────────────────────────────────────────────────────────────┐
│                      GRPOTrainer (Controller)                    │
│          协调 rollout 和 train 的交替执行                          │
└──────────┬─────────────────────────────────┬────────────────────┘
           │                                 │
           ▼                                 ▼
┌─────────────────────┐           ┌─────────────────────────────┐
│  Rollout (推理路径)  │           │     Actor (训练路径)          │
│  SGLang Engine       │           │     Megatron Engine          │
│  - 加载 HF 格式权重  │  ←─────── │     - 训练完后权重转换        │
│  - 生成 rollout 数据  │  NCCL     │       convert_bailingmoe_    │
│  - n_samples 采样    │  权重同步  │       to_hf() 转成 HF 格式   │
└─────────────────────┘           └─────────────────────────────┘
           │                                 ▲
           │  rollout 数据                    │ 计算 loss + 更新参数
           ▼                                 │
┌──────────────────────────────────────────────────────────────────┐
│  Reward 计算 → Advantage 估计 → PPO/GRPO Loss → 反向传播          │
└──────────────────────────────────────────────────────────────────┘
```

**RL 相比 SFT 额外需要的组件**：

| 组件                       | SFT    | RL   | 说明                                 |
| -------------------------- | ------ | ---- | ------------------------------------ |
| SGLang 推理引擎            | 不需要 | 需要 | 生成 rollout 数据                    |
| NCCL 权重同步              | 不需要 | 需要 | 训练权重 → 推理引擎                  |
| `convert_bailingmoe_to_hf` | 不需要 | 需要 | 权重名映射 + 格式转换                |
| Ref 模型                   | 不需要 | 需要 | 计算 KL 散度（可选）                 |
| Reward 函数                | 不需要 | 需要 | 计算奖励信号                         |
| RL 配置                    | 不需要 | 需要 | allocation_mode 含 sglang + megatron |

______________________________________________________________________

## 3. 适配任务清单

### 3.1 \[P0\] 创建 RL 配置文件

**新建**: `examples/bailing_moe_grpo.yaml`

基于 `gsm8k_grpo_megatron.yaml` 模板，结合 `bailing_moe_sft.yaml` 的模型配置。

#### 关键配置项

**资源分配**（最核心的决策）：

SFT 只有 actor：`megatron:d8p4t2e8`（64 GPUs）

RL 需要 rollout + actor 双路分配。两种方案：

| 方案                    | allocation_mode                                    | 节点数  | 说明                              |
| ----------------------- | -------------------------------------------------- | ------- | --------------------------------- |
| A. 独立节点（平坦模式） | `sglang:d8p1t8+megatron:d8p4t2e8`                  | 16 节点 | rollout 8 节点 + actor 8 节点     |
| B. 独立节点（异构模式） | `sglang:d8p1t8+megatron:(attn:d8p4t2\|ffn:d2p4e8)` | 16 节点 | attention 和 FFN 使用不同并行策略 |

推荐方案 A 作为初始配置，保持 actor 侧的并行配置和 SFT 一致（`d8p4t2e8`）， 减少调试变量。

**异构模式说明**：BailingMoe 的 MoE 只作用在 FFN 层，attention 层无 expert。异构 模式 `(attn:...|ffn:...)` 允许
attention 和 FFN 使用不同的并行度，例如 attention 不用 EP（节省通信），FFN 用 EP 分散专家。这是 Qwen3-MoE RL 配置
（`config_30b_moe_airline.yaml`）使用的模式。SFT 阶段用平坦模式已验证可用，RL 阶段可先用平坦模式跑通，再切换异构模式优化性能。

SGLang 侧 BailingMoe 是 MoE 模型，推理时需要较多显存。TP=8（每个推理实例占 1 个 完整节点的 8 张 GPU）是合理选择。

#### 配置模板

```yaml
experiment_name: bailing-moe-grpo
trial_name: trial0

seed: 1
enable_offload: false
total_train_epochs: 10
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 16                    # 8 rollout + 8 actor
  n_gpus_per_node: 8
  fileroot: /storage/openpsi/experiments/bailing-moe-grpo
  name_resolve:
    type: nfs
    nfs_record_root: /storage/openpsi/experiments/bailing-moe-grpo/name_resolve

# 8 rollout GPUs (sglang:d1p1t8) + 64 actor GPUs (megatron:d8p4t2e8)
allocation_mode: sglang:d8p1t8+megatron:d8p4t2e8

scheduler:
  type: ray

rollout:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  max_concurrent_rollouts: 128    # MoE 推理更重，比 dense 模型少
  queue_size: null
  consumer_batch_size: ${train_dataset.batch_size}
  max_head_offpolicyness: 2
  enable_rollout_tracing: false
  scheduling_spec: ${actor.scheduling_spec}
  fileroot: ${cluster.fileroot}
  tokenizer_path: ${tokenizer_path}
  dump_to_file: true

gconfig:
  n_samples: 4
  min_new_tokens: 0
  max_new_tokens: 2048
  greedy: false
  temperature: 1.0

actor:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: <MODEL_PATH>             # BailingMoe V2.5 HF checkpoint
  init_from_scratch: false
  disable_dropout: true
  gradient_checkpointing: true    # MoE 模型显存紧张，必须开
  dtype: bfloat16
  grad_reduce_dtype: float32      # MoE 梯度归约用 fp32，数值稳定性
  mb_spec:
    n_mbs: 1
    max_tokens_per_mb: 4096
  pad_to_maximum: true
  optimizer:
    type: adam
    lr: 3e-6                      # RL 学习率通常比 SFT 低
    weight_decay: 0.003
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    lr_scheduler_type: constant
    gradient_clipping: 1.0
    warmup_steps_proportion: 0.001
    min_loss_scale: 1.0
    loss_scale_window: 5.0
    hysteresis: 2
  # === Megatron 引擎配置 (MoE 关键参数) ===
  megatron:
    wrap_with_ddp: true
    ddp:
      grad_reduce_in_fp32: true
      overlap_grad_reduce: false
      overlap_param_gather: false
      use_distributed_optimizer: true
    use_deterministic_algorithms: true    # MoE RL 训练稳定性，强烈推荐
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 1
    moe_router_dtype: fp32               # Router 计算用 fp32
    moe_token_dispatcher_type: alltoall  # MoE token 分发方式
  # === RL 超参 ===
  eps_clip: 0.4
  temperature: ${gconfig.temperature}
  reward_scaling: 10.0
  reward_bias: -0.5
  reward_clip: 20.0                      # 防止极端 reward 导致训练不稳定
  kl_ctl: 0.0                            # 不使用 KL 散度 (省掉 ref 模型显存)
  ppo_n_minibatches: 1
  recompute_logprob: true
  use_decoupled_loss: true
  behave_imp_weight_cap: 5.0
  discount: 1.0
  gae_lambda: 1.0
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
  adv_norm:
    mean_level: batch
    std_level: batch
    std_unbiased: true
    eps: 1.0e-05
  max_new_tokens: ${gconfig.max_new_tokens}
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cpu: 4
      mem: 32
      cmd: python3 -m areal.infra.rpc.rpc_server
      env_vars:
        PYTHONPATH: "/AReaL/.venv/lib/python3.12/site-packages"

# Ref 模型：kl_ctl=0.0 时不参与前向计算，但仍需声明。
# BailingMoe 256 专家模型 colocation 会 OOM，此处设置最小配置。
# 如果后续需要 KL 散度 (kl_ctl > 0)，需要为 ref 单独分配节点。
ref:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: ${actor.path}
  init_from_scratch: false
  disable_dropout: true
  dtype: ${actor.dtype}
  mb_spec:
    n_mbs: 1
    max_tokens_per_mb: 4096
  optimizer: null
  enable_tree_training: false

sglang:
  model_path: ${actor.path}
  random_seed: ${seed}
  skip_tokenizer_init: true
  dtype: ${actor.dtype}
  context_length: 32768
  mem_fraction_static: 0.8
  max_prefill_tokens: 32768
  schedule_policy: lpm
  log_level: warning

train_dataset:
  batch_size: 128
  shuffle: true
  pin_memory: true
  num_workers: 4
  path: openai/gsm8k              # 替换为实际数据集
  type: rl                         # 关键：SFT 用 sft，RL 用 rl
  max_length: 2048

valid_dataset:
  batch_size: 128
  pin_memory: true
  num_workers: 4
  path: openai/gsm8k
  type: rl

saver:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: null

recover:
  mode: disabled
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600

evaluator:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: null

stats_logger:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  wandb:
    mode: disabled

perf_tracer:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  enabled: false
  session_tracer:
    enabled: false
```

**与 SFT 配置的关键差异**：

| 配置项                    | SFT                 | RL                                | 说明                                        |
| ------------------------- | ------------------- | --------------------------------- | ------------------------------------------- |
| `allocation_mode`         | `megatron:d8p4t2e8` | `sglang:d8p1t8+megatron:d8p4t2e8` | 新增 SGLang 推理节点                        |
| `train_dataset.type`      | `sft`               | `rl`                              | 数据格式不同                                |
| `rollout`                 | 无                  | 有                                | rollout 生成控制                            |
| `gconfig`                 | 无                  | 有                                | 采样超参                                    |
| `ref`                     | 无                  | 有                                | KL 散度参考模型                             |
| `sglang`                  | 无                  | 有                                | 推理引擎配置                                |
| `actor.megatron`          | 无                  | 有                                | MoE 引擎参数(router dtype, deterministic等) |
| `actor.grad_reduce_dtype` | 无                  | `float32`                         | MoE 梯度归约数值稳定性                      |
| `actor.reward_clip`       | 无                  | `20.0`                            | 防极端 reward                               |
| `actor.eps_clip` 等       | 无                  | 有                                | RL 超参                                     |
| `optimizer.lr`            | 2e-5                | 3e-6                              | RL 更低                                     |

______________________________________________________________________

### 3.2 \[P0\] 修复 expert_bias 参数名不一致

**问题**：NCCL 权重转换和 mbridge 权重加载使用了不同的 HF 参数名。

| 路径             | 代码位置                    | 映射到的 HF 名                     |
| ---------------- | --------------------------- | ---------------------------------- |
| NCCL 权重同步    | `megatron.py:640`           | `mlp.gate.e_score_correction_bias` |
| mbridge 权重加载 | `bailing_moe_bridge.py:119` | `mlp.gate.expert_bias`             |

`e_score_correction_bias` 是 DeepSeek-V3 的命名，BailingMoe HF 模型用的是 `expert_bias`。

**影响**：

- SFT 不受影响（SFT 不走 NCCL 权重同步路径）
- RL 受影响：训练完一步后 `convert_bailingmoe_to_hf` 会把 `mlp.router.expert_bias` 转成
  `mlp.gate.e_score_correction_bias` 发给 SGLang，但 SGLang 加载 BailingMoe 时期望的参数名是
  `mlp.gate.expert_bias`，导致权重丢失或报错

**修复**：

```python
# megatron.py convert_bailingmoe_to_hf() 中修改:
elif rest == "mlp.router.expert_bias":
    return [
        (f"model.layers.{layer_idx}.mlp.gate.expert_bias", param)  # 原为 e_score_correction_bias
    ]
```

**验证方式**：需要在集群上确认 SGLang 0.5.9 加载 BailingMoe 时实际使用的参数名。可通过：

```python
# 在 SGLang 加载 BailingMoe 时打印权重名
from sglang.srt.models.bailing_moe_linear import BailingMoeLinearForCausalLM
# 或检查 SGLang 源码中 bailing_moe 的 weight_map
```

______________________________________________________________________

### 3.3 \[P0\] 端到端 NCCL 权重同步验证

`convert_bailingmoe_to_hf` 函数已实现，但从未在 RL 场景下端到端测试过。需要验证：

1. **所有参数名匹配**：Lightning 层和 MLA 层的参数名是否都和 SGLang 期望一致
1. **QKV 格式转换正确性**：interleaved → concatenated 转换后，SGLang 能正确推理
1. **MoE expert 权重**：fc1 的 chunk(2) 拆分是否正确
1. **异构层处理**：不同层类型（Lightning vs MLA）产生不同的参数集合，SGLang 必须 正确接收

**验证步骤**：

```bash
# Step 1: 启动 SFT 训练 1 步，保存 mcore checkpoint
# Step 2: 调用 convert_bailingmoe_to_hf 转换所有参数
# Step 3: 比较转换后的参数名和 SGLang 加载 HF checkpoint 时的参数名
# Step 4: 启动 RL 训练，观察第一次 NCCL 权重同步是否成功
```

**重点关注的参数映射**（按层类型）：

Lightning 层:

```
mcore: self_attention.linear_qkv.weight     → HF: attention.query_key_value.weight  (需 reshape)
mcore: self_attention.linear_gate.weight    → HF: attention.g_proj.weight
mcore: self_attention.gate_norm.weight      → HF: attention.g_norm.weight
mcore: self_attention.linear_proj.weight    → HF: attention.dense.weight
```

MLA 层:

```
mcore: self_attention.linear_q_down_proj.weight  → HF: attention.q_a_proj.weight
mcore: self_attention.linear_kv_down_proj.weight → HF: attention.kv_a_proj_with_mqa.weight
mcore: self_attention.linear_proj.weight         → HF: attention.dense.weight
```

注意 BailingMoe 用 `attention.` 前缀而非常见的 `self_attn.`。

______________________________________________________________________

### 3.4 \[P1\] `bailing_hybrid` 加入 VALID_MOE_MODELS

**文件**: `areal/engine/core/model.py`

`bailing_hybrid` 在 `_CONVERSION_FN_REGISTRY` 中已注册（`megatron.py:740`），但不在
`VALID_MOE_MODELS` 中。如果有模型的 `model_type` 是 `bailing_hybrid`，`is_moe_model()` 会返回
False，跳过 MoE 专用逻辑（如 expert parallelism）。

**修复**：

```python
VALID_MOE_MODELS = [
    "qwen3_moe",
    "bailing_moe_v2",
    "bailing_moe_linear",
    "bailing_hybrid",      # 新增
]
```

______________________________________________________________________

## 4. 实施顺序

```
Phase 1: 修复代码问题
├── 3.2 修复 expert_bias 命名
└── 3.4 bailing_hybrid 加入 VALID_MOE_MODELS

Phase 2: 创建 RL 配置
├── 3.1a 创建 bailing_moe_grpo.yaml（多节点完整配置）
└── 3.1b 创建 bailing_moe_mini_grpo.yaml（单节点测试配置，EP=8，用 mini 模型）

Phase 3: 集群验证
├── 3.3 NCCL 权重同步端到端测试（先用 mini 模型单节点验证）
├── SGLang 加载 BailingMoe 推理验证
└── RL 训练 loss 下降验证
```

**mini RL 测试配置说明**：参考 `bailing_moe_mini_sft.yaml`（1 节点，EP=8），mini RL 配置使用
`sglang:d1p1t1+megatron:d1p1t1e8`（1 节点中 1 GPU 给 SGLang，其余给 Megatron）。但 BailingMoe 256
专家推理需要足够 GPU 显存，可能需要 2 节点： `sglang:d1p1t8+megatron:d1p1t1e8`（1 节点推理 + 1 节点训练）。具体取决于 mini
模型的参数量。

**预估工作量**：

| 阶段             | 代码量           | 耗时       |
| ---------------- | ---------------- | ---------- |
| Phase 1 代码修复 | ~10 行           | 0.5h       |
| Phase 2 RL 配置  | ~200 行 yaml × 2 | 1h         |
| Phase 3 集群验证 | 无新代码         | 视集群排队 |

______________________________________________________________________

## 5. 风险与注意事项

### 5.1 SGLang 参数名兼容性

SGLang 0.5.9 内置的 BailingMoe 支持可能和 HF 原始模型的参数命名有差异。NCCL 权重 同步时 SGLang 接收端是按 HF
参数名匹配的，任何名字不一致都会导致权重丢失。

**缓解措施**：在集群上打印 SGLang 端加载 BailingMoe 时的完整 weight map，逐一对比 `convert_bailingmoe_to_hf`
的输出。

### 5.2 MoE 模型推理显存

BailingMoe 有 256 个专家，推理时需要加载所有专家权重（不像训练时可以 EP 分散）。 SGLang 的 TP=8 可以将显存压力分散到 8 张 GPU，但
`mem_fraction_static` 可能需要 调高。

### 5.3 Lightning Attention 不支持 Context Parallelism

`bailing_moe.py` 中已有校验（CP 时报错），RL 配置中不应使用 CP。当前配置模板中 未使用 CP，无风险。

### 5.4 Ref 模型显存压力

BailingMoe 256 专家模型参数量大，Ref 模型和 Actor colocation 会 OOM。当前配置模板 已设置 `kl_ctl: 0.0`（不计算 KL
散度），Ref 模型不参与前向计算，不占显存。

如果后续需要 KL 散度（`kl_ctl > 0`），有两个选择：

- Ref 模型使用独立节点部署（增加 allocation_mode 中的节点数）
- Ref 模型使用 SGLang 推理引擎部署（需要额外配置）

______________________________________________________________________

## 6. 文件变更清单

| 操作 | 文件                                          | 说明                 |
| ---- | --------------------------------------------- | -------------------- |
| 修改 | `areal/engine/megatron_utils/megatron.py:640` | expert_bias 命名修复 |
| 修改 | `areal/engine/core/model.py:38-42`            | 添加 bailing_hybrid  |
| 新建 | `examples/bailing_moe_grpo.yaml`              | 多节点 RL 配置       |
| 新建 | `examples/bailing_moe_mini_grpo.yaml`         | 单节点 RL 测试配置   |
