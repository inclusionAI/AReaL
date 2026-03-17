# BailingMoeV2.5 适配指南

## 模型架构

BailingMoeV2.5 是混合专家（MoE）模型，采用异构注意力机制：

- **Lightning Attention**：线性注意力，O(n) 复杂度，使用 ALiBi geometric slopes 作为衰减因子
- **MLA（Multi-Latent Attention）**：低秩 KV 压缩，标准 softmax attention + RoPE
- **MoE**：稀疏激活的混合专家层，sigmoid routing + grouped TopK

### 层结构

层按 `layer_group_size` 分组，每组最后一层用 MLA，其余用 Lightning Attention：

```
layer_group_size=5:  L L L L M | L L L L M | L L L L M | L L L L M
layer_group_size=8:  L L L L L L L M | L L L L L L L M | ...

L = Lightning Attention, M = MLA
```

第一层（layer 0）为 Dense MLP，其余为 MoE MLP（由 `first_k_dense_replace` 控制）。

### 支持的模型

| 模型 | model_type | architectures | 规模 |
|------|-----------|---------------|------|
| Mini | bailing_moe_linear | BailingMoeLinearForCausalLM | 20层, H=2048, 256 experts |
| Flash | bailing_hybrid | BailingMoeV2_5ForCausalLM | 32层, H=4096, 256 experts |

两个 model_type 共用同一个 Bridge（`BailingMoeBridge`），自动适配。

## 关键配置参数

### MLA 层 RoPE

| 参数 | AReaL 值 | 说明 |
|------|---------|------|
| `rotary_percent` | 1.0 | megatron-core MLA 传入 `qk_pos_emb_head_dim`（64），1.0 保证全部 64 维应用 RoPE |
| `apply_rope_fusion` | False | 使用 Python 实现的 RoPE，避免 TE 版本差异导致的数值问题 |
| `rope_type` | "rope" | 标准 RoPE（非 YaRN） |
| `rotary_base` | 从 HF config 读取 | Mini=10000, Flash=6000000 |

**关于 `rotary_percent=1.0`**：HybridEngine 配置中 `rotary_percent=0.5`，但其 Megatron-LM fork 的 MLA
传入 `kv_channels=128`（head_dim），所以 `128 * 0.5 = 64`。AReaL 的 megatron-core MLA 直接传入
`qk_pos_emb_head_dim=64`，所以需要 `1.0` 才能得到相同的 RoPE dim=64。改为 0.5 会导致 RoPE dim=32，
经实验验证会显著恶化 MLA 层精度。

### Lightning Attention

| 参数 | 来源 | 说明 |
|------|-----|------|
| `partial_rotary_factor` | HF config | 0.5，RoPE 只作用于 head_dim 的一半 |
| `attn_head_dim` | HF config `head_dim` | Mini=128, Flash=128 |
| `linear_attn_norm_group_size` | HF config `group_norm_size` | GroupRMSNorm 分组大小 |
| g_gamma | ALiBi geometric slopes | `2^(-8/H * [1,2,...,H]) * layer_scale`，layer_scale = `1-(layer_idx-1)/(N-1)+1e-5` |

Lightning Attention 调用 `fla.ops.simple_gla.chunk_simple_gla` 内核，**不使用**
`fla.ops.lightning_attn.chunk_lightning_attn`（后者内置线性 slopes，非 ALiBi）。

### MoE

| 参数 | 来源 | 说明 |
|------|-----|------|
| `num_moe_experts` | HF config `num_experts` | 256 |
| `moe_router_topk` | HF config `num_experts_per_tok` | 8 |
| `moe_router_score_function` | HF config `scoring_func` | sigmoid |
| `moe_router_num_groups` | HF config `n_group` | 8 |
| `moe_router_group_topk` | HF config `topk_group` | 4 |
| `moe_router_topk_scaling_factor` | HF config `routed_scaling_factor` | 2.5 |
| `moe_router_enable_expert_bias` | 固定 True | |
| `moe_grouped_gemm` | 固定 True | |
| `moe_token_dispatcher_type` | 固定 "alltoall" | |

### Mini 与 Flash 模型的关键差异

| 参数 | Mini | Flash |
|------|------|-------|
| `q_lora_rank` | null（直接 Q 投影） | 1536（低秩 Q 压缩） |
| `rope_theta` | 10000 | 6000000 |
| `layer_group_size` | 5 | 8 |
| `max_position_embeddings` | 4096 | 262144 |
| `rope_interleave` | 无 | true |
| `partial_rotary_factor` | 无 | 0.5 |

Flash 模型的 `rope_interleave=true` 对应 megatron-core 的 `rotary_interleaved`，但 megatron-core
**不允许** MLA + rotary_interleaved 同时使用（会报 ValueError）。AReaL 保持默认 `rotary_interleaved=False`。

## 环境依赖

### 必需：flash-linear-attention (fla)

Lightning Attention 依赖 `fla` 库提供的 CUDA 内核：

```bash
pip install flash-linear-attention==0.4.2
```

当前 `fla` 安装在 AReaL 的 venv 中（`.venv/lib/python3.12/site-packages/`），不在系统镜像中。

**SFT 训练时**，如果通过 Slurm/Ray launcher 提交，需要在 `env_vars` 中添加 venv 的 site-packages 路径：

```yaml
actor:
  scheduling_spec:
    - env_vars:
        PYTHONPATH: "/AReaL/.venv/lib/python3.12/site-packages:${PYTHONPATH}"
```

或直接在镜像中安装 `fla`。

### 版本信息

| 组件 | AReaL 当前版本 |
|------|--------------|
| TransformerEngine | 2.11.0 |
| flash_attn | 2.8.1 |
| flash-linear-attention (fla) | 0.4.2 |
| torch | 2.6.0+cu126 |

TE/flash_attn 版本差异会影响 MLA 层 core attention 的数值精度（不影响正确性），Lightning Attention
层不受 TE/flash_attn 版本影响（走 fla 内核）。

## SFT 配置示例

```yaml
experiment_name: bailing-moe-sft
trial_name: flash-8node

seed: 1
total_train_epochs: 1
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 8
  n_gpus_per_node: 8
  fileroot: /path/to/output

# 64 GPUs: DP=8, PP=4, TP=2, EP=8
allocation_mode: megatron:d8p4t2e8

actor:
  path: /path/to/bailing-moe-v2.5-hf-model
  dtype: bfloat16
  gradient_checkpointing: true
  optimizer:
    type: adam
    lr: 2e-5
    weight_decay: 0.05
  scheduling_spec:
    - task_type: worker
      gpu: 1
      cmd: python3 -m areal.infra.rpc.rpc_server
      env_vars:
        PYTHONPATH: "/AReaL/.venv/lib/python3.12/site-packages:${PYTHONPATH}"
```

### 并行策略建议

- **EP（Expert Parallel）**：256 experts，EP=8 是自然选择（每卡 32 experts）
- **TP（Tensor Parallel）**：MLA 层需要 TP，Flash 模型 32 heads 建议 TP=2 或 TP=4
- **PP（Pipeline Parallel）**：异构层已支持 PP slicing，PP=4 经测试可用
- **CP（Context Parallel）**：Lightning Attention 不支持 CP（O(n) 无需 CP），不要启用

## 代码结构

```
areal/models/mcore/
├── bailing_moe_bridge.py    # mbridge Bridge：权重映射（HF ↔ mcore）
├── bailing_moe.py           # HF config → MLATransformerConfig，异构 layer spec 构建
├── lightning_attention.py   # Lightning Attention 实现（LightningSelfAttention）
└── hf_load.py               # HF 权重加载辅助
```

### 权重映射

BailingMoeBridge 支持以下权重映射：

**Lightning Attention 层**：
- HF `attention.query_key_value.weight` → mcore `self_attention.linear_qkv.weight`
  - 需要从 `[3,H,D]` 转为 `[H,3,D]` 格式（megatron interleaved QKV）
- HF `attention.g_proj.weight` → mcore `self_attention.linear_gate.weight`
- HF `attention.g_norm.weight` → mcore `self_attention.gate_norm.weight`

**MLA 层（q_lora_rank=None，如 Mini）**：
- HF `attention.q_proj.weight` → mcore `self_attention.linear_q_proj.weight`

**MLA 层（q_lora_rank!=None，如 Flash）**：
- HF `attention.q_a_proj.weight` → mcore `self_attention.linear_q_down_proj.weight`
- HF `attention.q_b_proj.weight` → mcore `self_attention.linear_q_up_proj.weight`
- HF `attention.q_a_layernorm.weight` → mcore `self_attention.linear_q_up_proj.layer_norm_weight`

**MLA 层公共**：
- HF `attention.kv_a_proj_with_mqa.weight` → mcore `self_attention.linear_kv_down_proj.weight`
- HF `attention.kv_b_proj.weight` → mcore `self_attention.linear_kv_up_proj.weight`
- HF `attention.kv_a_layernorm.weight` → mcore `self_attention.linear_kv_up_proj.layer_norm_weight`

## 已知限制

1. **CP 不支持**：Lightning Attention 不兼容 megatron-core 的 Context Parallelism
2. **VPP 不支持**：异构层结构不兼容 Virtual Pipeline Parallelism
3. **rotary_interleaved 不支持**：megatron-core 禁止 MLA + rotary_interleaved
4. **TE 版本敏感**：不同 TE 版本的 MLA core attention 数值行为略有差异，不影响训练收敛

## 与 HybridEngine 对比验证

Mini 模型对比结果（真实文本，seq_len=512，EP=8, TP=1, PP=1）：

| 指标 | AReaL | HybridEngine | 差异 |
|------|-------|-------------|------|
| Loss | 1.512 | 1.491 | 1.33% |
| PPL | 4.54 | 4.44 | 2.2% |

逐层对比：
- **Lightning 层（0-3）**：cosine 0.999+，几乎完全一致
- **MLA 层（4,9,14,19）**：attention cosine 0.92-0.95，差异来自 TE/flash_attn 版本不同
- **整体 hidden states**：cosine 0.996+

Loss 差异主要由 MLA 层的 TE 版本差异累积而来，不影响训练正确性（mini 模型 SFT 初始 loss=0.567，
远低于目标 ≤2）。
