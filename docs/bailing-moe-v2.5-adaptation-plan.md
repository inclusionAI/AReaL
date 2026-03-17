# Plan: BailingMoeV2_5ForCausalLM Support in AReaL

## Context

需要在当前分支 `chucai.dzq/bailing-sft-v1.0.1` 上适配 bailing-2.5 系列模型
(`BailingMoeV2_5ForCausalLM`)，用于 SFT 训练。

**模型架构特点**:

- 混合注意力: Lightning Attention (大部分层) + MLA (每 `layer_group_size` 层)
- MoE: sigmoid routing, grouped TopK (n_group=8, topk_group=4), shared experts
- 不依赖 antllm，在 AReaL 中自行实现

**现状**:

- AReaL Megatron 路径仅支持 Qwen3ForCausalLM
- **megatron-core 0.13.1** (PyPI, 非本地 0.11.0) 已有:
  - MLA: `MLASelfAttention`, `MLATransformerConfig` (q_lora_rank, kv_lora_rank,
    qk_head_dim 等)
  - MoE: `TopKRouter` (sigmoid + group-limited routing + expert bias), SharedExpertMLP,
    `moe_layer_freq`
  - Heterogeneous: `HeterogeneousTransformerConfig` (per-layer attention/MLP
    no-op/linear replacement)
  - MTP: `multi_token_prediction.py`
  - **没有 Lightning Attention** 和 `layer_group_size`
- **SGLang 0.5.9** (PyPI) 已支持 `BailingMoeV2_5ForCausalLM`，只需升级版本
- **flash-linear-attention (`fla`) v0.3.0** 提供 `chunk_lightning_attn` Triton 内核，可直接调用
  - 路径: `/storage/openpsi/codes/chucai.dzq/gh/flash-linear-attention`
  - API:
    `fla.ops.lightning_attn.chunk_lightning_attn(q, k, v, g_gamma, layer_idx, num_layers, scale, ...)`
  - 张量形状: `[B, T, H, K]` (需从 Megatron 的 `[S, B, H]` 转换)
- `get_gpt_decoder_block_spec()` 支持 MoE/Dense 混合，但所有层用**相同 attention 类型**
- **NCCL 权重更新**: `areal/engine/megatron_utils/megatron.py` 中的 `_CONVERSION_FN_REGISTRY`
  需要添加 BailingMoe 条目

______________________________________________________________________

## 实施计划

### Phase 0: 前置验证 (集群上执行)

验证 `mbridge==0.13.0` 是否支持 BailingMoe 权重映射：

```python
from mbridge.core import Bridge
bridge = Bridge.from_pretrained("/path/to/bailing-moe-v2.5-hf")
# 检查是否正确解析 config 和权重映射
```

- **支持** → 现有 `hf_load.py`/`hf_save.py` 通过 mbridge 自动工作
- **不支持** → 需要实现独立权重转换模块 (Phase 5)

______________________________________________________________________

### Phase 1: SGLang 版本升级

**目标**: 让 SGLang 推理侧支持 BailingMoeV2_5ForCausalLM

**方案**: 升级 `pyproject.toml` 中 sglang 版本从 `0.5.7` 到 `>=0.5.9`。

PyPI sglang 0.5.9 已内置 `BailingMoeV2_5ForCausalLM`
支持（`sglang/srt/models/bailing_moe_linear.py`），无需从 theta-sglang 移植。

**修改文件**: `pyproject.toml` (line 138)

```diff
- "sglang[tracing]==0.5.7",
+ "sglang[tracing]>=0.5.9",
```

**AReaL 的 `sglang_remote.py` 无需修改** — 它是模型无关的 HTTP 客户端。

______________________________________________________________________

### Phase 2: Lightning Attention 模块实现

**新建文件**: `areal/models/mcore/lightning_attention.py`

megatron-core 没有 Lightning Attention，使用 **`fla` (flash-linear-attention) v0.3.0** 库的
Triton 内核实现。

**依赖**: `fla` 库 (pip installable)，提供 `chunk_lightning_attn` Triton JIT 内核

- API:
  `fla.ops.lightning_attn.chunk_lightning_attn(q, k, v, g_gamma, layer_idx, num_layers, scale, initial_state, output_final_state, cu_seqlens)`
- 输入形状: `[B, T, H, K]` (B=batch, T=seq_len, H=num_heads, K=head_dim)
- **需要从 Megatron 的 `[S, B, H]` 布局转换**

**核心组件**:

```python
@dataclass
class LightningAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type]      # fused QKV projection
    core_attention: Union[ModuleSpec, type]   # Lightning attention kernel (fla)
    linear_proj: Union[ModuleSpec, type]      # output projection (dense)
    linear_gate: Union[ModuleSpec, type]      # gate projection (g_proj)
    q_layernorm: Union[ModuleSpec, type]      # Q RMSNorm
    k_layernorm: Union[ModuleSpec, type]      # K RMSNorm
    gate_layernorm: Union[ModuleSpec, type]   # gate norm (g_norm)

class LightningCoreAttention(MegatronModule):
    """调用 fla.ops.lightning_attn.chunk_lightning_attn 的核心计算"""
    # 处理张量布局转换: [S, B, H] -> [B, T, num_heads, head_dim] -> fla kernel -> 转回

class LightningSelfAttention(Attention):
    """Lightning Self-Attention, 继承 megatron Attention 基类"""
    # fused QKV -> Q/K/V split -> RoPE -> Q/K LayerNorm -> fla kernel -> gate -> output
```

**关键参数**:

- `g_gamma`: 每 head 的常数 log decay，在 `__init__` 中按 `layer_idx / num_layers` 计算
- `linear_attn_norm_group_size`: 输出的 group normalization 大小
- QKV 是融合的 (`query_key_value`), 不同于 MLA 的分离投影
- 有 gate projection (`g_proj`) 和 gate norm (`g_norm`)

______________________________________________________________________

### Phase 3: Config 转换 + 异构 Block Spec 构建

**新建文件**: `areal/models/mcore/bailing_moe.py`

#### 3a. HF Config -> MLATransformerConfig

```python
def hf_to_mcore_config_bailing_moe(
    hf_config: PretrainedConfig, dtype: torch.dtype
) -> MLATransformerConfig:
```

**字段映射**:

| HF Config                                | MLATransformerConfig                  |
| ---------------------------------------- | ------------------------------------- |
| `num_hidden_layers`                      | `num_layers`                          |
| `hidden_size`                            | `hidden_size`                         |
| `num_attention_heads`                    | `num_attention_heads`                 |
| `num_key_value_heads`                    | `num_query_groups`                    |
| `q_lora_rank`                            | `q_lora_rank`                         |
| `kv_lora_rank`                           | `kv_lora_rank`                        |
| `qk_nope_head_dim`                       | `qk_head_dim`                         |
| `qk_rope_head_dim`                       | `qk_pos_emb_head_dim`                 |
| `v_head_dim`                             | `v_head_dim`                          |
| `num_experts`                            | `num_moe_experts`                     |
| `num_experts_per_tok`                    | `moe_router_topk`                     |
| `n_group`                                | `moe_router_num_groups`               |
| `topk_group`                             | `moe_router_group_topk`               |
| `moe_intermediate_size`                  | `moe_ffn_hidden_size`                 |
| `num_shared_experts * intermediate_size` | `moe_shared_expert_intermediate_size` |
| `first_k_dense_replace`                  | → `moe_layer_freq` list               |
| `scoring_func="sigmoid"`                 | `moe_router_score_function="sigmoid"` |
| `rope_theta=600000`                      | `rotary_base=600000`                  |

复用 `common.py:hf_to_mcore_base_args()` 构建基础参数，再叠加 MLA + MoE 特有参数。

#### 3b. 异构 Layer Spec 构建

```python
def is_lightning_layer(layer_number: int, layer_group_size: int) -> bool:
    """layer_group_size=4: layers 0,1,2 是 Lightning, layer 3 是 MLA, 循环"""
    if layer_group_size <= 1:
        return False
    return (layer_number + 1) % layer_group_size != 0

def make_mcore_layer_specs_bailing_moe(
    tf_config: MLATransformerConfig,
    hf_config: PretrainedConfig,
    use_te: bool = True,
) -> TransformerBlockSubmodules:
```

**不使用** `get_gpt_decoder_block_spec()` (它只支持同一种 attention)，而是直接构建 `layer_specs` 列表:

```python
# 复用 megatron-core 的 get_gpt_layer_with_transformer_engine_spec / get_gpt_layer_local_spec
# 传不同的 multi_latent_attention 参数来生成不同 attention 的 spec

# 4 种变体:
lightning_dense = get_gpt_layer_spec(multi_latent_attention=False, num_experts=None, ...)  # Lightning 需自定义
lightning_moe   = get_gpt_layer_spec(multi_latent_attention=False, num_experts=N, ...)
mla_dense       = get_gpt_layer_spec(multi_latent_attention=True,  num_experts=None, ...)  # 复用现有 MLA
mla_moe         = get_gpt_layer_spec(multi_latent_attention=True,  num_experts=N, ...)

# 按层分配:
for i in range(num_layers):
    is_lightning = is_lightning_layer(i, layer_group_size)
    is_moe = i >= first_k_dense_replace
    layer_specs.append(选择对应的 spec)
```

4 种 layer spec 变体:

1. Lightning Attention + Dense MLP (前 first_k_dense_replace 层中的 Lightning 层)
1. Lightning Attention + MoE MLP (大部分层)
1. MLA Attention + Dense MLP (前 first_k_dense_replace 层中的 MLA 层)
1. MLA Attention + MoE MLP (每 layer_group_size 层)

注: MLA 层直接复用 megatron-core 的 `MLASelfAttention`; Lightning 层需自定义 attention module (Phase
2\)

______________________________________________________________________

### Phase 4: AReaL Model Registry 集成

**修改文件**: `areal/models/mcore/registry.py`

在 `make_hf_and_mcore_config()` 和 `make_mcore_layer_specs()` 中添加
`BailingMoeV2_5ForCausalLM` 分支:

```python
from areal.models.mcore.bailing_moe import (
    hf_to_mcore_config_bailing_moe,
    make_mcore_layer_specs_bailing_moe,
)

# In make_hf_and_mcore_config:
elif architecture == "BailingMoeV2_5ForCausalLM":
    return hf_config, hf_to_mcore_config_bailing_moe(hf_config, dtype)

# In make_mcore_layer_specs:
elif architecture == "BailingMoeV2_5ForCausalLM":
    return make_mcore_layer_specs_bailing_moe(tf_config, hf_config, use_te=True)
```

**修改文件**: `areal/engine/core/model.py`

```python
VALID_MOE_MODELS = ["qwen3_moe", "bailing_moe_v2"]
```

______________________________________________________________________

### Phase 5: 权重转换

#### 5a. 磁盘权重加载/保存 (HF ↔ mcore via mbridge)

取决于 Phase 0 结果：

- **mbridge 支持** → 现有 `hf_load.py`/`hf_save.py` 自动工作
- **mbridge 不支持** → 需要实现独立权重转换，可能修改 `hf_load.py` 中的 `_weight_to_mcore_tp()` 添加 Lightning
  attention 分支

#### 5b. NCCL 在线权重更新 (Megatron → HF，用于推理)

**修改文件**: `areal/engine/megatron_utils/megatron.py`

在 `_CONVERSION_FN_REGISTRY` (line 535) 中添加 BailingMoe 条目：

```python
_CONVERSION_FN_REGISTRY = {
    "qwen3_moe": convert_qwen3moe_to_hf,
    "qwen2": convert_qwen2_to_hf,
    "qwen3": convert_qwen2_to_hf,
    "deepseekv3": convert_deepseekv3_to_hf,
    "bailing_moe_v2": convert_bailingmoe_to_hf,  # 新增
}
```

**新增函数**: `convert_bailingmoe_to_hf()`

由于 BailingMoe 有两种 attention 类型，需要按层判断转换逻辑：

```python
def convert_bailingmoe_to_hf(mcore_state_dict, hf_config) -> dict:
    # 通用部分 (复用 convert_deepseekv3_to_hf 模式):
    # - embedding, output_layer, final_layernorm
    # - MoE: router, expert fc1 (gate+up split), fc2, shared_experts, expert_bias

    # 按层区分 attention 转换:
    for layer_idx in range(num_layers):
        if is_lightning_layer(layer_idx, layer_group_size):
            # Lightning 层: fused QKV → query_key_value
            #   self_attention.linear_qkv.weight → self_attn.query_key_value.weight
            #   self_attention.linear_gate.weight → self_attn.g_proj.weight
            #   self_attention.linear_proj.weight → self_attn.dense.weight
            #   q_layernorm/k_layernorm/gate_layernorm → q_norm/k_norm/g_norm
        else:
            # MLA 层 (复用 convert_deepseekv3_to_hf 的 MLA 映射):
            #   linear_q_down_proj → q_a_proj
            #   linear_q_up_proj → q_b_proj
            #   linear_kv_down_proj → kv_a_proj_with_mqa
            #   linear_kv_up_proj → kv_b_proj
            #   linear_proj → o_proj
            #   q_layernorm → q_a_layernorm, k_layernorm → kv_a_layernorm
```

**参考模板**:

- MLA 层映射: `convert_deepseekv3_to_hf` (line 380-530)
- MoE expert 映射: `convert_qwen3moe_to_hf` (line 133-286)
- Lightning 层 QKV: HybridEngine `bailing_moe_linear.py` 的反向映射

**关键权重映射** (mcore -> HF):

Lightning 层:

- `self_attention.linear_qkv.weight` → `self_attn.query_key_value.weight`
- `self_attention.linear_gate.weight` → `self_attn.g_proj.weight`
- `self_attention.linear_proj.weight` → `self_attn.dense.weight`
- `q_layernorm/k_layernorm/gate_layernorm` → `q_norm/k_norm/g_norm`

MLA 层:

- `self_attention.linear_q_down_proj.weight` → `self_attn.q_a_proj.weight`
- `self_attention.linear_q_up_proj.weight` → `self_attn.q_b_proj.weight`
- `self_attention.linear_kv_down_proj.weight` → `self_attn.kv_a_proj_with_mqa.weight`
- `self_attention.linear_kv_up_proj.weight` → `self_attn.kv_b_proj.weight`

______________________________________________________________________

### Phase 6: SFT 配置和示例

**新建**: `examples/bailing_moe_sft.yaml` (参考现有配置模板)

______________________________________________________________________

## 关键文件清单

| 操作            | 文件                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| **新建**        | `areal/models/mcore/lightning_attention.py` — Lightning Attention 模块 (基于 fla 库)                     |
| **新建**        | `areal/models/mcore/bailing_moe.py` — Config 转换 + Layer Spec 构建                                      |
| **新建**        | `examples/bailing_moe_sft.yaml` — SFT 配置                                                               |
| **修改**        | `areal/models/mcore/registry.py` — 添加 BailingMoeV2_5 分支                                              |
| **修改**        | `areal/engine/core/model.py` — 添加 bailing_moe 到 VALID_MOE_MODELS                                      |
| **修改**        | `areal/engine/megatron_utils/megatron.py` — 添加 `convert_bailingmoe_to_hf` 到 `_CONVERSION_FN_REGISTRY` |
| **修改**        | `pyproject.toml` — SGLang 升级到 >=0.5.9, 添加 fla 依赖                                                  |
| **修改** (可能) | `areal/models/mcore/hf_load.py` — Lightning attention 权重加载 (如 mbridge 不支持)                       |

## 复用的现有代码

| 文件                                                         | 复用内容                                |
| ------------------------------------------------------------ | --------------------------------------- |
| `areal/models/mcore/common.py:hf_to_mcore_base_args()`       | 基础 TransformerConfig 字段映射         |
| `areal/models/mcore/common.py:check_and_construct_configs()` | Config 构造 + 兼容性检查                |
| `areal/models/mcore/qwen3.py`                                | 参考模式：config 转换 + layer spec 构建 |
| `fla.ops.lightning_attn.chunk_lightning_attn`                | Lightning Attention Triton 内核         |
| `megatron.core.transformer.multi_latent_attention`           | MLA 实现 (MLA 层直接复用)               |
| `megatron.core.models.gpt.gpt_layer_specs`                   | Layer spec 构建模式                     |
| `megatron.core.transformer.moe`                              | MoE 路由、expert、dispatcher            |
| `convert_deepseekv3_to_hf` (megatron.py:380)                 | MLA 权重映射模板                        |
| `convert_qwen3moe_to_hf` (megatron.py:133)                   | MoE expert 权重映射模板                 |

## 实施顺序

```
Phase 0 (集群验证 mbridge) ─┐
                              ├── Phase 2 (Lightning Attention + fla) ──→ Phase 3 (Config + Specs) ──→ Phase 4 (Registry)
Phase 1 (SGLang 升级) ───────┘                                                                             │
                                                                                                            ↓
                                                                          Phase 5a (磁盘权重, 依赖 Phase 0) + Phase 5b (NCCL 权重更新)
                                                                                                            │
                                                                                                            ↓
                                                                                                   Phase 6 (SFT 配置)
```

## 验证方案

1. **单元测试**: Lightning Attention 模块的前向/反向正确性 (随机输入)
1. **模型构建测试**: 从 HF config 创建完整 GPTModel，验证参数数量和形状
1. **权重加载测试**: 加载 HF checkpoint，比较关键层输出
1. **SGLang 推理测试**: 启动 SGLang server，验证 generate 接口
1. **端到端 SFT 测试**: 小数据集 + 小 batch，验证 loss 下降

______________________________________________________________________

## 代码审查结果

### 第一轮 (2026-03-16): TP 支持问题 — 已修复

TP>1 时的 3 个 Critical 问题（QKV reshape 使用全局 heads、g_gamma 未 TP 切片、
GroupRMSNorm weight 大小错误）已在提交 `d5423ae7` 中修复。

### 第二轮 (2026-03-16): 深度 Review — 含验证结论

**涉及提交**:

```
0b189036 fix: add PP layer spec slicing for BailingMoeV2.5 heterogeneous layers
d5423ae7 feat: add TP/EP support for BailingMoeV2.5 Lightning Attention + MLA
34c99ac5 feat: add BailingMoeV2_5ForCausalLM support with Lightning Attention + MLA
d43c62ce adpt bailing sft
```

______________________________________________________________________

#### Critical — 需修复

##### ~~C1. g_gamma 计算公式与 fla 官方实现不一致~~ → 已验证：公式正确

**状态**: **Review 错误，无需修复**

Review 文档比较了我们的 ALiBi geometric slopes 与 fla 的 `chunk_lightning_attn` 内部
简化公式 `-(8/H) * (1 - layer_idx/num_layers) * h`，认为不一致。

**验证结论**: ALiBi geometric slopes 是 Lightning Attention-2 论文的官方公式，
在以下参考实现中完全一致：

| 实现 | 文件 | 公式 |
|------|------|------|
| HF 参考模型 | `antllm/models/bailing_moe/modeling_bailing_moe_v2_5.py:754` | `_build_slope_tensor()` → ALiBi slopes |
| Megatron-LM | `megatron/core/transformer/attention.py:1744` | `_build_slope_tensor()` → ALiBi slopes |
| AReaL | `lightning_attention.py:136-146` | `_build_alibi_slopes()` → ALiBi slopes |

fla 的 `chunk_lightning_attn` 简化公式是 fla 自己的默认值，**不是 BailingMoe 模型的
实际行为**。我们正确使用 `chunk_simple_gla` + 自行计算的 `g_gamma`。

fla 的 `g_gamma` 参数（shape `[H]`，per-head 常数 log decay）与 HF 模型的 `g` 参数
（shape `[B,T,H]`，展开后语义相同）是等价的，`g_gamma` 路径更高效。

##### C2. gate_norm 应用位置错误 ✅ 需修复

**文件**: `lightning_attention.py:454-463`

```python
# 当前代码（错误）:
gate = self.gate_norm(gate)              # L454: norm gate
attn_output = self.gate_norm(attn_output)  # L462: 又 norm attn_output
output = attn_output * gate.sigmoid()      # L463
```

**HF 参考实现** (`modeling_bailing_moe_v2_5.py:878-881`):

```python
o = self.g_norm(o)                    # g_norm 只用于 attn_output
g_proj = self.g_proj(hidden_states)   # gate 不做 norm
o = o * torch.sigmoid_(g_proj)        # sigmoid(raw gate)
```

**修复**: 移除 L454 `gate = self.gate_norm(gate)`，保留 L462 对 attn_output 的 norm。

> **注意**: 原 review 文档的修复建议方向反了（建议只 norm gate，移除 attn_output 的 norm）。
> 经 HF 参考实现验证，正确做法是：**norm attn_output，不 norm gate**。

##### C3. HF 权重加载缺少 QKV 格式转换 ✅ 需修复（TP>1）

**文件**: `hf_load.py:149-157`

HF 存储格式: `[Q_all, K_all, V_all]` concatenated
Megatron 期望格式: `[q0,k0,v0, q1,k1,v1, ...]` 按 head interleaved

`_slice_generic_weight()` 只做简单维度切片，不做格式转换。

- **TP=1**: 两种格式的 shape 完全相同 `[3*H*D, hidden]`，但内部排列不同
  （Q/K/V 对应错误的 head），不过实际影响需要进一步验证
- **TP>1**: 切片逻辑也不正确（应按 head 交错切而非连续段切片）

**建议修复**:

```python
# HF [Q,K,V] concatenated -> megatron interleaved
x = hf_weights[0]
x = x.view(3, num_heads, head_dim, hidden)
x = x.permute(1, 0, 2, 3).contiguous().view(-1, hidden)  # [H, 3, D, hidden]
if tp_size > 1:
    heads_per_tp = num_heads // tp_size
    x = x.view(num_heads, 3 * head_dim, hidden)
    x = x[tp_rank * heads_per_tp : (tp_rank + 1) * heads_per_tp].reshape(-1, hidden)
```

##### C4. q_lora_rank 默认值不一致 ⚠️ 建议统一

**文件**: `bailing_moe.py:103` 默认 512 vs `bailing_moe_bridge.py:141` 默认 None

**实际模型**: `q_lora_rank = None`（属性显式存在，`getattr` 不会用到默认值）。
当前模型不受影响，但默认值应统一为 None。

______________________________________________________________________

#### High — 特定并行策略影响

##### H1. Bridge MLA 映射缺 q_down/q_up — 当前模型不需要

当前模型 `q_lora_rank=None`，只需 `linear_q_proj`。映射已标注
`"q_lora_rank=None variant"`。后续支持 `q_lora_rank != None` 模型时需补充。

##### H2. expert_bias HF 名称可能与 SGLang 不一致

`megatron.py` 中用 `e_score_correction_bias`（DeepSeekV3 命名），Bridge 中用
`expert_bias`。需确认 SGLang 实际使用的参数名。

______________________________________________________________________

#### Medium — 建议修复

| # | 问题 | 建议 |
|---|------|------|
| M1 | `_build_alibi_slopes()` ~~冗余~~ → 实际是正确的，保留 | 无需修改 |
| M2 | 每个 Lightning 层独立创建 RotaryEmbedding，浪费显存 | 共享实例通过 ModuleSpec.params 传入 |
| M3 | `kv_channels` 未显式设置 | `base_args["kv_channels"] = qk_rope_head_dim` |
| M4 | `except Exception:` 过于宽泛 | 缩小为 `except (RuntimeError, AttributeError):` |
| M5 | SP 下 gate 输入行为需确认 | TP=2 测试已通过，暂无问题 |

______________________________________________________________________

#### Low — 代码质量

| # | 问题 | 建议 |
|---|------|------|
| L1 | 无效的 self-assignment (`lightning_rotary = lightning_rotary`) | 删除 no-op |
| L2 | plan doc 权重名注释不一致 | 统一 |
| L3 | 示例配置硬编码个人路径 | 使用占位符 |
| L4 | `convert_bailingmoe_to_hf` 未过滤 `_extra_state` | 添加过滤 |

______________________________________________________________________

#### 修复优先级

1. **C2** (gate_norm) — 直接影响数值正确性
2. **C3** (QKV 格式转换) — 影响权重加载正确性
3. **C4** (q_lora_rank 统一) — 代码健壮性
4. **L1/L4/M4** — 代码清理

______________________________________________________________________

#### 测试验证

修复后按顺序验证：

1. **模型构建**: `GPTModel(config, block_spec)` 不报错，参数数量与 HF 模型一致
2. **权重加载 (TP=1)**: 从 HF checkpoint 加载，比较关键层输出
3. **权重加载 (TP>1)**: 验证 QKV interleaved 格式转换 + TP 切片正确性
4. **Forward pass**: Lightning 层和 MLA 层输出 shape 正确
5. **SFT 训练**: 小数据集上 loss 正常下降
