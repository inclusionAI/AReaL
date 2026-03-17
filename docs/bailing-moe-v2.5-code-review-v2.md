# BailingMoeV2.5 适配 Code Review (v2)

**日期**: 2026-03-16
**范围**: 对比上一次 review (`docs/bailing-moe-v2.5-code-review.md`) 后的未提交修改

______________________________________________________________________

## 上次 Review 问题修复状态

### Critical 问题

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| C1 | g_gamma 公式 (ALiBi vs fla 线性) | **需要重新评估** | 经 CP review 发现 Megatron-LM 参考实现也用 ALiBi slopes，当前代码可能正确。见下方新分析。 |
| C2 | gate_norm 双重应用 | **已修复** ✓ | 移除了对 gate 的 `gate_norm` 调用，现在只 norm `attn_output`，gate 只过 `sigmoid`。与参考实现一致。 |
| C3 | HF 权重加载缺少 concatenated→interleaved 转换 | **已修复** ✓ | 新增 `_convert_fused_qkv_weight()` 函数，实现 `[3,H,D]→[H,3,D]` 转换 + TP 按 head 切片。 |
| C4 | q_lora_rank 默认值不一致 | **已修复** ✓ | `bailing_moe.py` 默认值从 `512` 改为 `None`，与 `bailing_moe_bridge.py` 一致。 |

### High 问题

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| H1 | Bridge MLA 映射缺少 q_lora_rank!=None 的权重名 | **已修复** ✓ | 新增 `_MLA_ATTENTION_MAPPING_Q_LORA` (q_a_proj, q_a_layernorm, q_b_proj)，按 `q_lora_rank` 动态选择。 |
| H2 | expert_bias HF 名称 | **未修改** | 仍然映射为 `e_score_correction_bias`，需确认 SGLang 使用的名称。 |
| H3 | head_dim 依赖 kv_channels | **未修改** | `kv_channels` 仍然可能为 None。 |

### Medium / Low 问题

| # | 问题 | 状态 |
|---|------|------|
| M4 | 异常捕获过于宽泛 | **已修复** ✓ — `except Exception` 改为 `except (RuntimeError, AttributeError)` 和 `except (ImportError, TypeError, RuntimeError)` |
| L1 | 无效 self-assignment | **已修复** ✓ — 删除了 `lightning_rotary = lightning_rotary` 等 no-op |
| L4 | convert_bailingmoe_to_hf 未过滤 _extra_state | **已修复** ✓ — 函数开头添加 `if "_extra_state" in name: return []` |

______________________________________________________________________

## 新增修改的 Review

### 1. `_convert_fused_qkv_weight()` — 正确 ✓

**文件**: `areal/models/mcore/hf_load.py:73-105`

```python
# [3*H*D, hidden] -> [3, H, D, hidden] -> [H, 3, D, hidden] -> [H*3*D, hidden]
x = x.view(3, num_heads, head_dim, hidden)
x = x.permute(1, 0, 2, 3).contiguous().view(-1, hidden)
```

这个转换与 `convert_bailingmoe_to_hf` 中的逆操作精确互逆：

```python
# megatron.py:648-651 (save 路径):
# [H*3*D, hidden] -> [H, 3, D, hidden] -> [3, H, D, hidden] -> [3*H*D, hidden]
param_rearranged = param.view(num_heads, 3, head_dim, hidden_size)
param_rearranged = param_rearranged.permute(1, 0, 2, 3).contiguous()
```

load 是 `[3,H,D] → [H,3,D]`，save 是 `[H,3,D] → [3,H,D]`。互逆，正确。

TP>1 时按 head 切片也正确：先 reshape 为 `[H, 3*D, hidden]`，取 `[heads_per_tp, 3*D, hidden]`。

**一个小问题**: `head_dim` 的推导 (`x.shape[0] // (num_heads * 3)`) 假设 Q/K/V
的 head_dim 相同。对于 Lightning Attention 这是成立的（fused QKV），但如果未来有
GQA 变体（K/V head 数不同），需要调整。当前不影响。

______________________________________________________________________

### 2. gate_norm 修复 — 正确 ✓

**文件**: `areal/models/mcore/lightning_attention.py:454-458`

```python
# 旧代码 (错误):
gate = self.gate_norm(gate)     # gate 被 norm
attn_output = self.gate_norm(attn_output)  # attn_output 也被同一个 norm

# 新代码 (正确):
attn_output = self.gate_norm(attn_output)  # 只 norm attn_output
output = attn_output * gate.sigmoid()       # gate 只过 sigmoid
```

与 Megatron-LM 参考实现 (`attention.py:2487-2491`) 一致：

```python
core_attn_out = self.pre_gate_norm(core_attn_out)
gate = F.sigmoid(gate)
core_attn_out = core_attn_out * gate
```

______________________________________________________________________

### 3. Bridge MLA mapping 拆分 — 正确 ✓

**文件**: `areal/models/mcore/bailing_moe_bridge.py:43-77`

将原来的 `_MLA_ATTENTION_MAPPING` 拆分为三部分：

- `_MLA_ATTENTION_MAPPING_Q_DIRECT`: `q_lora_rank=None` 时 (`linear_q_proj`)
- `_MLA_ATTENTION_MAPPING_Q_LORA`: `q_lora_rank!=None` 时 (`linear_q_down_proj` + `linear_q_up_proj`)
- `_MLA_ATTENTION_MAPPING_COMMON`: KV 投影等通用映射

在 `_weight_name_mapping_attention` 中动态合并：

```python
q_lora_rank = getattr(self.hf_config, "q_lora_rank", None)
q_mapping = _MLA_ATTENTION_MAPPING_Q_LORA if q_lora_rank is not None else _MLA_ATTENTION_MAPPING_Q_DIRECT
mapping = {**_MLA_ATTENTION_MAPPING_COMMON, **q_mapping}
```

逻辑正确，覆盖了两种 q_lora_rank 情况。

______________________________________________________________________

### 4. megatron.py 新增 q_lora MLA 转换 — 正确 ✓

**文件**: `areal/engine/megatron_utils/megatron.py:705-710`

新增了 `q_a_proj`, `q_a_layernorm`, `q_b_proj` 的 NCCL 路径转换，与 bridge 映射
和 DeepSeekV3 路径一致。

`_extra_state` 过滤也已添加 (line 557-558)。

`bailing_hybrid` 注册也已添加 (line 751)。

______________________________________________________________________

### 5. `@register_model("bailing_hybrid")` — 正确 ✓

**文件**: `areal/models/mcore/bailing_moe_bridge.py:82`

新增注册名，同时在 `_CONVERSION_FN_REGISTRY` 中也添加了对应条目。

______________________________________________________________________

## 仍存在的问题

### R1 [需确认] — g_gamma 公式

**状态**: 需要确认，可能已正确。

当前代码使用 ALiBi slopes + layer scaling：

```python
# lightning_attention.py:138-141
alibi_slopes = _build_alibi_slopes(num_heads_global)
layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5
g_gamma_global = -alibi_slopes * layer_scale
```

Megatron-LM 参考实现 (`attention.py:1744`) 使用**同样的公式**：

```python
slope = -self._build_slope_tensor(self.config.num_attention_heads) \
        * (1 - (self.global_layer_number - 1) / (self.config.num_layers - 1) + 1e-5)
```

两者行为一致（都是 ALiBi geometric slopes × layer-dependent scaling）。
当前代码调用的是 `chunk_simple_gla(g_gamma=self.g_gamma)`，手动传入 g_gamma，
**不会触发 fla 的内置公式**。

**结论: g_gamma 公式正确。** 上一次 review 中 C1 的结论撤回。

______________________________________________________________________

### R2 [仍存在] — expert_bias HF 名称待确认

**文件**: `areal/engine/megatron_utils/megatron.py:637-639`

```python
elif rest == "mlp.router.expert_bias":
    return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
```

`e_score_correction_bias` 是 DeepSeekV3 命名。BailingMoe HF 使用的可能是
`expert_bias`（Bridge 的 `_MLP_MAPPING` line 100 映射为 `mlp.gate.expert_bias`）。

两处不一致：

- Bridge (mbridge 路径): `mlp.gate.expert_bias`
- NCCL 转换 (megatron.py): `mlp.gate.e_score_correction_bias`

**影响**: NCCL 路径推送权重到 SGLang 时，SGLang 可能找不到 bias。
需要确认 SGLang `bailing_moe_linear.py` 中实际使用的参数名。

______________________________________________________________________

### R3 [仍存在] — kv_channels 可能为 None

**文件**: `areal/models/mcore/bailing_moe.py:91-98` (经 `common.py`)

`hf_to_mcore_base_args()` 中 `kv_channels = getattr(hf_config, "head_dim", None)`。
BailingMoe HF config 没有通用 `head_dim` 字段。

当前影响范围有限（MLA 层用自己的 `v_head_dim`，Lightning 层用 `attn_head_dim`），
但如果 megatron-core 内部某处用 `kv_channels` 做计算，可能出错。

**建议**: 在 `hf_to_mcore_config_bailing_moe()` 的 `base_args` 覆盖中显式设置：

```python
base_args["kv_channels"] = getattr(hf_config, "attn_head_dim", None) or \
                           getattr(hf_config, "v_head_dim", 128)
```

______________________________________________________________________

### R4 [Low] — 示例配置硬编码路径

**文件**: `examples/bailing_moe_sft.yaml:13,33`

仍包含 `/storage/openpsi/codes/chucai.dzq/` 个人路径。

______________________________________________________________________

## 新发现的问题

### N1 [Medium] — `_convert_fused_qkv_weight` 缺少 GQA (linear_attn_num_query_groups) 处理

**文件**: `areal/models/mcore/hf_load.py:90-91`

```python
num_heads = hf_config.num_attention_heads
head_dim = x.shape[0] // (num_heads * 3)
```

这里假设 Q/K/V 的 head 数都等于 `num_attention_heads`。参考 Megatron-LM
`LinearAttention` (`attention.py:1727-1728`)：

```python
linear_attn_num_query_groups = self.config.linear_attn_num_query_groups \
    if self.config.linear_attn_num_query_groups > 0 else self.config.num_query_groups
```

如果 `linear_attn_num_query_groups` 与 `num_attention_heads` 不同（即 Lightning
层使用 GQA），fused QKV 的 shape 不是 `[3*H*D, hidden]` 而是
`[(H_q + H_kv + H_kv)*D, hidden]`。

**当前影响**: BailingMoe v2.5 flash 模型 `linear_attn_num_query_groups=32` 等于
`num_attention_heads=32`（1:1 GQA ratio），所以当前不影响。但如果未来有不同配置的
模型，此处需要调整。

**建议**: 添加注释说明此假设，或从 hf_config 读取 `linear_attn_num_query_groups`。

______________________________________________________________________

### N2 [Low] — Bridge 中 `@register_model("bailing_hybrid")` 但 registry.py 未更新

**文件**: `areal/models/mcore/registry.py:119-122,134`

`make_hf_and_mcore_config()` 和 `make_mcore_layer_specs()` 中只检查了
`BailingMoeV2_5ForCausalLM` 和 `BailingMoeLinearForCausalLM`，没有检查
`BailingHybridForCausalLM` (对应 `bailing_hybrid` model_type)。

如果有模型使用 `bailing_hybrid` architecture，非 mbridge 路径会报
`ValueError: Architecture not registered`。

**建议**: 在 registry.py 的 `elif` 分支中添加 `BailingHybridForCausalLM`。

______________________________________________________________________

## 汇总

| 等级 | 数量 | 状态 |
|------|------|------|
| 上次 Critical (4) | 3 已修复, 1 重新评估后确认正确 | **全部解决** |
| 上次 High (3) | 1 已修复, 2 未修改 | R2 (expert_bias) 仍需确认 |
| 上次 Medium (5) | 1 已修复, 其余未改但非阻塞 | R3 (kv_channels) 建议修复 |
| 上次 Low (4) | 2 已修复, 2 未改 | 不阻塞 |
| 新发现 | 1 Medium + 1 Low | N1 (GQA) 当前不影响, N2 (registry) 需补充 |

**阻塞项**: 无 Critical 阻塞项。R2 (expert_bias 名称) 如果与 SGLang 不一致会导致
NCCL 推送后推理结果错误，但仅影响 NCCL 路径，磁盘路径 (mbridge) 不受影响。

**建议优先级**:

1. 确认 R2 expert_bias 的 SGLang 命名（查 SGLang `bailing_moe_linear.py` 的
   weight loading 代码）
2. 修复 R3 kv_channels 显式设置
3. 补充 N2 registry.py 的 `BailingHybridForCausalLM` 分支
