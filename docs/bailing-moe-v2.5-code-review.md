# BailingMoeV2.5 适配 Code Review

**日期**: 2026-03-16
**分支**: `chucai.dzq/bailing-sft-v1.0.1`
**参考实现**: `/storage/openpsi/codes/chucai.dzq/Asystem-HybridEngine`
**涉及提交**:

```
0b189036 fix: add PP layer spec slicing for BailingMoeV2.5 heterogeneous layers
d5423ae7 feat: add TP/EP support for BailingMoeV2.5 Lightning Attention + MLA
34c99ac5 feat: add BailingMoeV2_5ForCausalLM support with Lightning Attention + MLA
d43c62ce adpt bailing sft
```

______________________________________________________________________

## 涉及文件

| 操作   | 文件                                       | 说明                          |
| ------ | ------------------------------------------ | ----------------------------- |
| 新建   | `areal/models/mcore/lightning_attention.py` | Lightning Attention 模块      |
| 新建   | `areal/models/mcore/bailing_moe.py`        | Config 转换 + Layer Spec 构建 |
| 新建   | `areal/models/mcore/bailing_moe_bridge.py`  | mbridge Bridge 权重映射       |
| 新建   | `examples/bailing_moe_sft.yaml`            | SFT 示例配置                  |
| 修改   | `areal/engine/megatron_utils/megatron.py`  | NCCL 权重转换函数             |
| 修改   | `areal/models/mcore/registry.py`           | 模型注册                      |
| 修改   | `areal/engine/core/model.py`               | MoE 模型列表                  |
| 修改   | `areal/models/mcore/hf_load.py`            | HF 权重加载 TP 切片           |

______________________________________________________________________

## Critical — 会导致错误结果或运行时崩溃

### C1. g_gamma 计算公式与 fla 官方实现不一致

**文件**: `areal/models/mcore/lightning_attention.py:136-146`

当前使用 ALiBi-style geometric slopes：

```python
alibi_slopes = _build_alibi_slopes(num_heads_global)  # geometric sequence: 2^(-(2^-(log2(n)-3)))
layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5
g_gamma = -alibi_slopes * layer_scale
```

而 fla 库 `chunk_lightning_attn`（`fla/ops/lightning_attn/chunk.py:71`）的官方实现是：

```python
g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, dtype=torch.float32)
```

即 `g_gamma[h] = -(8/H) * (1 - layer_idx/num_layers) * h`，是**简单线性递增**。

两个公式差异：

| 维度     | 当前实现 (ALiBi)              | fla 官方                             |
| -------- | ----------------------------- | ------------------------------------ |
| 头间分布 | 2 的负幂指数几何级数          | 线性递增 `0, 1, 2, ...`             |
| 层间缩放 | `1 - idx/(N-1) + 1e-5`       | `1 - idx/N`（分母不同，无 eps）      |
| 值域     | 各 head 值域差异极大（指数级）| 各 head 值域线性等间距               |

**影响**: 加载已有 HF 预训练权重后推理结果与原始模型不一致；从头训练时收敛行为也会不同。

**建议修复**: 使用 fla 官方公式，或直接调用 `chunk_lightning_attn` 代替 `chunk_simple_gla`：

```python
# 方案 A：直接调用 chunk_lightning_attn
from fla.ops.lightning_attn import chunk_lightning_attn
output, _ = chunk_lightning_attn(
    q, k, v, layer_idx=self.layer_idx,
    num_layers=self.num_layers, scale=self.scale,
)

# 方案 B：手动计算 g_gamma 后调用 chunk_simple_gla
H = num_heads_global
g_gamma_global = -(8.0 / H * (1 - self.layer_idx / self.num_layers)) * torch.arange(H, dtype=torch.float32)
# TP 切片
g_gamma = g_gamma_global[tp_rank * local_heads : (tp_rank + 1) * local_heads]
```

______________________________________________________________________

### C2. gate_norm 被应用了两次（同时 norm gate 和 attn_output）

**文件**: `areal/models/mcore/lightning_attention.py:454-463`

```python
# L454: gate 被 gate_norm normalize
gate = self.gate_norm(gate)

# L462-463: attn_output 也被同一个 gate_norm normalize
attn_output = self.gate_norm(attn_output)
output = attn_output * gate.sigmoid()
```

同一个 `self.gate_norm`（共享 learnable weight）被同时用于两种不同的信号。

**参考实现对比**: HybridEngine 中 g_norm 参数命名为 `pre_gate_norm`（含义为"gate 之前的 norm"），只应用于 gate path。

**影响**:

- gate_norm 的可学习参数同时承担两种不同语义的归一化，训练不稳定
- 与预训练权重的行为不一致，加载后推理结果错误

**建议修复**: 只对 gate 应用 gate_norm，移除对 attn_output 的 gate_norm 调用：

```python
gate = self.gate_norm(gate)
# 直接使用 attn_output，不再 norm
output = attn_output * gate.sigmoid()
```

如果确实需要对 attn_output 也做 norm，应创建独立的 norm 实例。

______________________________________________________________________

### C3. HF 权重加载缺少 concatenated→interleaved 格式转换

**文件**: `areal/models/mcore/hf_load.py:149-157`

```python
if len(hf_weights_safe_slice) == 3:
    res = _merge_qkv_weights(...)  # 标准 QKV：3 个独立 tensor 合并
else:
    # Lightning Attention fused QKV：直接按 shape 切片
    res = _slice_generic_weight(mcore_param_shape, hf_weights_safe_slice, tp_rank, tp_size)
```

问题：

- **HF 存储格式**: `[Q_all, K_all, V_all]` concatenated（参考 HybridEngine 中
  `transform_lightning_attention_and_layernorm_weights` 的转换逻辑确认）
- **Megatron 期望格式**: `[q0,k0,v0, q1,k1,v1, ...]` 按 head interleaved

`_slice_generic_weight()` 只做简单的维度切片，**不会做格式转换**。

**影响**:

- **TP=1**: shape 完全相同（`[3*H*D, hidden]`），加载成功但**权重排列错误**，
  前向计算结果是垃圾
- **TP>1**: 切片也不正确（应按 head 交错切而非连续段切片）

**建议修复**: 在 fused QKV 分支中实现格式转换：

```python
else:
    # Fused QKV (Lightning Attention): HF [Q,K,V] concatenated -> megatron interleaved
    x = hf_weights_safe_slice[0]
    x = x[:] if not isinstance(x, torch.Tensor) else x
    num_heads = hf_config.num_attention_heads
    head_dim = x.shape[0] // (num_heads * 3)
    hidden = x.shape[1]
    # [3, H, D, hidden] -> [H, 3, D, hidden] -> flatten
    x = x.view(3, num_heads, head_dim, hidden)
    x = x.permute(1, 0, 2, 3).contiguous().view(-1, hidden)
    if tp_size > 1:
        # TP 切片：按 head 切
        heads_per_tp = num_heads // tp_size
        x = x.view(num_heads, 3 * head_dim, hidden)
        x = x[tp_rank * heads_per_tp : (tp_rank + 1) * heads_per_tp]
        x = x.reshape(-1, hidden)
    res = x
```

______________________________________________________________________

### C4. q_lora_rank 默认值不一致且可能不正确

**文件 1**: `areal/models/mcore/bailing_moe.py:103`

```python
"q_lora_rank": getattr(hf_config, "q_lora_rank", 512),  # 默认 512
```

**文件 2**: `areal/models/mcore/bailing_moe_bridge.py:141`

```python
q_lora_rank=getattr(hf_config, "q_lora_rank", None),  # 默认 None
```

**参考实现**: 所有 BailingMoe v2.5 模型的 `q_lora_rank` 均为 **1536**。

**影响**:

- `bailing_moe.py` 默认 512 → MLA 层的 q_down_proj/q_up_proj shape 错误，权重加载 shape
  mismatch 崩溃
- `bailing_moe_bridge.py` 默认 None → MLA 层使用 `linear_q_proj` 而非
  `linear_q_down_proj` + `linear_q_up_proj`，结构完全不同
- 两处默认值不一致，行为不可预测

**建议修复**:

1. 不使用默认值，显式断言字段存在：

```python
assert hasattr(hf_config, "q_lora_rank"), (
    "BailingMoe HF config must have q_lora_rank field"
)
```

2. 或统一默认值为 1536（匹配已知模型）。

______________________________________________________________________

## High — 影响特定并行策略或权重路径

### H1. Bridge MLA 映射缺少 q_lora_rank 不为 None 时的权重名

**文件**: `areal/models/mcore/bailing_moe_bridge.py:44-61`

当前 `_MLA_ATTENTION_MAPPING` 只有 `linear_q_proj`（对应 `q_lora_rank=None`）：

```python
_MLA_ATTENTION_MAPPING = {
    "self_attention.linear_q_proj.weight": [
        "model.layers.{layer_number}.attention.q_proj.weight"
    ],
    ...
}
```

参考实现确认 `q_lora_rank=1536`，此时 megatron 使用 `linear_q_down_proj` +
`linear_q_up_proj`。

**缺少以下映射**：

```python
"self_attention.linear_q_down_proj.weight": [
    "model.layers.{layer_number}.attention.q_a_proj.weight"
],
"self_attention.linear_q_up_proj.layer_norm_weight": [
    "model.layers.{layer_number}.attention.q_a_layernorm.weight"
],
"self_attention.linear_q_up_proj.weight": [
    "model.layers.{layer_number}.attention.q_b_proj.weight"
],
```

**影响**: mbridge 路径加载/保存 MLA 层权重时抛出 `NotImplementedError`。

______________________________________________________________________

### H2. expert_bias 的 HF 名称可能与 SGLang 不一致

**文件**: `areal/engine/megatron_utils/megatron.py:636-639`

```python
elif rest == "mlp.router.expert_bias":
    return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
```

`e_score_correction_bias` 是 DeepSeekV3 的命名。BailingMoe HF 中对应字段可能是
`expert_bias`（Bridge 中 `_MLP_MAPPING` line 100 使用的就是
`model.layers.{layer_number}.mlp.gate.expert_bias`）。

**影响**: NCCL 路径推送权重到 SGLang 后，SGLang 找不到对应的 bias 参数，router
没有 expert bias 修正。

**建议**: 确认 SGLang `bailing_moe_linear.py` 中实际使用的参数名，统一命名。

______________________________________________________________________

### H3. convert_bailingmoe_to_hf 中 head_dim 计算依赖 kv_channels 可能为 None

**文件**: `areal/engine/megatron_utils/megatron.py:645-647`

```python
num_heads = tf_config.num_attention_heads
hidden_size = param.shape[-1]
head_dim = param.shape[0] // (num_heads * 3)
```

Lightning 层的 QKV 转换直接从 param shape 推导 head_dim，这本身没问题。但代码中
**没有单独的 fallback 路径**来处理 MLA 层中可能需要 `kv_channels` 的场景（尽管
BailingMoe 的 MLA 层目前通过 `linear_q_down_proj` 等直接传递 param，不需要 reshape）。

结合 M3（kv_channels 可能为 None），如果后续有代码路径依赖 `tf_config.kv_channels`
做 reshape，会出现 `TypeError: unsupported operand type(s) for //: 'NoneType' and
'int'`。

______________________________________________________________________

## Medium — 功能可用但不够健壮

### M1. `_build_alibi_slopes` 函数冗余

**文件**: `areal/models/mcore/lightning_attention.py:64-90`

如果采用 C1 的修复（使用 fla 官方 g_gamma 公式），整个 `_build_alibi_slopes()` 函数
不再需要，应删除以减少维护负担。

______________________________________________________________________

### M2. 每个 Lightning 层创建独立的 RotaryEmbedding 实例

**文件**: `areal/models/mcore/lightning_attention.py:369-383`

64 层模型中约 48 个 Lightning 层各自创建一份 `RotaryEmbedding`，包含独立的 cos/sin
cache。对于长序列（如 32K tokens），每份 cache 约占 `2 * seq_len * rotary_dim *
dtype_size` 字节，48 份累计显存浪费不可忽略。

**建议**: 在 `bailing_moe.py` 的 `make_mcore_layer_specs_bailing_moe()` 中创建共享
`RotaryEmbedding` 实例，通过 `ModuleSpec.params` 传入各 Lightning 层。

______________________________________________________________________

### M3. kv_channels 未在 config 转换中显式设置

**文件**: `areal/models/mcore/bailing_moe.py:91-98`

`hf_to_mcore_base_args()` 中 `kv_channels = getattr(hf_config, "head_dim", None)`。
BailingMoe HF config 使用 `attn_head_dim`（Lightning 层）和
`qk_nope_head_dim`/`v_head_dim`（MLA 层），没有通用的 `head_dim` 字段。

**影响**: `kv_channels=None` 可能导致 MLA 层 RotaryEmbedding 初始化时 `kv_channels`
回退到 `hidden_size // num_attention_heads`，而这在 MLA 中并不总是正确的。

**建议修复**: 在 config 转换中显式设置：

```python
base_args["kv_channels"] = getattr(hf_config, "qk_rope_head_dim", 64)
```

______________________________________________________________________

### M4. 异常捕获过于宽泛

**文件**: `areal/models/mcore/lightning_attention.py:46-50`, `:55-60`, `:379`

```python
except Exception:
    pass
```

`_get_tp_world_size()`、`_get_tp_rank()` 和 RotaryEmbedding 创建中的 `except
Exception` 会吞掉 CUDA OOM、NCCL 超时等致命错误。

**建议**: 缩小捕获范围：

```python
except (RuntimeError, AttributeError):
    pass
```

______________________________________________________________________

### M5. Sequence Parallel 下的 gate 输入行为未验证

**文件**: `areal/models/mcore/lightning_attention.py:448-453`

```python
gate, _ = self.linear_gate(hidden_states)  # hidden_states 可能是 [S/TP, B, H]
gate = gate.view(seq_len, batch_size, self.num_heads_per_partition, self.attn_head_dim)
```

SP 开启时，`hidden_states` 的 seq 维是 `S/TP`。`linear_gate`（TEColumnParallelLinear）
内部会 all-gather 恢复到 `S`。但 `seq_len` 从 `qkv.shape[0]` 获取（L417），如果
`linear_qkv` 和 `linear_gate` 的 all-gather 行为一致，则 `seq_len` 匹配。

需要确认 TE 的 `ColumnParallelLinear` 在 SP 模式下是否总是 all-gather 输入序列维，
且返回 full sequence 的输出。如果某些配置下不是这样，gate 的 reshape 会崩溃。

______________________________________________________________________

## Low — 代码质量 / 可维护性

### L1. 无效的 self-assignment

**文件**: `areal/models/mcore/lightning_attention.py:434`, `:442`

```python
lightning_rotary = lightning_rotary  # no-op
rotary_pos_emb = rotary_pos_emb     # no-op
```

这两行是 `isinstance(..., tuple)` 为 True 时的 else 分支。条件为 True 意味着已经是
tuple，赋值给自身没有任何效果。

**建议**: 直接删除这两行 no-op。

______________________________________________________________________

### L2. plan doc 注释中权重名与实际映射不一致

**文件**: `docs/bailing-moe-v2.5-adaptation-plan.md:300-301`

文档中提到 Lightning 层 Q/K norm 映射为 `q_norm/k_norm`，但代码正确地映射到了
`query_layernorm/key_layernorm`（匹配 HF naming convention）。

仅注释/文档问题，不影响功能。建议统一。

______________________________________________________________________

### L3. 示例配置中硬编码个人路径

**文件**: `examples/bailing_moe_sft.yaml:13-14`, `:33`

```yaml
fileroot: /storage/openpsi/codes/chucai.dzq/tmp/bailing-moe-sft-trial0
path: /storage/openpsi/models/moe-mini-v25-...
```

示例配置不应包含个人/集群路径，应使用占位符如 `/path/to/model`。

______________________________________________________________________

### L4. convert_bailingmoe_to_hf 未过滤 \_extra_state

`bailing_moe_bridge.py:194` 有 `assert "_extra_state" not in mcore_weights_name`，但
`convert_bailingmoe_to_hf()` 函数本身没有这个过滤。遇到 FP8 tensor 的 `_extra_state`
参数时会走到末尾的 `raise ValueError`。

**建议**: 在函数开头添加过滤：

```python
if "_extra_state" in name:
    return []
```

______________________________________________________________________

## 汇总

| 等级     | 数量 | 关键问题                                                     |
| -------- | ---- | ------------------------------------------------------------ |
| Critical | 4    | g_gamma 公式错误、gate_norm 双重应用、QKV 加载格式转换缺失、q_lora_rank 默认值 |
| High     | 3    | MLA Bridge 映射不完整、expert_bias 名称待确认、head_dim 依赖 |
| Medium   | 5    | 冗余代码、显存浪费、kv_channels 缺失、异常处理、SP 验证      |
| Low      | 4    | 代码清理、注释、硬编码路径、extra_state 过滤                  |

**最优先修复**: C1 (g_gamma 公式) 和 C2 (gate_norm 双重应用) 直接影响数值正确性。
C3 (QKV 加载格式) 和 C4 (q_lora_rank) 影响权重加载。建议先修这 4 个 Critical 问题后
再跑测试。

______________________________________________________________________

## 测试验证清单

修复后应按顺序验证：

1. **模型构建**: `GPTModel(config, block_spec)` 不报错，参数数量与 HF 模型一致
2. **权重加载 (TP=1)**: 从 HF checkpoint 加载，比较关键层输出与 HF transformers
   推理结果
3. **权重加载 (TP>1)**: 验证 QKV interleaved 格式转换 + TP 切片正确性
4. **NCCL 权重推送**: `convert_bailingmoe_to_hf()` 输出的 HF 权重名与 SGLang 一致
5. **Forward pass**: 随机输入通过所有层，Lightning 层和 MLA 层输出 shape 正确
6. **SFT 训练**: 小数据集上 loss 正常下降
