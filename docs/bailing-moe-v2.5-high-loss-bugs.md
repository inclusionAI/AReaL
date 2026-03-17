# BailingMoeV2.5 高初始 Loss 问题排查

**日期**: 2026-03-16
**现象**: 初始 loss 7.8，正常应 ≤ 2
**根因**: 发现 2 个 Critical bug + 1 个 High bug，均导致前向计算与 HF 参考实现不一致

**对比基准**:
`/storage/openpsi/codes/chucai.dzq/gh/antllm/antllm/models/bailing_moe/modeling_bailing_moe_v2_5.py`
(`BailingMoeV2_5LinearAttention`, line 705+)

______________________________________________________________________

## Bug 1 [Critical] — slope 传参位置错误：用了 `g_gamma` 而非 `g`

**这个 bug 最可能是高 loss 的主因。**

### HF 参考实现 (modeling_bailing_moe_v2_5.py:870-876)

```python
o, recurrent_state = self.lightning_attn_ops[mode](
    q=query_states,
    k=key_states,
    v=value_states,
    g=self.slope[None, None, :].expand(bsz, q_len, self.num_heads),  # ← 用的是 g
    initial_state=recurrent_state,
    output_final_state=use_cache,
)
```

slope 传给了 **`g` 参数**（shape `[B, T, H]`，per-token 的 forget gate，在 log
空间中）。

### AReaL 当前实现 (lightning_attention.py:174-179)

```python
output, _ = chunk_simple_gla(
    q=q,
    k=k,
    v=v,
    g_gamma=self.g_gamma,  # ← 用的是 g_gamma
    scale=self.scale,
)
```

slope 传给了 **`g_gamma` 参数**（shape `[H]`，per-head 的 data-independent log decay）。

### `g` 和 `g_gamma` 的区别

查看 `chunk_simple_gla` 签名 (`fla/ops/simple_gla/chunk.py:189-215`):

```python
def chunk_simple_gla(
    q, k, v,
    g: torch.Tensor | None = None,       # Forget gates [B, T, H] — per-token
    g_gamma: torch.Tensor | None = None,  # Log decay [H] — per-head, data-independent
    ...
)
```

文档说明:

> `g`: Forget gates of shape `[B, T, H]`. Compared to GLA, the gating is **head-wise
> instead of elementwise**.
>
> `g_gamma`: Log decay of shape `[H]`. **Head-wise data-independent decay** is used if
> `g_gamma` is provided. **Only one of `g` or `g_gamma` should be provided.**

两者在 Triton kernel 中的行为完全不同:

- **`g` (per-token)**: 每个 token 位置有独立的 gate 值。在 chunk 内部，每个 token
  的 decay 由 `exp(cumsum(g))` 决定。虽然 slope 本身是常数，但 expand 为
  `[B, T, H]` 后以 per-token 方式参与 cumsum。
- **`g_gamma` (per-head)**: 一个标量 decay rate 应用于整个 chunk。在 Triton kernel
  中直接用 `g_gamma * (arange(BT) + 1)` 生成 cumulative decay。

**数学上它们不等价。** 使用 `g_gamma` 时 decay 从 chunk 边界重新开始计算，而 `g`
是连续累积的。虽然两者都能表达常数 decay，**但 kernel 内部的实现路径完全不同**,
导致数值结果不一致。

### 修复

```python
# LightningCoreAttention.forward 中:
# 将 slope 作为 g 传入 (expand 到 [B, T, H])
g = self.g_gamma[None, None, :].expand(q.shape[0], q.shape[1], -1)  # [B, T, H]
output, _ = chunk_simple_gla(
    q=q, k=k, v=v,
    g=g,          # ← 改为 g
    scale=self.scale,
)
```

______________________________________________________________________

## Bug 2 [Critical] — QKV split 格式与 HF 不一致

### HF 参考实现 (modeling_bailing_moe_v2_5.py:835-839)

```python
qkv = self.query_key_value(hidden_states)
# [B, T, (N+2K)*D] -> [B, T, N+2K, D]
qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
# split on head dim: [N, K, K]
query_states, key_states, value_states = qkv.split(
    [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
)
```

HF 权重是 **concatenated `[Q_all, K_all, V_all]`** 格式。QKV split 方式是
`view` + `split` on head dimension (dim=-2)，分为 `[N, K, K]` 个 head。

### Megatron-LM 参考实现 (attention.py:2696-2723)

```python
# [sq, b, hp] -> [sq, b, ng, (np/ng + 2) * hn]
mixed_qkv = mixed_qkv.view(*new_tensor_shape)
(query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
```

Megatron 权重是 **per-group interleaved** 格式:
`[grp0_Q | grp0_K | grp0_V | grp1_Q | grp1_K | grp1_V | ...]`

### AReaL 当前实现 (lightning_attention.py:419-424)

```python
# Split into Q, K, V from interleaved layout [H_local, 3, D]
qkv = qkv.view(
    seq_len, batch_size, self.num_heads_per_partition, 3, self.attn_head_dim
)
query = qkv[:, :, :, 0, :].contiguous()  # [S, B, H_local, D]
key = qkv[:, :, :, 1, :].contiguous()
value = qkv[:, :, :, 2, :].contiguous()
```

AReaL 假设权重是 **per-head interleaved** `[H, 3, D]` 格式，即
`[q0,k0,v0, q1,k1,v1, ...]`。

### 三种格式对比

| 格式 | 内存布局 (dim=0) | 使用方 |
|------|-------------------|--------|
| HF concatenated | `[Q_all \| K_all \| V_all]` | HF modeling code |
| Per-group interleaved | `[grp0_Q, grp0_K, grp0_V, grp1_Q, ...]` | Megatron-LM LinearAttention |
| Per-head interleaved | `[q0,k0,v0, q1,k1,v1, ...]` | AReaL 当前实现 |

**对于 BailingMoe 的 Lightning Attention 层**，`num_key_value_heads == num_attention_heads`
（没有 GQA），每个 group 只有 1 个 head。因此 **per-group interleaved 和 per-head
interleaved 是等价的**。

所以 `_convert_fused_qkv_weight` 中的 `[3,H,D] → [H,3,D]` 转换是正确的，
AReaL 的 `view(H, 3, D)` split 也与转换后的格式一致。

**结论: Bug 2 实际上不存在。** 因为 Lightning 层没有 GQA (`num_kv_heads == num_heads`)，
per-group 和 per-head interleaved 等价。QKV 格式转换和 split 是正确的。

______________________________________________________________________

## Bug 3 [High] — GroupRMSNorm 的输入 shape 不一致

### HF 参考实现 (modeling_bailing_moe_v2_5.py:878-879)

```python
o = o.reshape(bsz, q_len, -1)  # [B, T, H*D] — flatten heads
o = self.g_norm(o)               # GroupRMSNorm on flattened [B, T, H*D]
```

g_norm 接收的是 **[B, T, H*D]** 三维 tensor，在 H*D 维度上做 group RMSNorm。

### AReaL 当前实现 (lightning_attention.py:457-458)

```python
# attn_output shape: [S, B, num_heads_local, head_dim]
attn_output = self.gate_norm(attn_output)  # GroupRMSNorm on [S, B, H, D]
```

gate_norm 接收的是 **[S, B, H, D]** 四维 tensor。

### GroupRMSNorm 内部处理 (lightning_attention.py:220-228)

```python
x = x.view(*original_shape[:-2], self.num_groups, self.group_size, self.head_dim)
rms = x.float().pow(2).mean(dim=(-2, -1), keepdim=True).add(self.eps).rsqrt()
x = (x.float() * rms).to(x.dtype)
x = x.view(*original_shape)
weight = self.weight.view(self.num_heads, self.head_dim)
return x * weight
```

当输入是 `[S, B, H, D]` 时：
- `view` 变为 `[S, B, num_groups, group_size, head_dim]`
- `mean(dim=(-2, -1))` 在 `(group_size, head_dim)` 上做 RMS
- `weight.view(H, D)` 按 `[H, D]` broadcast

当输入是 `[B, T, H*D]` 时（HF 做法）：
- `view` 变为 `[B, T, num_groups, group_size, head_dim]`
- 同样在 `(group_size, head_dim)` 上做 RMS

**数学上**，只要 `original_shape[-2:]` 能正确 reshape 为 `(num_groups, group_size, head_dim)`，
两种 input shape 的 norm 结果是相同的。AReaL 的输入 `[S, B, H, D]` 在 `[-2:]` 维度上
是 `(H, D)`，而 HF 的 `[B, T, H*D]` 在 `[-1]` 维度上是 `(H*D)`。

AReaL 的 `view(*original_shape[:-2], num_groups, group_size, head_dim)` 展开后是
`[S, B, num_groups, group_size, head_dim]`——与 HF 展开后一样。

最后的 weight 乘法也一致（`weight.view(H, D)` broadcast 到 `[..., H, D]`）。

**结论: Bug 3 在 `linear_attn_norm_group_size` 整除 `num_heads` 的情况下不存在。**
两种 input shape 数学等价。但有一个**边界情况**：如果 `original_shape[-2:] == (H, D)`
而 `view` 的 `[-2:]` 实际上不是 `(H, D)`，会出错。当前代码应该没问题。

______________________________________________________________________

## 其他可能影响 loss 的问题

### O1 [Medium] — layer_idx 起点差异

HF 参考实现 (modeling_bailing_moe_v2_5.py:754):

```python
slope = -build_slope_tensor(self.num_heads) * (
    1 - (self.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
)
```

注意这里用的是 `self.layer_idx - 1`。HF 的 `layer_idx` 从 0 开始
（`BailingMoeV2_5DecoderLayer.__init__` 中传入 `layer_idx` 参数，
对应 `enumerate(range(config.num_hidden_layers))` 的 0-indexed 值）。

所以 HF 的第 0 层: `layer_scale = 1 - (-1)/(N-1) + 1e-5 = 1 + 1/(N-1) + 1e-5`
（大于 1！）

AReaL 实现 (lightning_attention.py:139-141):

```python
# megatron layer_number is 1-indexed; convert to 0-indexed
self.layer_idx = layer_number - 1
layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5
```

AReaL 的第 0 层: `layer_scale = 1 - 0/(N-1) + 1e-5 = 1 + 1e-5`

**差异**: HF 第 0 层的 `layer_scale ≈ 1 + 1/(N-1)`，AReaL 的 `layer_scale ≈ 1`。
对于 64 层模型，差异约 0.016。**整个 slope 数组都会偏移。**

**修复**: 保持与 HF 完全一致，用 `self.layer_idx - 1` (即原始的 0-indexed layer_idx
减 1)：

```python
self.layer_idx = layer_number - 1  # 0-indexed
# HF uses (layer_idx - 1), so: (0-indexed - 1) = -1 for first layer
layer_scale = 1.0 - (self.layer_idx - 1) / max(self.num_layers - 1, 1) + 1e-5
```

### O2 [Medium] — HF g_norm 权重 shape 可能与 AReaL 不匹配

HF 的 `BailingMoeV2_5GroupRMSNorm` (需要确认) 的权重 shape 可能是 `[H*D]`
（全局 head 数），而 AReaL 的 `GroupRMSNorm.weight` 是
`[num_heads_per_partition * head_dim]`（TP-local）。

如果 mbridge 加载时 shape 对不上会报错或静默错误。但这个问题在 TP=1 时不会暴露。

______________________________________________________________________

## 优先修复顺序

```
1. [Critical] Bug 1: slope 从 g_gamma 改为 g        ← 最可能修复高 loss
2. [Medium]  O1: layer_idx 偏移量对齐 HF
3. [High]    确认 g_norm weight shape 在 TP>1 时正确
```

## 验证方法

修复 Bug 1 后，用 TP=1 的配置训练 10 步，观察：
- 初始 loss 是否降到 ≤ 2
- Loss 是否正常下降

如果初始 loss 仍然高，继续排查 O1 (layer_idx) 和权重加载路径。
