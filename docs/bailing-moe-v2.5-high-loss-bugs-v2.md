# BailingMoeV2.5 高初始 Loss 根因分析 (v3)

**日期**: 2026-03-16
**现象**: 初始 loss 7.8（使用 `g_gamma`）；改为 `g` 后 loss 升至 14.42
**期望**: 初始 loss ≤ 2

**基准对比**:

- `antllm/models/bailing_moe/modeling_bailing_moe_v2_5.py`
  (`BailingMoeV2_5LinearAttention`, line 705+)
- `antllm/models/bailing_moe/configuration_bailing_moe_v2_5.py`
  (`BailingMoeV2_5Config`)

______________________________________________________________________

## 上一次 review 的错误纠正

### 撤回: "slope 应该用 `g` 而非 `g_gamma`"

**之前的结论是错误的。** `g` 和 `g_gamma` 对于常数 decay 在数学上完全等价：

- `g_gamma` 路径：kernel 内部计算 `b_g = gamma * (arange(BT) + 1)` =
  `[γ, 2γ, 3γ, ..., BT·γ]`
- `g` 路径：`chunk_local_cumsum(constant)` 产生 `[c, 2c, 3c, ..., BT·c]`

两者数值相同。**但改为 `g` 后 loss 反而升高到 14.42**，原因是 `g` 路径的 backward
会对 `g` tensor 计算梯度（`chunk_local_cumsum(dg, reverse=True)`,
`fla/ops/simple_gla/chunk.py:182`），对一个本应是常数的 buffer 产生非零梯度噪声，
干扰训练。

**结论：应该改回 `g_gamma`，这是正确且高效的传参方式。**
Megatron-LM 参考实现也使用 `g_gamma=self.slope` (`attention.py:2188`)。

______________________________________________________________________

## Bug 1 [Critical] — `layer_idx` 偏移量错误，所有层的 slope 都不对

**这是最可能导致高初始 loss 的根因。**

### HF 参考实现 (`modeling_bailing_moe_v2_5.py:754-756`)

```python
slope = -build_slope_tensor(self.num_heads) * (
    1 - (self.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
)
```

HF 的 `layer_idx` 是 **0-indexed**（从 `enumerate` 得到）。公式中用
`(self.layer_idx - 1)`，所以：

| HF layer_idx | 公式中的值 | layer_scale       |
| ------------ | ---------- | ----------------- |
| 0            | -1         | **1.0526** (N=20) |
| 1            | 0          | **1.0000**        |
| 19           | 18         | **0.0527**        |

### AReaL 当前实现 (`lightning_attention.py:125,141`)

```python
self.layer_idx = layer_number - 1   # megatron 1-indexed -> 0-indexed
layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5
```

| AReaL layer_idx | layer_scale |
| --------------- | ----------- |
| 0               | **1.0000**  |
| 1               | **0.9474**  |
| 19              | **0.0000**  |

### 差异分析

| 层   | HF layer_scale | AReaL layer_scale | 相对误差              |
| ---- | -------------- | ----------------- | --------------------- |
| 0    | 1.0526         | 1.0000            | **5.3%**              |
| 1    | 1.0000         | 0.9474            | **5.3%**              |
| 18   | 0.1053         | 0.0527            | **50%**               |
| 19   | 0.0527         | 0.0000            | **∞ (decay 为零！)**  |

**最后一层的 decay 在 AReaL 中为零**，意味着 Lightning Attention 完全没有衰减，
变成了无 decay 的线性 attention。这会导致该层的注意力输出完全错误。

对于更大的模型（如 64 层），最后几层的 slope 偏差同样极大。

### 修复

```python
# 匹配 HF: slope = -slopes * (1 - (layer_idx - 1) / (N - 1) + 1e-5)
# self.layer_idx 已经是 0-indexed
layer_scale = 1.0 - (self.layer_idx - 1) / max(self.num_layers - 1, 1) + 1e-5
```

______________________________________________________________________

## Bug 2 [Critical] — 改 `g` 传参导致 loss 从 7.8 升至 14.42

### 当前代码（已改为 `g`，需要回退）

`lightning_attention.py:174-185`:

```python
g = self.g_gamma[None, None, :].expand(q.shape[0], q.shape[1], -1)  # [B, T, H]
output, _ = chunk_simple_gla(q=q, k=k, v=v, g=g, scale=self.scale)
```

### 问题

1. `g` 路径在 backward 中运行 `chunk_local_cumsum(dg, reverse=True)`
   (`fla/ops/simple_gla/chunk.py:182`)，对 `expand` 后的 tensor 计算梯度。
   虽然 `g_gamma` buffer 本身 `requires_grad=False`，但 backward 产生的
   梯度噪声会通过 autograd 图影响上游参数。
2. `g` expand 为 `[B, T, H]` 需要额外显存和 `chunk_local_cumsum` kernel 开销。

### 修复

改回 `g_gamma`：

```python
output, _ = chunk_simple_gla(
    q=q, k=k, v=v,
    g_gamma=self.g_gamma,  # [H], 高效且正确
    scale=self.scale,
)
```

______________________________________________________________________

## Bug 3 [Medium] — `ffn_hidden_size` 可能配错

### HF config

```python
intermediate_size = 5120           # Dense MLP FFN hidden size
moe_intermediate_size = 512        # MoE expert FFN hidden size
```

### AReaL `common.py:33`

```python
"ffn_hidden_size": hf_config.intermediate_size,  # = 5120
```

`ffn_hidden_size=5120` 被 `base_args` 设置。`bailing_moe.py:122` 另外设置了
`moe_ffn_hidden_size=512`。如果 megatron-core 的 MoE expert MLP 初始化时使用的是
全局 `ffn_hidden_size`（5120）而非 `moe_ffn_hidden_size`（512），expert 权重的
shape 会完全错误，导致权重加载失败或随机初始化。

**需要确认**：在 megatron-core 的 `SequentialMLP` / `GroupedMLP` 初始化中，
是否优先使用 `moe_ffn_hidden_size`。

______________________________________________________________________

## Bug 4 [Medium] — `num_query_groups` 对 Lightning 层权重加载的潜在影响

### HF config

```python
num_attention_heads = 16
num_key_value_heads = 4            # MLA 层的 GQA
# Lightning 层: num_key_value_heads = num_attention_heads = 16 (无 GQA)
```

### AReaL `common.py:32`

```python
"num_query_groups": hf_config.num_key_value_heads,  # = 4
```

全局 `num_query_groups=4` 对于 MLA 层正确，但 Lightning 层没有 GQA。Lightning 层
使用自定义 `LightningSelfAttention`，不走 megatron-core 的 QKV split 逻辑，所以
**forward 不受影响**。但某些权重加载/保存路径（如 `_weight_to_mcore_tp`）可能用
`num_query_groups` 做 TP 切片计算，导致 Lightning 层权重切片错误。

**需要确认**: TP>1 时 `hf_load.py` 中 `_convert_fused_qkv_weight` 是否依赖
`num_query_groups`（当前代码中不依赖，用的是 `num_attention_heads`，应该没问题）。

______________________________________________________________________

## 已确认正确的部分

| 组件                                       | 状态     | 验证细节                                                                                                                                                  |
| ------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QKV 格式转换 (`_convert_fused_qkv_weight`) | ✓ 正确   | `[3,H,D]→[H,3,D]` 与 save 路径互逆；Lightning 无 GQA，per-head = per-group                                                                              |
| gate_norm 逻辑                             | ✓ 正确   | 只 norm attn_output，gate 只过 sigmoid，与 HF L878-881 一致                                                                                               |
| GroupRMSNorm 实现                          | ✓ 正确   | HF: `(group_norm_size, H*D/group_norm_size)` mean(-1)；AReaL: `(num_groups, group_size, head_dim)` mean(-2,-1)。当 `group_norm_size=4, H=16, D=128` 时等价 |
| RoPE 实现                                  | ✓ 正确   | `multi_latent_attention=False` 非 interleaved RoPE，与 HF Lightning 层一致。`rotary_dim = 128*0.5 = 64`                                                   |
| `_build_alibi_slopes`                      | ✓ 正确   | 与 HF `build_slope_tensor` 完全相同                                                                                                                       |
| Bridge MLA q_lora 映射                     | ✓ 正确   | 动态选择 Q_DIRECT 或 Q_LORA                                                                                                                              |
| `_extra_state` 过滤                        | ✓ 正确   | `convert_bailingmoe_to_hf` 开头已处理                                                                                                                     |
| `g` vs `g_gamma` 数学等价性               | ✓ 已验证 | 常数 decay 下两者在 Triton kernel 中产生相同的 `b_g` 值，但 `g_gamma` 路径无反向梯度，更稳定                                                              |

______________________________________________________________________

## 修复优先级

| #   | 等级         | 问题                                            | 预期效果                                           |
| --- | ------------ | ----------------------------------------------- | -------------------------------------------------- |
| 1   | **Critical** | 改回 `g_gamma`（撤回 `g` 改动）                | 恢复 7.8 的 loss 基线（消除 14.42 的退化）         |
| 2   | **Critical** | `layer_idx` 偏移：改为 `(self.layer_idx - 1)`   | 修复所有层 slope，**预期 loss 从 7.8 降到 ≤ 2**    |
| 3   | Medium       | 确认 `ffn_hidden_size` vs `moe_ffn_hidden_size` | 排除 expert MLP shape 错误                         |
| 4   | Medium       | 确认 `num_query_groups=4` 对 Lightning 层无影响  | 预防性检查                                         |

______________________________________________________________________

## 建议操作顺序

```
1. 回退 g -> g_gamma 改动                        ← 恢复到 loss 7.8 基线
2. 修复 layer_idx: 改为 (self.layer_idx - 1)      ← 预期 loss 降到 ≤ 2
3. 跑 10 步验证 loss
4. 如果仍然高，排查 Bug 3 (ffn_hidden_size) 和 Bug 4 (num_query_groups)
```

______________________________________________________________________

## 验证方法

修复后，用 TP=1 PP=1 的配置训练 10 步，观察：

- 初始 loss 是否降到 ≤ 2
- loss 是否随 step 正常下降

如果初始 loss 仍然高于 2，下一步排查：

1. 打印加载后的权重统计（mean/std）与 HF 对比
2. 单层 forward 对比：用相同输入，比较 HF 和 AReaL 单层输出是否一致
3. 检查 embedding layer 和 output_layer 权重是否正确映射
