# BailingMoeV2.5 CP 设计方案 Review

**日期**: 2026-03-16
**Review 对象**: `docs/bailing-moe-v2.5-cp-support-plan.md`

**交叉对比的三个代码库**:

| 代码库 | 路径 | 关键文件 |
|--------|------|----------|
| Megatron-LM (LinearAttention) | `/storage/openpsi/codes/chucai.dzq/gh/Megatron-LM/` | `megatron/core/transformer/attention.py:1685-2589` |
| HybridEngine | `/storage/openpsi/codes/chucai.dzq/Asystem-HybridEngine/` | `asystem_runtime/rl_function/actor_function.py` |
| AReaL 现有 CP | 本仓库 | `areal/engine/megatron_utils/packed_context_parallel.py`, `areal/api/alloc_mode.py` |

______________________________________________________________________

## 总体评估

**方案方向正确，细节需要修正。**

All-to-All CP (策略 A) 的选择是正确的。Megatron-LM `LinearAttention` 的
`lightning_attn_all_to_all_cp=True` 路径 (`attention.py:2198-2258`) 使用的就是
这个方案。

但 **策略 B (State-Passing) 并非 "暂不实现" 的简单方案**——它是参考实现中的另一条
**生产级路径** (`attention.py:2260-2467`)，包含自定义 Triton kernel
`inter_attention_triton`（436 行），且已在实际训练中使用。两者性能/精度 tradeoff
可后续评估，当前选 A 做第一版合理。

______________________________________________________________________

## 方案中正确的部分

| # | 内容 | 验证依据 |
|---|------|----------|
| ✓1 | All-to-All CP 策略选择 | 数学上正确（线性注意力 head 独立），参考 `attention.py:2198-2258` 确认 |
| ✓2 | `H/TP % CP == 0` 约束 | 参考实现 `attention.py:2230` 也做了同样的 `divide` 校验 |
| ✓3 | RoPE 在 all-to-all 之前应用 | 参考实现流程：QKV projection → RoPE → all-to-all → kernel，与文档一致 |
| ✓4 | gate 维度不受 CP 影响 | gate 在 `hidden_states`(`S/CP` 维度) 上操作，all-to-all 仅在 attention 内部 |
| ✓5 | g_gamma CP slice 逻辑 | 参考 `attention.py:2229-2231`: `cp_slope = self.slope[cp_rank * H_per_cp : (cp_rank+1) * H_per_cp]` |
| ✓6 | SP+CP 维度分析 | TEColumnParallelLinear 只 gather TP 的 SP 切分，CP 切分保留 |

______________________________________________________________________

## 需要修正的问题

### P1 [Critical] — g_gamma 公式需重新确认

文档中 g_gamma CP slice 逻辑本身是对的，**但前提是 g_gamma 的值本身正确**。

Megatron-LM 参考实现 (`attention.py:1744`) 使用的是 **ALiBi-style geometric slopes**：

```python
# Megatron-LM LinearAttention.__init__, line 1744:
slope = -self._build_slope_tensor(self.config.num_attention_heads) \
        * (1 - (self.global_layer_number - 1) / (self.config.num_layers - 1) + 1e-5)
```

其中 `_build_slope_tensor` (`attention.py:1843-1860`) 生成的是 ALiBi 论文中 2 的
负幂几何级数，与当前 AReaL `lightning_attention.py` 中的 `_build_alibi_slopes`
一致。

**这意味着前一次 code review (C1) 中的结论需要修正**：如果目标是与 Megatron-LM
训练一致（而非 fla 默认公式），**ALiBi slopes 是正确的**。

但 fla 库的 `chunk_lightning_attn` 内部**硬编码了不同的公式**
(`g_gamma = -(8/H) * (1 - layer_idx/num_layers) * arange(H)`)。如果 AReaL 调用的是
`chunk_lightning_attn` 而非 `chunk_simple_gla(g_gamma=...)`，会导致 g_gamma 被覆盖。

**行动**: 确认 AReaL 调用的是 `chunk_simple_gla` + 手动传入 ALiBi g_gamma。
如果是，则当前代码正确。

______________________________________________________________________

### P2 [Critical] — undo/redo_attention_load_balancing 签名错误 + 实现缺失

文档中的签名是：

```python
def _undo_attention_load_balancing(input_: torch.Tensor, cp_size: int, cp_rank: int)
```

但参考实现 (`attention.py:2214`) 的调用是：

```python
x = undo_attention_load_balancing(x, cp_size, cu_seqlens_q)
```

第三个参数是 **`cu_seqlens_q`**（packed sequence 的累积长度），**不是 `cp_rank`**。

参考 Megatron-LM `utils.py:1967-2025` 的实际签名：

```python
def undo_attention_load_balancing(
    output: Tensor,
    cp_size: int,
    cu_seqlens_q: Optional[Tensor] = None,  # packed sequences 需要 per-sequence 重排
) -> Tensor:
```

AReaL 使用 packed sequences (THD format)，zigzag 的 undo 需要 **per-sequence**
处理，不能简单地对整个 tensor 做 chunk 重排。

**行动**: 按参考实现修正签名为 `(output, cp_size, cu_seqlens_q)`，实现 THD 格式的
per-sequence zigzag 重排。注意参考实现区分了 THD（用 `thd_get_partitioned_indices`
+ `index_select`）和非 THD（直接 chunk 重排）两种路径。

______________________________________________________________________

### P3 [Critical] — all-to-all 缺少 autograd 包装

文档在"开放问题 3"中提到了这个问题，但正文的实现代码直接调用了裸的
`torch.distributed.all_to_all_single`。**反向传播会断掉。**

参考 Megatron-LM `mappings.py:632-713` 中 `all_to_all_cp2hp` / `all_to_all_hp2cp`
是通过 `torch.autograd.Function` 包装的（forward 做 cp2hp，backward 做 hp2cp）。

AReaL 的 FSDP engine 中已有 `all_to_all_single_autograd` 实现
(`areal/models/fsdp/ulysses.py`)，可以直接复用。

**行动**: 用 `torch.autograd.Function` 包装 `_all_to_all_cp2hp` /
`_all_to_all_hp2cp`，使得 forward 和 backward 互为逆操作。或直接复用
`ulysses.py:all_to_all_single_autograd`。

______________________________________________________________________

### P4 [High] — 与 AReaL 现有 packed_context_parallel 的交互未分析

文档中完全没有分析 Lightning Attention 的 All-to-All CP 与 AReaL 现有
`packed_context_parallel_forward` 的交互。

**分析结论：交互是安全的。** 原因：

1. AReaL 在 **input level** 做 zigzag split (`preprocess_packed_seqs_context_parallel`)，
   所有层收到的 `hidden_states` 都是 `S/CP` 维度
2. Lightning Attention 层的 all-to-all CP 在**层内部**完成序列↔head 交换，
   输出维度仍然是 `S/CP`（因为 `hp2cp` 恢复了 CP-split 维度）
3. MLA 层由 megatron-core/TE 内部的 ring-attention 处理，也在层内部完成 CP 通信
4. `postprocess_packed_seqs_context_parallel` 在 pipeline 最后一个 stage 做
   all-gather + 反向 zigzag，与层内 CP 通信无关

**行动**: 在文档中补充此分析，明确层间维度一致性。

______________________________________________________________________

### P5 [High] — gate_norm 逻辑与参考实现不一致（非 CP 问题但阻塞验证）

参考 Megatron-LM `attention.py:2476-2492` 的 gate_norm 流程：

```python
def gate_norm(hidden_states, core_attn_out):
    gate = self.get_gate_tensors(hidden_states)       # gate projection
    core_attn_out = self.pre_gate_norm(core_attn_out)  # norm 只作用于 attn output
    gate = F.sigmoid(gate)                              # gate 只过 sigmoid
    core_attn_out = core_attn_out * gate               # 逐元素相乘
    return core_attn_out
```

即：**`pre_gate_norm` 只应用于 `core_attn_out`，gate 只过 sigmoid 不过 norm**。
当前 AReaL 实现中 `gate_norm` 同时作用于 gate 和 attn_output 是错误的。

**此问题与 CP 无关，但在 CP 实现之前必须先修正**，否则 CP 验证时的基线 (CP=1)
结果就是错的，无法判断 CP 引入的差异。

**行动**: 在 `lightning_attention.py` 中修正 gate_norm 逻辑，使其只 norm
attn_output。

______________________________________________________________________

### P6 [Medium] — _all_to_all_cp2hp 伪代码与参考实现不一致

文档中写的是 **4D tensor** `[S/CP, B, H, D]` 直接做 all-to-all：

```python
input_ = input_.view(seq_len, batch, cp_size, num_heads // cp_size, head_dim)
input_ = input_.permute(2, 0, 1, 3, 4).contiguous()
```

参考实现 (`attention.py:2204-2221`) 的流程不同——先 **flatten heads+dim** 为一个维度
`[S/CP, B, H_local*D]`，然后调用 `all_to_all_cp2hp`（在 `mappings.py` 中操作
**3D tensor** `[S/CP, B, H_total]`），再 reshape 回 `[S, B, H_local/CP, D]`：

```python
# 参考实现流程:
x = x.reshape(*x.shape[:-2], head_num * d)       # [S/CP, B, H*D]
x = all_to_all_cp2hp(x, self.cp_group)            # [S, B, H*D/CP]
x = undo_attention_load_balancing(x, cp_size, cu_seqlens_q)
x = x.reshape(*x.shape[:-1], head_num_per_cp, d) # [S, B, H/CP, D]
x = x.transpose(0, 1)                             # [B, S, H/CP, D]
```

两种都可以实现，但建议对齐参考实现的 3D 方案以减少出错概率。

**行动**: 修改伪代码为先 flatten 再 all-to-all 再 reshape 的 3D 方案。

______________________________________________________________________

### P7 [Medium] — packed sequences 的 cu_seqlens 传递路径未说明

参考实现 `attention.py:2242` 在 all-to-all 后调用 `chunk_simple_gla` 时传入了
`cu_seqlens=cu_seqlens_q`。all-to-all 后序列恢复为全长 `S`，此时**原始 cu_seqlens
仍然适用**（因为 all-to-all 恢复的就是原始全序列顺序，经过 undo zigzag 后）。

AReaL 中 `packed_context_parallel.py` 的原始 `cu_seqlens` 传入了
`PackedSeqParams`，但文档未说明如何在 `LightningSelfAttention.forward` 中获取
这个值。

**行动**: 补充从 `packed_seq_params.cu_seqlens_q` 提取原始 cu_seqlens 并传递给
attention kernel 的方案。

______________________________________________________________________

## 开放问题的答案

### 问题 1：AReaL 是否使用 zigzag 切分？

**是的。** `packed_context_parallel.py:preprocess_packed_seqs_context_parallel` 使用
标准的 2-chunk zigzag 策略（与 TE issue #1368 描述一致）：每个 CP rank 取前半段的第
`cp_rank` 个 chunk 和后半段的倒数第 `cp_rank+1` 个 chunk。

```python
# packed_context_parallel.py line 46-55:
splitted[start_idx : start_idx + half_seqlen] = d[
    half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
]
remain_start = input_lens[i] - half_seqlen * (cp_rank + 1)
remain_end = input_lens[i] - half_seqlen * cp_rank
splitted[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
    remain_start:remain_end
]
```

因此 **必须实现** `undo/redo_attention_load_balancing`。

### 问题 2：AReaL SFT 是否使用 packed sequences？

**是的。** Megatron engine 的 forward 路径始终通过
`packed_context_parallel_forward`，该函数检查 `cu_seqlens` 并构建
`PackedSeqParams(qkv_format="thd")`。SFT 数据通过 `pad_mb_list` 对齐后以 packed
格式传入。

因此 `undo/redo_attention_load_balancing` 需要处理 **THD format** 的 per-sequence
zigzag 重排。

### 问题 3：Autograd 兼容性

**必须包装。** 直接调用 `all_to_all_single` 不会传播梯度。可复用 AReaL 已有的
`all_to_all_single_autograd` (`areal/models/fsdp/ulysses.py`)，或仿照
Megatron-LM `mappings.py` 的 `torch.autograd.Function` 包装。

______________________________________________________________________

## 修正优先级汇总

| # | 等级 | 问题 | 行动 |
|---|------|------|------|
| P1 | Critical | g_gamma 公式需确认 | 确认调用的是 `chunk_simple_gla` + 手动 g_gamma（ALiBi slopes 可能正确） |
| P2 | Critical | undo/redo 签名错误 + 实现缺失 | 修正签名为 `(output, cp_size, cu_seqlens_q)`，实现 THD zigzag 重排 |
| P3 | Critical | all-to-all 缺少 autograd 包装 | 用 `autograd.Function` 包装或复用 `ulysses.py` |
| P4 | High | 与 packed_context_parallel 交互未分析 | 补充分析（已确认安全） |
| P5 | High | gate_norm 逻辑错误（阻塞基线） | 修正为只 norm attn_output |
| P6 | Medium | cp2hp 伪代码 reshape 不一致 | 对齐参考实现的 3D flatten 方案 |
| P7 | Medium | cu_seqlens 传递路径未说明 | 补充从 packed_seq_params 提取的方案 |

______________________________________________________________________

## 建议实施顺序

```
1. 修正 gate_norm (P5, 非 CP 但阻塞基线) ──────────────────────┐
2. 确认 g_gamma 公式 (P1, 决定用 chunk_simple_gla               │
   还是 chunk_lightning_attn)                                   │
                                                                ↓
3. 实现 autograd-wrapped all-to-all (P3) ──→ 4. 实现 undo/redo (P2, THD 格式)
                                                                │
                                                                ↓
5. 集成到 LightningSelfAttention.forward (含 cu_seqlens 传递, P7)
                                                                │
                                                                ↓
6. 修改 bailing_moe.py 校验 ──→ 7. 验证 CP=2+TP=2 训练
```
