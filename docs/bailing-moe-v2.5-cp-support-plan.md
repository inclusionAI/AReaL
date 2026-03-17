# BailingMoeV2.5 Lightning Attention CP (Context Parallelism) 支持设计

**日期**: 2026-03-16
**分支**: `chucai.dzq/bailing-sft-v1.0.1`
**参考实现**: `/storage/openpsi/codes/chucai.dzq/gh/Megatron-LM/megatron/core/transformer/attention.py` (`LinearAttention` 类, line 1685+)

______________________________________________________________________

## 背景

BailingMoeV2.5 的 Lightning Attention 目前不支持 Context Parallelism (CP)。在
`bailing_moe.py` 中遇到 CP>1 时直接 raise ValueError。

要支持异构并行策略 `megatron:(attn:d2p1t4c2|ffn:d2p1e8)` 中的 `c2`，需要为
Lightning Attention 添加 CP 支持。

**目标**: 支持 All-to-All CP 策略，使 Lightning Attention 层可以在 CP>1 下正常训练。

______________________________________________________________________

## CP 策略分析

### 策略 A: All-to-All CP (Head Parallelism) — 推荐

**原理**: 通过 all-to-all 通信将 "序列切分+全部 head" 转换为 "全序列+部分 head"。
每个 CP rank 处理完整序列但只负责 `H/CP` 个 head。线性注意力是 head 独立的，
所以这在数学上是精确的。

**流程** (参考 `attention.py:2198-2258`):

```
输入: QKV [S/CP, B, H/TP, D]
  ↓ flatten heads → all_to_all_cp2hp
中间: QKV [S, B, H/TP/CP, D]
  ↓ undo_attention_load_balancing (zigzag → sequential)
  ↓ g_gamma slice: slope[cp_rank * H_per_cp : (cp_rank+1) * H_per_cp]
  ↓ chunk_simple_gla(q, k, v, g_gamma=cp_slope)
  ↓ redo_attention_load_balancing (sequential → zigzag)
输出: [S, B, H/TP/CP, D]
  ↓ all_to_all_hp2cp
最终: [S/CP, B, H/TP, D]
```

**优点**:
- 实现简单，fla kernel 不需要改动
- 数学上精确（线性注意力 head 独立）
- 通信量 = 序列 × head_dim，与注意力计算量比很小

**约束**:
- `num_heads_per_tp_partition % cp_size == 0`（H/TP 需被 CP 整除）

### 策略 B: State-Passing CP — 暂不实现

**原理**: 利用线性注意力的递推特性，每个 CP rank 计算局部 intra-chunk attention，
然后 all-gather 递推状态做 inter-chunk 修正。

**复杂度**: 需要自定义 Triton kernel (`inter_attention_triton`)，处理 packed sequences
的 cu_seqlens，代码量大 (~300 行)。

**暂缓**: 策略 A 已足够，且实现更可靠。

______________________________________________________________________

## 修改文件清单

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `areal/models/mcore/lightning_attention.py` | 大 | 添加 all-to-all CP 通信原语 + forward 逻辑 |
| `areal/models/mcore/bailing_moe.py` | 小 | 移除 CP>1 的 ValueError，添加 head 整除校验 |

______________________________________________________________________

## 实现细节

### 1. 添加 CP 通信原语

**文件**: `areal/models/mcore/lightning_attention.py`

AReaL 的 megatron-core 没有 `all_to_all_cp2hp` / `all_to_all_hp2cp`。需要从参考实现
移植，或直接在 `lightning_attention.py` 中实现（~40 行）。

**参考来源**: `megatron/core/tensor_parallel/mappings.py:632-713`

```python
def _all_to_all_cp2hp(input_: torch.Tensor, cp_group) -> torch.Tensor:
    """Convert context-parallel to head-parallel layout.

    [S/CP, B, H_local, D] -> [S, B, H_local/CP, D]

    通过 all-to-all 交换序列维度和 head 维度。
    """
    cp_size = torch.distributed.get_world_size(group=cp_group)
    # input: [S/CP, B, H_local, D]
    seq_len, batch, num_heads, head_dim = input_.shape
    # reshape: [S/CP, B, CP, H_local/CP, D] -- 将 heads 分成 CP 组
    input_ = input_.view(seq_len, batch, cp_size, num_heads // cp_size, head_dim)
    # permute: [CP, S/CP, B, H_local/CP, D]
    input_ = input_.permute(2, 0, 1, 3, 4).contiguous()
    # flatten for all-to-all: [CP * S/CP * B * H_local/CP * D]
    output = torch.empty_like(input_)
    torch.distributed.all_to_all_single(output, input_, group=cp_group)
    # output: [CP, S/CP, B, H_local/CP, D] -> [S, B, H_local/CP, D]
    output = output.view(cp_size * seq_len, batch, num_heads // cp_size, head_dim)
    return output


def _all_to_all_hp2cp(input_: torch.Tensor, cp_group) -> torch.Tensor:
    """Convert head-parallel to context-parallel layout (逆操作).

    [S, B, H_local/CP, D] -> [S/CP, B, H_local, D]
    """
    cp_size = torch.distributed.get_world_size(group=cp_group)
    full_seq, batch, heads_per_cp, head_dim = input_.shape
    seq_per_cp = full_seq // cp_size
    # reshape: [CP, S/CP, B, H_local/CP, D]
    input_ = input_.view(cp_size, seq_per_cp, batch, heads_per_cp, head_dim).contiguous()
    output = torch.empty_like(input_)
    torch.distributed.all_to_all_single(output, input_, group=cp_group)
    # output: [CP, S/CP, B, H_local/CP, D] -> [S/CP, B, H_local, D]
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    output = output.view(seq_per_cp, batch, cp_size * heads_per_cp, head_dim)
    return output
```

同时需要 `undo_attention_load_balancing` 和 `redo_attention_load_balancing`
(参考 `megatron/core/utils.py:1967-2025`)，用于将 CP zigzag 切分转换为顺序排列
（线性注意力需要顺序序列）。

```python
def _undo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """将 CP zigzag 切分恢复为顺序排列.

    CP 使用 zigzag 切分以均衡因果注意力负载:
    rank 0: chunks [0, 2*CP-1, 1, 2*CP-2, ...]
    rank 1: chunks [2, 2*CP-3, 3, 2*CP-4, ...]

    线性注意力需要顺序序列，需要恢复。
    """
    # 实现 zigzag → sequential 的 index 重排
    ...


def _redo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """将顺序排列恢复为 CP zigzag 切分（逆操作）."""
    ...
```

### 2. 修改 LightningCoreAttention

**文件**: `areal/models/mcore/lightning_attention.py` — `LightningCoreAttention`

**`__init__`** 改动:
- g_gamma 存储完整的 `H_local` 个值（TP-sliced）
- forward 时根据 `cp_rank` 动态 slice 取 `H_local/CP` 段

```python
# 当前: g_gamma shape [H_local] (TP-sliced)
# CP 时: forward 中传入 cp_rank 参数
# g_gamma_cp = self.g_gamma[cp_rank * H_per_cp : (cp_rank+1) * H_per_cp]
```

**`forward`** 改动:
- 添加可选 `cp_rank` 参数
- CP>1 时：对 g_gamma 做 CP slice 后传给 `chunk_simple_gla`

### 3. 修改 LightningSelfAttention.forward

核心改动，在 `core_attention` 调用前后添加 CP 通信:

```python
def forward(self, hidden_states, ...):
    # ... QKV projection, layernorm, RoPE ...
    # (都在 [S/CP, B, ...] 或 SP all-gather 后的维度上操作)

    cp_size = _get_cp_world_size()

    if cp_size > 1:
        cp_group = mpu.get_context_parallel_group()
        cp_rank = _get_cp_rank()

        # 1. QKV: all-to-all 交换 seq↔head
        #    [S/CP, B, H/TP, D] -> [S, B, H/TP/CP, D]
        query = _all_to_all_cp2hp(query, cp_group)
        key = _all_to_all_cp2hp(key, cp_group)
        value = _all_to_all_cp2hp(value, cp_group)

        # 2. Undo zigzag load balancing (线性注意力需要顺序序列)
        query = _undo_attention_load_balancing(query, cp_size, cp_rank)
        key = _undo_attention_load_balancing(key, cp_size, cp_rank)
        value = _undo_attention_load_balancing(value, cp_size, cp_rank)

        # 3. Core attention (full sequence, fewer heads, CP-sliced g_gamma)
        attn_output = self.core_attention(query, key, value, cp_rank=cp_rank)

        # 4. Redo load balancing
        attn_output = _redo_attention_load_balancing(attn_output, cp_size, cp_rank)

        # 5. all-to-all 恢复
        #    [S, B, H/TP/CP, D] -> [S/CP, B, H/TP, D]
        attn_output = _all_to_all_hp2cp(attn_output, cp_group)
    else:
        attn_output = self.core_attention(query, key, value)

    # ... gate, gate_norm, output projection ...
```

### 4. Gate Projection 在 CP 下的维度处理

**关键问题**: gate projection 在 `hidden_states` 上操作。SP+CP 时：

- `hidden_states` 输入维度: `[S/(TP*CP), B, H]` (SP+CP 切分)
- `linear_gate` (TEColumnParallelLinear) 输出: `[S/CP, B, H_local*D]`
  (only gather TP part, CP part 保留)
- attention output 经过 `hp2cp` 后: `[S/CP, B, H_local, D]`

两者 seq 维度都是 `S/CP`，维度一致，不需要额外处理。

但 gate 的 head 维度是 `H_local`（全部 local heads），而 attn_output 在 CP all-to-all
前后 head 维度也是 `H_local`，所以维度匹配。

### 5. RoPE 适配

Lightning Attention 使用自己的 `RotaryEmbedding`。CP 下：

- `self.lightning_rotary_emb(seq_len)` 中 `seq_len = qkv.shape[0]`
- SP+CP 时这个值是 `S/CP`
- RoPE 生成的 cos/sin 频率从 position 0 开始

**问题**: CP rank 0 持有 positions [0, S/CP)，rank 1 持有 [S/CP, 2*S/CP)，
但 zigzag 切分下位置不连续。

**解决方案**: All-to-All CP 中，RoPE 在 all-to-all **之前**应用（在 `S/CP` 维度上），
此时每个 rank 持有的 tokens 经过 zigzag 切分，位置已经是正确的（CP 框架已处理）。
all-to-all 后变成全序列、少 head，不再需要 position encoding。

参考实现也是这个顺序：RoPE → all-to-all → attention kernel。

### 6. 修改 bailing_moe.py

移除 CP>1 的 ValueError，替换为 head 数整除校验:

```python
# 替换原来的 ValueError:
if tf_config.context_parallel_size > 1 and n_lightning > 0:
    tp_size = tf_config.tensor_model_parallel_size
    cp_size = tf_config.context_parallel_size
    heads_per_tp = tf_config.num_attention_heads // tp_size
    if heads_per_tp % cp_size != 0:
        raise ValueError(
            f"For Lightning Attention with CP, num_heads_per_tp_partition "
            f"({heads_per_tp}) must be divisible by context_parallel_size ({cp_size})."
        )
```

______________________________________________________________________

## SP + CP 维度交互分析

当 TP>1 时 SP (Sequence Parallelism) 自动开启。关键问题：
TEColumnParallelLinear 在 SP+CP 环境下的输出维度。

**分析**:

```
输入 hidden_states: [S/(TP*CP), B, H]  (SP+CP 切分)
  ↓ TEColumnParallelLinear (SP: all-gather TP 部分)
输出: [S/CP, B, H_local * 3 * D]       (仅 gather TP 切分，CP 切分保留)
  ↓ reshape
QKV: [S/CP, B, H_local, D]
  ↓ all_to_all_cp2hp
转换: [S, B, H_local/CP, D]            (全序列，CP 切分到 head 维度)
```

SP 只 gather TP 的序列切分，CP 的切分保留。所以 `qkv.shape[0]` 是 `S/CP`，
与 all-to-all 输入要求一致。

______________________________________________________________________

## CP 辅助函数：获取 CP 信息

```python
def _get_cp_world_size() -> int:
    """Get context parallel world size."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_cp_rank() -> int:
    """Get context parallel rank."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_rank()
    except (RuntimeError, AttributeError):
        pass
    return 0
```

______________________________________________________________________

## 开放问题

1. **CP zigzag load balancing**: AReaL 的 CP 实现是否使用了 zigzag 切分？
   如果没有（即简单的连续切分），则不需要 `undo/redo_attention_load_balancing`。
   需要检查 AReaL megatron-core 中 CP 的序列分发方式。

2. **packed sequences (cu_seqlens)**: 参考实现中 all-to-all CP 也处理了 cu_seqlens
   （变长序列）。AReaL 的 SFT 数据是否使用 packed sequences？如果是，all-to-all
   需要额外处理 cu_seqlens 的重排。

3. **Autograd 兼容性**: `all_to_all_single` 需要包装为 `torch.autograd.Function`
   以支持反向传播。参考实现中通过 `_AllToAll` autograd function 实现。

______________________________________________________________________

## 验证方案

### 测试配置

```yaml
# /tmp/bailing_moe_sft_cp2tp2.yaml
experiment_name: bailing-moe-sft-cp2tp2
allocation_mode: megatron:(attn:d2p1t2c2|ffn:d2p1e4)
# world_size = D * TP * PP * CP = 2 * 2 * 1 * 2 = 8 GPUs
```

### 验证命令

```bash
LOG_DIR=/storage/openpsi/codes/chucai.dzq/tmp/bailing-moe-sft-cp2tp2
mkdir -p $LOG_DIR
PYTHONPATH=$PWD:$PYTHONPATH python -m areal.infra.launcher.local \
    examples/math/gsm8k_sft.py \
    --config /tmp/bailing_moe_sft_cp2tp2.yaml \
    2>&1 | tee $LOG_DIR/train.log
```

### 验证清单

- [ ] CP=2 + TP=2 训练不 crash
- [ ] Loss 与 CP=1 接近（对比 `bailing_moe_sft_tp2.yaml` 的 loss 曲线）
- [ ] 异构模式 `(attn:d2p1t2c2|ffn:d2p1e4)` 可运行
- [ ] 梯度正确性：小 batch 上比较 CP=1 和 CP=2 的梯度差异在数值精度范围内
