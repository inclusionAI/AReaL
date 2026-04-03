# FSDP2 流水线并行

本文档描述 AReaL 中 FSDP2 流水线并行（PP）支持，该功能将 FSDP 训练后端从 3D 并行（DP x SP x TP）扩展为 4D 并行（PP x DP x SP x TP）。

## 概述

流水线并行将模型的层分布到多个 GPU 组（称为**阶段**）上，每个阶段处理不同的层子集。训练过程中，微批次（microbatch）在阶段之间以流水线方式流动，前向和反向传播交叠执行以最大化吞吐量。

AReaL 的 FSDP2 流水线并行基于 PyTorch 原生的 `torch.distributed.pipelining` API 构建，并适配了 HuggingFace 模型架构。这使得已经使用 FSDP 后端的用户无需切换到 Megatron 或 Archon 即可将训练扩展到更大的模型。

### 为什么需要 FSDP2 + PP？

在没有 PP 的情况下，FSDP 后端支持数据并行（DP）、序列/上下文并行（SP）和 tensor 并行（TP）。虽然这些方式对许多工作负载已经足够，但存在以下限制：

- **DP** 在每个 rank 上复制完整模型 —— 对超大模型存在显存瓶颈。
- **TP** 跨 GPU 切分单个算子，需要高带宽互连（如 NVLink），且带来通信开销。
- **SP** 切分序列长度，但受注意力头数整除性约束。

流水线并行通过跨阶段切分**层**来补充这些并行方式，阶段之间仅需传递激活张量，通信量极小。这使得 PP 在跨节点扩展（节点间带宽较低）场景下尤其有效。

## 架构

### 4D 设备网格

当启用 PP 时，设备网格布局变为：

```
("pp", "dp", "sp", "tp")
```

这与 torchtitan 的方式一致，PP 作为最外层维度。训练任务的 GPU 总数为：

```
world_size = pp × dp × sp × tp
```

例如，使用 `fsdp:d4p2t2` 在 16 个 GPU 上：

```
pp=2, dp=4, sp=1, tp=2
world_size = 2 × 4 × 1 × 2 = 16
```

### PP 如何与 FSDP2 集成

集成过程在初始化时按以下顺序执行：

1. **模型切分**：将 HuggingFace 模型按 transformer 层分配到各流水线阶段。Embedding 层和输出层分别分配给第一阶段和最后一阶段。

2. **逐阶段并行化**：每个模型部分（阶段）独立接收 TP + FSDP2 分片。这意味着每个阶段在其自身的 DP x SP x TP 子网格内进行完全分片和 tensor 并行化。

3. **调度构建**：流水线调度（如 1F1B、Interleaved1F1B）编排微批次在训练期间如何流经各阶段。

4. **Runner 创建**：`FSDPPipelinedRunner` 封装调度并处理训练和评估的前向/反向执行。

### 关键优化：`reshard_after_forward=False`

启用 PP 时，FSDP2 配置为 `reshard_after_forward=False`。在标准 FSDP2（未启用 PP）中，参数在每次前向传播后会被 reshard（释放）以节省显存。然而在 PP 模式下，多个微批次会依次通过同一阶段。如果每个微批次的前向传播后都 reshard 参数，则下一个微批次需要重新进行 all-gather —— 这会带来显著的通信开销。

通过在前向传播后保持参数 gathered 状态，每个阶段避免了跨微批次的重复 all-gather。这是从 torchtitan 和 verl 引入的 FSDP2 + PP 关键优化。

> **权衡**：这会增加每个阶段的峰值显存使用量（参数保持未分片状态），但避免了冗余通信。请参阅[处理 OOM 问题](../best_practices/handling_oom.md)获取显存调优指南。

## 配置

### Backend 字符串

要在 FSDP 后端启用 PP，在 backend 字符串中添加 `p` 维度：

```yaml
actor:
  backend: "fsdp:d4p2"       # 4 DP × 2 PP = 8 GPU
```

可以将 PP 与 TP 和 SP 组合使用：

```yaml
actor:
  backend: "fsdp:d2p2t2"     # 2 DP × 2 PP × 2 TP = 8 GPU
```

### FSDPEngineConfig PP 字段

所有 PP 相关字段位于 actor/critic 配置的 `fsdp` 部分下：

#### `pp_schedule`

流水线调度类型。默认值：`"Interleaved1F1B"`。

| 调度                     | 每 Rank 阶段数 | 描述                                   |
| ------------------------ | -------------- | -------------------------------------- |
| `1F1B`                   | 1              | 经典一前向一反向调度                    |
| `Interleaved1F1B`        | >= 2           | 交错调度，每 rank 多个虚拟阶段          |
| `InterleavedZeroBubble`  | >= 2           | 带零气泡优化的交错调度                  |
| `ZBVZeroBubble`          | 2（V 形）      | V 形零气泡调度（阶段按 V 形分配）       |

**单阶段调度**（`1F1B`）为每个 PP rank 分配恰好 1 个虚拟阶段。
**多阶段调度**（`Interleaved1F1B`、`InterleavedZeroBubble`、`ZBVZeroBubble`）为每个 rank 分配 2 个或更多虚拟阶段，以提高流水线利用率。

**V 形调度**（`ZBVZeroBubble`）为每个 rank 分配恰好 2 个阶段，按 V 形模式排列：rank 0 获得阶段 (0, N-1)，rank 1 获得阶段 (1, N-2)，以此类推。这平衡了前向和反向传播的计算量。

#### `pp_layers_per_stage`

每个虚拟流水线阶段的 transformer 层数。默认值：`None`。

- 如果设置，虚拟阶段数按 `ceil((num_layers + first_less + last_less) / pp_layers_per_stage)` 计算。
- 计算得到的虚拟阶段数必须能被 PP 度数整除。
- 如果为 `None`，每 rank 的阶段数根据调度类型推断：`1F1B` 为 1，交错/ZBV 调度为 2。

#### `pp_first_stage_less_layers`

从第一阶段减去的等效层数，用于补偿 embedding 层的开销。默认值：`1`。

由于第一阶段还承载 `model.embed_tokens`，因此会分配较少的 transformer 层以平衡各阶段的计算量。

#### `pp_last_stage_less_layers`

从最后一阶段减去的等效层数，用于补偿输出头的开销。默认值：`1`。

由于最后一阶段还承载 `model.norm` 和 `lm_head`（或 critic 模型的 `score`），因此会分配较少的 transformer 层。

### 示例：层分布

对于一个有 28 个 transformer 层的模型，`pp_size=4`（默认 `first_less=1`、`last_less=1`）：

```
有效层数 = 28 + 1 + 1 = 30
每阶段层数 = 30 / 4 = 7（+ 2 个余数阶段获得 8 层）

阶段 0: embed_tokens + 7 层  (layers 0-6)   — 1 层让给 embed
阶段 1: 8 层                 (layers 7-14)
阶段 2: 8 层                 (layers 15-22)
阶段 3: 5 层 + norm + lm_head (layers 23-27) — 1 层让给输出
```

## 支持的 PP 调度

### 1F1B（一前向一反向）

经典流水线调度。每个 rank 持有恰好 1 个阶段。微批次进入流水线，当流水线填满后，每个 rank 交替执行一次前向和一次反向。

- **优点**：简单，显存占用低（每 rank 仅 1 个阶段）。
- **缺点**：预热和收尾阶段存在流水线气泡。

```yaml
actor:
  fsdp:
    pp_schedule: "1F1B"
```

### Interleaved1F1B

每个 rank 持有多个虚拟阶段（默认 2 个）。微批次在虚拟阶段之间交错执行，减少流水线气泡。

- **优点**：相比 1F1B 减少气泡。
- **缺点**：更高的显存占用（每 rank 多个阶段），调度更复杂。

```yaml
actor:
  fsdp:
    pp_schedule: "Interleaved1F1B"
```

### InterleavedZeroBubble

带零气泡优化的交错调度，通过将一个微批次的反向计算与另一个微批次的前向计算重叠来进一步减少空闲时间。

- **优点**：接近零气泡。
- **缺点**：由于保留激活而增加显存；可能使用 `retain_graph`。

```yaml
actor:
  fsdp:
    pp_schedule: "InterleavedZeroBubble"
```

### ZBVZeroBubble

V 形零气泡调度，每个 rank 持有恰好 2 个阶段，按 V 形模式分配（rank *i* 获得阶段 *i* 和 *N-1-i*）。这提供了出色的负载均衡和最小的流水线气泡。

- **优点**：最佳气泡减少，各 rank 计算量均衡。
- **缺点**：要求每 rank 恰好 2 个阶段；显存占用较高。

```yaml
actor:
  fsdp:
    pp_schedule: "ZBVZeroBubble"
```

## 完整配置示例

32 GPU 配置，使用 4D 并行训练 70B 模型：

```yaml
rollout:
  backend: "sglang:d4t4"       # 4 × 4 = 16 GPU 用于推理

actor:
  backend: "fsdp:d2p2t4"       # 2 DP × 2 PP × 4 TP = 16 GPU 用于训练
  gradient_checkpointing: true
  fsdp:
    pp_schedule: "Interleaved1F1B"
    pp_layers_per_stage: null   # 自动：每个 PP rank 2 个虚拟阶段
    pp_first_stage_less_layers: 1
    pp_last_stage_less_layers: 1
    memory_efficient_load: true
```

最小 8 GPU 配置，仅使用 PP（无 TP）：

```yaml
rollout:
  backend: "sglang:d4"          # 4 GPU 用于推理

actor:
  backend: "fsdp:d2p2"          # 2 DP × 2 PP = 4 GPU 用于训练
  fsdp:
    pp_schedule: "1F1B"
```

## 限制与已知问题

- **不支持 PP + EP 组合**：FSDP Engine 不支持将流水线并行与专家并行组合使用。PP + EP 工作负载请使用 Archon Engine。

- **需要 HuggingFace 模型结构**：PP 实现假设标准 HuggingFace 模型结构（`model.embed_tokens`、`model.layers.*`、`model.norm`、`lm_head`/`score`）。自定义模型架构可能需要适配。

- **微批次数量**：微批次数量应 >= 虚拟阶段总数（`num_virtual_stages = stages_per_rank × pp_degree`），以避免过多的流水线气泡。不满足此条件时会记录警告日志。

- **`reshard_after_forward` 强制为 `False`**：启用 PP 时，FSDP 参数在前向传播后保持未分片状态以避免重复 all-gather。相比非 PP 的 FSDP，这会增加显存使用量。

- **视觉语言模型**：PP 支持针对纯解码器 LLM 设计。PP 与 VLM 的组合尚未验证。

- **逐层优化器步进**：`per_layer_optim_step` 与 PP 尚未联合验证，请谨慎使用。

## 另请参阅

- [分配模式参考](alloc_mode.md) — Backend 字符串语法和 GPU 分配
- [处理 OOM 问题](../best_practices/handling_oom.md) — 训练显存调优
- [Archon：PyTorch 原生训练引擎](../tutorial/archon.md) — 支持 PP + EP 的替代后端