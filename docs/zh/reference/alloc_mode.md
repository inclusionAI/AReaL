# 分配模式

本文档描述 AReaL 的分配模式系统，该系统控制分布式 RL 训练期间 GPU 在推理和训练后端之间的分配方式。

## 概述

`allocation_mode` 配置选项是一个基于模式的字符串，用于指定：

- 使用哪些后端进行推理（SGLang、vLLM）和训练（FSDP、Megatron、Archon）
- 每个后端的并行化策略
- 所需的 GPU 总数

AReaL 将此字符串解析为 `AllocationMode` 对象，在整个集群中协调资源分配。

## 语法

### 基本格式

```
<backend>:<parallelism_dims>
```

### 双组件格式（推理 + 训练）

```
<inference_backend>:<dims> + <training_backend>:<dims>
```

`+` 运算符分隔在**独立 GPU 池**上运行的组件。

### 并行维度

| 维度     | 缩写 | 描述                 | 适用于           |
| -------- | ---- | -------------------- | ---------------- |
| Data     | `d`  | 模型副本数量         | 所有后端         |
| Tensor   | `t`  | 跨 GPU 分割操作      | 所有后端         |
| Pipeline | `p`  | 跨 GPU 阶段分割层    | Megatron、Archon |
| Context  | `c`  | 跨 GPU 分割序列长度  | 所有后端         |
| Expert   | `e`  | 跨 GPU 分割 MoE 专家 | Megatron、Archon |

维度指定为 `<缩写><大小>`，例如 `d4t2` 表示数据并行大小为 4，tensor 并行大小为 2。

## 计算 GPU 需求

组件的 GPU 总数计算如下：

```
world_size = dp × tp × pp × cp
```

专家并行（`e`）不会增加 world size —— 它在现有 GPU 网格内重新分配专家的放置位置。

### 示例

| 分配模式                          | 推理 GPU | 训练 GPU | 总计 |
| --------------------------------- | -------- | -------- | ---- |
| `d8`                              | -        | 8        | 8    |
| `sglang:d2t4`                     | 8        | -        | 8    |
| `sglang:d2t4 + fsdp:d4t2`         | 8        | 8        | 16   |
| `sglang:d4t4 + megatron:d2p2t4e4` | 16       | 16       | 32   |

## 后端选择

### 推理后端

| 后端     | 支持的维度    |
| -------- | ------------- |
| `sglang` | `d`, `t`      |
| `vllm`   | `d`, `t`, `p` |

对于推理，`d` 表示独立服务器实例的数量，每个实例使用 `t × p` 个 GPU。

请注意，内部后端配置不影响 AReaL 如何分配 GPU。给定分配模式 `sglang:d4t4`，你还可以配置
`sglang.dp_size=4`、`sglang.ep_size=4` 和 `sglang.enable_dp_attention=True`。在这种情况下，我们启动 4
个模型副本，每个副本 4 个 GPU。在每个实例内，SGLang 仍然使用 DP attention 和专家并行来分配注意力层和专家层中的计算。

### 训练后端

| 后端       | 支持的维度              | 使用场景                      |
| ---------- | ----------------------- | ----------------------------- |
| `fsdp`     | `d`, `t`, `c`           | 简单并行的默认选择            |
| `megatron` | `d`, `t`, `p`, `c`, `e` | 流水线或专家并行必需          |
| `archon`   | `d`, `t`, `p`, `c`, `e` | Megatron 的替代方案（实验性） |

省略后端时，AReaL 根据并行配置自动选择：

- **FSDP**：当仅指定 `d`、`t`、`c` 时使用
- **Megatron**：当 `p > 1` 或 `e > 1` 时使用

```
# 等效形式
d4t2           # 自动选择 FSDP
fsdp:d4t2      # 显式 FSDP

d2p2t4         # 自动选择 Megatron（pp > 1）
megatron:d2p2t4  # 显式 Megatron
```

## MoE 混合并行

对于 Mixture-of-Experts 模型，Megatron/Archon 支持使用混合语法为注意力和 FFN（专家）模块使用不同的并行策略：

```
megatron:(attn:<attn_dims>|ffn:<ffn_dims>)
```

这启用了
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)，可以降低组合上下文和专家并行的最低
GPU 要求。

### 约束

- 流水线并行大小（`p`）对 `attn` 和 `ffn` 必须相同
- World size 必须匹配（如果 `ffn` 中省略 `d`，则自动派生）
- 专家并行（`e`）仅在 `ffn` 部分有效

### 示例

```
megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)
```

| 模块 | dp  | pp  | tp  | cp  | ep  | World Size |
| ---- | --- | --- | --- | --- | --- | ---------- |
| attn | 4   | 2   | 2   | 2   | -   | 32         |
| ffn  | 2   | 2   | 4   | -   | 2   | 32         |

## 另见

- [Fine-tuning Large MoE Models](../tutorial/megatron.md) - Megatron 后端教程
- [Archon: PyTorch-Native Training Engine](../tutorial/archon.md) - Archon 后端教程
- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
