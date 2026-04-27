# LoRA 参考

LoRA 是一种参数高效的微调技术，会在预训练权重中注入可训练的低秩矩阵， 通常作用在线性层附近。与全参数微调相比，LoRA 可以显著降低显存占用和计算开销， 从而让大模型的
RL 微调在硬件资源有限的条件下也更具可行性。

在 AReaL 中，LoRA 尤其适用于以下场景：

- 在相对有限的硬件条件下进行超大模型的强化学习训练，例如使用 8 x 80 GB GPU 训练 70B+ 规模模型，
- 由于显存压力更低，可以支持更大的 batch size，
- 模型迁移与部署更加简单，因为只需要保存和分发 LoRA adapter，
- \[Future\] 更高效地并行微调多个 LoRA adapter，以提升硬件利用率（参见 RFC
  [#609](https://github.com/inclusionAI/AReaL/issues/609)）。

本文档说明如何在 RL 训练中启用 LoRA，并配置相关参数。

## 后端支持

AReaL 当前的 LoRA 支持矩阵如下：

| Engine   | vLLM | SGLang |
| -------- | ---- | ------ |
| FSDP2    | ✅   | ✅     |
| Megatron | ✅   | ❌     |
| Archon   | ❌   | ❌     |

**示例脚本：**

| Engine       | Example script                                    |
| ------------ | ------------------------------------------------- |
| FSDP2        | `examples/math/gsm8k_grpo_lora.yaml`              |
| FSDP2 Delta  | `examples/math/gsm8k_grpo_lora_delta_sync.yaml`   |
| Megatron     | `examples/math/gsm8k_grpo_megatron_lora.yaml`     |
| Megatron MoE | `examples/math/gsm8k_grpo_megatron_lora_moe.yaml` |

对于 Megatron + vLLM，AReaL 现在支持：

- 在 Qwen3 MoE 等 MoE 架构上进行 LoRA 微调，并通过 XCCL 更新 LoRA 权重。
- 当 Megatron 与 rollout group 横跨多个节点时进行跨节点 LoRA 训练。

## 核心 LoRA 参数

| 参数              | 作用                                                               | 常见取值              |
| ----------------- | ------------------------------------------------------------------ | --------------------- |
| `use_lora`        | 是否启用 LoRA 微调模式。                                           | `true` / `false`      |
| `lora_rank` (`r`) | 低秩适配器的秩。`r` 越大，表达能力越强，但显存与计算开销更高。     | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA 缩放系数。通常可理解为有效缩放与 `alpha / r` 成正比。         | `16`, `32`, `64`      |
| `target_modules`  | 指定注入 LoRA 的目标子模块。这是最关键、且与模型结构强相关的配置。 | 例如 \[`all-linear`\] |
| `peft_type`       | PEFT 方法类型。在 AReaL 配置中为 LoRA。                            | `lora`                |

## LoRA 增量权重同步（Delta Sync）

### 概述

在标准 LoRA 权重同步流程中，训练引擎会将 LoRA adapter 权重合并到基座模型中，然后在每个训练步骤
将**完整的合并后权重**传输到推理引擎。对于大模型来说，这一传输过程代价很高，容易成为性能瓶颈。

**LoRA Delta Sync（增量权重同步）** 采用不同的策略：基座模型权重仅在首次同步时传输一次，此后的 每次同步仅传输 LoRA adapter
权重（`lora_A` / `lora_B` 矩阵）。由于 adapter 参数量通常 **不到模型总参数量的 1%**，因此可以大幅减少权重同步时间和网络带宽消耗。

### 适用场景

LoRA Delta Sync 适用于以下组合：

- **训练引擎：** FSDP (FSDP2)
- **推理引擎：** SGLang
- **微调方式：** LoRA（`use_lora: true`）
- **权重更新模式：** 磁盘

> **注意：** Delta Sync 不支持 vLLM 推理引擎或 Megatron 训练引擎。

### 工作原理

同步过程分为两个阶段：

1. **首次同步（阶段一）**

   - **阶段 1a** -- FSDP 引擎将**基座模型权重**（不含 LoRA 参数）保存为 HuggingFace safetensors 格式到磁盘，SGLang
     通过 `/update_weights_from_disk` 端点加载。
   - **阶段 1b** -- 紧接着，LoRA adapter 权重被保存到磁盘， SGLang 通过 `/load_lora_adapter` 端点加载
     adapter。SGLang 服务端将 adapter 加载到基座模型之上。

1. **后续同步（阶段二）**

   - 仅传输**更新后的 LoRA adapter 权重**，通过 `/load_lora_adapter` 端点完成。基座模型权重已驻留在推理引擎的 GPU
     显存中，无需重复传输。

这种两阶段设计意味着，在初始的全量同步之后，每次后续权重更新仅传输极小的 adapter 增量， 从而显著加快每次迭代的速度。

### 配置说明

要启用 LoRA Delta Sync，请在 YAML 配置文件的 actor（训练引擎）部分设置 `lora_delta_sync: true`，并配合标准的 LoRA
设置：

```yaml
actor:
  backend: "fsdp:d4"
  path: Qwen/Qwen2.5-1.5B-Instruct

  # 权重更新模式：启用 lora_delta_sync 后，disk
  weight_update_mode: disk

  # 标准 LoRA 设置
  use_lora: true
  peft_type: lora
  lora_rank: 16
  lora_alpha: 16
  target_modules: [all-linear]

  # 启用增量同步
  lora_delta_sync: true
```

SGLang 部分需要启用 LoRA 支持并设置最大 LoRA 秩：

```yaml
sglang:
  model_path: ${actor.path}
  dtype: ${actor.dtype}
  enable_lora: ${actor.use_lora}
  max_lora_rank: ${actor.lora_rank}
  mem_fraction_static: 0.8
```

完整的可运行示例请参考 `examples/math/gsm8k_grpo_lora_delta_sync.yaml`。

### 关键参数

| 参数                   | 取值要求 / 说明                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `use_lora`             | `true`                                                                                |
| `lora_delta_sync`      | `true` -- 启用增量同步路径。                                                          |
| `weight_update_mode`   | `disk`                                                                                |
| `delta_sync_dir`       | 可选，多节点共享文件系统路径（如 NFS/CPFS）。单节点可省略（默认 `~/.cache/areal/`）。 |
| `lora_rank`            | 训练与推理配置中需保持一致（例如 `16`）。                                             |
| `lora_alpha`           | LoRA 缩放系数，与标准 LoRA 相同。                                                     |
| `sglang.enable_lora`   | `true` -- SGLang 服务端必须启用 LoRA 支持。                                           |
| `sglang.max_lora_rank` | 必须 >= 训练引擎使用的 `lora_rank`。                                                  |

## 实践建议

- 可先从 `r=16` 或 `r=32` 开始，再按效果和资源逐步调参。
- `target_modules` 需与具体模型的层命名保持一致。
- 对于 Megatron 后端，LoRA 需要使用 `megatron-bridge`，而不是 `mbridge`。
