# 使用 EAGLE 进行推测解码

## 概述

推测解码（Speculative Decoding）是一种加速自回归文本生成的技术。它使用一个轻量级的
**草稿模型（Draft Model）**并行提出多个候选 token，然后由完整的**目标模型（Target Model）**
在一次前向传播中进行验证。当候选 token 被接受时，有效吞吐量可显著提升（通常 2-3 倍），
且不改变输出分布。

AReaL 集成了 **EAGLE**（Extrapolation Algorithm for Greater Language-model Efficiency）
作为推测解码后端。EAGLE 利用目标模型的隐藏状态通过小型辅助头预测未来 token，特别适合
RL 训练循环中策略模型持续演化的场景。

### 为什么在 RL 训练中使用推测解码？

在 RLHF / GRPO 训练流水线中，rollout 生成通常是吞吐瓶颈。推测解码通过以下方式直接
解决这一问题：

- 降低 rollout 阶段每个样本的生成延迟
- 提高推理阶段的 GPU 利用率
- 保持完全一致的输出质量（验证步骤是精确的）

结合 **MTP（多 Token 预测）在线训练**，草稿模型能与不断演化的策略保持对齐，在整个
训练过程中维持较高的接受率。

## 前提条件

启用推测解码前，请确保：

1. **带 MTP 层的模型**：基座模型必须包含 MTP（多 Token 预测）头层。`Qwen/Qwen3-0.6B`
   等 Qwen3 系列模型自带 MTP 层，可作为 EAGLE 草稿头使用。

2. **SGLang 后端**：推测解码需要 SGLang 推理后端。请确保已安装并配置 SGLang：

   ```bash
   pip install "sglang[all]>=0.4.7"
   ```

3. **Megatron-Core >= 0.12.0**：MTP 在线训练需要 Megatron-Core 0.12.0 或更高版本，
   该版本包含了内置梯度隔离（embedding detach 和 functional_call lm_head）的
   `MultiTokenPrediction` 模块。这确保 MTP 损失梯度仅更新 MTP 层参数，不会污染
   主策略模型的权重。

4. **充足的 GPU 显存**：草稿模型会在推理 GPU 上增加少量显存开销。如需要，可降低
   `sglang.mem_fraction_static`（例如从 `0.85` 降至 `0.80`）。

## 配置说明

### SGLang EAGLE 配置

推测解码在实验 YAML 的 `sglang` 部分进行配置。关键字段位于 `SGLangConfig` 中：

```yaml
sglang:
  model_path: ${actor.path}
  dtype: bfloat16
  mem_fraction_static: 0.80
  context_length: 32768

  # --- 推测解码配置 ---
  speculative_algorithm: "EAGLE"         # 或 "EAGLE3"
  speculative_draft_model_path: null     # null = 使用内置 MTP 头
  speculative_num_steps: 3              # 每次迭代的草稿步数
  speculative_eagle_topk: 1             # 草稿 token 选择的 top-k 值
  speculative_num_draft_tokens: 4       # 每步提出的草稿 token 数
  speculative_attention_mode: null      # null 使用默认注意力机制
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `speculative_algorithm` | `null` | 算法名称：`"EAGLE"` 或 `"EAGLE3"`。`null` 禁用推测解码。 |
| `speculative_draft_model_path` | `null` | 外部草稿模型路径。`null` 复用目标模型内置的 MTP 层。 |
| `speculative_num_steps` | `3` | EAGLE 在验证前执行的自回归草稿步数。 |
| `speculative_eagle_topk` | `1` | 每个草稿步保留的 top-k 候选数。 |
| `speculative_num_draft_tokens` | `4` | 每次推测迭代中馈入验证器的总草稿 token 数。 |
| `speculative_attention_mode` | `null` | 覆盖草稿阶段使用的注意力核。`null` 使用引擎默认值。 |

### MTP 在线训练配置

为保持草稿模型与训练中的策略对齐，请在 `actor` 部分启用 MTP 在线训练：

```yaml
actor:
  backend: "megatron:d4p1t1"
  path: Qwen/Qwen3-0.6B

  # --- MTP 在线训练 ---
  enable_mtp_training: true
  mtp_num_layers: 1                    # 必须匹配模型的 MTP 架构
  mtp_loss_scaling_factor: 0.1         # MTP 损失相对于主 RL 损失的权重

  # Megatron 特定的 MTP 设置（在 actor.megatron 中）
  megatron:
    mtp_num_layers: 1                  # 与 actor.mtp_num_layers 一致
    mtp_loss_scaling_factor: 0.1       # 与 actor.mtp_loss_scaling_factor 一致
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `enable_mtp_training` | `false` | MTP 在线训练的总开关。 |
| `mtp_num_layers` | `1` | 训练的 MTP 头层数。启用时必须 > 0。 |
| `mtp_loss_scaling_factor` | `0.1` | MTP 辅助损失的权重。必须在 (0, 1.0] 范围内。 |

当 `enable_mtp_training` 为 `true` 时，训练器会在 MTP 头上计算辅助的下一 token
预测损失，并将其（按比例缩放后）加到主 RL 目标中。这确保了草稿头随策略变化持续
提升预测准确性。

## 完整示例

以下是一个使用 4 GPU 的最小 GRPO + EAGLE GSM8K 配置：

```yaml
experiment_name: gsm8k-grpo-eagle
trial_name: trial0
seed: 42
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 1
  n_gpus_per_node: 4

actor:
  backend: "megatron:d2p1t1"
  path: Qwen/Qwen3-0.6B
  enable_mtp_training: true
  mtp_num_layers: 1
  mtp_loss_scaling_factor: 0.1

sglang:
  model_path: ${actor.path}
  speculative_algorithm: "EAGLE"
  speculative_num_steps: 3
  speculative_num_draft_tokens: 4
  mem_fraction_static: 0.80

train_dataset:
  path: openai/gsm8k
  type: rl
  batch_size: 128
```

完整配置文件请参见
[`examples/math/gsm8k_grpo_megatron_eagle.yaml`](https://github.com/inclusionAI/AReaL/blob/main/examples/math/gsm8k_grpo_megatron_eagle.yaml)。

## 监控

### 关键指标

训练过程中，请在日志或 WandB 面板中关注以下指标：

1. **推测接受率（Speculative Accept Rate）**
   - 日志中记录为 `spec_accept_rate`（= `spec_accept_token_num / spec_draft_token_num`）
   - 对齐良好的草稿模型的健康接受率为 **0.6 - 0.9**
   - 如果接受率降至 **0.4** 以下，说明草稿模型正在落后于策略

2. **MTP 损失（MTP Loss）**
   - 训练统计中记录为 `mtp_loss`
   - 应随时间下降；MTP 损失上升表明训练不稳定
   - 典型范围：**0.5 - 2.0**，取决于模型大小和任务

3. **生成吞吐量（Generation Throughput）**
   - 对比启用和禁用推测解码时的 tokens/秒
   - 预期加速比：**1.5x - 3x**，取决于接受率和模型架构

### 接受率趋势解读

| 趋势 | 含义 | 建议操作 |
|---|---|---|
| 稳定在 0.7 以上 | 草稿模型对齐良好 | 无需操作 |
| 逐渐下降 | 策略演化速度快于草稿模型 | 增大 `mtp_loss_scaling_factor` |
| 突然下降 | 可能是学习率突变或数据分布变化 | 检查训练稳定性 |
| 极低（<0.3） | 草稿模型无效 | 验证 MTP 层是否在训练 |

## 故障排除

### 接受率很低

1. **验证 MTP 训练已启用**：检查是否设置了 `actor.enable_mtp_training: true`。
   未启用在线训练时，草稿模型会很快过时。

2. **检查 MTP 层数**：确保 `actor.mtp_num_layers` 与模型架构匹配。Qwen3 模型
   通常有 1 个 MTP 层。

3. **增大 MTP 损失权重**：如果接受率随时间下降，尝试将 `mtp_loss_scaling_factor`
   从 `0.1` 增加到 `0.2` 或 `0.3`。

### 推理阶段显存不足（OOM）

1. **降低显存比例**：将 `sglang.mem_fraction_static` 调低（例如 `0.75`）。

2. **减少草稿 token 数**：将 `speculative_num_draft_tokens` 从 `4` 降至 `2`。

3. **减少草稿步数**：将 `speculative_num_steps` 从 `3` 降至 `2`。

### 训练速度低于预期

1. **检查 GPU 分配**：确保推理和训练 GPU 正确分离。在 4 GPU 上可使用
   `sglang:d2p1t1` 配合 `megatron:d2p1t1` 以实现均衡分配。

2. **分析流水线**：启用 `perf_tracer.enabled: true` 以识别瓶颈是在生成、训练
   还是数据加载阶段。

3. **临时禁用推测解码**：设置 `speculative_algorithm: null` 并对比吞吐量，以
   判断开销是否来自推测本身。

### MTP 损失不下降

1. **验证模型支持 MTP**：并非所有模型架构都包含 MTP 头。检查模型配置中是否包含
   MTP 层定义。

2. **检查学习率**：MTP 头与 actor 共享优化器。如果基础学习率过低，MTP 训练可能
   停滞。

3. **检查梯度流**：确保 `actor.gradient_checkpointing` 未影响 MTP 梯度计算。

## 高级配置

### 使用外部草稿模型

除了依赖内置 MTP 层，您也可以提供独立的草稿模型：

```yaml
sglang:
  speculative_algorithm: "EAGLE"
  speculative_draft_model_path: /path/to/eagle-draft-model
```

注意：使用外部草稿模型时，通常应将 `enable_mtp_training` 设为 `false`，除非外部
模型的权重也在训练中更新。

### EAGLE3 算法

EAGLE3 是一种改进变体，支持更灵活的树形结构推测：

```yaml
sglang:
  speculative_algorithm: "EAGLE3"
  speculative_num_steps: 5
  speculative_eagle_topk: 2
  speculative_num_draft_tokens: 8
```

EAGLE3 通常能达到更高的接受率，但扩展的草稿树会消耗更多显存。

### 草稿权重 CPU 备份

当使用共置的训练和推理模式（即相同 GPU 同时服务两者）时，草稿模型权重可能在 GPU
显存回收时丢失。启用 CPU 备份：

```yaml
sglang:
  enable_draft_weights_cpu_backup: true
```

这会保留草稿权重的 CPU 副本，在每个训练步之后恢复。

## 参考资料

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [SGLang 文档](https://sgl-project.github.io/)
- [AReaL Megatron 后端教程](megatron.md)
- [AReaL 分配模式参考](../reference/alloc_mode.md)
