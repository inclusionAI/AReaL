# SGLang 流水线并行（PP）支持

本文档介绍 AReaL 在 SGLang 推理后端中对流水线并行（Pipeline Parallelism, PP）的支持。PP
将模型的各层拆分到多个 GPU 上依次执行，使得单卡显存不足以容纳的大模型也能进行推理。

## 概述

流水线并行将模型层划分为若干顺序阶段（stage），每个阶段分配到独立的 GPU 上。推理时，micro-batch
依次流经各个阶段完成前向计算。AReaL 在已有的数据并行（DP）和张量并行（TP）基础上，扩展了
SGLang 集成以支持 PP 维度。

在 RL 训练循环中引入 PP 的核心挑战在于**权重同步**：每个训练步结束后，需要将 Megatron
训练后端更新的模型权重传输到 SGLang 推理服务器。启用 PP 后，每个流水线阶段仅持有部分层的参数，
因此权重更新必须按阶段进行协调。

AReaL 通过**逐 PP-rank NCCL 通信组**解决这一问题——每个流水线阶段在对应的训练 rank
和该阶段的所有推理 worker 之间建立独立的 NCCL 组。

## 架构：逐 PP-Rank NCCL 通信组

### 无 PP 时（PP=1）

未使用 PP 时，单个 NCCL 组连接训练 rank 和所有推理 worker：

```
训练 Rank 0                         SGLang Worker
(全部层)                            (全部层)
      │                                  │
      └──── NCCL 通信组 ────────────────┘
            (所有参数)
```

### 启用 PP 时（以 PP=2 为例）

当 PP>1 时，每个训练 PP rank 创建独立的 NCCL 组，仅持有相同流水线阶段的 SGLang worker
加入该组：

```
训练 PP Rank 0                      SGLang PP Rank 0
(第 0..N/2 层)                      (第 0..N/2 层)
      │                                  │
      └── NCCL 组 0 ────────────────────┘
          (group_name: "update_weight_group_0")

训练 PP Rank 1                      SGLang PP Rank 1
(第 N/2..N 层)                      (第 N/2..N 层)
      │                                  │
      └── NCCL 组 1 ────────────────────┘
          (group_name: "update_weight_group_1")
```

每个逐 PP-rank 通信组的属性如下：

- **World size** = `n_servers * tp_size + 1`（该 PP rank 下所有 DP 实例的全部 TP worker，
  加上一个训练 rank）
- **Rank offset** 仅基于 `tp_size` 计算（而非 `tp_size * pp_size`），因为每个组仅包含
  同一 PP rank 的 worker
- 初始化请求中包含 `pp_rank` 字段，使 SGLang 能够判断哪些 worker 应加入该组

## 工作流程

下图展示了启用 PP 后端到端的执行流程：

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                          训练步骤                                    │
 │                                                                     │
 │  1. 推理采样阶段                                                     │
 │     Controller ──> SGLang 服务器生成轨迹                              │
 │     (各 PP 阶段在每个服务器内依次处理)                                  │
 │                                                                     │
 │  2. 训练阶段                                                         │
 │     Controller ──> Megatron worker（PP 流水线调度）                    │
 │     前向：Stage 0 → Stage 1 → ... → Stage K                         │
 │     反向：Stage K → ... → Stage 1 → Stage 0                         │
 │     各阶段执行优化器更新                                               │
 │                                                                     │
 │  3. 权重同步阶段                                                      │
 │     a. 暂停 SGLang 服务器                                             │
 │     b. 对每个 PP rank r = 0, 1, ..., K：                              │
 │        训练 PP rank r ──NCCL broadcast──> SGLang PP rank r            │
 │        (组名: "update_weight_group_{r}")                              │
 │     c. 恢复 SGLang 服务器                                             │
 │                                                                     │
 │  4. 版本更新                                                          │
 │     在训练端和推理端同步更新模型版本号                                    │
 └─────────────────────────────────────────────────────────────────────┘
```

### 权重更新组初始化

在初始化阶段，AReaL 对每个 SGLang 服务器的每个 PP rank 调用 `init_weights_update_group`。
请求负载包含以下字段：

| 字段             | 说明                                                |
| ---------------- | --------------------------------------------------- |
| `master_address` | 训练端的 NCCL master 地址                            |
| `master_port`    | NCCL master 端口                                    |
| `rank_offset`    | `1 + server_idx * tp_size`                          |
| `world_size`     | `n_servers * tp_size + 1`                           |
| `group_name`     | `"update_weight_group_{pp_rank}"`                   |
| `pp_rank`        | 流水线阶段索引（仅在 PP>1 时包含）                    |

### 权重传输

每次权重同步时，训练端逐层广播参数。每个训练 PP rank 仅广播其持有的层，并使用对应的 NCCL
组完成传输。此过程由 `RolloutController` 中已有的 `update_weights_from_distributed`
路径透明地处理。

## 配置指南

### Backend 字符串格式

流水线并行通过 backend 字符串中的 `p` 维度指定：

```
sglang:d<DP>p<PP>t<TP>
megatron:d<DP>p<PP>t<TP>
```

每个引擎的总 GPU 数量为 `DP * PP * TP`。

### 配置示例

#### 8 卡：DP=2, PP=2, TP=1

```yaml
rollout:
  backend: "sglang:d2p2t1"    # 2 × 2 × 1 = 4 GPU

actor:
  backend: "megatron:d2p2t1"   # 2 × 2 × 1 = 4 GPU
```

两个 SGLang 服务器实例，每个占用 2 张 GPU（2 个流水线阶段）。四个 Megatron worker
组成 2 个数据并行组，每组包含 2 级流水线。

#### 8 卡：DP=1, PP=2, TP=2

```yaml
rollout:
  backend: "sglang:d1p2t2"    # 1 × 2 × 2 = 4 GPU

actor:
  backend: "megatron:d1p2t2"   # 1 × 2 × 2 = 4 GPU
```

一个 SGLang 服务器实例占用 4 张 GPU（2 个流水线阶段，每阶段 2 路张量并行）。
Megatron 端使用相同的并行布局。

#### 16 卡：DP=2, PP=2, TP=2

```yaml
cluster:
  n_nodes: 1
  n_gpus_per_node: 16

rollout:
  backend: "sglang:d2p2t2"    # 2 × 2 × 2 = 8 GPU

actor:
  backend: "megatron:d2p2t2"   # 2 × 2 × 2 = 8 GPU
```

### 已验证配置矩阵

下表列出了经过测试的 DP、PP、TP 组合。rollout 和 actor 的 PP、TP 值必须一致，
以确保权重更新组正确对齐。

| DP  | PP  | TP  | 单引擎 GPU 数 | 说明                      |
| --- | --- | --- | ------------- | ------------------------- |
| 1   | 2   | 1   | 2             | 最小 PP 配置               |
| 2   | 2   | 1   | 4             | PP + 数据并行              |
| 1   | 2   | 2   | 4             | PP + 张量并行              |
| 2   | 2   | 2   | 8             | 三维并行                   |
| 1   | 4   | 1   | 4             | 更深的流水线               |
| 2   | 4   | 2   | 16            | 大规模配置                 |

> **约束**：`rollout.backend` 中的 PP 和 TP 值必须与 `actor.backend` 一致。
> DP 值可以不同，但通常保持相等。

## 前置条件

### SGLang PP 补丁

SGLang 的流水线并行权重更新组功能需要使用包含 PP 补丁的 SGLang 版本。该补丁使 SGLang
worker 具备以下能力：

1. 在 `/init_weights_update_group` 接口中接受 `pp_rank` 参数
2. 根据自身 PP rank 过滤需要加入的 NCCL 组
3. 在 `/update_weights` 调用时处理逐 PP-rank 的权重更新

请确保使用支持 PP 的 SGLang 版本。可通过以下命令检查：

```bash
# 检查 SGLang 版本
python -c "import sglang; print(sglang.__version__)"
```

具体兼容版本请参阅 AReaL 安装指南。

### Megatron-LM

Megatron 训练后端需要配置匹配的流水线并行参数。AReaL 的 Megatron 集成原生支持
PP，训练端无需额外补丁。

## 启用 PP 运行

### 使用示例配置

```bash
# 本地启动（8 卡）
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron_pp.yaml \
    scheduler.type=local

# Ray 集群
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron_pp.yaml \
    scheduler.type=ray

# Slurm 集群
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron_pp.yaml \
    scheduler.type=slurm
```

### 命令行覆盖 PP 配置

可以通过命令行参数在任何已有配置上启用 PP：

```bash
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron.yaml \
    rollout.backend="sglang:d2p2t1" \
    actor.backend="megatron:d2p2t1" \
    scheduler.type=local
```

### 运行测试配置

用于 CI 或快速验证：

```bash
python examples/math/gsm8k_rl.py \
    --config examples/test/gsm8k_grpo_megatron_pp_test.yaml \
    scheduler.type=local
```

该配置使用 Qwen3-0.6B 小模型，配合较小的 batch size 和序列长度以加速执行。

## 常见问题排查

### NCCL 组初始化失败

**现象**：`init_weights_update_group` 阶段超时或报错。

**可能原因**：

- SGLang 版本不支持 `pp_rank` 参数，请确认使用了兼容 PP 的构建版本。
- 训练进程与推理 GPU 进程之间网络不通，请检查所有 GPU 之间的 NCCL 通信是否正常。
- 防火墙阻断了 NCCL master 端口，请确认 `master_port` 可访问。

**排查步骤**：

```bash
# 开启 NCCL 调试日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 权重更新挂起

**现象**：第一个训练步后，在权重同步阶段卡住。

**可能原因**：

- rollout 和 actor 的 PP 值不匹配，两者必须使用相同的 PP 值。
- world size 计算错误，每个逐 PP-rank 组预期恰好有 `n_servers * tp_size + 1` 个参与者。

**排查步骤**：

1. 确认 `rollout.backend` 和 `actor.backend` 中 `p` 值一致。
2. 检查日志中 NCCL 组名和 world size 是否正确。
3. 确保所有 SGLang 服务器进程健康（`/health` 接口可正常访问）。

### 显存不足（OOM）

**现象**：推理或训练阶段出现 CUDA OOM 错误。

**可能原因**：

- `mem_fraction_static` 设置过高，PP 分摊后单卡可用显存不足。
- 上下文长度过大，超出了 PP 分层后单卡的承载能力。

**解决方案**：

- 降低 `sglang.mem_fraction_static`（例如从 0.8 降至 0.7）。
- 减小 `sglang.context_length`。
- 增大 PP 值，将层分布到更多 GPU 上。

### 性能注意事项

流水线并行会引入流水线气泡（pipeline bubble）——部分阶段在等待 micro-batch
时处于空闲状态。对于推理场景，这通常是可接受的，但需注意以下几点：

- **延迟**：与纯 TP 配置相比，PP 由于顺序执行各阶段会增加单请求延迟。
- **吞吐量**：在足够的请求并发度（持续批处理）下，PP 仍可获得较高吞吐。
- **显存**：PP 降低了单卡显存占用，可支持更大的模型或更长的上下文。

在 PP 和 TP 之间选择时，若模型能放入 TP GPU 的聚合显存中，优先选择 TP。当模型所需
GPU 数量超出纯 TP 所能提供的范围，或跨节点通信带宽更适合 PP 的点对点通信模式时，使用 PP。

## 参阅

- [分配模式](alloc_mode.md) - Backend 字符串与并行维度的完整参考
- [在 GSM8K 上运行 GRPO](../tutorial/gsm8k_grpo.md) - 基础 GRPO 工作流教程
- [使用 Megatron 训练大模型](../tutorial/megatron.md) - Megatron 后端教程
