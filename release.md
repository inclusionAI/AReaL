# AReaL v1.0.0：统一智体 RL、PyTorch 原生引擎、AI 辅助开发三箭齐发，通用为本，原生致简

**本期聚焦：AReaL v1.0.0 正式发布**

经过数月的迭代优化，AReaL 迎来了里程碑式的 v1.0.0 版本。在支持稳定、高效异步强化学习训练的基础上，本次发布带来了**三大核心特性**：

1. **Unified Agentic RL（统一智能体强化学习）**：通过 OpenAI/Anthropic API 代理，任意 Agent 框架只需修改一个
   `base_url` 即可接入训练，真正实现"协议统一，框架无关"，同时保证训推 Token 一致性。

1. **Archon 引擎**：PyTorch 原生 5D 并行训练引擎，支持 DP、TP、PP、CP、EP 全维度并行，`uv sync`
   即装即用，`torch.compile` 默认开启。在 0.5.0 训练万亿参数模型 Ring-1T 的基础上，1.0.0 还能在 **仅仅6 台机器（48 卡
   H200）** 上成功训练 **235B MoE** 模型，并在 Tau2-bench 上取得 **SOTA** 表现。

1. **大规模 AI 辅助编程**：得益于大规模 AI 编程工具的可靠使用，Archon 引擎的开发仅用了 **1 人·月的工作量**。更令人兴奋的是，我们已将这些驾驭AI
   coding"武功秘籍"完全开源，让每位开发者都能借助“专家团队”，在AReaL中加速自己的 Agent RL 应用开发。

# 核心特性一：Unified Agentic RL（统一智能体强化学习）

## 背景与演进

在 v0.5.0 版本中，我们已经实现了 Agent 逻辑与训练逻辑的解耦，通过"Agent 独立运行 + 训练逻辑外置"的设计理念，让 Agent 不感知自身正在被用于训练。

然而，在实际落地过程中，我们仍然发现了两个关键痛点：

1. **框架适配成本高**：虽然训练逻辑已外置，但不同 Agent 框架（LangChain、OpenAI Agents SDK、CAMEL-AI
   等）的接口各异，每接入一个新框架都需要编写适配代码。

1. **Token 一致性难保证**：Agent 框架通过高层 API 与 LLM 交互，无法直接获取 token 级别的信息。如果在训练时重新
   tokenize，可能导致训推 token 不一致，影响算法正确性。

v1.0.0 通过 **Proxy Worker 架构** 彻底解决了这两个问题：任意支持 OpenAI/Anthropic 协议的 Agent 框架，只需修改一个
`base_url` 即可接入训练，同时 token 级信息在推理时直接捕获，完全避免 tokenization mismatch。

## Proxy Worker 架构

v1.0.0 引入了一个关键组件：**Proxy Worker**。它作为 Agent 与推理引擎之间的桥梁，提供标准化的 API 端点，同时完成 token 级轨迹收集。

```
┌─────────────────────────────────────────────────────────────────┐
│                         PPOTrainer                              │
│         (检测 Agent workflow，初始化 Proxy Workers)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RolloutController                            │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │   Rollout    │     │    Proxy     │  FastAPI Server          │
│  │   Worker     │◄────│    Worker    │  /v1/chat/completions    │
│  │              │     │              │  /v1/responses           │
│  │ SGLang/vLLM  │     │              │  /v1/messages            │
│  └──────────────┘     └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    标准 OpenAI/Anthropic API 调用
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
    │ LangChain │      │Claude Code│      │ OpenAI    │
    │   Agent   │      │   Agent   │      │Agents SDK │
    └───────────┘      └───────────┘      └───────────┘
```

Proxy Worker 实现了三套主流 API 协议：

| 端点                                     | 协议               | 用途         |
| ---------------------------------------- | ------------------ | ------------ |
| `POST /{session_id}/v1/chat/completions` | OpenAI Chat        | 标准聊天补全 |
| `POST /{session_id}/v1/responses`        | OpenAI Responses   | 新版响应 API |
| `POST /{session_id}/v1/messages`         | Anthropic Messages | Claude 兼容  |

在启动 Proxy Worker 后，用户可以像平常使用 OpenAI/Anthropic
官方模型一样编写自己的智能体，只要将它接入AReaL的代理地址，你的智能体就可以开始自己的强化学习训练了！

以下是一个典型的 Agent 接入示例。注意，这段代码**完全不依赖 AReaL**——它可以直接用 OpenAI 官方 API 运行，也可以无缝接入 AReaL 训练：

```python
class MyAgent:
    async def run(self, data, **extra_kwargs):
        # AReaL 自动注入 base_url 和 http_client
        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")

        # 标准 OpenAI SDK 用法，无需任何修改
        client = AsyncOpenAI(
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )

        response = await client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        # 返回 reward（float 或 dict）
        return compute_reward(response, data["answer"])
```

## 实现原理

#### 推理数据流

那么，AReaL 是如何在不修改 Agent 代码的情况下，完成 token 级轨迹收集的呢？

核心在于 **四进程协作架构**：Controller 负责任务调度，Rollout Worker 管理 Agent 执行，Proxy Worker 提供 API 代理并缓存
token 信息，GPU Process 执行实际推理。

```
│ Controller Process │  │ Rollout Worker (RPC) │  │ Proxy Worker │  │ GPU Process │
│                    │  │                      │  │              │  │             │
│ RolloutController  │  │  Flask HTTP Server   │  │ FastAPI HTTP │  │ SGLang/vLLM │
│        │           │  │        │             │  │    Server    │  │      │      │
│        ▼           │  │   /call endpoint     │  │ OpenAI API   │  │ Inference   │
│ BatchTaskDispatcher│  │        │             │  │ compatible   │  │   Engine    │
│   (bg thread)      │  │        ▼             │  │      │       │  │      │      │
│        │           │  │   Engine Thread      │  │      │       │  │      │      │
│        │           │  │        │             │  │      │       │  │      │      │
│        │    HTTP   │  │        ▼             │  │      │       │  │      │      │
│ submit ├────POST───┼─>│   RemoteInfEngine    │  │      │       │  │      │      │
│ task 1 │           │  │        │             │  │      │       │  │      │      │
│        │           │  │        ▼             │  │      │       │  │      │      │
│ submit │           │  │ OpenAIProxyWorkflow  │  │      │       │  │      │      │
│ task 2 │           │  │        │             │  │      │       │  │      │      │
│        │           │  │  OpenAIProxyClient ──┼──┼──────┤       │  │      │      │
│ submit │           │  │        │             │  │      │       │  │      │      │
│ task 3 │           │  │   agent.run()        │  │      │       │  │      │      │
│        │           │  │        │             │  │      │       │  │      │      │
│        │           │  │        ▼             │  │      │       │  │      │      │
│        │           │  │   OpenAI API call ───┼──┼─>  /chat/ ───┼──┼─> generate  │
│        │           │  │        │             │  │ completions  │  │    tokens   │
│        │           │  │        │             │  │      │       │  │      │      │
│        │           │  │  ChatCompletion <────┼──┼──────<───────┼──┼──────┘      │
│        │           │  │        │             │  │   (cached)   │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │     reward           │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │   set_reward() ──────┼──┼─>  /rl/      │  │             │
│        │           │  │        │             │  │ set_reward   │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │     ...              │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │    trajectory        │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│    collect<────────┼──┼────────┘             │  │      │       │  │             │
│                    │  │                      │  │              │  │             │
└────────────────────┴──┴──────────────────────┴──┴──────────────┴──┴─────────────┘
```

整体数据流有以下步骤组成：

1. **任务提交**：Controller 将训练任务分发给 Rollout Worker
1. **Agent 执行**：`agent.run()` 被调用，Agent 像往常一样发起 OpenAI API 请求
1. **请求拦截**：Proxy Worker 拦截请求，转发给 GPU 推理引擎
1. **Token 缓存**：推理结果（包含 token IDs 和 logprobs）被缓存在 `InteractionCache` 中
1. **Reward 标注**：Agent 返回 reward 后，通过 `/rl/set_reward` 关联到对应的交互
1. **轨迹导出**：训练时导出完整的 token 级轨迹，包含 input_ids、logprobs、rewards

这种设计的精妙之处在于：**Agent 完全不感知 AReaL 的存在**，它只是在调用一个"看起来像 OpenAI"的 API。而 AReaL 在背后默默完成了所有 RL
训练所需的数据收集工作。

#### 训练数据导出

换存在 `InteractionCache` 中的数据如何被用于训练呢？在实际的 Agentic 场景中，一次任务执行往往包含多轮 LLM
交互，并且这些交互**很可能不是线性的**，例如，一个智能体可能在读代码阶段启动多个子智能体探索代码仓库，这些子智能体和主智能体使用同样的模型，因此探索的过程同样可以被用于训练。在这种模式下，我们不能简单将多个请求按照时间顺序“拼起来”作为一条完整的智能体轨迹。

为了兼容非线性的智能体轨迹，AReaL 采用 **Individual Mode** 将每轮交互**独立导出**为训练样本，并通过折扣因子将最终 reward
反向传播到每一轮：

```
Turn 1: [system, user]           → output_1 → reward = 0.81  (0.9 × 0.9)
Turn 2: [system, user, asst]     → output_2 → reward = 0.9   (0.9 × 1.0)
Turn 3: [system, user, asst, ...] → output_3 → reward = 1.0   (final)
```

这样，早期的决策也能获得合理的信用分配，让模型学会"为长远目标做出正确的早期选择"。

传统方案中，推理时的文本需要在训练时重新 tokenize，可能因 tokenizer 配置差异导致 token 序列不一致。AReaL
从根本上避免了这个问题：**推理时产生的 token IDs 直接被缓存，训练时原样使用**。发送给推理引擎的 tokens 就是用于梯度计算的 tokens，100% 一致。

#### 针对智能体的优化：动态树状注意力机制

在 Agentic RL 场景下，同一个 prompt 可能产生多条不同轨迹（如多次采样），并且每条逻辑上的轨迹也会被 AReaL
打散成为多条独立的输入输出。一个批次的数据之间往往存在**大量共享前缀**。传统训练方式对每条轨迹独立计算，造成大量冗余计算。

AReaL 引入了基于 Trie（前缀树）的序列打包方案：

1. **构建 Trie 结构**：将共享前缀的序列压缩到同一个树结构中
1. **稀疏注意力计算**：通过 FlexAttention / Triton Kernel 实现完整的树状注意力 forward-backward
   方案，让共享前缀仅计算一次

树状注意力带来了显著的性能提升，单 Worker 训练吞吐最高提升 **8.31x**，集群整体吞吐最高提升 **6.20x**，相比于基线方案减少超过了 50%
的GPU显存占用。

具体的测试数据、算法实现，请参考论文：[AREAL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models](https://www.arxiv.org/pdf/2602.00482)

______________________________________________________________________

# 核心特性二：Archon 训练引擎

## 背景与动机

在大规模 RL 训练中，Megatron-LM 是业界标杆级的训练框架。然而，在实际使用过程中，我们遇到了以下痛点：

1. **安装复杂**：需要 Docker 环境，`transformer_engine` `apex` 等组件需要 C++ 编译，环境配置耗时
1. **调试困难**：代码层层嵌套，调用栈深，出现问题时难以定位

与此同时，PyTorch 生态在分布式训练方面已经非常成熟：

- **DTensor**：分布式张量抽象，支持声明式并行
- **DeviceMesh**：灵活的设备拓扑管理
- **torch.distributed.pipelining**：原生流水线并行支持

基于 PyTorch 官方大模型训练参考实现 **torchtitan**，我们开发了 **Archon 引擎**——一个 PyTorch 原生的 5D 并行训练引擎。

## 引擎对比

| 特性            | MegatronEngine | ArchonEngine                     |
| --------------- | -------------- | -------------------------------- |
| 后端            | Megatron-Core  | PyTorch 原生                     |
| 模型来源        | Megatron 模型  | AReaL中自定义 Archon 模型        |
| torch.compile   | 不支持         | **默认开启**                     |
| 数据并行 (DP)   | Megatron DP    | FSDP2                            |
| 张量并行 (TP)   | Megatron TP    | PyTorch DTensor                  |
| 流水线并行 (PP) | 支持 (VPP)     | **支持** (1F1B, Interleaved1F1B) |
| 专家并行 (EP)   | 完整 EP/ETP    | **完整 EP/ETP**                  |
| 上下文并行 (CP) | Megatron CP    | Ulysses SP                       |
| 安装方式        | Docker + 编译  | **`uv sync` 即用**               |

## 5D 并行实现详解

Archon 的核心价值在于：**使用 PyTorch 原生 API 实现完整的 5D 并行**。下面我们逐一介绍每种并行的实现方式。

### 张量并行（Tensor Parallel, TP）

**使用 API**：`torch.distributed.tensor.parallel` 模块

张量并行将模型参数在张量维度上切分到多个 GPU：

- `ColwiseParallel`：列切分，用于 Q/K/V 投影、FFN 的 w1/w3
- `RowwiseParallel`：行切分，用于 wo 输出投影、FFN 的 w2
- `SequenceParallel`：序列维度切分，用于 LayerNorm

**核心代码示例**（来自 `areal/experimental/models/archon/qwen3/infra/parallelize.py`）：

```python
layer_plan = {
    # Attention 层
    "attention.wq": ColwiseParallel(use_local_output=False),  # Q投影列切分
    "attention.wk": ColwiseParallel(use_local_output=False),  # K投影列切分
    "attention.wv": ColwiseParallel(use_local_output=True),   # V投影列切分
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)), # 输出投影行切分

    # FFN 层
    "feed_forward.w1": ColwiseParallel(),   # gate
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),  # down
    "feed_forward.w3": ColwiseParallel(),   # up

    # LayerNorm
    "attention_norm": SequenceParallel(),
    "ffn_norm": SequenceParallel(),
}
parallelize_module(block, tp_mesh, layer_plan)
```

### 上下文并行（Context Parallel, CP）

**实现方式**：Ulysses Sequence Parallelism

**使用 API**：`torch.distributed._functional_collectives.all_to_all_single_autograd`

Ulysses SP 通过 All-to-All 通信实现长序列的分布式处理：

1. **Attention 前**：gather sequence, scatter heads
1. **Attention 计算**：每个 rank 处理完整序列的部分 head
1. **Attention 后**：gather heads, scatter sequence

**核心代码**（来自 `areal/models/fsdp/ulysses.py`）：

```python
# 进入 attention 前
xq = gather_seq_scatter_heads(xq, seq_dim=1, head_dim=2, group=cp_group)
xk = gather_seq_scatter_heads(xk, seq_dim=1, head_dim=2, group=cp_group)
xv = gather_seq_scatter_heads(xv, seq_dim=1, head_dim=2, group=cp_group)

# ... attention 计算 ...

# attention 后
output = gather_heads_scatter_seq(output, head_dim=2, seq_dim=1, group=cp_group)
```

### 流水线并行（Pipeline Parallel, PP）

**使用 API**：`torch.distributed.pipelining` 模块

- `PipelineStage`：流水线阶段封装
- `get_schedule_class`：获取调度策略

**支持的调度策略**：

- `1F1B`：单虚拟阶段，内存效率高
- `Interleaved1F1B`：多虚拟阶段，通信效率更高

**阶段划分**：

- 第一阶段：`tok_embeddings` + 前几层 transformer blocks
- 中间阶段：若干层 transformer blocks
- 最后阶段：后几层 transformer blocks + `norm` + `output`

### 专家并行（Expert Parallel, EP）

**使用 API**：

- `torch.distributed._functional_collectives.all_to_all_single`：Token dispatch/combine
- `torch.distributed.tensor.distribute_tensor`：权重分片
- `torch._grouped_mm`：分组矩阵乘法

**工作流程**（来自 `areal/experimental/models/archon/expert_parallel.py`）：

```python
class ExpertParallel(BaseExpertParallel):
    """Expert Parallelism with ETP=1"""

    def _partition_fn(self, name, module, device_mesh):
        """Shard expert weights on expert dimension (dim 0)"""
        for param_name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(param_name, dist_param)

    def _token_dispatch(self, module, inputs, device_mesh):
        """Dispatch tokens to EP ranks via all-to-all"""
        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()

        # Step 1: Exchange token counts via all-to-all
        num_tokens_per_expert_received = all_to_all_single(
            num_tokens_per_expert, None, None, group=group
        )

        # Step 2: All-to-all to dispatch tokens
        routed_input = all_to_all_single_autograd(
            routed_input, self.output_splits, self.input_splits, group
        )

        return routed_input, aligned_num_tokens

    def _token_combine(self, module, output, device_mesh):
        """Combine expert outputs via all-to-all back"""
        return all_to_all_single_autograd(
            output, self.input_splits, self.output_splits, group
        )
```

**2D 分片（ETP）**：支持 Expert + Tensor 双维度切分

```python
class ExpertTensorParallel(ExpertParallel):
    """Expert Parallelism with Tensor Parallelism (EP + ETP)"""

    def _partition_fn(self, name, module, device_mesh):
        # w1: [num_experts, dim, hidden_dim] -> [Shard(0), Shard(1)]
        module.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(0), Shard(1)]))
        )
        # w2: [num_experts, hidden_dim, dim] -> [Shard(0), Shard(2)]
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(0), Shard(2)]))
        )
```

### 数据并行（Data Parallel, DP）

**使用 API**：`torch.distributed.fsdp.fully_shard`（FSDP2）

```python
def apply_fsdp(model, dp_mesh, param_dtype, reduce_dtype, ...):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # 逐层应用 FSDP
    for transformer_block in model.layers.values():
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=True)

    fully_shard(model, **fsdp_config)
```

### 并行化应用顺序

Archon 按照以下顺序应用各种并行：

```
TP → EP → CP → Activation Checkpointing → torch.compile → FSDP
```

这个顺序确保各种并行策略正确组合，避免冲突。

## 实战案例：48 卡训练 235B MoE

基于 Archon 引擎，我们在 **6 台机器（48 卡 H200）** 上成功训练了 **Qwen3-235B-A22B MoE** 模型，并在
**Tau2-bench** 上取得了 **SOTA** 表现。

### 配置示例

```yaml
# Archon config for Qwen3-235B MoE on 6 nodes (48 GPUs)
# SGLang: 4 replicas × 4 TP = 16 GPUs
# Archon: 1 DP × 4 PP × (attn: TP2×CP2, ffn: TP1×EP4) = 32 GPUs

experiment_name: archon-tau2-grpo
trial_name: trial-0

cluster:
  n_nodes: 6
  n_gpus_per_node: 8

allocation_mode: "sglang:d4t4+archon:(attn:d1p4t2c2|ffn:d1p4t1e4)"

actor:
  path: Qwen/Qwen3-235B-A22B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

详细实验结果请参考论文：**arXiv:2601.22607v1**

______________________________________________________________________

# 核心特性三：AI 辅助编程实践

## 引入：Archon 的开发效率奇迹

前面介绍的 Archon 引擎，实现了完整的 5D 并行（DP、TP、PP、CP、EP），支持 235B 规模的 MoE
模型训练。这样一个复杂的分布式训练引擎，从实现到验证正确性，仅用了 **1 人 1 个多月** 的时间。

这归功于我们在开发过程中大规模、可靠地使用了 AI 编程工具。

更令人兴奋的是，**AReaL 已经将这些使用 AI 工具的"武功秘籍"完全开源**，让每位开发者都能借助同样的方法论加速自己的 Agent RL 应用开发。

## AI 辅助开发配置

AReaL 在 `.claude/` 目录下提供了完整的 AI 辅助开发配置，支持 Claude Code 等 LLM 编程工具：

```
AReaL/
├── CLAUDE.md              # 项目上下文与约束
└── .claude/
    ├── agents/            # 专业化 AI 助手
    │   ├── planner.md
    │   ├── code-verifier.md
    │   ├── fsdp-engine-expert.md
    │   ├── archon-engine-expert.md
    │   ├── megatron-engine-expert.md
    │   ├── algorithm-expert.md
    │   └── launcher-scheduler-expert.md
    ├── skills/            # 引导式开发工作流
    │   ├── add-dataset/
    │   ├── add-workflow/
    │   ├── add-reward/
    │   ├── add-unit-tests/
    │   └── debug-distributed/
    ├── commands/          # 自动化操作
    │   ├── create-pr.md
    │   ├── gen-commit-msg.md
    │   └── pr-review.md
    └── rules/             # 代码质量标准
        ├── api-config.md
        ├── code-style.md
        ├── distributed.md
        └── testing.md
```

## 三类开发工具

### Agents（智能助手）

Agents 是专业化的 AI 助手，根据上下文自动激活，提供领域专业知识：

**通用 Agents**：

| Agent                  | 专长领域           | 激活时机                           |
| ---------------------- | ------------------ | ---------------------------------- |
| `planner`              | 复杂任务实现规划   | 多文件修改、新功能开发、架构决策前 |
| `code-verifier`        | 代码格式检查与测试 | 代码修改后，commit 前              |
| `simple-code-reviewer` | 快速代码质量检查   | 代码修改后，commit 前              |

**领域专家 Agents**：

| Agent                       | 专长领域             | 激活时机                          |
| --------------------------- | -------------------- | --------------------------------- |
| `fsdp-engine-expert`        | FSDP2 配置与内存优化 | FSDPEngine 代码修改或问题咨询     |
| `archon-engine-expert`      | MoE 训练与专家并行   | ArchonEngine 代码修改或问题咨询   |
| `megatron-engine-expert`    | 流水线并行与大模型   | MegatronEngine 代码修改或问题咨询 |
| `algorithm-expert`          | GRPO/PPO/DAPO 算法   | RL 算法相关问题                   |
| `launcher-scheduler-expert` | Slurm/Ray/K8s 配置   | 集群配置与调度问题                |

### Skills（引导式工作流）

Skills 提供分步骤的引导式开发工作流，确保遵循 AReaL 的约定：

| 技能                 | 用途                                       | 触发方式                   |
| -------------------- | ------------------------------------------ | -------------------------- |
| `/add-dataset`       | 添加新数据集加载器到 `areal/dataset/`      | 用户请求添加数据集时       |
| `/add-workflow`      | 创建新 RolloutWorkflow 实现                | 用户请求创建新 workflow 时 |
| `/add-reward`        | 实现新奖励函数                             | 用户请求添加奖励函数时     |
| `/add-unit-tests`    | 为新功能添加单元测试                       | 用户请求添加测试时         |
| `/debug-distributed` | 排查分布式训练问题（hang、OOM、NCCL 错误） | 遇到分布式训练问题时       |

每个 Skill 都会引导完成整个流程：文件创建、注册、测试，以及常见陷阱的规避。

### Commands（自动化命令）

Commands 是自动化工作流，通过 `/` 前缀调用：

| 命令              | 功能                                  | 使用场景          |
| ----------------- | ------------------------------------- | ----------------- |
| `/create-pr`      | 自动 rebase、squash commits 并创建 PR | 开发完成后提交 PR |
| `/gen-commit-msg` | 从 staged changes 生成 commit 信息    | 准备 commit 时    |
| `/pr-review`      | 智能代码审查与风险分析                | Review PR 时      |

`/pr-review` 特别强大：它使用动态模板分析 PR 变更，检测风险等级（CRITICAL/HIGH/MEDIUM/LOW），并按需启动针对性的子 Agent
进行审查。

## 典型开发流程

```
用户: 我想添加一个新的代码执行奖励函数

Claude: [planner agent 激活，创建实现计划]
        让我分析一下这个任务...
        计划如下：
        1. 创建 areal/reward/code_execution.py
        2. 实现 CodeExecutionReward 类
        3. 在 __init__.py 注册
        4. 添加单元测试

用户: /add-reward code_execution

Claude: [引导创建奖励函数]
        步骤 1/4: 创建文件结构...
        步骤 2/4: 实现奖励计算逻辑...
        步骤 3/4: 注册到模块导出...
        步骤 4/4: 生成测试用例...

用户: 帮我验证一下修改

Claude: [code-verifier agent 激活]
        运行 pre-commit...
        运行测试...
        ✓ 所有检查通过

用户: /create-pr

Claude: [执行 create-pr command]
        Rebasing from origin/main...
        Squashing commits...
        Creating PR...
        ✓ PR created: https://github.com/inclusionAI/AReaL/pull/xxx
```

## 赋能 Agent RL 开发

这套 AI 辅助开发配置不仅用于 AReaL 自身的开发，**用户也可以直接使用这些配置来加速自己的 Agent RL 应用开发**：

- 使用 `/add-workflow` 快速创建自定义 Agent 工作流
- 使用 `/add-reward` 实现特定领域的奖励函数
- 使用 `/debug-distributed` 排查分布式训练问题
- 借助领域专家 Agent（如 `algorithm-expert`）获取 GRPO/PPO/DAPO 算法指导

## 阶段性 Agent 使用建议

我们建议按以下阶段使用不同的 Agent：

1. **规划阶段**（编码前）：使用 `planner` 进行架构设计和实现规划
1. **格式检查阶段**（编码后）：使用 `code-verifier` 自动运行 formatting、linting 和测试，快速捕获语法错误和风格问题
1. **质量检查阶段**（格式化后）：使用 `simple-code-reviewer` 进行快速代码质量检查，关注逻辑问题和代码异味

______________________________________________________________________

# 总结与展望

## v1.0.0 三大核心价值

1. **Unified Agentic RL**：一个 `base_url` 连接万千 Agent 框架，在保障训推 Token 一致、减轻前端开发负担的同时，利用 Tree
   Attention 优化带来最高 8x 吞吐提升

1. **Archon 引擎**：PyTorch 原生 5D 并行，`uv sync` 即用，`torch.compile` 默认开启，48 卡训练 235B MoE 达
   SOTA

1. **AI 辅助开发**：开源"武功秘籍"，1 人月打造复杂分布式引擎的实践方法论

## 未来方向

- Archon 引擎生产级稳定性完善
- 更多 Agent 框架的开箱即用示例
- AI 辅助开发配置的持续优化

______________________________________________________________________

# 资源链接

- **GitHub**: https://github.com/inclusionAI/AReaL
- **Tree Attention 论文**: https://www.arxiv.org/abs/2602.00482
- **Tau2 训练论文**: https://www.alphaxiv.org/overview/2601.22607v1
- **AI 辅助开发文档**: `docs/reference/ai_assisted_dev.md`

______________________________________________________________________
