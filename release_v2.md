## AReaL v1.0.0：统一智能体强化学习训练，通用为本，原生致简

经过近一年的迭代打磨，AReaL 迎来了里程碑式的 **v1.0.0 稳定版本**。这不仅是一个版本号的跃迁，更标志着 AReaL 从"能用"走向"好用"的成熟蜕变。

AReaL 的愿景很简单：**让每个人都能轻松、低成本地训练自己的 AI Agent**。

在这个愿景的驱动下，我们构建了一套面向大规模推理与智能体模型的全异步强化学习训练系统。它不是又一个"能跑通 demo"的框架，而是一个经过万亿参数模型实战检验、支撑多个
SOTA 成果的生产级系统。

## 一、AReaL v1.0.0 解决什么问题

在深入功能细节之前，让我们先理解 AReaL 试图解决的核心痛点。

**痛点一：传统 RL 框架在 Agent 场景下效率不足**

主流的强化学习框架大多将 Agent 实现耦合在框架中，采用批量训练模式：Agent
生成轨迹，等待所有模型生成和工具调用完毕，然后统一训练，再生成下一批轨迹。这种交错生成模式的问题在 Agent
场景下尤为突出——当你的推理集群有数百张卡时，一个长尾样本就能让整个系统空转等待。

**痛点二：Agent 框架接入成本高，缺乏统一标准**

将一个 Agent 框架接入 RL 训练却往往需要大量适配工作。LangChain、OpenAI Agents SDK、CAMEL-AI、Claude
Code……每个框架的接口都不一样，每接入一个新框架就要写一套新的胶水代码。更麻烦的是，Agent 框架通过高层 API 与 LLM 交互，无法直接获取 token
级别的信息，导致训推 token 不一致的隐患。

**痛点三：框架代码量膨胀，人类维护成本攀升**

随着功能不断迭代，RL 框架的代码量不可避免地膨胀。AReaL
目前已有数万行代码，涵盖分布式训练、推理引擎、算法实现、调度器等多个模块。对于新加入的开发者，理解这些代码需要数周甚至数月；对于核心维护者，每一次重构都是牵一发而动全身的挑战。

______________________________________________________________________

## 二、AReaL 能支持什么

理解了 AReaL 要解决的问题，让我们概览它的核心能力。这一节将简要介绍各项功能，为后续的深入展开做铺垫。

**全异步分离式强化学习架构**

AReaL 的核心架构创新在于训推完全解耦。推理引擎流式生成轨迹，训练引擎持续消费样本，两者完全解耦、并行运行。通过精心设计的 off-policyness
控制机制，我们在保证训练稳定性的同时，实现了相比同步方案 **2倍以上** 的吞吐提升。

这种设计让系统组件更模块化、功能化——例如，你可以独立扩展推理或训练资源，而无需重启整个实验。

**通过统一协议支持任意Agent框架接入强化学习训练**

AReaL 不去适配每一个框架，也不创造一个专门用于强化学习的新 Agent 框架，而是将"协议"作为统一标准。

任何支持 OpenAI/Anthropic API 协议的 Agent 框架，只需修改一个 `base_url` 即可接入训练。Token 级信息在推理时直接捕获，从根本上避免
tokenization mismatch，并通过训练时树状注意力机制提升训练性能。

**下一代 Infra 范式：AI 辅助编程工具箱**

当框架代码量膨胀到一定程度，传统的文档和示例已经不足以帮助开发者快速上手。AI 理解代码的速度远超人类——一个训练有素的 AI
编程助手可以在几秒内扫描整个代码库，理解模块间的依赖关系，定位关键实现细节。这让我们意识到：**人类的核心价值应该聚焦于"做什么"（需求定义、架构决策），而"怎么做"（代码实现、调试排错）可以更多地交给
AI**。

我们构建了一套 AI 辅助开发体系：领域专家 Agents
理解每个模块的架构和约定，引导式工作流带领开发者完成常见任务，自动化命令处理重复性工作。开发者只需描述"想做什么"，AI 负责"怎么做"。

**多后端、多硬件支持**

在训练侧，AReaL 提供三种引擎选择：Megatron 引擎在大规模 MoE 训练上久经考验、功能完备；FSDP 引擎基于 HuggingFace 模型、兼容性最广（如
VLM）；Archon 引擎 PyTorch 原生、安装最简。在推理侧，SGLang 或 vLLM，不管你熟悉哪一个都可以快速切换。

算法性能上，AReaL v1.0.0 能够用 48 张 H200 卡训练 235B 参数的 Qwen3-MoE 模型，并在 Tau²-bench 上取得了超越 Gemini
3.0 Pro 和 GPT 5.2 的 SOTA 表现\[1\]。

硬件方面，除了 NVIDIA GPU，我们也支持 Ascend NPU，相关代码在 `ascend` 分支持续维护。

______________________________________________________________________

## 三、v1.0.0 核心特性深度解析

前文概述了 AReaL 要解决的三大痛点以及对应的解决方案。接下来，让我们深入探索 v1.0.0 版本如何将这些理念落地为具体的技术实现。

如果说 v0.5.0 的核心是"解耦"——让 Agent 与训练逻辑分离，那么 v1.0.0 的核心则是\*\*"统一"与"原生"\*\*：

- **统一**：通过标准 API 代理，将千差万别的 Agent 框架统一到同一套训练范式中
- **原生**：用 PyTorch 原生 API 构建分布式训练引擎，告别 Megatron 的层层依赖

更令人兴奋的是，我们在构建这些能力的过程中，验证了**大规模 AI 辅助编程**的可行性，并将这套方法论完全开源。

### 3.1 统一智能体强化学习

#### 背景与演进

在 v0.5.0 版本中，我们已经实现了 Agent 逻辑与训练逻辑的解耦，通过"Agent 独立运行 + 训练逻辑外置"的设计理念，让 Agent 不感知自身正在被用于训练。

然而，在实际落地过程中，我们仍然发现了两个关键痛点：

1. **框架适配成本高**：虽然训练逻辑已外置，但不同 Agent 框架（LangChain、OpenAI Agents SDK、CAMEL-AI
   等）的接口各异，每接入一个新框架都需要编写适配代码。

1. **Token 一致性难保证**：Agent 框架通过高层 API 与 LLM 交互，无法直接获取 token 级别的信息。如果在训练时重新
   tokenize，可能导致训推 token 不一致，影响算法正确性。

v1.0.0 通过 **Proxy Worker 架构** 彻底解决了这两个问题：**我们将"协议"而非"框架"作为统一标准**，任意支持 OpenAI/Anthropic
协议的 Agent 框架，只需修改一个 `base_url` 即可接入训练，同时 token 级信息在推理时直接捕获，完全避免 tokenization mismatch。

#### Proxy Worker 架构

v1.0.0 引入了一个关键组件：**Proxy Worker**。它作为 Agent 与推理引擎之间的桥梁，提供标准化的 API 端点，同时完成 token 级轨迹收集。

![Proxy Worker 架构](assets/fig-1.png)

Proxy Worker 实现了三套主流 API 协议：

| 端点                                     | 协议               | 用途         |
| ---------------------------------------- | ------------------ | ------------ |
| `POST /{session_id}/v1/chat/completions` | OpenAI Chat        | 标准聊天补全 |
| `POST /{session_id}/v1/responses`        | OpenAI Responses   | 新版响应 API |
| `POST /{session_id}/v1/messages`         | Anthropic Messages | Claude 兼容  |

在启动 Proxy Worker 后，用户可以像平常使用 OpenAI/Anthropic 官方模型一样编写自己的智能体，只要将它接入 AReaL
的代理地址，你的智能体就可以开始自己的强化学习训练了！

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

我们也准备了其他案例，例如[langchain](https://github.com/inclusionAI/AReaL/tree/main/areal/workflow/langchain),
[openai agent sdk](https://github.com/inclusionAI/AReaL/blob/main/areal/workflow/openai/math_agent.py)，**甚至是训练你自己的[claude-code](https://github.com/inclusionAI/AReaL/blob/main/examples/agent_workflow/config_claude.yaml)**。

#### 实现原理

那么，AReaL 是如何在不修改 Agent 代码的情况下，完成 token 级轨迹收集的呢？

核心在于 **四进程协作架构**：Controller 负责任务调度，Rollout Worker 管理 Agent 执行，Proxy Worker 提供 API 代理并缓存
token 信息，GPU Process 执行实际推理。

![四进程数据流架构](assets/fig-2.png)

整体数据流由以下步骤组成：

1. **任务提交**：Controller 将训练任务分发给 Rollout Worker
1. **Agent 执行**：`agent.run()` 被调用，Agent 像往常一样发起 OpenAI API 请求
1. **请求拦截**：Proxy Worker 拦截请求，转发给 GPU 推理引擎
1. **Token 缓存**：推理结果（包含 token IDs 和 logprobs）被缓存在 `InteractionCache` 中
1. **Reward 标注**：Agent 返回 reward 后，通过 `/rl/set_reward` 关联到对应的交互
1. **轨迹导出**：训练时导出完整的 token 级轨迹，包含 input_ids、logprobs、rewards

这种设计的精妙之处在于：**Agent 完全不感知 AReaL 的存在**，它只是在调用一个"看起来像 OpenAI"的 API。而 AReaL 在背后默默完成了所有 RL
训练所需的数据收集工作。

缓存在 `InteractionCache` 中的数据如何被用于训练呢？在实际的 Agentic 场景中，一次任务执行往往包含多轮 LLM
交互，并且这些交互**很可能不是线性的**。例如，一个智能体可能在读代码阶段启动多个子智能体探索代码仓库，这些子智能体和主智能体使用同样的模型，因此探索的过程同样可以被用于训练。在这种模式下，我们不能简单地将多个请求按照时间顺序"拼起来"作为一条完整的智能体轨迹。

为了兼容非线性的智能体轨迹，AReaL 采用 **Individual Mode** 将每轮交互**独立导出为训练样本**，并通过折扣因子将最终 reward
反向传播到每一轮：

```
Turn 1: [system, user]           → output_1 → reward = 0.81  (0.9 × 0.9)
Turn 2: [system, user, asst]     → output_2 → reward = 0.9   (0.9 × 1.0)
Turn 3: [system, user, asst, ...] → output_3 → reward = 1.0   (final)
```

这样，早期的决策也能获得合理的信用分配，让模型学会"为长远目标做出正确的早期选择"。

传统方案中，推理时的文本需要在训练时重新 tokenize，可能因 tokenizer 配置差异导致 token 序列不一致。AReaL
从根本上避免了这个问题：**推理时产生的 token IDs 直接被缓存，训练时原样使用**。发送给推理引擎的 tokens 就是用于梯度计算的 tokens，100% 一致。

然而，同一个 prompt 可能产生多条不同轨迹（如多次采样），并且每条逻辑上的轨迹也会被 AReaL
打散成为多条独立的输入输出。一个批次的数据之间往往存在**大量共享前缀**。传统训练方式对每条轨迹独立计算，造成大量冗余计算。

AReaL 还在此之上，引入了基于 Trie（前缀树）的序列打包方案：

1. **构建 Trie 结构**：将共享前缀的序列压缩到同一个树结构中
1. **稀疏注意力计算**：通过 FlexAttention / Triton Kernel 实现完整的树状注意力 forward-backward
   方案，让共享前缀仅计算一次

树状注意力带来了显著的性能提升：单 Worker 训练吞吐最高提升 **8.31x**，集群整体吞吐最高提升 **6.20x**，相比于基线方案减少超过 50% 的 GPU
显存占用。

具体的测试数据和算法实现，请参考论文：[AREAL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models](https://www.arxiv.org/pdf/2602.00482)

______________________________________________________________________

### 3.2 大规模 AI 辅助编程

#### Archon 引擎的诞生

在大规模 RL 训练领域，Megatron-LM 是业界标杆。然而，它的安装需要 Docker 环境和繁琐的 C++
编译，代码层层嵌套，难以调试和扩展。我们一直在思考：**能否用 PyTorch 原生 API 实现同等能力的分布式训练引擎？**

答案是 **Archon 引擎**——一个支持完整 5D 并行（DP、TP、PP、CP、EP）的 PyTorch 原生训练引擎。

令人惊讶的是，这样一个复杂的分布式系统，从零开始实现到验证正确性，**仅用了 1 人·月的工作量**。

这背后的秘密是 AReaL 中**大规模、可靠地使用 AI 编程专家**。

更令人兴奋的是，我们也将这些驾驭 AI coding 的"武功秘籍"**完全开源**，让每位开发者都能借助"专业团队"，在 AReaL 中加速自己的 Agent RL
应用开发。

#### Archon 引擎：AI 辅助编程的硬核成果

**为什么需要 Archon？**

| 痛点     | Megatron-LM                           | Archon 解决方案      |
| -------- | ------------------------------------- | -------------------- |
| 安装复杂 | Docker + transformer_engine/apex 编译 | **只需要安装 torch** |
| 调试困难 | 调用栈深，难以定位                    | PyTorch 原生，栈清晰 |
| 编译优化 | 不支持 torch.compile                  | **默认开启**         |

**5D 并行能力一览。**

Archon 使用 PyTorch 原生 API 实现了与 Megatron 同等的并行能力：

- **数据并行 (DP)**：基于 FSDP2 `fully_shard`，相比 Megatron 默认的数据并行进一步拆分了模型参数
- **流水线并行 (PP)**：基于 `torch.distributed.pipelining`，支持 1F1B 和 Interleaved1F1B 调度
- **张量并行 (TP)**：基于 DTensor，使用 `ColwiseParallel` / `RowwiseParallel` 切分权重
- **上下文并行 (CP)**：基于 Ulysses Sequence Parallelism，通过 all-to-all 分布式处理长序列
- **专家并行 (EP)**：基于 all-to-all + `grouped_mm`，支持 EP + ETP 2D 分片

#### AI 辅助开发的"武功秘籍"

Archon 引擎能在 1 人·月内完成，得益于我们在 AReaL 中构建的一套完整的 AI 辅助开发体系。这套体系现已完全开源，包含三类核心工具：

**领域专家 Agents。**

AI 编程最大的挑战是**让 AI 理解你的代码库**。我们为 AReaL 的每个核心模块配备了专业化的 AI 助手：

![AReaL AI 专家团队](assets/fig-3.png)

每个专家 Agent 都内置了对应模块的架构知识、代码约定和常见陷阱。当你修改相关代码时，对应的专家会自动激活，提供精准指导。

**从"写代码"到"说需求"。**

除了领域专家，我们还为常见的开发任务设计了**引导式工作流**。想添加一个新的数据集加载器？只需说一句 `/add-dataset`，AI
会引导你完成文件创建、接口实现、模块注册、测试生成的全流程。想实现一个奖励函数？`/add-reward` 会确保你遵循 AReaL 的约定，避开常见的坑。遇到分布式训练
hang 住或 OOM？`/debug-distributed` 提供系统化的排查路径，从 NCCL 超时到显存泄漏，逐一定位。

日常开发中那些重复性的工作，同样可以交给 AI 来完成。`/create-pr` 会自动帮你 rebase 最新代码、squash 零散的 commits、生成清晰的 PR
描述；`/gen-commit-msg` 能从你 staged 的改动中智能提炼出 commit 信息；`/pr-review` 则提供多维度的代码审查，自动识别潜在风险。

**一个真实的开发场景。**

假设你想为 AReaL 添加一个代码执行奖励函数。在传统的开发模式下，你需要先阅读现有奖励函数的实现，理解接口约定，手动创建文件，实现逻辑，注册到模块，编写测试，最后提交
PR。这个过程可能需要来回翻阅文档、查看示例代码、处理各种细节。

而在 AI 辅助开发模式下，你只需要告诉 AI"我想添加一个代码执行奖励函数"。**planner agent** 会自动分析任务，制定实现计划。然后你调用
`/add-reward code_execution`，AI 会引导你逐步完成文件结构创建、奖励逻辑实现、模块注册和测试生成。完成后，**code-verifier
agent** 自动运行格式检查和测试。最后一句 `/create-pr`，一个规范的 Pull Request 就创建好了。

**整个过程，你专注于"做什么"，AI 负责"怎么做"。**

**你也可以用起来。**

这套 AI 辅助开发配置不仅用于 AReaL 自身的开发，**你也可以直接使用它来加速自己的 Agent RL 应用开发**。想快速搭建一个自定义 Agent
工作流？`/add-workflow` 可以帮你。需要实现特定领域的奖励函数？`/add-reward`
遵循最佳实践。遇到分布式训练的疑难杂症？`/debug-distributed` 提供系统化的排查思路。对 GRPO、PPO、DAPO
等算法有疑问？`algorithm-expert` 随时待命。

所有配置文件都在 `.claude/` 目录下，欢迎参考和复用。

______________________________________________________________________

## 四、未来展望

回顾 AReaL 的发展历程，我们看到了一条清晰的从"解耦"到"统一"演进路线。

在 1.0.0 版本之前，AReaL 的核心理念是"解耦"——不论是训推分离、Agent 逻辑分离，还是 controller-worker
分离，都让每个系统组件变得更模块化、功能化和易维护。

而从 1.0.0 版本开始，AReaL 希望基于完善的子模块，在应用层外拓，在引擎层内修，既能容纳万千生态，又能追求极致性能，走在 Agentic RL
框架易用性、可靠性和扩展性的山脊之上。

AReaL 的愿景是成为 **Agentic AI 时代的高性能 RL 运行时服务**。展望未来，我们将持续推进：

- **Archon 引擎生产级效率优化**：更完善的功能支持（如 LoRA）、更高效的通信和并行优化
- **可用性提升**：让 AReaL 中的系统组件灵活拼接、即插即用，不再需要复杂的配置
- **VLM/Omni 模型 Agent 训练**：集成更多模态 Agentic 的强化学习训练
- **AI 辅助开发能力持续进化**：让 AI 专家团队更懂 AReaL，也更懂你的需求

我们坚信，**当训练框架足够简单、当 Agent 接入足够统一、当 AI 能够辅助系统开发**，Agentic RL 的大规模落地将不再是少数团队的专属能力。

欢迎每一位对强化学习和大语言模型感兴趣的开发者使用 AReaL，并提供宝贵的反馈与建议，一起推动强化学习系统持续创新！

______________________________________________________________________

## 五、资源链接

📦 **GitHub 仓库**：https://github.com/inclusionAI/AReaL

⭐ 欢迎 Star 和 Fork，也期待您的 PR！

📄 **相关论文**：

\[1\] Tau² 235B MoE 训练：https://www.arxiv.org/abs/2601.22607 \[2\] Dynamic Tree
Attention：https://www.arxiv.org/abs/2602.00482

📚 **文档资源**：

- 快速开始：https://inclusionai.github.io/AReaL/tutorial/quickstart.html
- Agentic RL 教程：https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html
- Archon 引擎教程：https://inclusionai.github.io/AReaL/tutorial/archon.html
- AI 辅助开发指南：https://inclusionai.github.io/AReaL/reference/ai_assisted_dev.html

也欢迎持续关注蚂蚁百灵模型的新发布：

______________________________________________________________________
