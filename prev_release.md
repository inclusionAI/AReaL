# AReaL v0.5.0：强化学习框架的架构革新，执一驭万，智体同协

**<font style="color:rgb(0, 0, 0);">💻</font>\*\*\*\*<font style="color:rgb(0, 0, 0);">
ASystem 核心技术解析系列 | 每周四硬核解密</font>**

<font style="color:rgb(0, 0, 0);">欢迎回到我们为您精心策划的「ASystem 系统开源」核心技术解析系列的最新一期！</font>

<font style="color:rgb(0, 0, 0);">ASystem 是我们为支撑万亿级思考模型
</font>**<font style="color:rgb(0, 0, 0);">Ring-1T</font>**<font style="color:rgb(0, 0, 0);">
等大规模 RL 训练而构建的完整技术底座。在超大规模 RL 训练中，系统工程的复杂性极易反噬算法开发效率和灵活性。</font>

**<font style="color:rgb(0, 0, 0);">本期聚焦：RL 算法开发的高效基石 AReaL</font>**

<font style="color:rgb(0, 0, 0);">本期，我们将深度解析 ASystem
中面向算法设计、以</font>**<font style="color:rgb(0, 0, 0);">开发效率和灵活性</font>**<font style="color:rgb(0, 0, 0);">为核心的关键用户层技术——</font>**<font style="color:rgb(0, 0, 0);">AReaL
强化学习框架</font>**<font style="color:rgb(0, 0, 0);">。</font>

<font style="color:rgb(0, 0, 0);">AReaL 通过极简的 API
和可扩展的插件机制，将算法设计者从繁琐的系统细节中解放出来，专注于算法本身。</font>

<font style="color:rgb(0, 0, 0);">我们希望通过 AReaL
的架构设计与实践，推动将强化学习框架从“完整应用”转化为“高性能后端依赖”，为业界大规模 RL 训练提供社区友好的思路。</font>

<font style="color:rgb(0, 0, 0);">硬核解密持续放送！锁定我们，每周四不见不散！</font>

# 前言

AReaL 是一个面向算法设计，以开发效率和灵活性为核心的强化学习框架。它通过极简的 API
和可扩展的插件机制降低用户的学习曲线和使用心智负担，让开发者聚焦于算法本身而非系统细节，具备大规模扩展、无侵入 Agentic RL、故障感知与自恢复等能力。在蚂蚁内部，作为
ASystem 的关键用户层技术，我们使用 AReaL 支持了 Ring 1T 万亿参数 MoE 模型的强化学习（ Reinforcement Learning ，下文简称
RL）后训练，相关代码已在以下链接开源。

> [https://github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)

**AReaL 本次 12 月 “ASystem 疯狂星期四” 开源的新版本 v0.5.0 带来了解耦式 Agentic RL，以及 Single Controller
架构两个核心特性**，本文会介绍它们的设计理念和解决方案。

- **解耦式 Agentic RL**：**AReaL 通过 OpenAI API
  代理，提供了一套解耦化的智能体训练服务解决方案**，便于环境提供者、算法开发者和系统开发者形成复杂工程中的零障碍流水线，极大提升了开发效率与系统可维护性。
- **Single Controller 架构**：**消除了 SPMD (Single Program, Multiple Data)
  模式的长尾和数据不均匀问题**，这种分层设计既能提升推理扩展性、增强系统层面精细化控制，又能保留算法编排的灵活易用性，降低算法开发者代码迁移的成本；

我们希望通过 AReaL
架构设计及实践，推动将强化学习框架从“完整应用”到“<font style="background-color:#FBDE28;">运行时服务</font>”的转化，为业界大规模强化学习提供一些社区友好的思路。

# 核心特性一：解耦式 Agentic RL 架构

## 背景介绍

在当前智能体（Agent）系统快速演进的背景下，如何高效地将强化学习（Reinforcement Learning, RL）能力融入到复杂、灵活的 Agent
逻辑中，成为一个关键挑战。传统的 Agentic RL 实现方式往往将 RL 的训练逻辑深度耦合在 Agent 的编排框架中，导致代码复用性差、调试困难、且难以快速迭代和部署。

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/164456455/1765200486262-e997f4b0-d6eb-47ee-90ee-7de4cbceea26.png)

（注：图片为 AI 生成）

为解决这一问题，我们在 AReaL 中自研了一个**面向 Agentic RL 的解耦式训练框架**。AReaL 的核心设计理念是 **"Agent 独立运行 +
训练逻辑外置"**，通过高度抽象的架构设计，实现了 Agent 逻辑与 RL 训练逻辑的完全分离，从而在保证灵活性的同时，极大提升了开发效率与系统可维护性。

## 核心设计思想：解耦与透明

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/193056530/1765175383627-5e444467-2bb6-45c7-aa70-ba4472f30567.png)

AReaL Agentic RL 的架构设计建立在两个核心原则之上：

1. **Agent 完全独立运行（Agent Autonomy）** 在 AReaL 框架中，Agent 本身不依赖任何 RL
   框架的组件，也不感知自身正在被用于训练。它只是一个标准的、基于大语言模型（LLM）的决策系统，按照既定的编排逻辑接收输入、调用工具、生成动作并输出结果。这种设计确保了
   Agent 的纯净性与可移植性——同一个 Agent 实现既可以用于在线推理，也可以无缝接入离线训练，真正做到“一套代码，两处复用”。
1. **RL 训练作为外部观察者（RL as Observer）** AReaL 不主动干预 Agent 的执行流程，而是通过“代理请求”的方式，监听并记录 Agent
   与环境交互的完整轨迹（Trajectory）。这些轨迹包括：用户输入、Agent
   的思维链（Thought）、调用的动作（Action）、环境反馈（Observation）以及最终的奖励信号（Reward）。通过这种方式，AReaL 将复杂的
   Agent 执行过程转化为标准的 RL 训练数据，从而可以使用任意成熟的 RL 算法进行策略优化。

## Agent 架构工作流程详解

AReaL 的工作流程可分为以下几个关键阶段：

### 1. Agent 启动与代理封装

为了让 AReaL 能够驱动用户编写的 Agent，并进行 RL 训练，用户需要提供一个接口函数来告知 AReaL 如何执行 Agent，以及 Agent 所获得的
reward。该函数固定命名为 `run_agent_return_reward`，输入 `data` 可以是任意类型，它是用户提供的数据集中的一条数据，输出
`reward`，表示本次 Agent 运行得到的奖励。

```python
async def run_agent_return_reward(data: Any) -> float:
    """
    Input:
        data: Any, One sample of data in the dataset.
    Return:
        reward: float, The final reward of the result of this agent run.
    """
```

**该接口是用户将 Agent 接入 AReaL 唯一需要编写的接口。**

### 2. 轨迹收集（Trajectory Collection）

当 AReaL 开始进行 Agentic RL 训练时，框架会为每一次 Agent 运行自动开启一个会话（Session），并记录从输入到输出的每一步交互。由于我们优化的对象是
LLM 的输出，能够参与参数更新的数据仅有 LLM 自主生成的 token，而其他部分作为 LLM 的输入被记录下来

- **输入捕获**：用户原始 query、工具调用结果、其他 Agent 的输入等，以 token 形式被保存下来；
- **输出缓存**：LLM 输出的 token（包括思维链、调用工具的参数、自我反思等等）在被解码成字符串返回到 agent 之前，被缓存到 AReaL 中；
- **环境反馈**：Agent 自行调用外部环境和工具进行迭代；
- **奖励标注**：计算 Agent 最终的或每一步的奖励分数，并上传到 AReaL；

所有这些信息被组织为一个结构化的轨迹（Trajectory），用于后续训练。

### 3. 强化学习训练（RL Training）

一旦积累了足够的轨迹数据，AReaL 便进入 RL 训练流程。AReaL
会收集每一次会话中所有交互的输入、输出和奖励值，按照时间顺序排序，并根据用户指定的折扣因子（discount
factor）将最终奖励进行反向传播，得到一个完整的可以训练的批次。

训练流程基于 AReaL 框架实现，支持多种 RL 算法，并且和推理完全异步，充分利用硬件资源。训练过程后，AReaL 会自动将策略更新后的模型权重回传给 Agent 的
LLM 后端，从而实现策略迭代。

### 4. 模型部署与闭环迭代

训练完成的模型可直接导出为标准模型文件（如 HuggingFace 格式），并部署到生产环境中的 Agent 系统中。由于 Agent
本身不依赖训练框架，部署过程无需任何代码修改。同时，生产环境中的新交互数据又可被 AReaL 持续收集，形成“收集 → 训练 → 部署 → 再收集”的闭环优化循环。

## 核心优势总结

| 特性           | 说明                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| **低侵入性**   | 无需修改现有 Agent 代码，仅需提供启动函数即可接入训练。                  |
| **高复用性**   | 训练好的模型可直接用于线上服务，避免训练与推理不一致问题。               |
| **灵活性强**   | 支持任意 Agent 框架（LangChain、LlamaIndex、自研框架等）与任意 RL 算法。 |
| **可观测性好** | 完整记录 Agent 决策轨迹，便于调试、分析与人工审核。                      |
| **易于扩展**   | 可集成多种奖励模型、支持多任务联合训练、支持分布式数据收集。             |

## 快速体验

为了方便用户快速体验，我们提供了一个通过多轮对话解决数学问题的实验脚本，可以通过如下方式启动实验

```plain
python3 -m areal.launcher.ray examples/multi-turn-math/gsm8k_rl_mt.py \
    --config examples/multi-turn-math/gsm8k_grpo_mt.yaml \
    experiment_name=gsm8k-grpo-multiturn trial_name=trial0
```

训练的 reward 如下图，红色为通过两轮迭代解决问题，黄色为单轮迭代，可以看到多轮反思确实能够提升性能。

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/164456455/1765200136395-230edae4-b3e3-4b0d-8346-8f2d4bb23a7d.png)

_在解耦式架构下数学智能体的训练奖励曲线_

# 核心特性二：Single Controller 架构

## 背景介绍

AReaL 最初沿用主流预训练框架所采用的 **SPMD
执行模型**：多个训练进程运行相同代码，数据与模型参数在进程间分片。这种模式在预训练中问题不大，但在强化学习场景下却面临两大核心问题：

1、长尾问题

强化学习训练包含两个关键阶段：**Rollout** 与 **Training**。在当前 SPMD 架构中：

```
- 每个训练进程独立提交固定数量的 prompt 至推理引擎；
- 由于生成长度不可预测，部分进程可能由于长尾任务，需长时间等待；
- 所有进程必须同步完成生成，训练才能进入下一步，尤其是训练刚开始时。
```

这导致“快”的进程长时间空转，严重浪费 GPU 算力，整体吞吐受限于最慢的生成任务；

2、精细化控制

在大规模 RL 训练中，软硬件故障是不可避免的问题，势必要考虑以下问题

```
- 推理单实例故障如何快速替换而不是重启整个实验
- 推理引擎瓶颈时如何快速横向扩展
```

由于 SPMD 所有进程都运行相同代码，要实现上述精细化的控制场景会让代码逻辑非常复杂，到处夹杂着控制流与计算流。为了优雅解决上述问题，我们引入了 **"单控制器 +
分布式引擎"** 的分层设计，将系统划分为 **控制平面** 与 **数据平面**，实现精细化的工作流控制与高效的数据流转。

## 架构设计

![画板](https://intranetproxy.alipay.com/skylark/lark/0/2025/jpeg/149585/1765186241079-9e4d5b2c-fda7-48db-a976-89c2230e6699.jpeg)

_Single Controller 架构_

### 设计理念

在架构设计上，**自上而下分为 Controller、Worker、Engine 三层**：

- Controller：跑在 CPU 节点上，封装了分布式的细节，包括分布式调度、分布式数据聚合与分发等，暴露了与 Engine 相同的接口，用户的训练脚本可以像调用
  Engine 一样调用 Controller 的接口。
- Worker：位于 Controller 之下 ，它提供了 Engine Deploy 的能力，在部署形态上，Engine 可以是 Embedding 模式与 Worker
  跑在同一进程，也可以独立进程部署，之所以我们支持独立进程部署是为了解耦 AReaL 框架与引擎的包依赖，同时让外部 Engine 能更方便地接入到 AReaL
  中，而不需要开发者理解 AReaL 的代码细节；同时我们将分布式数据流处理也放在这一层，对应 Controller 分发的 metadata，解决 Single
  Controller 单点瓶颈问题。
- Engine 层：专注于并行计算，完全兼容原生 SGLang、FSDP、Megatron 等训推引擎。

### 核心组件

| 组件                                    | 类型     | 职责                                                                              | 优势                               |
| --------------------------------------- | -------- | --------------------------------------------------------------------------------- | ---------------------------------- |
| **TrainController / RolloutController** | 控制平面 | 在用户脚本中运行，负责启动/终止引擎、调度任务、传输元数据、聚合结果               | 集中控制，简化用户接口             |
| **TrainWorker / RolloutWorker**         | 数据平面 | 与 Engine 对等部署，分布式数据流处理，执行分布式计算，通过 RPC 接口与 Engine 交互 | 数据本地化，避免通信瓶颈           |
| **TrainEngine / InferenceEngine**       | 数据平面 | 在 GPU 节点上运行，执行训推并行计算、暴露 RPC 服务接口                            | 训推并行计算，完全兼容原生训推引擎 |

## API Demo

```python
# 需通过 launcher 启动
# python3 -m areal.launcher.local script.py --config xxx.yaml

def main(args):
	actor = FSDPPPOActor(config=config.actor)
	actor.create_process_group(parallel_strategy=parallel_strategy)

	rollout = RemoteSGLangEngine(config.rollout)
	rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

	# 手动在 head rank 加载数据并广播
	batch = None
	if actor.is_data_parallel_head():
    	batch = rollout.prepare_batch(...)
    	batch = tensor_container_to(batch, actor.device)

	batch = broadcast_tensor_container(batch, src_rank=actor.current_data_parallel_head(), group=...)
```

```python
# 直接运行脚本，无需 launcher
# python script.py --config xxx.yaml

def main(args):
	# 使用控制器包装引擎
	actor = TrainController(
    	engine=FSDPPPOActor(config=config.actor),
    	scheduler=LocalScheduler(...)
	)
	rollout = RolloutController(
    	engine=RemoteSGLangEngine(config.rollout),
    	scheduler=LocalScheduler(...)
	)

	# 自动返回 DistributedBatch，无需手动广播
	batch = rollout.prepare_batch(...)
    # 控制器自动处理数据分发
```

可以看到 Single Controller 与 SPMD 的 Trainer 在用户 trainer
编程界面上基本一致，但是简化了一些分布式细节，一个好的例子是，我们将\*\*「分布式数据流」抽象为 DistributedBatch\*\*。DistributedBatch
是整个架构的"数据护照"，它不携带实际张量，仅保存元信息：

```python
@dataclass
class TensorMetadata:
    """Metadata for a tensor field."""
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"

@dataclass
class ShardMetadata:
    """Metadata for a single (sub-)shard stored on one node.
    A logical batch can be composed of multiple shards, and a single physical
    shard can be split into multiple logical sub-shards via offset and batch_size.
    """
    node_id: str
    node_addr: str
    shard_id: str
    batch_size: int
    offset: int = 0
    fields: dict[str, TensorMetadata] = field(default_factory=dict)


@dataclass
class BatchMetadata:
    """Metadata for a distributed batch sharded across multiple nodes."""
    batch_id: str
    global_step: int
    total_batch_size: int
    shards: list[ShardMetadata] = field(default_factory=list)
```

从数据流理解 RL 流程，大致如下：

1. Rollout Controller 收集各推理引擎生成的元数据；
1. Train Controller 将元数据按数据并行策略分片，发送给 Worker；
1. 各训练 Worker 在首次访问时，**按需通过 RPC 拉取本地/远端所需张量**；

该设计通过 metadata 而不是实际 Tensor 驱动整个 RL 训练，解决了 Single Controller 架构下的单点瓶颈问题，可以大幅提升 RL
训练的规模与效率。

# 未来展望

AReaL 当前已支持基础的 Agentic RL 训练流程 和Single Controller 模式，未来将进一步支持并完善：

- Single Controller 模式下的高效数据流和分布式启动能力
- 自动扩缩容、故障修复和高可用训练
- 轨迹数据版本管理与可视化分析平台
- 提升智能体场景下的训推性能

通过 AReaL，我们**期望为 Agentic AI
的研发提供一个\*\*\*\*<font style="background-color:#FBDE28;">高效、可靠、可扩展的强化学习运行时服务框架</font>**，推动智能体系统从“能用”向“好用”持续演进。

欢迎每一位对强化学习和大语言模型感兴趣的开发者使用 AReaL，并提供宝贵的反馈与建议，一起推动强化学习系统持续创新！

📦 GitHub 仓库：

[https://github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)

⭐ 欢迎 Star 和 Fork，也期待您的 PR！

也欢迎持续关注蚂蚁百灵模型的新发布：

🤗 Hugging Face：https://huggingface.co/inclusionAI

🤖 魔搭社区：https://www.modelscope.cn/organization/inclusionAI
