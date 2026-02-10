# 撰写微信公众号推广文章

## 简介

我要为下次 AReaL 仓库（这个仓库）v1.0.0版本的发布撰写一篇公众号文章进行推广，希望你帮助我完成。

## 要求

### 内容

你可以通过git记录看到最近两周之内文档的变化。目前最新的文档呈现在 "./docs" 文件夹下和 README 中。

你应该重点关注以下内容：

1. agentic RL

你应该起个名字叫xxx agentic rl，重点参考 "docs/tutorial/agentic_rl.md" 的内容

- 上一次的release中（下文有文章链接），已经提到AReaL将 agent 运行被解耦到训练流程之外，目前的架构进一步通过 LLM 代理提供OpenAI
  API接口，让任意agent框架（包括langchain，claude code等等）构成的agent应用，都能被AReaL训练。

- 应当深入介绍agent trajectory是怎么通过individual mode导出，并接入训练的

- 深入介绍proxy worker的整体infra架构，将"docs/tutorial/agentic_rl.md"的数据流图，利用 ”/pptx"
  技能画在pptx里面，生成一张漂亮的架构图；你不用放在文章里面，之后我自己放

- 介绍tree attention帮助我们在agentic training中做了训练优化（和 "docs/tutorial/agentic_rl.md"
  同样的逻辑），你应该结合这篇论文（https://www.arxiv.org/pdf/2602.00482）和代码一起来看
  @areal/engine/megatron_engine.py

2. Archon引擎

你需要根据 "docs/tutorial/archon.md" 介绍下AReaL中新实现的archon训练引擎。

基于新引擎，我们可以6台机器48卡H200训练235B MoE模型，并在tau2
bench上获得sota表现（参考论文https://www.alphaxiv.org/overview/2601.22607v1）

训打模型需要5D parallel，最广泛使用的框架是megatron，但是megatron 和相关依赖安装复杂，代码层层嵌套，难以使用和修改。

鉴于PyTorch生态逐渐成熟，在AI辅助编程的帮助下，AReaL实现了一个PyTorch原生的训练引擎，支持以下功能：

- 完整的5D 并行支持
- 细粒度的activation checkpointing
- torch.compile支持，极简代码实现极致优化
- 无需docker，`uv sync` 即可使用 （其他功能请参考 "docs/tutorial/archon.md" ）

3. AI辅助编程

Archon这么复杂的分布式训练架构其实实现+验证正确性只用了一个人一个月多一点的时间，而归功于大规模、可靠化的AI 编程工具使用。

你需要说明AReaL甚至已经将这些使用AI工具的"武功秘籍"开源，参考 @docs/reference/ai_assisted_dev.md
去描述AI辅助编程的内容，并且告诉用户也能够用这些内容帮助自己agent RL引用的开发。

### 文字风格

请参考[上一次AReaL Release的微信公众号风格](./prev_release.md) 和 另外一个系统
[AState的公众号风格](https://mp.weixin.qq.com/s/QHcO9IjxqTje8pWygurP2g)

你可以忽略排版，也不包括每次必有的文章抬头内容，重点关注文字风格：中心思想（需要推广的feature）需要简明扼要，措辞需要关注技术特点。

1. 引导性问题：从用户视角提出"如何实现"的疑问
1. 架构概述：简洁说明四进程协作架构的职责
1. 数据流图：从 docs/reference/agent_workflow.md 搬运的详细流程图
1. 关键数据流说明：用 6 个步骤解释数据如何流动
1. 设计精髓总结：强调 Agent 无感知的核心优势
