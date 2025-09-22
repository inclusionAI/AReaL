# Tool-Integrated Reasoning (TIR) Agent

## 项目概述

本项目在AReaL框架中实现了一个Tool-Integrated Reasoning智能体，该智能体能够在数学推理过程中通过多轮工具调用来解决复杂问题。智能体使用Python代码执行、数学计算等工具，并通过强化学习进行端到端训练。

## 核心特性

- **多轮工具调用**: 支持在推理过程中进行多轮工具调用
- **流式生成**: 实时检测工具调用意图，无缝集成工具执行
- **安全执行**: 沙箱环境执行Python代码，确保系统安全
- **强化学习训练**: 基于AReaL框架的GRPO训练，端到端优化
- **模块化设计**: 易于扩展新工具和功能

## 代码组成逻辑

### 1. 核心组件

#### 1.1 TIRWorkflow (`tir_workflow.py`)
- **功能**: 核心工作流，管理多轮推理过程
- **关键特性**:
  - 继承自AReaL的`RolloutWorkflow`基类
  - 支持多轮工具调用的推理过程
  - 实现流式生成和工具调用检测
  - 集成奖励函数计算

#### 1.2 ToolManager (`tool_manager.py`)
- **功能**: 工具管理器，负责协调工具调用
- **支持的工具**:
  - **Python执行器**: 执行Python代码进行数学计算
  - **计算器**: 基础数学运算
- **关键特性**:
  - 工具注册和路由机制
  - 安全的代码执行环境
  - 统一的工具调用接口

#### 1.3 工具实现 (`tools/`)
- **BaseTool** (`tools/base.py`): 工具基类，定义工具接口
- **QwenPythonTool** (`tools/python_tool.py`): Python代码执行工具
- **CalculatorTool** (`tools/calculator_tool.py`): 数学计算工具

#### 1.4 训练脚本 (`train_tir.py`)
- **功能**: 完整的训练流程实现
- **特性**:
  - 集成AReaL的GRPO训练框架
  - 支持分布式训练

### 2. 工具调用机制

#### 2.1 工具调用格式

执行Python代码
```python
# Initialize the count of concave numbers
count = 0

# Iterate over all possible values for A (hundreds place)
for A in range(2, 10):
    # For each A, iterate over all possible values for B (tens place)
    for B in range(0, A):
        # For each B, iterate over all possible values for C (ones place)
        for C in range(B + 1, A):
            # Increment the count for each valid concave number
            count += 1

# The final count of distinct three-digit concave numbers
print(count)
```

```python
output: 120
```


数学计算
```
<calculator>1 + 2 * 3</calculator>
```
```python
output: 6
```

#### 2.2 流式生成与工具调用检测
主要流程：
1. 模型生成到工具调用标记时暂停
2. 检测并解析工具调用内容
3. 在安全环境中执行工具
4. 将工具结果整合到对话中
5. 继续生成后续内容

核心逻辑见`examples/tir/tir_workflow.py`

```python
async def _multi_round_response(self, engine, prompt_ids, data):
  prompt_str = self.tokenizer.decode(prompt_ids)
  completions_str = ""
  has_tool = False
  tool_call_count = 0
  tool_success_count = 0
  stop_reason = None
  max_len = 3000
  turn = 0
  # State flag for each episode: whether waiting for tool start marker
  waiting_for_tool_start = True
  tool_start_idx = -1

  # initialize seq, logprobs, loss_mask, versions
  context_ids = copy.deepcopy(prompt_ids)
  seq = copy.deepcopy(prompt_ids)
  logprobs = [0.0] * len(context_ids)
  loss_mask = [0] * len(context_ids)
  versions = [-1] * len(context_ids)
  output_ids = []

  while turn <= self.max_turns:
      if len(context_ids) >= max_len:
          break

      # Generate response
      resp, stop_reason = await self._generate_response(engine, context_ids, max_len, waiting_for_tool_start)

      context_ids.extend(resp.output_tokens)
      seq.extend(resp.output_tokens)
      logprobs.extend(resp.output_logprobs)
      loss_mask.extend([1] * resp.output_len)
      versions.extend(resp.output_versions)
      
      cur_completions_str = self.tokenizer.decode(resp.output_tokens)
      completions_str += cur_completions_str
      output_ids.extend(resp.output_tokens)
  
      # End token, truncate
      if context_ids[-1] in [
          self.tokenizer.pad_token_id,
          self.tokenizer.eos_token_id,
      ]:
          break

      # If answer appears, truncate immediately
      if re.search(ANSWER, cur_completions_str):
          break

      # State transition logic: detect if tool start marker is encountered
      if waiting_for_tool_start and stop_reason == "stop":
          # Check if tool start marker is detected
          tool_start_marker = self._detect_tool_start_marker(cur_completions_str)
          if tool_start_marker:
              waiting_for_tool_start = False
              tool_start_idx = len(completions_str) - len(tool_start_marker)
              # Continue generating until tool end marker
              continue

      # If tool call is detected, execute tool call
      if not waiting_for_tool_start and stop_reason == "stop" and tool_start_idx != -1:
          tool_results, tool_status = self._execute_tools(completions_str[tool_start_idx:])
          if tool_status == ToolCallStatus.NOT_FOUND:
              # No match found, continue generating until next tool end marker
              continue
          turn += 1
          has_tool = True
          tool_call_count += 1  # Increment tool call count
          if tool_status == ToolCallStatus.SUCCESS:
              tool_success_count += 1
          tool_results = self._process_tool_result(tool_results)
          # Append tool response token IDs
          tool_rsp_token_ids=self.tokenizer.encode(tool_results, add_special_tokens=False)
          # Concatenate to seq
          # Build tool mask
          context_ids.extend(tool_rsp_token_ids)
          seq.extend(tool_rsp_token_ids)
          logprobs.extend([0.0] * len(tool_rsp_token_ids))
          loss_mask.extend([0] * len(tool_rsp_token_ids))
          versions.extend([-1] * len(tool_rsp_token_ids))
          completions_str += tool_results
          
          # After tool execution completes, reset state flag to prepare for next tool call detection
          waiting_for_tool_start = True

  reward = await self.async_reward_fn(
      prompt_str,
      completions_str,
      prompt_ids,
      output_ids,
      tool_using=has_tool,
      tool_status=tool_call_count,
      **data
  )
  
  # Record tool call count to stats_tracker
  stats_tracker.get(self.rollout_stat_scope).scalar(
      tool_call_count=tool_call_count,
      tool_success_count=tool_success_count
  )

  res = dict(
      input_ids=torch.tensor(seq[:max_len]).unsqueeze(0),
      logprobs=torch.tensor(logprobs[:max_len]).unsqueeze(0),
      loss_mask=torch.tensor(loss_mask[:max_len]).unsqueeze(0),
      versions=torch.tensor(versions[:max_len]).unsqueeze(0),
      attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(0),
      rewards=torch.tensor([float(reward)]),
  )
  return TensorDict(res, batch_size=[1])

```


## 如何使用

### 1. 环境准备

1. 参考`docs/tutorial/installation.md`进行基本环境安装
2. 安装qwen_agent: `pip install qwen_agent -y`

### 2. 数据准备

项目使用数学推理数据集，以ToRL数据集为例，数据格式如下：

```json
{"messages": [{"role": "user", "content": "What is 15 + 27?"}], "answer": "42"}
{"messages": [{"role": "user", "content": "Calculate 3 * 4 + 2 * 5"}], "answer": "22"}
```

### 3. 配置训练

编辑 `tir_config.yaml` 配置文件：

```yaml
# 模型配置
actor:
  path: /path/to/your/model
  dtype: bfloat16

# 数据集配置
train_dataset:
  path: /path/to/train/data.parquet
  batch_size: 64

valid_dataset:
  path: /path/to/valid/data.parquet
  batch_size: 64

# tir相关配置
tir:
  max_turns: 2
  max_length: 3000
  tool_timeout: 30
  enable_tools: python;calculator
```


### 4. 运行训练


**单机多GPU训练**

```bash
python3 -m areal.launcher.local \
  examples/tir/train_tir.py \
  --config examples/tir/tir_config.yaml
```

**多机多GPU训练**

TODO

### 5. 测试脚本

TODO

## 训练效果

### 1. 实验设置

- 训练使用了[Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) 作为基模型。
- 奖励仅用结果是否正确。
- 训练Prompt参考[ToRL](https://arxiv.org/pdf/2503.23383), 仅提示模型可以用编程工具，具体可以查看`examples/tir/prompts.py`

### 2. 训练曲线


训练过程中的关键指标变化：

- **奖励曲线**:

  **grpo_actor/task_reward**
  
  <img src="figure/task_reward.png" alt="奖励曲线" width="600"/>

  黄色线为TIR的reward, 可以看到相对纯GRPO训练有15%左右的正确率优势。

- **工具使用频率**:

  <img src="figure/tool_call_count.png" alt="奖励曲线" width="600"/>

  工具调用次数和成功率变化，随训练进行，单个回答调用tool次数0.9->1.2, tool调用成功率没有明显变化。

### 3. 评估

TODO

## 文件结构

```
examples/tir/
├── README.md                   # 项目说明文档
├── tir_workflow.py             # 核心工作流实现
├── tool_manager.py             # 工具管理器
├── tir_config.yaml             # 配置文件
├── train_tir.py                # 训练脚本
├── test_tir.py                 # 测试脚本
├── tools/                      # 工具实现
│   ├── __init__.py
│   ├── base.py                 # 工具基类
│   ├── python_tool.py          # Python执行器
│   └── calculator_tool.py      # 计算器
├── data/                       # 数据文件
│   └── sample_math.jsonl       # 示例数据
└── utils/                      # 工具函数
    └── __init__.py
```


## TODOs
- [ ] 支持异步工具调用.
- [ ] 支持多机训练
- [ ] 调优, 提供Intruct模型的Prompt模板