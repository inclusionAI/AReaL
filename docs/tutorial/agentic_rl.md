# Agentic Reinforcement Learning

**Agentic Reinforcement Learning (Agentic RL)** is a training paradigm that uses
reinforcement learning to elevate Large Language Models (LLMs) from passive text
predictors into autonomous agents. Instead of optimizing for a single, correct response,
this approach trains agents over extended, interactive episodes where they must learn to
plan, use tools, and reason through multiple steps. By learning from trial-and-error
feedback in a dynamic environment, Agentic RL aims to develop models that can
independently strategize and execute complex, long-horizon tasks.

This guide demonstrates how to use AReaL to train agentic models with popular agent
frameworks. AReaL provides seamless integration with agent frameworks like
[CAMEL-AI](https://github.com/camel-ai/camel) and
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python), enabling you to
leverage their agent orchestration capabilities while using AReaL's distributed
reinforcement learning training system.

## Overview

Agentic frameworks provide powerful abstractions for building multi-agent systems with
features like agent coordination, tool calling, handoffs, and structured interactions.
These frameworks excel at complex tasks requiring sequential reasoning, such as
mathematical problem-solving, code generation, and multi-step planning.

While these frameworks are powerful out of the box, reinforcement learning (RL) training
can significantly improve their performance by optimizing task-specific behavior,
learning from feedback signals, and adapting to domain-specific requirements.

However, these frameworks cannot be directly used for reinforcement learning training
for several reasons:

1. **Lack token-level access**: Agent frameworks interact with language models through
   high-level APIs (e.g., OpenAI's chat completion API), which do not expose token IDs
   and log probabilities needed for RL training. RL algorithms require token-level
   information to compute policy gradients.

1. **No reward mechanism**: Agent frameworks are designed for inference and do not have
   built-in reward functions. RL training requires reward signals to guide policy
   optimization, which must be computed based on task-specific metrics (e.g., answer
   correctness for math problems).

1. **Limited parallelization**: Standard agent usage involves sequential execution,
   making it difficult to efficiently collect diverse trajectories needed for RL
   training.

AReaL addresses these limitations by providing:

1. **OpenAI-compatible client with token-level tracking**: AReaL's `ArealOpenAI` client
   is a drop-in replacement for OpenAI's `AsyncOpenAI` client that routes all LLM calls
   to AReaL's inference engine (SGLang or vLLM). Every interaction (completion/response)
   is automatically tracked with complete token-level information including input
   tokens, output tokens, and associated log probabilities (see the
   [OpenAI-Compatible Workflows](openai_workflows.md) guide for details). This enables
   RL algorithms to access all the granular data for policy gradient computation.

1. **Reward assignment and propagation**: AReaL provides a flexible reward system that
   allows you to assign rewards to specific interactions or entire trajectories. The
   system automatically builds conversation trees based on message role sequences and
   supports reward backpropagation with customizable discounting factors, enabling
   automatic reward assignment across multi-turn conversations.

1. **Parallel trajectory collection**: AReaL's workflow system enables parallel
   execution of multiple agent instances, allowing you to collect diverse trajectories
   for each query. This is essential for effective RL training, as it increases sample
   diversity and improves policy gradient estimates.

## Prerequisites

Before starting, ensure you have:

1. Completed the [installation guide](installation.md)

1. Installed the agent framework you want to use:

```bash
# For CAMEL-AI
pip install camel-ai

# For OpenAI Agents SDK
pip install openai-agents
```

## Training with CAMEL

CAMEL‑AI is an open‑source, modular framework for building intelligent multi‑agent
systems. It provides a flexible agent architecture that can handle complex dialogue
flows, tool calling, and multi-agent interactions.

### Building a Trainable CAMEL Agent

We'll build a trainable CAMEL agent step by step, starting from the simplest example and
gradually adding complexity. By the end, you'll have a complete agent integrated into
AReaL's training pipeline.

#### Step 1: Writing a CAMEL Agent

A typical CAMEL agent is straightforward to write. Here's a simple example that uses
CAMEL's `ChatAgent` to solve math problems:

```python
from camel.agents import ChatAgent

# Create a basic CAMEL agent
agent = ChatAgent(
    system_message="You are a helpful math assistant.",
    model="gpt-4o-mini",
)

# Run the agent
response = await agent.astep("Solve: 2 + 2 = ?")
print(response.msg.content)
```

#### Step 2: Converting to an RL-Trainable Agent

To make this agent trainable with AReaL, simply replace the model with AReaL's
OpenAI-compatible model:

```python
from camel.agents import ChatAgent
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI

# Create AReaL's OpenAI-compatible client
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)

# Replace the model with AReaL's OpenAI-compatible model
agent = ChatAgent(
    system_message="You are a helpful math assistant.",
    model=AReaLOpenAICompatibleModel(
        openai_client=client,
        tokenizer=tokenizer,
        model_type="areal"
    )
)

# Now the client (ArealOpenAI) records token-level information and can be used for RL training
response = await agent.astep("Solve: 2 + 2 = ?")
```

#### Step 3: Adding Reward Evaluation

Next, we need to evaluate and assign rewards. After the agent responds, we check if the
answer is correct and set the reward:

```python
def math_reward_fn(result, answer):
    """Simple reward function: 1.0 if correct, 0.0 otherwise."""
    return 1.0 if result.strip() == answer.strip() else 0.0

# Run the agent
response = await agent.astep("Solve: 2 + 2 = ?")

# Evaluate and set reward
reward = math_reward_fn(response.msg.content, "4")
client.set_final_reward(reward)
```

#### Step 4: Wrapping the Agent in a Reusable Class

To integrate the agent into AReaL's training pipeline, wrap it in a class that manages
the agent lifecycle and reward evaluation. This makes it easier to reuse the agent in
different training workflows. Here's how to structure it:

```python
from areal.api.reward_api import AsyncRewardWrapper
from transformers import PreTrainedTokenizerFast

class CamelMathAgent:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        # Wrap reward function for async execution
        self.async_reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        """Run agent on a dataset sample.

        Parameters
        ----------
        data : Dict
            Dataset sample with 'messages' (conversation history) and 'answer' (ground truth)
        client : ArealOpenAI
            Client that tracks token information
        """
        # Create agent with AReaL OpenAI-compatible model
        agent = ChatAgent(
            system_message="You are a helpful math assistant.",
            model=AReaLOpenAICompatibleModel(...),
        )

        # Run agent
        response = await agent.astep(data["messages"][-1]["content"])
        content = response.msg.content

        # Evaluate reward and set reward on client for RL training
        reward = await self.async_reward_fn(result=content, answer=data["answer"])
        client.set_final_reward(reward)

        return reward
```

#### Step 5: Creating the Rollout Workflow

Finally, we integrate our agent into AReaL's `RolloutWorkflow`. Here, we can collect
multiple trajectories in parallel, which is essential for effective RL training:

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
import asyncio

class CamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        n_trajs: int = 2,  # Collect 2 trajectories per query
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.n_trajs = n_trajs

        # Create our agent wrapper
        self.agent = CamelMathAgent(tokenizer=self.tokenizer)

    async def arun_episode(self, engine, data):
        """Run one training episode: collect trajectories and return training data."""
        # Create one client per trajectory (enables parallel collection)
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Run agents in parallel
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(data=data, client=clients[i])
                for i in range(self.n_trajs)
            ]
        )

        # Export all interactions with rewards
        interactions_with_reward = {}
        for client in clients:
            # Apply reward discounting for multi-turn conversations
            client.apply_reward_discount(turn_discount=0.9)
            # Export interactions with token-level data
            interactions = client.export_interactions(style="individual")
            interactions_with_reward.update(interactions)

        return interactions_with_reward
```

**Key points:**

- **Parallel episode execution**: AReaL's training loop calls `arun_episode` in parallel
  across multiple samples in a batch, enabling parallel trajectory collection at the
  batch level.
- **Parallel trajectory collection within episodes**: Each `arun_episode` call creates
  multiple `ArealOpenAI` clients and runs agents in parallel using `asyncio.gather()`,
  collecting diverse trajectories for each query.
- **Reward discounting**: For multi-turn conversations, rewards are discounted backward
  through the conversation tree.
- **Interactions export**: All interactions with token-level data and rewards are
  exported in a format ready for RL training.

This workflow is then integrated into AReaL's standard training loop, which handles
rollout collection, advantage computation, and policy updates.

#### Step 6: Running the Training Example

Now you can use this workflow in AReaL's training loop. The workflow integrates
seamlessly with AReaL's actor and training infrastructure:

```python
# In your training script
workflow = CamelRLVRWorkflow(
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    n_trajs=2,
)

# AReaL will call workflow.arun_episode() for each batch
# The workflow handles rollout collection, and AReaL handles training
```

That's it! Your CAMEL agent is now fully integrated into AReaL's training pipeline. See
the
[complete train script](https://github.com/inclusionAI/AReaL/blob/main/examples/camel/train.py)
for a full working implementation.

### Full Working Example

The full working CAMEL training example is located in
[**`examples/camel/`**](https://github.com/inclusionAI/AReaL/blob/main/examples/camel/).
To run the example on a single node:

```bash
python3 -m areal.launcher.local examples/camel/train.py \
    --config examples/camel/config.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```

### Customization

#### Using Different CAMEL Agent Types

You can customize the CAMEL agent by using different agent types from the CAMEL library.
For example, to use a `TaskPlannerAgent` with tool calling capabilities:

```python
from camel.agents import TaskPlannerAgent

class CamelTaskAgent:
    async def run_agent(self, data, client: ArealOpenAI):
        agent = TaskPlannerAgent(
            model=AReaLOpenAICompatibleModel(
                openai_client=client,
                tokenizer=self.tokenizer,
                model_type="areal"
            ),
            tools=[your_tools],  # Add tool calling
        )
        # ... rest of the logic
```

#### Modifying Agent Behavior

Customize the agent's behavior through CAMEL's configuration options:

```python
agent = ChatAgent(
    model=AReaLOpenAICompatibleModel(...),
    system_message="...",
    token_limit=...,
    # ... other CAMEL parameters
)
```

Refer to the [CAMEL-AI documentation](https://github.com/camel-ai/camel) for available
agent types and configuration options.

## Training with OpenAI Agents

Using the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) with AReaL
follows a similar pattern to using CAMEL. The key difference is that instead of using
CAMEL's `AReaLOpenAICompatibleModel` adapter, you can use AReaL's `ArealOpenAI` client
directly with the SDK's `RunConfig`.

#### Step 1: Basic Agent Implementation

```python
from agents import Agent as OpenAIAgent
from agents import Runner as OpenAIRunner

# Create a basic agent with OpenAI client
agent = OpenAIAgent(
    name="MathAgent",
    instructions="You are a helpful assistant that solves math problems."
)

# Run the agent
result = await OpenAIRunner.run(
    agent,
    input="Solve: 2 + 2 = ?"
)
print(result.final_output)
```

#### Step 2: Enabling RL Training

Replace the standard OpenAI client with AReaL's `ArealOpenAI` client:

```python
from agents import Agent as OpenAIAgent
from agents import OpenAIProvider, RunConfig
from agents import Runner as OpenAIRunner
from areal.experimental.openai import ArealOpenAI

# Create AReaL's OpenAI-compatible client
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)

# Create agent with OpenAI Agents SDK
agent = OpenAIAgent(
    name="MathAgent",
    instructions="You are a helpful assistant that solves math problems."
)

# Configure runner to use ArealOpenAI (instead of OpenAI client)
run_config = RunConfig(
    model_provider=OpenAIProvider(openai_client=client),
    tracing_disabled=True,
)

# Run agent - all LLM calls go through ArealOpenAI
result = await OpenAIRunner.run(
    agent,
    input="Solve: 2 + 2 = ?",
    run_config=run_config
)

# The ArealOpenAI client automatically captures token-level data suitable for RL training
```

#### Step 3: Adding Reward Evaluation

```python
def math_reward_fn(result, answer):
    """Simple reward function: 1.0 if correct, 0.0 otherwise."""
    return 1.0 if result.strip() == answer.strip() else 0.0

# Run the agent
result = await OpenAIRunner.run(
    agent,
    input="Solve: 2 + 2 = ?",
    run_config=run_config
)

# Compute reward and associate it with the interaction
reward = math_reward_fn(result.final_output, "4")
client.set_final_reward(reward)
```

#### Step 4: Wrapping the Agent in a Reusable Class

```python
from areal.api.reward_api import AsyncRewardWrapper
from transformers import PreTrainedTokenizerFast

class OpenAIMathAgent:
    def __init__(self):
        # Wrap reward function for async execution
        self.async_reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        """Run agent on a dataset sample.

        Parameters
        ----------
        data : Dict
            Dataset sample with 'messages' (conversation history) and 'answer' (ground truth)
        client : ArealOpenAI
            Client that tracks token information
        """
        # Create agent with OpenAI Agents SDK
        agent = OpenAIAgent(
            name="MathAgent",
            instructions="You are a helpful assistant that solves math problems."
        )

        # Configure runner to use ArealOpenAI
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
        )

        # Run agent
        result = await OpenAIRunner.run(
            agent,
            input=data["messages"][-1]["content"],
            run_config=run_config
        )

        # Evaluate reward and set reward on client for RL training
        reward = await self.async_reward_fn(
            result=result.final_output,
            answer=data["answer"]
        )
        client.set_final_reward(reward)

        return reward
```

#### Step 5: Creating the Rollout Workflow

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
import asyncio

class RLVRAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        n_trajs: int = 2,  # Collect 2 trajectories per query
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.n_trajs = n_trajs

        # Create our agent wrapper
        self.agent = OpenAIMathAgent()

    async def arun_episode(self, engine, data):
        """Run one training episode: collect trajectories and return training data."""
        # Create one client per trajectory (enables parallel collection)
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Run agents in parallel
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(data=data, client=clients[i])
                for i in range(self.n_trajs)
            ]
        )

        # Export all interactions with rewards
        interactions_with_reward = {}
        for client in clients:
            # Apply reward discounting for multi-turn conversations
            client.apply_reward_discount(turn_discount=0.9)
            # Export interactions with token-level data
            interactions = client.export_interactions(style="individual")
            interactions_with_reward.update(interactions)

        return interactions_with_reward
```

#### Step 6: Incorporating into Training

```python
workflow = RLVRAgentWorkflow(
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    n_trajs=2,
)
```

For a complete implementation, refer to the
[complete training script](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/train_agents.py).

### Complete Example

The full working OpenAI Agents training example is located in
[**`examples/openai-agents/`**](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/).
To run the example on a single node:

```bash
python3 -m areal.launcher.local examples/openai-agents/train_agents.py \
    --config examples/openai-agents/config.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```

### Customization

For comprehensive details on agent instructions, handoffs, `ModelSettings`, and
additional configuration options, refer to the
[OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/).
