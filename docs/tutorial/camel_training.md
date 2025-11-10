# Training with CAMEL

This guide demonstrates how to use AReaL to train language models with
[CAMEL-AI](https://github.com/camel-ai/camel), a framework for building agentic
workflows. The CAMEL integration allows you to leverage CAMEL's agent capabilities while
using AReaL's distributed reinforcement learning training system.

## Overview

CAMEL‑AI is an open‑source, modular framework for building intelligent multi‑agent
systems. It provides a flexible agent architecture that can handle complex dialogue
flows, tool calling, and multi-agent interactions. CAMEL agents excel at tasks requiring
sequential reasoning, such as mathematical problem-solving, code generation, and
multi-step planning.

While CAMEL agents are powerful out of the box, reinforcement learning (RL) training can
significantly improve their performance by optimizing task-specific behavior, learning
from feedback signals, and adapting to domain-specific requirements.

However, CAMEL agents cannot be directly trained with reinforcement learning for several
reasons:

1. **Lack token-level access**: CAMEL agents interact with language models through
   high-level APIs (e.g., OpenAI's chat completion API), which do not expose token IDs
   needed for RL training. RL algorithms require token-level information to compute
   policy gradients.

1. **No reward mechanism**: CAMEL agents are designed for inference and do not have
   built-in reward functions. RL training requires reward signals to guide policy
   optimization, which must be computed based on task-specific metrics (e.g., answer
   correctness for math problems).

1. **Limited parallelization**: Standard CAMEL usage involves sequential agent
   execution, making it difficult to efficiently collect diverse trajectories needed for
   RL training.

AReaL addresses these limitations by integrating CAMEL agents with its training
infrastructure:

1. **OpenAI-compatible client with token-level tracking**: AReaL's `ArealOpenAI` client
   provides a drop-in replacement for OpenAI's API that automatically records
   token-level information. Each interaction (completion/response) is cached with its
   input tokens, output tokens, and associated log probabilities (see the
   [OpenAI-Compatible Workflows](openai_workflows.md) guide for details). This enables
   RL algorithms to access the granular data needed for policy gradient computation.

1. **Reward system integration**: AReaL allows you to define custom reward functions and
   automatically propagate rewards through conversation trees. The `ArealOpenAI` client
   supports reward assignment at any point in the conversation, with automatic backward
   discounting for multi-turn interactions.

1. **Parallel trajectory collection**: AReaL's workflow system enables parallel
   execution of multiple CAMEL agent instances, allowing you to collect diverse
   trajectories for each query. This is essential for effective RL training, as it
   increases sample diversity and improves policy gradient estimates.

## Prerequisites

Before starting, ensure you have:

1. Completed the [installation guide](installation.md)
1. Installed CAMEL-AI:

```bash
pip install camel-ai
```

## Building a Trainable CAMEL Agent

We'll build a trainable CAMEL agent step by step, starting from the simplest example and
gradually adding complexity. By the end, you'll have a complete agent integrated into
AReaL's training pipeline.

### Writing a CAMEL Agent

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

### Converting to an RL-Trainable Agent

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

### Adding Reward Evaluation

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

### Wrapping the Agent in a Reusable Class

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

### Creating the Rollout Workflow

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

### Running the Training Example

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

## Full Working Example

The full working CAMEL training example is located in
[**`examples/camel/`**](https://github.com/inclusionAI/AReaL/blob/main/examples/camel/).
To run the example on a single node:

```bash
python3 -m areal.launcher.local examples/camel/train.py \
    --config examples/camel/config.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```

## Customization

### Using Different CAMEL Agent Types

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

### Modifying Agent Behavior

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
