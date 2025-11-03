# Training with CAMEL

This guide demonstrates how to use AReaL to train language models with
[CAMEL-AI](https://github.com/camel-ai/camel), a framework for building agentic
workflows. The CAMEL integration allows you to leverage CAMEL's agent capabilities while
using AReaL's distributed reinforcement learning training system.

## Overview

The CAMEL training example combines:

- **CAMEL's ChatAgent**: For building multi-turn conversational agents with flexible
  agent behaviors
- **AReaL's GRPO Algorithm**: For reinforcement learning training with efficient
  distributed rollout and training
- **OpenAI-Compatible API**: Seamless integration through AReaL's OpenAI-compatible
  interface

This integration enables you to:

- Use CAMEL's agent framework for complex multi-turn conversations
- Train agents with RL using AReaL's asynchronous training pipeline
- Collect multiple trajectories per query for more diverse training data
- Automatically track rewards and propagate them through conversation trees

## Prerequisites

Before running the CAMEL training example, ensure you have:

1. Completed the [installation guide](installation.md)
1. Installed CAMEL-AI:

```bash
pip install camel-ai
```

## Quick Start

The CAMEL training example is located in
[**`examples/camel/`**](https://github.com/inclusionAI/AReaL/blob/main/examples/camel/).
To run the example on a single node:

```bash
python3 -m areal.launcher.local examples/camel/train.py \
    --config examples/camel/config.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```

## How It Works

### Architecture Overview

The CAMEL training workflow consists of three main components:

1. **CamelMathAgent**: A wrapper around CAMEL's `ChatAgent` that implicitly integrates
   with AReaL's inference engine
1. **CamelRLVRWorkflow**: A custom `RolloutWorkflow` that orchestrates agent
   interactions and reward collection
1. **Training Loop**: Standard AReaL GRPO training with CAMEL agents for rollout

### Key Components

#### CamelMathAgent

The `CamelMathAgent` class wraps CAMEL's `ChatAgent` with AReaL's OpenAI-compatible
interface:

```python
class CamelMathAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        messages = data["messages"].copy()

        # Create CAMEL agent
        agent = ChatAgent(
            model=AReaLOpenAICompatibleModel(
                openai_client=client,
                tokenizer=self.tokenizer,
                model_type="areal"
            ),
            token_limit=self.max_total_tokens,
        )

        # Run CAMEL agent in asynchronous mode
        response = await agent.astep(messages[-1]["content"])
        content = response.msg.content

        # Evaluate and set reward
        reward = await self.async_reward_fn(result=content, answer=data["answer"])
        client.set_final_reward(reward)
        return reward
```

Key features:

- Uses `AReaLOpenAICompatibleModel` to connect CAMEL agents with AReaL's inference
  engine
- Wraps reward functions with `AsyncRewardWrapper` for non-blocking evaluation
- Sets final rewards on the client for RL training

#### CamelRLVRWorkflow

The workflow class collects multiple trajectories per query and exports them with
discounted rewards:

```python
class CamelRLVRWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )

        # Export completions with reward discounting
        interactions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            interactions = client.export_interactions(style="individual")
            interactions_with_reward.update(interactions)
        return interactions_with_reward
```

This workflow:

- Creates multiple `ArealOpenAI` clients to collect diverse trajectories
- Runs agents in parallel using `asyncio.gather`
- Applies turn-level discounting to rewards (default: 0.9)
- Exports all interactions with their rewards for training

## Configuration

The example configuration file
([`examples/camel/config.yaml`](https://github.com/inclusionAI/AReaL/blob/main/examples/camel/config.yaml))
includes several CAMEL-specific parameters:

```yaml
n_trajs: 2  # Number of trajectories to collect per query
max_tokens_per_trajectory: 8192  # Maximum tokens per trajectory
```

## Customizing the CAMEL Agent

### Using Different CAMEL Agents

You can replace `ChatAgent` with other CAMEL agent types. For example, to use a
different agent with tool calling:

```python
from camel.agents import TaskPlannerAgent

class CamelTaskAgent:
    async def run_agent(self, data, client: ArealOpenAI):
        messages = data["messages"].copy()
        agent = TaskPlannerAgent(
            model=AReaLOpenAICompatibleModel(
                openai_client=client,
                tokenizer=self.tokenizer,
                model_type="areal"
            ),
            tools=[your_tools],
        )
        # ... agent-specific logic ...
```

### Modifying Agent Behavior

CAMEL agents support various configuration options. You can customize them by passing
additional parameters when creating the agent:

```python
agent = ChatAgent(
    model=AReaLOpenAICompatibleModel(...),
    token_limit=max_total_tokens,
    system_message="You are a helpful math assistant.",
    # ... other CAMEL agent parameters ...
)
```

Refer to the [CAMEL-AI documentation](https://github.com/camel-ai/camel) for available
agent types and configuration options.
