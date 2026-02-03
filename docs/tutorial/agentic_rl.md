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

CAMEL-AI is an open-source, modular framework for building intelligent multi-agent
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
client.set_last_reward(reward)
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
        client.set_last_reward(reward)

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
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer

        # Create our agent wrapper
        self.agent = CamelMathAgent(tokenizer=self.tokenizer)

    async def arun_episode(self, engine, data):
        """Run one training episode: collect trajectories and return training data."""
        # Create one client per trajectory (enables parallel collection)
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # Run agents in parallel
        reward = await self.agent.run_agent(data=data, client=client)

        # Apply reward discounting for multi-turn conversations
        client.apply_reward_discount(turn_discount=0.9)
        # Export interactions with token-level data
        return client.export_interactions(style="individual")
```

**Key points:**

- **Parallel episode execution**: AReaL's training loop calls `arun_episode` in parallel
  across multiple samples in a batch, enabling parallel trajectory collection at the
  batch level.
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
python3 examples/camel/train.py \
    --config examples/camel/config.yaml \
    scheduler.type=local \
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
demonstrates a **modular, configuration-based approach** that separates agent definition
from training logic. Instead of defining agents directly in your training workflow, you
create reusable agent builder functions and reference them via configuration.

**Key architectural advantages:**

1. **Modularity**: Agent definitions live in separate, reusable modules
1. **Configuration-driven**: Specify agent builders and reward functions via config
   paths
1. **Generic workflow**: One workflow class works with any agent builder
1. **Easy experimentation**: Test different agents by changing config, not code

### Building An Agent with OpenAI SDK

#### Step 1: Create an Agent Builder Function

First, create an agent builder function that returns an OpenAI `Agent` object. This can
be a simple single agent or a complex multi-agent workflow with handoffs:

```python
# In areal/workflow/openai_agent/math_agent.py
from agents import Agent as OpenAIAgent
from agents import handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

def build_math_agent() -> OpenAIAgent:
    """Create a multi-agent workflow for math problem solving."""

    # Create specialized agents for different reasoning stages
    problem_analyzer = OpenAIAgent(
        name="Problem Analyzer",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a math problem analyzer. Your job is to:
        1. Carefully read and understand the math problem
        2. Identify the type of problem (algebra, geometry, arithmetic, etc.)
        3. Break down the problem into key components
        ...
        """
    )

    solution_specialist = OpenAIAgent(
        name="Solution Specialist",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a math solution specialist...
        """
    )

    # Create main orchestrator with handoffs
    main_agent = OpenAIAgent(
        name="Math Problem Solver",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are a math problem solving coordinator...
        """,
        handoffs=[
            handoff(agent=problem_analyzer, ...),
            handoff(agent=solution_specialist, ...),
        ]
    )

    return main_agent
```

For the complete multi-agent implementation, see
[`areal/workflow/openai_agent/math_agent.py`](https://github.com/inclusionAI/AReaL/blob/main/areal/workflow/openai_agent/math_agent.py).

#### Step 2: Configure the Training

In your config file, specify the paths to your agent builder and reward function:

```yaml
# examples/openai_agents/config.yaml
reward_fn_path: "areal.reward.gsm8k.gsm8k_reward_fn"
agent_builder_path: "areal.workflow.openai_agent.math_agent.build_math_agent"
agent_builder_kwargs: {}  # Optional kwargs for the builder function

gconfig:
  n_samples: 4  # Collect 4 trajectories per query for diversity
  max_tokens: 8192  # Maximum tokens per agent interaction
  temperature: 1.0
```

**Key parameters:**

- **`agent_builder_path`**: Python import path to your agent builder function
- **`reward_fn_path`**: Python import path to your reward function
- **`agent_builder_kwargs`**: Optional dictionary of arguments to pass to the builder
- **`gconfig.n_samples`**: Number of parallel trajectories to collect per query
- **`gconfig.max_tokens`**: Maximum tokens for each agent interaction

#### Step 3: Use the Generic Workflow

AReaL provides a generic `OpenAIAgentWorkflow` that works with any agent builder:

```python
# In examples/openai_agents/train_agents.py
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils.dynamic_import import import_from_string

class OpenAIAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        agent_builder_path: str,
        agent_builder_kwargs: dict,
        reward_fn_path: str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer

        # Dynamically import agent builder and reward function
        self.agent = OpenAIAgentWrapper(
            agent_builder_path=agent_builder_path,
            agent_builder_kwargs=agent_builder_kwargs,
            reward_fn_path=reward_fn_path,
            temperature=gconfig.temperature,
            max_tokens=gconfig.max_tokens,
        )

    async def arun_episode(self, engine, data):
        """Run one training episode: collect trajectories and return training data."""
        # Create one client per trajectory (controlled by gconfig.n_samples)
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # Run agents in parallel
        reward = await self.agent.run_agent(data=data, client=client)

        # Export all interactions with rewards
        client.apply_reward_discount(turn_discount=0.9)
        return client.export_interactions(style="individual")
```

The `OpenAIAgentWrapper` handles the agent execution:

```python
class OpenAIAgentWrapper:
    def __init__(
        self,
        agent_builder_path: str,
        agent_builder_kwargs: dict,
        reward_fn_path: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ):
        # Dynamically import the builder and reward functions
        self.agent_builder = import_from_string(agent_builder_path)
        self.async_reward_fn = AsyncRewardWrapper(
            import_from_string(reward_fn_path)
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run_agent(self, data, client: ArealOpenAI):
        # Build agent using the imported builder function
        agent = self.agent_builder(**self.agent_builder_kwargs)

        # Configure to use ArealOpenAI
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )

        # Run agent
        result = await OpenAIRunner.run(
            agent,
            input=data["messages"][-1]["content"],
            run_config=run_config
        )

        # Compute and set reward
        reward = await self.async_reward_fn(
            completions=result.final_output,
            answer=data["answer"],
            prompt=None,
            prompt_ids=None,
            completion_ids=None,
        )
        client.set_last_reward(reward)

        return reward
```

#### Step 4: Run Training with PPOTrainer

Use AReaL's `PPOTrainer` for streamlined training:

```python
from areal.experimental.trainer import PPOTrainer

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        # Create workflow with config parameters
        workflow = OpenAIAgentWorkflow(
            agent_builder_path=config.agent_builder_path,
            agent_builder_kwargs=config.agent_builder_kwargs,
            reward_fn_path=config.reward_fn_path,
            gconfig=config.gconfig,
            tokenizer=tokenizer,
        )

        eval_workflow = OpenAIAgentWorkflow(
            agent_builder_path=config.agent_builder_path,
            agent_builder_kwargs=config.agent_builder_kwargs,
            reward_fn_path=config.reward_fn_path,
            gconfig=config.gconfig,
            tokenizer=tokenizer,
        )

        # Start training
        trainer.train(workflow, eval_workflow)
```

For a complete implementation, refer to the
[complete training script](https://github.com/inclusionAI/AReaL/blob/main/examples/openai_agents/train_agents.py).

### Complete Example

The full working OpenAI Agents training example is located in
[**`examples/openai_agents/`**](https://github.com/inclusionAI/AReaL/blob/main/examples/openai_agents/).
To run the example on a single node:

```bash
python3 examples/openai_agents/train_agents.py \
    --config examples/openai_agents/config.yaml \
    scheduler.type=local \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```

### Customization

The modular architecture makes it easy to customize your agent training:

#### Creating Custom Agent Builders

You can create any agent workflow by writing a new builder function:

```python
# In your custom module, e.g., my_package/custom_agent.py
from agents import Agent as OpenAIAgent

def build_custom_agent() -> OpenAIAgent:
    """Build your custom agent with any configuration."""
    agent = OpenAIAgent(
        name="CustomAgent",
        instructions="Your custom instructions...",
        # Add any OpenAI Agent SDK features:
        # - Tools
        # - Handoffs
        # - Custom configurations
    )
    return agent
```

Then reference it in your config:

```yaml
agent_builder_path: "my_package.custom_agent.build_custom_agent"
```

#### Passing Arguments to Agent Builders

Use `agent_builder_kwargs` to parameterize your agent builder:

```python
def build_parameterized_agent(
    system_message: str,
    enable_tools: bool = True
) -> OpenAIAgent:
    """Agent builder that accepts parameters."""
    agent = OpenAIAgent(
        name="ParameterizedAgent",
        instructions=system_message,
        tools=get_tools() if enable_tools else None,
    )
    return agent
```

Configure it in your config file:

```yaml
agent_builder_path: "my_package.agent.build_parameterized_agent"
agent_builder_kwargs:
  system_message: "You are a specialized assistant..."
  enable_tools: true
```

#### Using Custom Reward Functions

Similarly, you can specify custom reward functions:

```python
# In my_package/rewards.py
def custom_reward_fn(completions, answer, prompt, prompt_ids, completion_ids):
    """Custom reward function following AReaL's API."""
    # Your custom logic here
    return reward_value
```

Reference it in your config:

```yaml
reward_fn_path: "my_package.rewards.custom_reward_fn"
```

For comprehensive details on agent instructions, handoffs, `ModelSettings`, and
additional configuration options, refer to the
[OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/).
