# Agentic Reinforcement Learning

This guide demonstrates how to use AReaL to train agentic models with popular agent
frameworks, e.g., [CAMEL-AI](https://github.com/camel-ai/camel),
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python), etc, enabling you
to leverage their agent orchestration capabilities while using AReaL's distributed
reinforcement learning training system.

## Overview

The core design philosophy of agentic RL in AReaL is **unified training and
deployment**. In other words, we expect the user to use the same code both training and
evaluation without any change. However, it is usually difficult to do so because the
following features of agent programming frameworks:

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

1. **Proxy model client with token-level tracking**: AReaL sets up a HTTP proxy LLM
   server that routes all LLM calls to AReaL's inference engine (SGLang or vLLM). Every
   interaction (completion/response) is automatically tracked with complete token-level
   information including input tokens, output tokens, and associated log probabilities.
   This enables RL algorithms to access all the granular data for policy gradient
   computation, while retaining the frontend flexibility with various agent frameworks.

1. **Reward assignment and propagation**: AReaL provides a flexible reward system that
   allows you to assign rewards to specific interactions or entire trajectories via
   direct HTTP calls to the proxy server. The system automatically builds conversation
   trees based on message role sequences and supports reward backpropagation with
   customizable discounting factors, enabling automatic reward assignment across
   multi-turn conversations.

1. **Parallel trajectory collection**: AReaL's workflow system enables concurrent
   execution of multiple agent instances, allowing you to collect diverse trajectories
   for each query. This is essential for effective RL training, as it increases sample
   diversity and improves policy gradient estimates.

We demonstrate several concrete examples below. More examples can be found in the
[`workflow/` directory](../../areal/workflow/).

## Examples

### Training with OpenAI Agent

#### Step 1: Build a Runnable Agent with the ReaL OpenAI API

Implement a standard agent loop with tool calling. This code snippet has nothing to do
with AReaL. This agent can run with OpenAI's official models with a proper API key and
base URL.

While the following example just invokes the LLM for a single turn, you can add a
for-loop to generate multi-turn responses.

Place the following content in `my_agent.py` where AReaL can import:

```python
from agents import (
    Agent,
    OpenAIProvider,
    RunConfig,
    SQLiteSession,
    function_tool,
)
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))

class MathAgent:
    async def run(self, data, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None)
        # IMPORTANT: replace the `base_url`
        client = AsyncOpenAI(base_url=base_url, http_client=http_client, max_retries=0)
        content = data["messages"][-1]["content"]
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",  # no need to pass
            tracing_disabled=True,
        )
        agent = Agent(
            name="RLVR Math with Calculator",
            instructions="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
            tools=[add, multiply],
        )
        session = SQLiteSession("math")
        result = await OpenAIRunner.run(
            agent, input=content, session=session, run_config=run_config
        )
        # return reward
        return math_reward_fn(completions=result.final_output, answer=data["answer"])
```

#### Step 2: Integrating the Agent Directly

Passing the path of the above agent to the trainer, then everything is done!

```python
import sys

from areal.api.cli_args import load_expr_config, GRPOConfig
from areal.dataset import get_custom_dataset
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer

def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset
    ) as trainer:
        trainer.train(
            workflow="my_agent.MathAgent"
        )

if __name__ == "__main__":
    main(sys.argv[1:])
```

The full working OpenAI Agents training example is located in
[**`examples/experimental/proxy/`**](https://github.com/inclusionAI/AReaL/blob/main/examples/experimental/proxy/).
To run the example on a single node:

```bash
python3 examples/experimental/proxy/train.py \
    --config examples/experimental/proxy/config.yaml \
    scheduler.type=local workflow=my_agent.MathAgent
```

### Training with CAMEL-AI

CAMEL-AI is an open-source, modular framework for building intelligent multi-agent
systems. It provides a flexible agent architecture that can handle complex dialogue
flows, tool calling, and multi-agent interactions.

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

### More Examples

Beyond the two examples above, AReaL supports integration with various other agent
frameworks and SDKs:

- **Claude Agent SDK**: Train agents using Anthropic's Claude Agent SDK with MCP tools.
  See the
  [Claude example](https://github.com/inclusionAI/AReaL/blob/main/areal/workflow/anthropic/claude_math_agent.py)
  for a math agent with calculator tools.

- **LangChain**: Integrate LangChain agents with AReaL's training infrastructure. See
  the
  [LangChain example](https://github.com/inclusionAI/AReaL/blob/main/areal/workflow/langchain/math_agent.py)
  for details.

## Under the Hood

### Two Integration Paradigms

The two examples above demonstrate two distinct approaches to integrating agent
frameworks with AReaL's training infrastructure. Both produce a workflow object that can
be passed to the trainer, but they differ in their level of coupling with AReaL
internals.

**Proxy Approach (OpenAI Agent Example)**

The proxy approach keeps your agent code completely independent from AReaL. Your agent
uses the standard OpenAI SDK with a customized `base_url` pointing to AReaL's proxy
server. This is where the name "proxy" comes from—AReaL sets up an HTTP server that
implements the same endpoints as OpenAI's API (`/v1/chat/completions`, `/v1/responses`,
etc.) but routes inference requests to the model being trained via SGLang or vLLM.

This approach offers maximum flexibility: you can use any agent framework (OpenAI Agents
SDK, LangChain, OpenHands, etc.) without modification, as long as it allows configuring
the API base URL. The same agent code works unchanged for both training and deployment.

**Direct Approach (CAMEL-AI Example)**

The direct approach uses AReaL's `ArealOpenAI` client, which extends `AsyncOpenAI` and
binds directly to the inference engine:

```python
# ArealOpenAI wraps the engine and provides OpenAI-compatible API
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)
```

This requires modifying your agent code to accept the AReaL client instead of creating
its own. In exchange, you gain access to lower-level engine state (such as the current
rollout version) and avoid the overhead of HTTP communication. This approach is
well-suited when you're building a custom workflow that needs tight integration with
AReaL's internals.

**When to Use Each Approach**

| Consideration           | Proxy Approach                          | Direct Approach                       |
| ----------------------- | --------------------------------------- | ------------------------------------- |
| Code modification       | None (just change `base_url`)           | Must accept `ArealOpenAI` client      |
| Framework compatibility | Any OpenAI-compatible framework         | Frameworks that accept custom clients |
| Performance             | HTTP overhead (minimal)                 | Direct engine calls                   |
| Engine state access     | Limited                                 | Full access                           |
| Recommended for         | Existing agents, third-party frameworks | Custom workflows, tight integration   |

### Architecture Overview

AReaL's agentic training infrastructure consists of three layers that work together to
provide OpenAI-compatible APIs while capturing full token-level data for RL training.

**Layer 1: Training Controller**

At the top level, `PPOTrainer` orchestrates the training loop: rollout generation,
advantage computation, PPO updates, and weight synchronization. When an agent workflow
is detected (a class with a `run` method rather than a `RolloutWorkflow`), the
controller initializes proxy workers to handle OpenAI-compatible requests.

**Layer 2: Proxy Workers**

Each proxy worker runs a FastAPI server
([`proxy_rollout_server.py`](../../areal/experimental/openai/proxy/proxy_rollout_server.py))
that implements OpenAI-compatible endpoints. The server manages:

- **Session lifecycle**: Each agent execution gets a unique session via
  `/rl/start_session`. All subsequent requests include the session ID in the URL path
  (e.g., `/{session_id}/v1/chat/completions`).
- **Token tracking**: Every completion is stored with full token-level information—input
  token IDs, output token IDs, and log probabilities—in an `InteractionCache`.
- **Reward management**: Rewards can be assigned via `/rl/set_reward` endpoint, either
  to specific completions by ID or to the most recent completion.
- **Export**: The `/export_trajectories` endpoint applies reward discounting and returns
  all interactions in a format ready for training.

**Layer 3: Inference Engine**

The bottom layer consists of inference servers (SGLang or vLLM) that perform actual
model inference. The `ArealOpenAI` client handles tokenization, calls the engine's
`agenerate` method with token IDs, and detokenizes the output back to text for the
OpenAI-compatible response.

### Request Flow

A typical agentic rollout flows through the system as follows:

**1. Session Initialization**

Before executing an agent, the controller grants capacity and starts a session:

```
POST /grant_capacity          → Reserves a slot (staleness control)
POST /rl/start_session        → Returns unique session_id (e.g., "task-0-0")
```

The session ID namespaces all subsequent requests, enabling concurrent agent executions
without conflicts.

**2. Agent Execution**

The agent code runs with the proxy's base URL. Each LLM call goes through the proxy:

```
POST /{session_id}/v1/chat/completions
  → Proxy receives request
  → ArealOpenAI tokenizes messages using chat template
  → Engine generates tokens with log probabilities
  → Response detokenized and returned as ChatCompletion
  → Interaction stored in session's InteractionCache
```

For multi-turn conversations, the agent continues calling the same endpoint. Each
completion is stored with a reference to its parent (determined by message prefix
matching), building a conversation tree.

**3. Reward Assignment**

After the agent completes its task, rewards are assigned:

```
POST /{session_id}/rl/set_reward
  Body: {"reward": 1.0}                    → Sets reward on last completion
  Body: {"interaction_id": "...", "reward": 0.5}  → Sets reward on specific completion
```

The workflow can return a scalar (applied to last completion) or a dict mapping
completion IDs to rewards for fine-grained control.

**4. Export and Cleanup**

When the session ends, interactions are exported for training:

```
POST /{session_id}/rl/end_session   → Marks session as complete
POST /export_trajectories
  Body: {"session_id": "...", "discount": 0.99, "style": "individual"}
  → Applies reward backpropagation through conversation tree
  → Returns all interactions with token data and final rewards
```

The export process propagates rewards backward through the conversation tree: leaf nodes
keep their assigned rewards, while parent nodes receive discounted rewards from their
children.

### Reward Backpropagation

For multi-turn conversations, AReaL automatically builds a conversation tree and
propagates rewards from leaf nodes back to earlier turns.

**Tree Construction**

Parent-child relationships are determined by message role sequences. If one completion's
message list is a prefix of another's, it becomes the parent:

```
Completion A: [system, user]                    → root
Completion B: [system, user, assistant, user]   → child of A
Completion C: [system, user, assistant, user, assistant, user] → child of B
```

This creates a linear chain: A → B → C. Branching occurs when multiple completions share
the same parent (e.g., exploring different responses to the same context).

**Discount Application**

Rewards propagate backward with geometric discounting. Given a discount factor γ (e.g.,
0.99), a parent node's reward is computed as:

```
reward(parent) = assigned_reward(parent) + γ × mean(reward(children))
```

Processing occurs in reverse topological order (leaves first), ensuring children's
rewards are finalized before propagating to parents.

### Training with Agent Trajectories

A complete agentic episode may contain multiple LLM interactions (turns). For training,
these are treated as independent input-output-reward tuples:

```
Turn 1: [system, user]           → output_1 → reward_1 (discounted)
Turn 2: [system, user, asst, user] → output_2 → reward_2 (discounted)
Turn 3: [system, user, asst, user, asst, user] → output_3 → reward_3 (final)
```

Each tuple includes the full token-level data needed for policy gradient computation:
input token IDs, output token IDs, and log probabilities. The discounted rewards ensure
the RL objective correctly credits earlier actions for final outcomes.

**Token Consistency Guarantee**

Because AReaL stores the actual tokens used during inference (not re-tokenized text),
there is no risk of tokenization mismatch between rollout and training. The tokens sent
to the inference engine are exactly the tokens used for gradient computation.

**Efficient Training with Tree Attention**

Multi-turn trajectories often share long token prefixes, which can slow down training
due to redundant computation. AReaL addresses this with prefix-shared tree attention,
which computes attention over shared prefixes only once. See the tree attention
documentation for details.
