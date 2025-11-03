# Training with OpenAI Agents

This guide demonstrates how to use the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) with AReaL for
training agentic models. The OpenAI Agents SDK provides a high-level framework for
building multi-agent workflows with handoffs, tool calling, and structured agent
interactions, while AReaL handles the underlying RL training loop.

## Overview

AReaL's OpenAI Agents integration enables you to:

- **Use OpenAI Agents SDK**: Leverage the `openai-agents` library's `Agent`, `handoff`,
  `Runner`, etc. APIs to define complex multi-agent workflows
- **Automatic token-level tracking**: All agent interactions (completions/responses) are
  automatically tracked with token-level logging and reward assignment
- **Tool calling support**: Built-in support for function calling through the agents SDK
- **Multi-agent coordination**: Support for agent handoffs and coordination between
  specialized agents
- **Seamless RL training**: Convert agent trajectories directly into RL training data

## Architecture

The integration works by providing `ArealOpenAI`, a drop-in replacement for OpenAI's
`AsyncOpenAI` client that routes all LLM calls to AReaL's inference engine:

```
┌─────────────────────┐
│  OpenAI Agents SDK  │
│   (Agent, Runner)   │
└──────────┬──────────┘
           │ Uses
           ▼
┌─────────────────────┐
│     ArealOpenAI     │  ← Drop-in replacement for AsyncOpenAI
└──────────┬──────────┘
           │ Routes to
           ▼
┌─────────────────────┐
│   AReaL Inference   │
│       Engine        │
│    (SGLang/vLLM)    │
└─────────────────────┘
```

All completions and responses are cached with token-level information (tokens, logprobs,
etc.) and can be exported as training data with rewards.

## Quick Start

### Prerequisites

Install the OpenAI Agents SDK:

```bash
pip install openai-agents
```

### Running the Example

The complete example can be found in `examples/openai-agents/`. To run the math agent
training:

```bash
python3 -m areal.launcher.local examples/openai-agents/train_agents.py \
    --config examples/openai-agents/config.yaml \
    experiment_name=my_agent_experiment \
    trial_name=trial0
```

This will:

1. Train an agent using the OpenAI Agents SDK to solve GSM8K math problems
1. Use AReaL's inference engine (SGLang) for all LLM calls
1. Collect trajectories with rewards
1. Train the model using GRPO

### Configuration

The key configuration options for OpenAI Agents training are:

```yaml
# Agent-specific settings
agent_type: math  # or "multi_agent_math"
n_trajs: 2        # Number of trajectories to collect per prompt
max_turns: 8      # Maximum number of agent turns per trajectory
max_tokens_per_trajectory: 8192
```

See
[examples/openai-agents/config.yaml](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/config.yaml)
for the full configuration.

## Creating Your Own Agent Workflow

### Basic Single-Agent Workflow

Here's how to create a simple agent using OpenAI Agents SDK:

```python
from agents import Agent as OpenAIAgent
from agents import OpenAIProvider, RunConfig
from agents import Runner as OpenAIRunner
from areal.experimental.openai import ArealOpenAI
from areal.api.reward_api import AsyncRewardWrapper

class MyAgent:
    def __init__(self):
        # Wrap your reward function to make it async
        self.async_reward_fn = AsyncRewardWrapper(my_reward_function)

    async def run_agent(self, data, client: ArealOpenAI):
        # Define your agent with instructions
        agent = OpenAIAgent(
            name="MyAgent",
            instructions="You are a helpful assistant that solves problems."
        )

        # Configure the runner to use ArealOpenAI
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
        )

        # Run the agent
        result = await OpenAIRunner.run(
            agent,
            input=data["messages"][-1]["content"],
            run_config=run_config
        )

        # Evaluate and set reward
        reward = await self.async_reward_fn(
            result=result.final_output,
            answer=data["answer"]
        )
        client.set_final_reward(reward)

        return reward
```

Then create a workflow that uses the agent:

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI

class MyAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        ...
    ):
        self.agent = MyAgent()
        ...

    async def arun_episode(self, engine, data):
        # Create ArealOpenAI clients for each trajectory
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser="qwen25"  # or None if no tool calling
            )
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories in parallel
        rewards = await asyncio.gather(*[
            self.agent.run_agent(data=data, client=clients[i])
            for i in range(self.n_trajs)
        ])

        # Export interactions with rewards
        interactions_with_reward = {}
        for client in clients:
            # Apply reward discounting across turns
            client.apply_reward_discount(turn_discount=0.9)
            # Export all interactions for training
            interactions = client.export_interactions(style="individual")
            interactions_with_reward.update(interactions)

        return interactions_with_reward
```

### Multi-Agent Workflow

The OpenAI Agents SDK supports **agent handoffs**, allowing you to create specialized
agents that can delegate tasks to each other. Here's a simple example from the
multi-agent math workflow:

```python
from agents import Agent as OpenAIAgent
from agents import handoff

class MultiAgentMathAgent:
    ...

    def _create_agent_workflow(self) -> OpenAIAgent:
        # Create specialized agents
        problem_analyzer = OpenAIAgent(
            name="Problem Analyzer",
            instructions="""You analyze math problems and break them down.
            If you need help solving, hand off to the Solution Specialist."""
        )

        solution_specialist = OpenAIAgent(
            name="Solution Specialist",
            instructions="""You solve math problems step by step.
            If you need verification, hand off to the Verification Agent."""
        )

        verification_agent = OpenAIAgent(
            name="Verification Agent",
            instructions="You verify solutions and provide final answers."
        )

        # Create main orchestrator with handoffs
        main_agent = OpenAIAgent(
            name="Math Problem Solver",
            instructions="""You coordinate solving math problems.
            Use handoffs to specialized agents as needed.""",
            handoffs=[
                handoff(
                    agent=problem_analyzer,
                    tool_name_override="analyze_problem",
                    tool_description_override="Analyze problem structure"
                ),
                handoff(
                    agent=solution_specialist,
                    tool_name_override="solve_problem",
                    tool_description_override="Solve the problem step by step"
                ),
                handoff(
                    agent=verification_agent,
                    tool_name_override="verify_solution",
                    tool_description_override="Verify the solution"
                ),
            ],
        )

        return main_agent

    async def run_agent(self, data, client: ArealOpenAI):
        agent = self._create_agent_workflow()
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
        )

        result = await OpenAIRunner.run(
            agent,
            input=data["messages"][-1]["content"],
            run_config=run_config
        )

        # set reward
        ...

        return reward
```

### Tool Calling Support

AReaL supports function calling through the OpenAI Agents SDK. The tool calls are parsed
and integrated into the conversation tree:

```python
# When creating ArealOpenAI client, specify tool_call_parser
client = ArealOpenAI(
    engine=engine,
    tokenizer=tokenizer,
    # Parser for tool call format (e.g., "qwen25", "deepseekv3", "gpt-oss", etc.)
    tool_call_parser="qwen25",
)
```

The tool call parser handles extracting function names and arguments from the model
output using SGLang's `Tool Parser`. Refer to the
[SGLang documentation](https://docs.sglang.ai/advanced_features/tool_parser.html) for
more supported parsers.

## Reward Management

### Setting Rewards

Rewards can be set at different levels:

1. **Per-interaction reward**: Set reward for a specific interaction by ID

   ```python
   client.set_reward(interaction_id, reward_value)
   ```

1. **Final reward**: Set reward for the most recent interaction

   ```python
   client.set_final_reward(reward_value)
   ```

### Reward Discounting

When you have multi-turn conversations, you may want to discount rewards for earlier
turns. Use `apply_reward_discount`:

```python
# Apply geometric discounting: reward[i] += reward[i+1] * turn_discount
client.apply_reward_discount(turn_discount=0.9)
```

This propagates rewards backward through the conversation tree, so earlier turns receive
partial credit based on later success.

### Exporting Interactions

After setting rewards, export interactions for training:

```python
# Export all individual interactions
interactions = client.export_interactions(style="individual")

# Each interaction contains:
# - completion/response: The OpenAI API response
# - model_response: AReaL's model response with token-level data
# - reward: The assigned reward
# - to_tensor_dict(): Convert to training format
```

The `style="individual"` export returns all interactions separately. Use
`style="concat"` to export only leaf nodes of the built conversation tree (requires
`chat_template_type="concat"`).

## Integration with Training Script

To use your custom workflow in training, create or update `train_agents.py`:

```python
from areal.api.cli_args import AgentRLConfig

@dataclass
class AgentRLConfig(GRPOConfig):
    agent_type: str = field(default="my_agent")
    n_trajs: int = field(default=1)
    max_turns: int = field(default=8)
    max_tokens_per_trajectory: int = field(default=32768)

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    # ... setup actor, rollout, etc. ...

    # Create workflow based on agent type
    if config.agent_type == "my_agent":
        from my_workflow import MyAgentWorkflow

        workflow = MyAgentWorkflow(
            gconfig=config.gconfig,
            tokenizer=tokenizer,
            n_trajs=config.n_trajs,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
        )
    # ... continue with training loop ...
```

## Best Practices

### 1. Async Reward Functions

Always wrap blocking reward functions with `AsyncRewardWrapper`:

```python
from areal.api.reward_api import AsyncRewardWrapper

async_reward_fn = AsyncRewardWrapper(my_reward_function)
```

This ensures rewards don't block the async rollout loop.

### 2. Parallel Trajectory Collection

Use `asyncio.gather` to collect multiple trajectories in parallel:

```python
rewards = await asyncio.gather(*[
    self.agent.run_agent(data=data, client=clients[i])
    for i in range(self.n_trajs)
])
```

### 3. Reward Discounting Strategy

Consider the discount factor based on your task:

- **High discount (0.9-1.0)**: Early turns are nearly as important as final answer
- **Low discount (0.5-0.7)**: Focus learning on final outcomes

```python
client.apply_reward_discount(turn_discount=0.9)  # 10% per turn
```

### 4. Tool Call Parser Selection

Choose the tool call parser based on your model:

- **Qwen models**: Use `tool_call_parser="qwen25"`
- **Other models**: Use the appropriate parser (e.g., "deepseekv3", "gpt-oss", etc.)

### 5. Multi-Agent Coordination

When using handoffs:

- Keep agent instructions clear and focused
- Use descriptive tool names and descriptions
- Test agent handoffs independently before RL training

## Examples

Complete examples are available in `examples/openai-agents/`:

- [**`math_workflow.py`**](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/math_workflow.py):
  Single-agent math problem solving workflow
- [**`multi_agent_math_workflow.py`**](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/multi_agent_math_workflow.py):
  Multi-agent math problem solving workflow with handoffs
- [**`train_agents.py`**](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/train_agents.py):
  Training script that integrates workflows
- [**`config.yaml`**](https://github.com/inclusionAI/AReaL/blob/main/examples/openai-agents/config.yaml):
  Configuration file with all options

## Troubleshooting

### Import Errors

If you see import errors for `agents`:

```bash
pip install openai-agents
```

### Tool Calls Not Parsed Correctly

If tool calls aren't being extracted:

1. Verify `tool_call_parser` matches your model's format
1. Check that tools are provided in the correct format
1. Review model output to ensure it contains tool calls

### Rewards Not Propagating

If rewards aren't propagating through turns:

1. Ensure `client.set_final_reward()` is called before export
1. Check that `apply_reward_discount()` is called before export
1. Verify `export_interactions()` is called with the correct style

## Related Documentation

- [OpenAI-Compatible Workflows](openai_workflows.md): Lower-level API for
  OpenAI-compatible workflows
