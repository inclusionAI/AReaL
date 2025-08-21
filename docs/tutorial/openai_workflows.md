# OpenAI-Compatible Workflows

This guide introduces AReaL's experimental OpenAI-compatible API for writing rollout
workflows. This feature allows you to use familiar OpenAI client patterns while
leveraging AReaL's inference engine and reward system for RL training data collection.

## Overview

The OpenAI-compatible workflow feature provides:

- **Familiar API**: Use OpenAI's chat completions interface with AReaL's backend
- **Reward tracking**: Automatic reward assignment and backpropagation through
  conversation trees
- **Tool call support**: Compatible with OpenAI's function calling features
- **Multi-turn conversations**: Built-in support for conversation flows with reward
  discounting
- **Caching system**: Efficient storage and retrieval of completions with token-level
  data

## Key Components

### `ArealOpenAI` Client

The `ArealOpenAI` client is a drop-in replacement for OpenAI's `AsyncOpenAI` client that
routes requests to AReaL's inference engine:

```python
from areal.experimental.openai import ArealOpenAI

# Create client with AReaL engine and tokenizer
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)

# Use like standard OpenAI client
completion = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_completion_tokens=100
)
```

### Reward Management

The client automatically tracks completions and supports reward assignment:

```python
# Set reward for a completion
client.set_reward(completion.id, reward_value)

# Export all completions with propagated rewards
completions_with_rewards = client.export_completions(turn_discount=0.9)
```

## Creating OpenAI-Compatible Workflows

### Basic Workflow Structure

Here's the structure of an OpenAI-compatible workflow:

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI

class MyOpenAIWorkflow(RolloutWorkflow):
    def __init__(self, reward_fn, gconfig, tokenizer, **kwargs):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def _run_one_episode(self, engine, data, rid):
        # Create OpenAI-compatible client
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # Run conversation and collect rewards
        # ... workflow logic here ...

        # Export completions with rewards
        return client.export_completions()
```

### Multi-Turn Conversation Example

Let's examine the multi-turn workflow implementation:

```python
async def _run_one_episode(self, engine, data, rid):
    client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
    messages = deepcopy(data["messages"])

    t = reward = 0
    discount = 1

    while reward == 0 and t < self.max_turns:
        # Generate response
        completion = await client.chat.completions.create(
            messages=messages,
            temperature=self.gconfig.temperature,
            max_completion_tokens=self.gconfig.max_new_tokens,
            store=True  # Important: enables caching
        )

        # Get completion with token data
        comp_data = client.get_completions(completion.id)

        # Evaluate reward
        reward = await self.async_reward_fn(
            prompt, response, input_tokens, output_tokens, **data
        )

        t += 1

        # Continue conversation if needed
        if reward == 0 and t < self.max_turns:
            messages.append({
                "role": "assistant",
                "content": completion.choices[0].message.content
            })
            messages.append(self.reflection_msg)
            discount *= self.turn_discount

    # Set final reward
    client.set_reward(completion.id, reward * discount)

    return client.export_completions(), comp_data
```

## Important Parameters and Options

### Chat Completion Parameters

The client supports standard OpenAI parameters:

- `messages`: List of conversation messages
- `temperature`: Sampling temperature (0.0 to 2.0)
- `max_completion_tokens`: Maximum tokens in response
- `top_p`: Nucleus sampling parameter
- `frequency_penalty`: Frequency penalty for repetition
- `stop`: Stop sequences
- `tools`: Function calling tools (experimental)
- `store`: Whether to cache the completion for training (default: True)

### Tool Call Support

Enable function calling with tools:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

completion = await client.chat.completions.create(
    messages=messages,
    tools=tools,
    tool_choice="auto"  # or "none" to disable
)
```

## Reward Backpropagation

The system automatically builds conversation trees and backpropagates rewards:

### Tree Structure Based on Role Sequences

The system constructs conversation trees based on message **role sequences**. If one
completion's role structure is a prefix of another completion's role structure, it
becomes the parent node.

**How it works:**

- Each completion's messages are converted to a role sequence:
  `("system", "user", "assistant", "user", ...)`
- Parent-child relationships are determined by prefix matching of these role sequences
- The completion with the longest matching prefix becomes the parent

**Examples:**

```python
# Example 1: Simple linear conversation
# Completion 1: [system, user] -> role sequence: ("system", "user")
# Completion 2: [system, user, assistant, user] -> role sequence: ("system", "user", "assistant", "user")
# Completion 3: [system, user, assistant, user, assistant, user] -> role sequence: ("system", "user", "assistant", "user", "assistant", "user")
# Tree: C1 -> C2 -> C3 (linear chain)

# Example 2: Branching conversation
# Completion A: [system, user] -> ("system", "user")
# Completion B: [system, user, assistant, user] -> ("system", "user", "assistant", "user")
# Completion C: [system, user, assistant, user] -> ("system", "user", "assistant", "user")
# Completion D: [system, user, assistant, user, assistant, user] -> ("system", "user", "assistant", "user", "assistant", "user")
# Tree: A  ->  B  ->  D
#          \-> C ->/

# Example 3: Tool calling conversation
# Completion X: [system, user] -> ("system", "user")
# Completion Y: [system, user, assistant, tool, user] -> ("system", "user", "assistant", "tool", "user")
# Completion Z: [system, user, assistant, tool, user, assistant] -> ("system", "user", "assistant", "tool", "user", "assistant")
# Tree: X -> Y -> Z (tool messages are part of the role sequence)
```

**Reward Backpropagation Process:**

- All nodes can receive their explicitly set rewards
- All nodes also receive discounted average of their children's rewards
- Processing occurs in topological order (leaves first, then parents)

### Export with Discounting

```python
# Export with turn-level discounting
completions = client.export_completions(turn_discount=0.9)

# Access reward data
for completion_id, comp_data in completions.items():
    reward = comp_data.reward
    tensor_dict = comp_data.to_tensor_dict()  # Convert to training format
```

## Additional Examples

For more concrete examples and test cases, check the test suite at
`areal/experimental/tests/test_openai.py`.
