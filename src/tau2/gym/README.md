# Tau2 Gym

A Gymnasium-compatible environment for evaluating conversational agents in the τ²-bench framework. This module provides a standardized gym interface that allows you to run agents step-by-step in controlled simulation environments.

## Overview

The Tau2 Gym module extends the standard `gym.Env` interface to provide a conversational agent evaluation environment. It follows the [Gymnasium API standard](https://gymnasium.farama.org/) for reinforcement learning environments.

## Usage

### Basic Setup

```python
import gymnasium as gym
from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID

# Register the environment (only needed once)
register_gym_agent()

# Create environment for a specific domain and task
domain = "mock"
task_id = "create_task_1"

env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task_id)
```

### Simple Agent Interaction

```python
# Initialize the environment
observation, info = env.reset()
print(f"Initial observation: {observation}")

# Access available tools and policy from info
print(f"Available tools: {info['tools']}")
print(f"Agent policy: {info['policy']}")

# Execute agent actions step by step
action = "Hello! I'm here to help you with your request."
observation, reward, terminated, truncated, info = env.step(action)

print(f"Observation: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")

next_action = "create_task(user_id='user_1', title='Important Meeting')"
observation, reward, terminated, truncated, info = env.step(next_action)
print(f"Observation: {observation}")
print(f"Step reward: {reward}")
```

### Info Dictionary

Both `reset()` and `step()` methods return an `info` dictionary containing important context about the environment:

- **`tools`**: List of available tools/actions the agent can use in the current domain
- **`policy`**: The policy string that defines the agent's behavior and constraints
- **`simulation_run`**: JSON representation of the current simulation state (when available)

```python
# Example of accessing info contents
observation, info = env.reset()

# View available tools
for tool in info['tools']:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Parameters: {tool.parameters}")

# View agent policy
print(f"Agent must follow this policy: {info['policy']}")
```

### Actions Input Format

#### Tool Calls

For tool calls (actions that interact with the environment), you can use either JSON or functional format:

**JSON Format:**
```python
# JSON-formatted tool call
action = '{"name": "search_flights", "arguments": {"origin": "NYC", "destination": "LAX"}}'
observation, reward, terminated, truncated, info = env.step(action)
```

**Functional Format:**
```python
# Function-style tool call with keyword arguments
action = "search_flights(origin='NYC', destination='LAX')"
observation, reward, terminated, truncated, info = env.step(action)

# Another example with different parameters
action = "create_task(user_id='user_1', title='Important Meeting', priority='high')"
observation, reward, terminated, truncated, info = env.step(action)
```

#### Message to User

For communication with the user (non-tool actions), simply use a string:

```python
# Plain text message to the user
action = "Hello! I'm here to help you with your request."
observation, reward, terminated, truncated, info = env.step(action)

action = "I understand you need to book a flight. Let me search for available options."
observation, reward, terminated, truncated, info = env.step(action)
```

### Observation Format

The `observation` returned by `reset()` and `step()` is a string representation of the conversation history. Each message is formatted as `"role: content"` and messages are separated by newlines.

#### Message Types

**User Messages:**
- Plain text: `"user: Hello, I need help booking a flight"`
- Note: User tool calls are not visible in the observation

**Assistant Messages:**
- Plain text: `"assistant: I'll help you book a flight. Let me search for available options."`
- Tool calls: `"assistant: search_flights(origin='NYC', destination='LAX')"`

**Tool Results:**
- Tool call results: `"tool: {"name": "search_flights", "arguments": {"origin": "NYC", "destination": "LAX"}, "result": "Found 3 flights..."}"`
- Tool results are returned in JSON format with role "tool"

#### Example Observation

```python
observation, info = env.reset()
print(observation)
# Output might be:
# user: Hello, I need help booking a flight from New York to Los Angeles
# assistant: I'll help you book a flight. Let me search for available options.
# assistant: search_flights(origin='NYC', destination='LAX')
# tool: {"name": "search_flights", "arguments": {"origin": "NYC", "destination": "LAX"}, "result": "Found 3 flights: Flight 123, Flight 456, Flight 789"}
```

The observation string provides the complete conversation context, making it easy to understand the current state of the interaction and plan the next action accordingly.