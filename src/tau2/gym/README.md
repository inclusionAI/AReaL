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

# Or with additional configuration
env = gym.make(
    TAU_BENCH_ENV_ID, 
    domain=domain, 
    task_id=task_id,
    solo_mode=True,
    user_llm="gpt-4",
    user_llm_args={"temperature": 0.7}
)
```

### Environment Configuration

The `AgentGymEnv` supports several configuration options to customize the simulation behavior:

```python
from tau2.gym.gym_agent import AgentGymEnv

# Basic environment (normal mode with default user simulator)
env = AgentGymEnv(domain="retail", task_id="book_flight_1")

# Solo mode - agent works independently on tickets
env = AgentGymEnv(
    domain="retail", 
    task_id="book_flight_1",
    solo_mode=True
)

# Custom user LLM configuration  
env = AgentGymEnv(
    domain="retail", 
    task_id="book_flight_1",
    user_llm="gpt-4",
    user_llm_args={"temperature": 0.7, "max_tokens": 1000}
)

# Combined configuration
env = AgentGymEnv(
    domain="telecom",
    task_id="troubleshoot_mobile_data_1", 
    solo_mode=False,
    user_llm="claude-3-sonnet",
    user_llm_args={"temperature": 0.5}
)
```

#### Configuration Parameters

**`solo_mode` (bool, optional):** 
- **Default:** `False`
- **Description:** When `True`, the agent works independently on task tickets without user interaction. When `False`, the agent interacts with a user simulator.
- **Usage:** Set to `True` for independent problem-solving scenarios, `False` for conversational interactions.

**`user_llm` (str, optional):**
- **Default:** Uses system default user LLM
- **Description:** Specifies which language model to use for the user simulator (only applies when `solo_mode=False`).
- **Examples:** `"gpt-4"`, `"claude-3-sonnet"`, `"gpt-3.5-turbo"`

**`user_llm_args` (dict, optional):**
- **Default:** Uses system default LLM arguments
- **Description:** Additional parameters to pass to the user simulator LLM (only applies when `solo_mode=False`).
- **Common parameters:** `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, etc.

### Mode Comparison

#### Normal Mode (`solo_mode=False`)
- **Interaction:** Agent communicates with a user simulator
- **Use case:** Conversational scenarios, customer service, interactive problem-solving
- **Task input:** User scenario with persona and instructions
- **Example:** Customer service representative helping a user book a flight

```python
env = AgentGymEnv(domain="airline", task_id="book_flight_1", solo_mode=False)
observation, info = env.reset()
# Observation includes user messages and responses
action = "Hello! I'd be happy to help you book a flight. Where would you like to travel?"
observation, reward, terminated, truncated, info = env.step(action)
```

#### Solo Mode (`solo_mode=True`)
- **Interaction:** Agent works independently on task tickets (no user simulator)
- **Use case:** Technical troubleshooting, independent problem-solving, ticket resolution
- **Task input:** Task ticket defined in the task configuration (not shown in initial observation)
- **Initial observation:** Empty/None (no conversation history)
- **Example:** IT support agent troubleshooting a network issue from a ticket

```python
env = AgentGymEnv(domain="telecom", task_id="troubleshoot_network_1", solo_mode=True)
observation, info = env.reset()
# Observation includes the task ticket and any initial context
action = "check_network_status(user_id='user_123')"
observation, reward, terminated, truncated, info = env.step(action)
```

### Simple Agent Interaction

#### Normal Mode Example

```python
# Initialize environment in normal mode (default)
env = AgentGymEnv(domain="mock", task_id="create_task_1", solo_mode=False)
observation, info = env.reset()
print(f"Initial observation: {observation}")

# Access available tools and policy from info
print(f"Available tools: {len(info['tools'])} tools available")
print(f"Agent policy: {info['policy'][:100]}...")

# Execute agent actions step by step
action = "Hello! I'm here to help you with your request."
observation, reward, terminated, truncated, info = env.step(action)

print(f"Observation: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")

# Use a tool
next_action = "create_task(user_id='user_1', title='Important Meeting')"
observation, reward, terminated, truncated, info = env.step(next_action)
print(f"Observation: {observation}")
print(f"Step reward: {reward}")
```

#### Solo Mode Example

```python
# Initialize environment in solo mode
env = AgentGymEnv(domain="telecom", task_id="troubleshoot_mobile_data_1", solo_mode=True)
observation, info = env.reset()
print(f"Initial observation: {observation}")  # Will be empty/None in solo mode

# In solo mode, agent works independently on the ticket
# The task ticket information is available through the task definition, not the observation
action = "get_user_details(user_id='customer_123')"
observation, reward, terminated, truncated, info = env.step(action)

print(f"After getting user details: {observation}")

# Continue troubleshooting
next_action = "check_mobile_data_usage(user_id='customer_123')"
observation, reward, terminated, truncated, info = env.step(next_action)
print(f"Usage check result: {observation}")
```

#### Custom LLM Configuration Example

```python
# Use a specific LLM for user simulation
env = AgentGymEnv(
    domain="airline", 
    task_id="book_flight_1",
    solo_mode=False,
    user_llm="gpt-4",
    user_llm_args={
        "temperature": 0.8,
        "max_tokens": 500,
        "top_p": 0.9
    }
)

observation, info = env.reset()
# The user simulator will now use GPT-4 with the specified parameters
action = "I'd be happy to help you find and book a flight. Where would you like to go?"
observation, reward, terminated, truncated, info = env.step(action)
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

**Note:** In solo mode (`solo_mode=True`), the initial observation is empty (None), as there is no user interaction. The task ticket information is available through the task definition. Subsequent observations show the results of agent actions and tool calls without user interactions.

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