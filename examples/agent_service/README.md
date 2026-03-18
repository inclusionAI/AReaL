# Agent Service Demo — Tau2 with PydanticAI

## Overview

This example demonstrates AReaL's Agent Service running a **tau2 customer-service
agent** powered by **PydanticAI**. The agent handles multi-turn conversations, calls
tau2 environment tools (e.g. flight lookup, reservation booking), and maintains
conversation history across turns.

The Agent Service consists of four independent HTTP services:

```
Client → Gateway (8080) → Router (8081) → DataProxy (9100) → Worker (9000)
```

- **Gateway**: public entry point (WebSocket + OpenResponses HTTP bridge)
- **Router**: session-affine routing (DataProxy registration, round-robin)
- **DataProxy**: stateful session proxy (conversation history, forwards to Worker)
- **Worker**: stateless agent execution (loads AgentRunnable, runs one turn)

## Architecture

```
Client (HTTP/WS)
    │
    ▼
┌──────────┐  POST /route   ┌──────────┐
│ Gateway  │ ──────────────▶ │ Router   │
│ :8080    │ ◀────────────── │ :8081    │
└──────────┘  DataProxy addr └──────────┘
    │
    │ POST /session/{key}/turn
    ▼
┌──────────┐
│ DataProxy│
│ :9100    │  POST /run   ┌──────────┐
│ (history)│ ────────────▶│ Worker   │
└──────────┘              │ :9000    │
                          │ (agent)  │
                          └──────────┘
```

## Files

| File          | Description                                           |
| ------------- | ----------------------------------------------------- |
| `agent.py`    | `Tau2Agent` — PydanticAI agent with tau2 domain tools |
| `config.yaml` | Configuration: LLM endpoints, tau2 domain, data path  |
| `run_demo.py` | One-click: starts all services, runs tau2 demo        |

## Prerequisites

```bash
pip install pydantic-ai
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
```

## Configuration

Edit `config.yaml` to set your LLM endpoints and tau2 settings:

```yaml
tau2:
  domain: airline
  data_dir: /path/to/tau2-bench/data

agent_llm:
  model: openai:your-model-name
  base_url: http://localhost:8000/v1
  api_key: unused

user_llm:
  model: null   # set for user simulator, null for scripted messages
  base_url: null
  api_key: unused
```

Alternatively, set `TAU2_DATA_DIR` as an environment variable.

## Quick Start

### One-click demo

```bash
python examples/agent_service/run_demo.py                       # single task, airline
python examples/agent_service/run_demo.py --domain telecom      # different domain
python examples/agent_service/run_demo.py --full                # all tasks
python examples/agent_service/run_demo.py --config my.yaml      # custom config
```

This starts all four services in background threads and runs a multi-turn conversation
showing tool calls and history accumulation.

### Manual startup (separate terminals)

```bash
# Terminal 1: Router
python -m areal.experimental.agent_service.router --port 8081

# Terminal 2: Worker + DataProxy
python -m areal.experimental.agent_service.worker \
    --agent examples.agent_service.agent.Tau2Agent \
    --router-addr http://localhost:8081 \
    --worker-port 9000 \
    --proxy-port 9100

# Terminal 3: Gateway
python -m areal.experimental.agent_service.gateway \
    --router-addr http://localhost:8081 \
    --port 8080
```

### Send a request

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"type": "message", "content": "I need to change my flight AA123"}],
    "model": "tau2-agent",
    "user": "my-session"
  }'
```

## Implementing Your Own Agent

Create a class that satisfies the `AgentRunnable` protocol:

```python
from areal.experimental.agent_service.agent_worker import (
    AgentRequest, AgentResponse, EventEmitter,
)

class MyAgent:
    def __init__(self, **kwargs):
        # Configure LLM client, tools, etc.
        pass

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        # request.message — current user message
        # request.history — prior conversation turns
        # emitter — stream events back to client
        await emitter.emit_delta("Hello!")
        return AgentResponse(summary="Hello!")
```

Then start a worker with your agent:

```bash
python -m areal.experimental.agent_service.worker \
    --agent mypackage.myagent.MyAgent \
    --router-addr http://localhost:8081
```

## Multi-turn Conversations

The DataProxy automatically manages conversation history. Each turn:

1. DataProxy reads history for the session
1. Builds `AgentRequest` with `history` field populated
1. Forwards to Worker → Agent sees full conversation context
1. Appends user message + agent response to history
1. Tool calls and results are also recorded in history

The agent accesses history via `request.history`:

```python
async def run(self, request, *, emitter):
    for msg in request.history:
        print(f"{msg['role']}: {msg['content']}")
    # ... generate response using full context
```
