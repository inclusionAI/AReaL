# Agent Service

## Overview

The Agent Service provides **agent-level** capabilities on top of AReaL's model-level
proxy. It exposes complete agent sessions — multi-turn conversations with tool use,
memory, and pluggable agent frameworks — via independent HTTP microservices.

## Architecture

The Agent Service consists of four independent HTTP services that communicate via REST:

```
Client (HTTP/WS)
    │
    ▼
┌──────────┐  POST /route   ┌──────────┐
│ Gateway  │ ──────────────▶ │ Router   │
│          │                 │          │
└──────────┘                 └──────────┘
    │                             │
    │ POST /session/{key}/turn    │ returns DataProxy addr
    ▼                             │
┌──────────┐  ◀───────────────────┘
│ DataProxy│
│ (history)│  POST /run   ┌──────────┐
│          │ ────────────▶│ Worker   │
└──────────┘              │ (agent)  │
                          └──────────┘
```

### Components

**Gateway** — Public entry point. Accepts WebSocket connections (Gateway protocol) and
HTTP requests (OpenResponses bridge at `POST /v1/responses`). Routes to the appropriate
DataProxy via the Router.

**Router** — Session-affine routing service. DataProxy instances register at startup.
The Router assigns new sessions round-robin and maintains session → DataProxy affinity.

**DataProxy** — Stateful session proxy, paired 1:1 with a Worker. Manages per-session
conversation history. On each turn: reads history → constructs `AgentRequest` (with
history) → forwards to Worker → appends messages to history → returns response.

**Worker** — Stateless agent execution server. Loads an `AgentRunnable` implementation
at startup. Each `POST /run` request is a single turn — the agent receives the full
conversation history in the request and returns a response. The Worker has no session
state.

## Agent Protocol

Any class that satisfies the `AgentRunnable` protocol can run on the Worker:

```python
@runtime_checkable
class AgentRunnable(Protocol):
    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse: ...
```

### AgentRequest

```python
@dataclass
class AgentRequest:
    message: str                              # Current user message
    session_key: str                          # Session identifier
    run_id: str                               # Unique run identifier
    history: list[dict[str, str]]             # Prior conversation turns
    queue_mode: QueueMode = QueueMode.COLLECT
    metadata: dict[str, Any] = field(default_factory=dict)
```

### AgentResponse

```python
@dataclass
class AgentResponse:
    summary: str = ""                         # Agent reply text
    metadata: dict[str, Any] = field(default_factory=dict)
```

### EventEmitter

```python
class EventEmitter(Protocol):
    async def emit_delta(self, text: str) -> None: ...
    async def emit_tool_call(self, name: str, args: str) -> None: ...
    async def emit_tool_result(self, name: str, result: str) -> None: ...
```

## HTTP APIs

### Router

| Endpoint          | Method | Description                 |
| ----------------- | ------ | --------------------------- |
| `/health`         | GET    | Health check                |
| `/register`       | POST   | Register a DataProxy        |
| `/unregister`     | POST   | Unregister a DataProxy      |
| `/route`          | POST   | Get DataProxy for a session |
| `/remove_session` | POST   | Remove session affinity     |

### DataProxy

| Endpoint                 | Method | Description              |
| ------------------------ | ------ | ------------------------ |
| `/health`                | GET    | Health check             |
| `/session/{key}/turn`    | POST   | Send a message (turn)    |
| `/session/{key}/close`   | POST   | Close session            |
| `/session/{key}/history` | GET    | Get conversation history |

### Worker

| Endpoint  | Method | Description            |
| --------- | ------ | ---------------------- |
| `/health` | GET    | Health check           |
| `/run`    | POST   | Execute one agent turn |

### Gateway

| Endpoint        | Method | Description                |
| --------------- | ------ | -------------------------- |
| `/health`       | GET    | Health check               |
| `/ws`           | WS     | Gateway WebSocket protocol |
| `/v1/responses` | POST   | OpenResponses HTTP bridge  |

## Multi-turn Conversation Flow

```
Turn 1:
  Client → Gateway → Router (route session) → DataProxy
    DataProxy: history = []
    DataProxy → Worker: POST /run {message, history: []}
    Worker → Agent: run(request) → AgentResponse
    DataProxy: history = [user_msg, assistant_msg]
    DataProxy → Gateway → Client

Turn 2:
  Client → Gateway → Router (same DataProxy) → DataProxy
    DataProxy: history = [user_msg_1, assistant_msg_1]
    DataProxy → Worker: POST /run {message, history: [user_msg_1, assistant_msg_1]}
    Worker → Agent: run(request) → AgentResponse
    DataProxy: history = [..., user_msg_2, assistant_msg_2]
    DataProxy → Gateway → Client
```

## Code Organization

```
areal/experimental/agent_service/
├── __init__.py          # Public exports
├── README.md            # This document
├── protocol.py          # Gateway protocol frame types
├── config.py            # AgentServiceConfig dataclass
├── agent_worker.py      # Worker HTTP server + AgentRunnable protocol
├── data_proxy.py        # DataProxy HTTP server + DataProxyClient
├── agent_router.py      # Router HTTP server + RouterClient
├── agent_gateway.py     # Gateway HTTP server (WebSocket)
└── agent_bridge.py      # OpenResponses HTTP bridge

examples/agent_service/
├── agent.py             # DemoAgent example
├── start_router.py      # Launch Router
├── start_worker.py      # Launch Worker + DataProxy
├── start_gateway.py     # Launch Gateway
├── run_demo.py          # One-click demo
└── README.md            # Example documentation
```
