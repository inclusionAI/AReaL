# Plan 4: GatewayRolloutController — Parallel Implementation to RolloutController

## TL;DR

> **Quick Summary**: Build `GatewayRolloutController` as a **parallel, independent implementation** (NOT a subclass) of the rollout controller API. Unlike `RolloutController` which uses scheduler RPC for all engine communication, `GatewayRolloutController` routes **everything** through the gateway HTTP stack (Gateway → Router → Data Proxy → SGLang). Internally it uses `WorkflowExecutor` + `BatchTaskDispatcher` (same as `RemoteInfEngine`) with a `GatewayInfEngine` that implements `InferenceEngine` via HTTP calls to the gateway.
>
> **Key Design Principle**: `GatewayRolloutController` is duck-type compatible with `RolloutController` — it provides the same method signatures so the trainer can use either one. But internally it routes through gateway HTTP instead of scheduler RPC.
>
> **Deliverables**:
> - `areal/experimental/gateway/controller/__init__.py` — package exports
> - `areal/experimental/gateway/controller/config.py` — `GatewayControllerConfig` dataclass
> - `areal/experimental/gateway/controller/inf_engine.py` — `GatewayInfEngine` (implements InferenceEngine via gateway HTTP)
> - `areal/experimental/gateway/controller/controller.py` — `GatewayRolloutController` class
> - `areal/experimental/gateway/data_proxy/weight_update.py` — Weight update endpoint handlers
> - `areal/experimental/gateway/data_proxy/app.py` — Modified: register weight update routes
> - `areal/experimental/gateway/gateway/app.py` — Modified: add weight update + set_version broadcast endpoints
> - `tests/experimental/gateway/test_controller.py` — Unit tests for controller
> - `tests/experimental/gateway/test_data_proxy_weight_updates.py` — Unit tests for weight update endpoints
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 5 → Task 7

---

## Context

### Architecture Overview

```
Training Loop (PPOTrainer)
    │
    ▼
GatewayRolloutController
    │
    ├── WorkflowExecutor (in-process, same as RemoteInfEngine)
    │       │
    │       ├── BatchTaskDispatcher (producer-consumer with staleness)
    │       │
    │       └── GatewayInfEngine.agenerate()
    │               │
    │               ▼ HTTP POST /generate
    │           Gateway (:8080)
    │               │
    │               ▼ /route query
    │           Router (:8081)
    │               │
    │               ▼ forward to selected worker
    │           Data Proxy (:8082)
    │               │
    │               ▼ HTTP to co-located SGLang
    │           SGLang Server (:30000)
    │
    ├── Callback Server (Flask, for training controller compatibility)
    │       Receives: /callback/init_weights_group, /callback/update_weights_xccl,
    │                 /callback/update_weights_disk, /callback/pause_generation,
    │                 /callback/continue_generation
    │       → Translates to gateway HTTP broadcasts
    │
    └── Gateway Services (background threads)
            Router, Data Proxy (per worker), Gateway
```

### Comparison: RolloutController vs GatewayRolloutController

| Aspect | RolloutController | GatewayRolloutController |
|--------|-------------------|--------------------------|
| **Inference path** | Scheduler RPC → Worker → RemoteInfEngine → HTTP → SGLang | WorkflowExecutor → GatewayInfEngine → HTTP → Gateway → Router → Data Proxy → SGLang |
| **Constructor** | `(inf_engine, config, scheduler)` | `(config: GatewayControllerConfig, scheduler: Scheduler)` |
| **Task dispatch** | BatchTaskDispatcher → scheduler RPC submit/wait_for_task with callback | WorkflowExecutor (in-process) → agenerate via HTTP |
| **Weight updates** | Scheduler RPC collective calls to engines | Gateway HTTP broadcast to all data proxies |
| **Pause/resume** | Scheduler RPC collective calls | Gateway HTTP broadcast |
| **Callback server** | Flask server for training controller integration | Same — Flask server, but translates to gateway HTTP |
| **Subclass of** | Nothing | Nothing (parallel implementation) |
| **Config type** | `InferenceEngineConfig` | `GatewayControllerConfig` |

### Critical Integration Point: `callback_addr`

The training controller (`TrainController.connect_engine()`) creates `RolloutCallback(controller_addr=rollout.callback_addr)`. This callback sends HTTP POSTs to the rollout controller for weight updates and pause/continue. **GatewayRolloutController MUST expose a `callback_addr` property and run a callback Flask server** to maintain compatibility with the existing training side.

The callback server receives these routes:
- `/callback/init_weights_group` → calls `self.init_weights_update_group(meta)`
- `/callback/update_weights_xccl` → calls `self.update_weights_from_distributed(meta, param_specs)`
- `/callback/update_weights_disk` → calls `self.update_weights_from_disk(meta)`
- `/callback/pause_generation` → calls `self.pause_generation()`
- `/callback/continue_generation` → calls `self.continue_generation()`

### Key Decisions

- **Gateway-specific config**: `GatewayControllerConfig` dataclass with gateway ports, admin key, tokenizer path, etc.
- **HTTP via gateway for everything**: ALL operations (inference, weight updates, pause/continue) go through gateway HTTP.
- **Reuse WorkflowExecutor directly**: In-process WorkflowExecutor with GatewayInfEngine as the inference engine. No callback server for rollout completion — that's handled by WorkflowExecutor internally.
- **Callback server for training integration only**: The Flask callback server is needed ONLY for training controller compatibility (weight updates, pause/continue from training side).
- **Scheduler role is limited**: Only launches SGLang servers and gateway services. Not used for inference at all.
- **New independent test suite**: Tests mock HTTP endpoints, not scheduler RPC.

---

## Work Objectives

### Core Objective
Create `GatewayRolloutController` as a parallel implementation to `RolloutController` that routes all traffic through the gateway HTTP stack while maintaining API compatibility with the trainer.

### Must Have
- `GatewayControllerConfig` dataclass with all gateway-specific configuration fields
- `GatewayInfEngine` implementing `InferenceEngine` interface via gateway HTTP
- `GatewayRolloutController` with same API surface as `RolloutController` (duck-type compatible)
- `callback_addr` property and Flask callback server for training controller compatibility
- Weight update forwarding endpoints on data proxy
- Weight update broadcast endpoints on gateway
- `agenerate()` method routing through gateway HTTP
- `chat_completion` support via `/chat/completions` gateway endpoint
- `submit()`, `wait()`, `prepare_batch()`, `rollout_batch()` via WorkflowExecutor
- `pause()`, `resume()`, `set_version()`, `get_version()`, `get_capacity()`, `export_stats()`
- `destroy()` that stops gateway services and cleans up
- Unit tests for controller and weight update endpoints
- Python 3.10 compatible (lazy imports for `areal.api.*`)

### Must NOT Have
- NO changes to existing `RolloutController`, `rl_trainer.py`, `train_controller.py`, or any file outside `areal/experimental/gateway/` and `tests/experimental/gateway/`
- NO GPU requirements in unit tests
- NO abstract base classes or inheritance hierarchies — concrete classes only
- NO wildcard imports
- NO OpenAI proxy/agent workflow support (out of scope)
- NO multi-model support or model routing (out of scope)
- NO custom scheduler implementation — use existing `Scheduler` interface
- NO modifications to existing gateway/router endpoints — only ADD new ones

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest + conftest.py shim for Python 3.10)
- **Automated tests**: YES (tests-after)
- **Framework**: pytest
- **Test command**: `"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/ -v --tb=long --ignore=tests/experimental/gateway/test_data_proxy_integration.py --ignore=tests/experimental/gateway/test_gateway_integration.py`

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — config + foundation):
├── Task 1: GatewayControllerConfig dataclass [quick]
├── Task 2: GatewayInfEngine (InferenceEngine via gateway HTTP) [unspecified-high]
└── Task 3: Data proxy weight update endpoints [unspecified-high]

Wave 2 (After Wave 1 — gateway broadcast + controller core):
├── Task 4: Gateway broadcast endpoints for weight updates + set_version [unspecified-high]
└── Task 5: GatewayRolloutController class [deep]

Wave 3 (After Wave 2 — tests):
├── Task 6: Unit tests for data proxy weight update endpoints [unspecified-high]
└── Task 7: Unit tests for GatewayRolloutController [unspecified-high]

Wave FINAL (After ALL tasks):
└── Task 8: Full test pass + pre-commit verification [quick]

Critical Path: Task 1 → Task 2 → Task 5 → Task 7 → Task 8
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1: Config | — | 2, 3, 4, 5 | 1 |
| 2: GatewayInfEngine | 1 | 5 | 1 |
| 3: Data proxy weight endpoints | 1 | 4, 6 | 1 |
| 4: Gateway broadcast endpoints | 3 | 5, 6 | 2 |
| 5: Controller class | 1, 2, 4 | 7 | 2 |
| 6: Weight update tests | 3, 4 | 8 | 3 |
| 7: Controller tests | 5 | 8 | 3 |
| 8: Full verification | 6, 7 | — | FINAL |

### Agent Dispatch Summary

| Wave | Tasks | Categories |
|------|-------|------------|
| 1 | 3 | T1→`quick`, T2→`unspecified-high`, T3→`unspecified-high` |
| 2 | 2 | T4→`unspecified-high`, T5→`deep` |
| 3 | 2 | T6→`unspecified-high`, T7→`unspecified-high` |
| FINAL | 1 | T8→`quick` |

---

## TODOs

- [ ] 1. `GatewayControllerConfig` dataclass

  **What to do**:
  - Create `areal/experimental/gateway/controller/config.py`:
    - `@dataclass class GatewayControllerConfig:`
    - Fields:
      - `gateway_host: str = "0.0.0.0"` — gateway listen host
      - `gateway_port: int = 8080` — gateway listen port
      - `router_host: str = "0.0.0.0"` — router listen host
      - `router_port: int = 8081` — router listen port
      - `data_proxy_host: str = "0.0.0.0"` — data proxy listen host
      - `data_proxy_base_port: int = 8082` — base port for data proxies (incremented per worker)
      - `admin_api_key: str = "areal-admin-key"` — shared admin key for all services
      - `tokenizer_path: str = ""` — tokenizer path for data proxy
      - `model_path: str = ""` — model path (for SGLang server launch)
      - `routing_strategy: str = "round_robin"` — routing strategy for router
      - `poll_interval: float = 5.0` — router health poll interval
      - `request_timeout: float = 120.0` — HTTP request timeout
      - `max_resubmit_retries: int = 20` — SGLang backend resubmit retries
      - `resubmit_wait: float = 0.5` — wait between pause polls
      - `setup_timeout: float = 300.0` — timeout waiting for services to start
      - `log_level: str = "info"` — log level for gateway services
      - `consumer_batch_size: int = 16` — batch size for staleness manager
      - `max_concurrent_rollouts: int | None = None` — max concurrent rollouts
      - `max_head_offpolicyness: int = 0` — max staleness
      - `queue_size: int | None = None` — dispatcher queue size
      - `enable_rollout_tracing: bool = False` — enable rollout tracing
    - Use `from __future__ import annotations` for Python 3.10 compat
    - NO imports from `areal.api.*` at module level
  - Create `areal/experimental/gateway/controller/__init__.py`:
    ```python
    from areal.experimental.gateway.controller.controller import GatewayRolloutController
    from areal.experimental.gateway.controller.config import GatewayControllerConfig
    __all__ = ["GatewayRolloutController", "GatewayControllerConfig"]
    ```

  **Must NOT do**:
  - Do NOT import `areal.api.cli_args` at module level
  - Do NOT add fields that couple to scheduler internals

  **Recommended Agent Profile**: `quick`
  **Parallelization**: Wave 1. Blocks Tasks 2, 3, 4, 5.

  **References**:
  - `areal/experimental/gateway/gateway/config.py` — `GatewayConfig` pattern to follow
  - `areal/experimental/gateway/router/config.py` — `RouterConfig` pattern to follow
  - `areal/experimental/gateway/data_proxy/config.py` — `DataProxyConfig` pattern to follow
  - `areal/api/cli_args.py:InferenceEngineConfig` — fields to replicate (consumer_batch_size, max_concurrent_rollouts, etc.)

  **Acceptance Criteria**:
  - [ ] `from areal.experimental.gateway.controller.config import GatewayControllerConfig` works
  - [ ] `GatewayControllerConfig()` creates instance with all defaults
  - [ ] Config has all fields needed by controller, WorkflowExecutor, and gateway services

  **QA Scenarios**:
  ```
  Scenario: Config import and defaults
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  from areal.experimental.gateway.controller.config import GatewayControllerConfig
  c = GatewayControllerConfig()
  assert c.gateway_port == 8080
  assert c.router_port == 8081
  assert c.data_proxy_base_port == 8082
  assert c.consumer_batch_size == 16
  print('Config OK')
  "
    Expected Result: prints 'Config OK'
    Evidence: .sisyphus/evidence/task-1-config.txt
  ```

  **Commit**: YES (group with Tasks 2-4)
  - Message: `feat(gateway/controller): add GatewayControllerConfig, GatewayInfEngine, and weight update endpoints`

---

- [ ] 2. `GatewayInfEngine` — InferenceEngine implementation via gateway HTTP

  **What to do**:
  - Create `areal/experimental/gateway/controller/inf_engine.py`:
    - `class GatewayInfEngine:` — implements `InferenceEngine` interface (duck-type, no inheritance needed)
    - **Constructor**: `__init__(self, gateway_addr: str, config: GatewayControllerConfig)`
      - `self.gateway_addr` — gateway HTTP address (e.g., `http://127.0.0.1:8080`)
      - `self.config` — controller config
      - `self._version = 0`
      - `self._version_lock = Lock()`
      - `self._workflow_executor: WorkflowExecutor | None = None`
      - `self._initialized = False`
    - **`async agenerate(self, req: ModelRequest) -> ModelResponse`**:
      - Makes HTTP POST to `{gateway_addr}/generate` with the request payload
      - Handles the pause/resume resubmit loop (abort → wait → retry, matching SGLangBackend pattern)
      - Returns `ModelResponse` with accumulated tokens, logprobs, versions
      - Use `aiohttp` for async HTTP (same as `RemoteInfEngine.agenerate`)
      - Refer to `areal/infra/remote_inf_engine.py:703-868` for the exact pattern
    - **`set_version(self, version: int)`** / **`get_version(self) -> int`**: Thread-safe version management
    - **`pause(self)` / `resume(self)`**: Delegate to `workflow_executor.pause()` / `resume()`
    - **Properties**: `workflow_executor`, `initialized`
    - **`initialize(self, train_data_parallel_size=None)`**: Creates WorkflowExecutor + StalenessManager
    - **`destroy(self)`**: Cleans up WorkflowExecutor
    - **`submit()`, `wait()`, `rollout_batch()`, `prepare_batch()`**: Delegate to WorkflowExecutor
      - Follow exact pattern from `RemoteInfEngine` (lines 991-1170)
      - `submit()` takes `workflow: WorkflowLike`, resolves it via `_resolve_workflow()`, delegates to `workflow_executor.submit()`
      - `wait()` delegates to `workflow_executor.wait()`
      - `rollout_batch()` delegates to `workflow_executor.rollout_batch()`
      - `prepare_batch()` delegates to `workflow_executor.prepare_batch()`
    - Use lazy imports for ALL `areal.api.*` imports (inside methods or under TYPE_CHECKING)

  **Critical Design Notes**:
  - GatewayInfEngine does NOT manage SGLang servers — it only talks HTTP to the gateway
  - The `/generate` endpoint on the gateway already handles routing to the correct data proxy
  - Round-robin is done by the Router, not by the engine
  - The engine needs to handle the request/response format expected by gateway `/generate`
  - WorkflowExecutor takes the engine as `inference_engine` parameter. Need to verify WorkflowExecutor only uses `agenerate()` on it — if it accesses `RemoteInfEngine`-specific attributes, wrap accordingly

  **Must NOT do**:
  - Do NOT import `areal.api.*` at module level (Python 3.10 compat)
  - Do NOT manage SGLang processes or server addresses directly
  - Do NOT implement weight update logic (that's in the controller)

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 1 (parallel with Tasks 1, 3). Blocks Task 5.

  **References**:
  - `areal/infra/remote_inf_engine.py:703-868` — `agenerate()` pattern (abort/resubmit loop, token accumulation)
  - `areal/infra/remote_inf_engine.py:991-1170` — `submit()`, `wait()`, `rollout_batch()`, `prepare_batch()` delegation to WorkflowExecutor
  - `areal/infra/remote_inf_engine.py:525-639` — `_resolve_workflow()` for WorkflowLike resolution
  - `areal/infra/workflow_executor.py` — WorkflowExecutor class that this engine creates
  - `areal/api/engine_api.py:InferenceEngine` — interface to duck-type implement
  - `areal/api/io_struct.py:ModelRequest,ModelResponse` — request/response types
  - `areal/experimental/gateway/gateway/app.py` — gateway `/generate` endpoint format
  - `areal/experimental/gateway/data_proxy/backend.py` — SGLangBackend for request format reference

  **Acceptance Criteria**:
  - [ ] `from areal.experimental.gateway.controller.inf_engine import GatewayInfEngine` works
  - [ ] `GatewayInfEngine` has `agenerate`, `submit`, `wait`, `rollout_batch`, `prepare_batch`, `set_version`, `get_version`, `pause`, `resume`
  - [ ] `agenerate` makes HTTP POST to gateway `/generate` endpoint

  **QA Scenarios**:
  ```
  Scenario: GatewayInfEngine imports and has correct interface
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  from areal.experimental.gateway.controller.inf_engine import GatewayInfEngine
  methods = ['agenerate', 'submit', 'wait', 'rollout_batch', 'prepare_batch',
             'set_version', 'get_version', 'pause', 'resume', 'initialize', 'destroy']
  for m in methods:
      assert hasattr(GatewayInfEngine, m), f'Missing: {m}'
  print('GatewayInfEngine interface OK')
  "
    Expected Result: prints 'GatewayInfEngine interface OK'
    Evidence: .sisyphus/evidence/task-2-inf-engine.txt
  ```

  **Commit**: YES (group with Tasks 1, 3, 4)

---

- [ ] 3. Data proxy weight update endpoints

  **What to do**:
  - Create `areal/experimental/gateway/data_proxy/weight_update.py`:
    - Three endpoint handler functions that forward HTTP requests to the co-located SGLang server:
    - `async def update_weights_from_disk(request: Request) -> JSONResponse`:
      - Receives JSON `{"path": str, "type": "disk", ...}` (WeightUpdateMeta fields)
      - Forwards as HTTP POST to `{backend_addr}/update_weights_from_disk` on the co-located SGLang server
      - Returns SGLang's response
    - `async def update_weights_from_distributed(request: Request) -> JSONResponse`:
      - Receives JSON with `meta` and `param_specs` fields
      - Forwards as HTTP POST to `{backend_addr}/update_weights_from_distributed`
      - Returns SGLang's response
    - `async def init_weights_update_group(request: Request) -> JSONResponse`:
      - Receives JSON with `meta` fields
      - Forwards as HTTP POST to `{backend_addr}/init_weights_update_group`
      - Returns SGLang's response
    - `async def set_version(request: Request) -> JSONResponse`:
      - Receives JSON `{"version": int}`
      - Forwards to SGLang's `/set_version` (if SGLang has it) OR stores locally and returns OK
      - Also updates the data proxy's internal backend version tracking
    - Each handler accesses `request.app.state.backend` to get the SGLang backend address
  - Modify `areal/experimental/gateway/data_proxy/app.py`:
    - Import the new endpoint handlers
    - Register routes: `app.post("/update_weights_from_disk")(update_weights_from_disk)`
    - Register routes: `app.post("/update_weights_from_distributed")(update_weights_from_distributed)`
    - Register routes: `app.post("/init_weights_update_group")(init_weights_update_group)`
    - Register routes: `app.post("/set_version")(set_version)`

  **Critical Design Notes**:
  - These endpoints are pure HTTP-to-HTTP forwards — the data proxy does NOT participate in NCCL
  - The SGLang server has its own weight update HTTP endpoints that these forward to
  - The data proxy adds these as new routes alongside existing /generate, /chat/completions, etc.
  - Admin API key authentication should be enforced on these endpoints (these are control-plane operations)

  **Must NOT do**:
  - Do NOT modify existing data proxy endpoints
  - Do NOT implement NCCL logic — just HTTP forwarding
  - Do NOT remove or rename any existing routes

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 1 (parallel with Tasks 1, 2). Blocks Tasks 4, 6.

  **References**:
  - `areal/experimental/gateway/data_proxy/app.py` — existing app with endpoint registration pattern
  - `areal/experimental/gateway/data_proxy/pause.py` — pattern for forwarding to SGLang (pause/continue forward to SGLang)
  - `areal/infra/remote_inf_engine.py:870-990` — how weight update HTTP requests are built for SGLang
  - `areal/infra/controller/rollout_controller.py:997-1020` — how RolloutController handles weight updates

  **Acceptance Criteria**:
  - [ ] Data proxy app has `/update_weights_from_disk`, `/update_weights_from_distributed`, `/init_weights_update_group`, `/set_version` routes
  - [ ] All existing data proxy tests still pass (26 chat + 7 generate + 18 pause + 11 standalone = 62 tests)
  - [ ] New endpoints forward to SGLang server address

  **QA Scenarios**:
  ```
  Scenario: Existing data proxy tests still pass after adding weight update routes
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_data_proxy_chat.py tests/experimental/gateway/test_data_proxy_generate.py tests/experimental/gateway/test_data_proxy_pause.py tests/experimental/gateway/test_data_proxy_standalone.py -v --tb=long
    Expected Result: All 62 tests PASS
    Evidence: .sisyphus/evidence/task-3-existing-tests.txt
  ```

  **Commit**: YES (group with Tasks 1, 2, 4)

---

- [ ] 4. Gateway broadcast endpoints for weight updates + set_version

  **What to do**:
  - Modify `areal/experimental/gateway/gateway/app.py`:
    - Add new broadcast endpoints that fan out to ALL registered workers (same pattern as existing `/pause_generation` and `/continue_generation`):
    - `POST /update_weights_from_disk` — broadcasts to all workers' data proxy `/update_weights_from_disk`
    - `POST /update_weights_from_distributed` — broadcasts to all workers' data proxy `/update_weights_from_distributed`
    - `POST /init_weights_update_group` — broadcasts to all workers' data proxy `/init_weights_update_group`
    - `POST /set_version` — broadcasts to all workers' data proxy `/set_version`
    - All broadcast endpoints require admin API key authentication
    - Use the same pattern as existing `pause_generation` endpoint: query router for all worker addresses, fan out HTTP calls to each
    - For weight updates that need per-worker rank info (like `init_weights_update_group`), include the worker index in the forwarded request

  **Critical Design Notes**:
  - The gateway already has the pattern for broadcasting to all workers (see `/pause_generation` endpoint)
  - Weight update requests need to reach ALL data proxies simultaneously
  - The gateway queries the router for the list of all workers, then fans out requests
  - `init_weights_update_group` may need per-worker rank assignment (worker index)

  **Must NOT do**:
  - Do NOT modify existing gateway endpoints
  - Do NOT remove or rename any existing routes
  - Do NOT add NCCL logic — gateway is pure HTTP routing

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 2. Depends on Task 3. Blocks Tasks 5, 6.

  **References**:
  - `areal/experimental/gateway/gateway/app.py` — existing broadcast pattern in `/pause_generation` and `/continue_generation`
  - `areal/experimental/gateway/data_proxy/weight_update.py` — the data proxy endpoints this broadcasts to (created in Task 3)
  - `areal/experimental/gateway/gateway/app.py:140-180` — the broadcast implementation pattern (query router for workers, fan out)

  **Acceptance Criteria**:
  - [ ] Gateway has `/update_weights_from_disk`, `/update_weights_from_distributed`, `/init_weights_update_group`, `/set_version` broadcast endpoints
  - [ ] All existing gateway tests still pass (24 tests)
  - [ ] New endpoints broadcast to all workers

  **QA Scenarios**:
  ```
  Scenario: Existing gateway tests still pass after adding weight update broadcast routes
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_gateway.py -v --tb=long
    Expected Result: All 24 tests PASS
    Evidence: .sisyphus/evidence/task-4-existing-tests.txt
  ```

  **Commit**: YES (group with Tasks 1, 2, 3)

---

- [ ] 5. `GatewayRolloutController` class

  **What to do**:
  - Create `areal/experimental/gateway/controller/controller.py`:
    - `class GatewayRolloutController:` — NOT a subclass of anything
    - **Constructor**: `__init__(self, config: GatewayControllerConfig, scheduler: Scheduler)`
      - Store config and scheduler
      - Initialize state:
        - `self._version = 0`, `self._version_lock = Lock()`
        - `self._gateway_inf_engine: GatewayInfEngine | None = None`
        - `self._gateway_services_started = False`
        - `self._router_thread = None`, `self._data_proxy_threads = []`, `self._gateway_thread = None`
        - `self.workers = []`, `self.server_infos = []`
        - `self._worker_role: str`
        - `self._staleness_manager: StalenessManager | None = None`
        - Callback server state (same as RolloutController)
    - **`initialize(self, role: str, alloc_mode, server_args=None, server_infos=None, *args, **kwargs)`**:
      1. Store `self._worker_role = role`
      2. Use scheduler to create workers and launch SGLang servers (same pattern as RolloutController._async_initialize)
      3. Start Router as background thread (using `create_app` from `router.app`)
      4. Start one Data Proxy per worker (using `create_app` from `data_proxy.app`) — each pointed at its worker's SGLang server
      5. Start Gateway as background thread (using `create_app` from `gateway.app`)
      6. Register each data proxy as a worker in the Router
      7. Wait for all services to be healthy
      8. Create `GatewayInfEngine(gateway_addr, config)` and call `.initialize()`
      9. Create StalenessManager
      10. Start callback server (for training controller compatibility)
    - **`destroy(self)`**:
      1. Stop GatewayInfEngine
      2. Stop callback server
      3. Stop gateway services (Gateway, Data Proxies, Router — in reverse order)
      4. Delete workers via scheduler
    - **`submit(self, data, workflow, ...)`**: Delegate to `self._gateway_inf_engine.submit()`
    - **`wait(self, count, ...)`**: Delegate to `self._gateway_inf_engine.wait()`
    - **`rollout_batch(self, data, workflow, ...)`**: Delegate to `self._gateway_inf_engine.rollout_batch()`
    - **`prepare_batch(self, dataloader, workflow, ...)`**: Delegate to `self._gateway_inf_engine.prepare_batch()`
    - **`async agenerate(self, req: ModelRequest) -> ModelResponse`**: Delegate to `self._gateway_inf_engine.agenerate(req)`
    - **`set_version(self, version: int)`**: Update local version + HTTP POST to gateway `/set_version` broadcast
    - **`get_version(self) -> int`**: Return local version (thread-safe)
    - **`get_capacity(self) -> int`**: Delegate to staleness_manager
    - **`pause(self)`**: HTTP POST to gateway `/pause_generation` + pause dispatcher
    - **`resume(self)`**: HTTP POST to gateway `/continue_generation` + resume dispatcher
    - **`async pause_generation(self)`**: HTTP POST to gateway `/pause_generation`
    - **`async continue_generation(self)`**: HTTP POST to gateway `/continue_generation`
    - **`async init_weights_update_group(self, meta)`**: HTTP POST to gateway `/init_weights_update_group`
    - **`async update_weights_from_distributed(self, meta, param_specs)`**: HTTP POST to gateway `/update_weights_from_distributed`
    - **`async update_weights_from_disk(self, meta)`**: HTTP POST to gateway `/update_weights_from_disk`
    - **`export_stats(self) -> dict`**: Return WorkflowExecutor stats (local)
    - **`config_perf_tracer(self, config, role)`**: No-op or minimal implementation
    - **`save_perf_tracer(self, step, force)`**: No-op or minimal implementation
    - **`start_proxy(self)`**: No-op (gateway IS the proxy)
    - **`start_proxy_gateway(self)`**: No-op (gateway IS the proxy gateway)
    - **`@property proxy_gateway_addr`**: Return gateway address
    - **`@property callback_addr`**: Return callback server address
    - **`@property staleness_manager`**: Return `self._staleness_manager`
    - **`@property dispatcher`**: Return `self._gateway_inf_engine.workflow_executor.dispatcher`
    - **`@property runner`**: Return `self.dispatcher.runner`
    - **Callback server** (`_start_callback_server()`):
      - Same Flask callback server pattern as RolloutController (lines 530-623 of rollout_controller.py)
      - Routes: `/callback/init_weights_group`, `/callback/update_weights_xccl`, `/callback/update_weights_disk`, `/callback/pause_generation`, `/callback/continue_generation`
      - Each callback handler calls the corresponding method on the controller (which routes to gateway HTTP)
      - This is needed for `connect_engine()` compatibility — the training controller creates `RolloutCallback(controller_addr=rollout.callback_addr)`

  **Critical Design Notes**:
  - The controller's `submit/wait/rollout_batch/prepare_batch` go through WorkflowExecutor, which calls `GatewayInfEngine.agenerate()`, which calls gateway HTTP
  - Weight updates and pause/continue go through gateway HTTP broadcast endpoints (not scheduler RPC)
  - The callback server is a compatibility layer for the training controller — it receives callbacks from the training side and translates them to gateway HTTP calls
  - Scheduler is used ONLY in `initialize()` to create workers and launch SGLang servers
  - Gateway services run as background threads (uvicorn in separate threads)
  - The `server_infos` attribute should be populated during initialize() with the SGLang server addresses (for eval_rollout reuse)

  **Must NOT do**:
  - Do NOT use scheduler RPC for inference (submit, wait, agenerate)
  - Do NOT subclass RolloutController
  - Do NOT import `areal.api.*` at module level (Python 3.10 compat)
  - Do NOT add abstract base classes

  **Recommended Agent Profile**: `deep`
  **Parallelization**: Wave 2. Depends on Tasks 1, 2, 4. Blocks Task 7.

  **References**:
  - `areal/infra/controller/rollout_controller.py` — The parallel implementation to match API surface. Study EVERY public method.
  - `areal/infra/controller/rollout_controller.py:67-118` — Constructor fields and state
  - `areal/infra/controller/rollout_controller.py:150-300` — `initialize()` and `_async_initialize()` patterns
  - `areal/infra/controller/rollout_controller.py:301-334` — `destroy()` cleanup order
  - `areal/infra/controller/rollout_controller.py:530-623` — `_start_callback_server()` — MUST replicate this pattern
  - `areal/infra/controller/rollout_controller.py:850-968` — `submit()`, `wait()`, `rollout_batch()`, `prepare_batch()`
  - `areal/infra/remote_inf_engine.py:379-453` — `initialize()` pattern (creating WorkflowExecutor)
  - `areal/experimental/gateway/controller/inf_engine.py` — GatewayInfEngine (created in Task 2)
  - `areal/experimental/gateway/controller/config.py` — GatewayControllerConfig (created in Task 1)
  - `areal/experimental/gateway/router/app.py` — `create_app()` for starting Router service
  - `areal/experimental/gateway/data_proxy/app.py` — `create_app()` for starting Data Proxy service
  - `areal/experimental/gateway/gateway/app.py` — `create_app()` for starting Gateway service
  - `areal/api/scheduler_api.py:Scheduler` — scheduler interface for creating workers

  **Acceptance Criteria**:
  - [ ] `from areal.experimental.gateway.controller import GatewayRolloutController` works
  - [ ] Constructor accepts `(config: GatewayControllerConfig, scheduler: Scheduler)`
  - [ ] All public methods from RolloutController are present (duck-type compatible)
  - [ ] `callback_addr` property returns valid address after `_start_callback_server()`
  - [ ] Gateway services start in `initialize()` and stop in `destroy()`
  - [ ] `submit/wait/rollout_batch/prepare_batch` delegate to WorkflowExecutor
  - [ ] Weight updates route through gateway HTTP endpoints

  **QA Scenarios**:
  ```
  Scenario: GatewayRolloutController imports and has all methods
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  from areal.experimental.gateway.controller import GatewayRolloutController
  methods = ['initialize', 'destroy', 'submit', 'wait', 'rollout_batch',
             'prepare_batch', 'agenerate', 'set_version', 'get_version',
             'get_capacity', 'pause', 'resume', 'export_stats',
             'init_weights_update_group', 'update_weights_from_distributed',
             'update_weights_from_disk', 'pause_generation', 'continue_generation',
             'callback_addr', 'staleness_manager', 'dispatcher', 'runner',
             'start_proxy', 'start_proxy_gateway', 'proxy_gateway_addr']
  for m in methods:
      assert hasattr(GatewayRolloutController, m), f'Missing: {m}'
  print('All methods/properties present')
  "
    Expected Result: prints 'All methods/properties present'
    Evidence: .sisyphus/evidence/task-5-api.txt

  Scenario: GatewayRolloutController is NOT a subclass of RolloutController
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  from areal.experimental.gateway.controller import GatewayRolloutController
  # Should NOT be a subclass of RolloutController
  try:
      from areal.infra.controller.rollout_controller import RolloutController
      assert not issubclass(GatewayRolloutController, RolloutController)
      print('NOT a subclass: PASS')
  except ImportError:
      print('Cannot import RolloutController (expected in Python 3.10), skipping check')
  "
    Expected Result: prints 'NOT a subclass: PASS' or skips gracefully
    Evidence: .sisyphus/evidence/task-5-not-subclass.txt
  ```

  **Commit**: YES
  - Message: `feat(gateway/controller): implement GatewayRolloutController with WorkflowExecutor`

---

- [ ] 6. Unit tests for data proxy weight update endpoints

  **What to do**:
  - Create `tests/experimental/gateway/test_data_proxy_weight_updates.py`:
    - Test the new weight update endpoints added to the data proxy in Task 3
    - Mock the SGLang backend HTTP calls (same pattern as existing data proxy tests)
    - Tests to include:
      - `test_update_weights_from_disk_forwards_to_sglang` — verify HTTP forward
      - `test_update_weights_from_distributed_forwards_to_sglang` — verify HTTP forward
      - `test_init_weights_update_group_forwards_to_sglang` — verify HTTP forward
      - `test_set_version_forwards_to_sglang` — verify HTTP forward
      - `test_weight_update_requires_admin_key` — verify auth enforcement
      - `test_weight_update_handles_sglang_error` — error propagation
      - `test_weight_update_handles_sglang_timeout` — timeout handling
    - Use `httpx.AsyncClient` with `app=data_proxy_app` (same test pattern as existing data proxy tests)

  **Must NOT do**:
  - Do NOT require GPU or real SGLang server
  - Do NOT modify existing test files

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 3 (parallel with Task 7). Depends on Tasks 3, 4.

  **References**:
  - `tests/experimental/gateway/test_data_proxy_generate.py` — test pattern for data proxy endpoints
  - `tests/experimental/gateway/test_data_proxy_pause.py` — test pattern for control-plane endpoints
  - `tests/experimental/gateway/conftest.py` — Python 3.10 compat shim
  - `areal/experimental/gateway/data_proxy/weight_update.py` — endpoints under test (created in Task 3)

  **Acceptance Criteria**:
  - [ ] All new weight update tests pass
  - [ ] At least 7 tests total
  - [ ] No GPU requirements

  **QA Scenarios**:
  ```
  Scenario: Weight update tests pass
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_data_proxy_weight_updates.py -v --tb=long
    Expected Result: All PASSED, at least 7 tests
    Evidence: .sisyphus/evidence/task-6-weight-tests.txt
  ```

  **Commit**: YES (group with Task 7)
  - Message: `test(gateway/controller): unit tests for controller and weight update endpoints`

---

- [ ] 7. Unit tests for GatewayRolloutController

  **What to do**:
  - Create `tests/experimental/gateway/test_controller.py`:
    - Tests for GatewayRolloutController with mocked HTTP endpoints (no real gateway/router/data proxy)
    - Use `unittest.mock.patch`, `httpx.AsyncClient`, or `aioresponses` to mock HTTP calls
    - Test classes to include:
    - **`TestGatewayControllerConfig`**:
      - `test_config_defaults` — verify all default values
      - `test_config_custom_values` — verify custom values override defaults
    - **`TestGatewayInfEngine`**:
      - `test_agenerate_makes_http_post` — verify HTTP POST to gateway `/generate`
      - `test_set_version_get_version` — version management roundtrip
      - `test_pause_resume` — pause/resume delegation
    - **`TestGatewayRolloutControllerConstruction`**:
      - `test_constructor` — verify constructor accepts `(config, scheduler)`
      - `test_initial_state` — verify initial state (version=0, no workers, etc.)
    - **`TestGatewayRolloutControllerAPISurface`**:
      - `test_has_all_methods` — verify all required methods exist (duck-type check)
      - `test_not_subclass_of_rollout_controller` — verify NOT a subclass
    - **`TestGatewayRolloutControllerVersionManagement`**:
      - `test_get_version_initial` — initial version is 0
      - `test_set_version_updates_version` — set and get version
      - `test_set_version_broadcasts_to_gateway` — verify HTTP call to gateway
    - **`TestGatewayRolloutControllerPauseResume`**:
      - `test_pause_calls_gateway` — verify HTTP POST to gateway /pause_generation
      - `test_resume_calls_gateway` — verify HTTP POST to gateway /continue_generation
    - **`TestGatewayRolloutControllerWeightUpdates`**:
      - `test_init_weights_update_group_calls_gateway` — verify HTTP broadcast
      - `test_update_weights_from_distributed_calls_gateway` — verify HTTP broadcast
      - `test_update_weights_from_disk_calls_gateway` — verify HTTP broadcast
    - **`TestGatewayRolloutControllerCallbackServer`**:
      - `test_callback_addr_after_server_start` — verify callback_addr property
      - `test_callback_pause_generation` — verify callback route works
      - `test_callback_continue_generation` — verify callback route works
    - **`TestGatewayRolloutControllerExportStats`**:
      - `test_export_stats_returns_dict` — verify return type
    - At least 20 tests total

  **Must NOT do**:
  - Do NOT require GPU or real SGLang/gateway/router services
  - Do NOT modify existing test files
  - Do NOT modify `tests/test_rollout_controller.py`

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 3 (parallel with Task 6). Depends on Task 5.

  **References**:
  - `tests/test_rollout_controller.py` — test patterns to adapt (MockScheduler, test structure)
  - `tests/experimental/gateway/conftest.py` — Python 3.10 compat shim
  - `tests/experimental/gateway/test_gateway.py` — gateway test patterns with mocked HTTP
  - `areal/experimental/gateway/controller/controller.py` — class under test (created in Task 5)
  - `areal/experimental/gateway/controller/inf_engine.py` — engine under test (created in Task 2)

  **Acceptance Criteria**:
  - [ ] All controller tests pass
  - [ ] At least 20 tests total
  - [ ] No GPU requirements
  - [ ] Tests verify HTTP routing, not scheduler RPC

  **QA Scenarios**:
  ```
  Scenario: Controller tests pass
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_controller.py -v --tb=long
    Expected Result: All PASSED, at least 20 tests
    Evidence: .sisyphus/evidence/task-7-controller-tests.txt
  ```

  **Commit**: YES (group with Task 6)
  - Message: `test(gateway/controller): unit tests for controller and weight update endpoints`

---

- [ ] 8. Full verification pass

  **What to do**:
  - Run the full test suite to verify no regressions:
    `"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/ -v --tb=long --ignore=tests/experimental/gateway/test_data_proxy_integration.py --ignore=tests/experimental/gateway/test_gateway_integration.py`
  - Run pre-commit on all new/modified files
  - Verify import: `from areal.experimental.gateway.controller import GatewayRolloutController`

  **Recommended Agent Profile**: `quick`
  **Parallelization**: Wave FINAL. Depends on Tasks 6, 7.

  **Acceptance Criteria**:
  - [ ] ALL gateway tests pass (124 existing + new)
  - [ ] Pre-commit passes on all new/modified files
  - [ ] GatewayRolloutController has complete API surface

  **Commit**: NO (verification only)

---

- [ ] F1. **Full Test Suite Pass**
  Run ALL gateway tests: `"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/ -v --tb=long --ignore=tests/experimental/gateway/test_data_proxy_integration.py --ignore=tests/experimental/gateway/test_gateway_integration.py`
  Expected: ALL tests pass (124 existing + new controller tests + new weight update tests).

- [ ] F2. **Pre-commit Verification**
  Run pre-commit on all new/modified files.

---

## Commit Strategy

- **Commit 1** (after Tasks 1-4): `feat(gateway/controller): add GatewayControllerConfig, GatewayInfEngine, and weight update endpoints`
- **Commit 2** (after Task 5): `feat(gateway/controller): implement GatewayRolloutController with WorkflowExecutor`
- **Commit 3** (after Tasks 6-7): `test(gateway/controller): unit tests for controller and weight update endpoints`

---

## Success Criteria

```bash
# All gateway tests pass (existing + new)
"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/ -v --tb=long --ignore=tests/experimental/gateway/test_data_proxy_integration.py --ignore=tests/experimental/gateway/test_gateway_integration.py

# GatewayRolloutController has same API surface
"D:\Programs\Python\Python310\python.exe" -c "
from areal.experimental.gateway.controller import GatewayRolloutController
methods = ['initialize', 'destroy', 'submit', 'wait', 'rollout_batch',
           'prepare_batch', 'agenerate', 'set_version', 'get_version',
           'get_capacity', 'pause', 'resume', 'export_stats',
           'init_weights_update_group', 'update_weights_from_distributed',
           'update_weights_from_disk', 'pause_generation', 'continue_generation']
for m in methods:
    assert hasattr(GatewayRolloutController, m), f'Missing method: {m}'
print('All methods present')
"
```
