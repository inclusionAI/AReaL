# Plan 4: GatewayController — Drop-in Replacement for RolloutController

## TL;DR

> **Quick Summary**: Build `GatewayController` as a **drop-in replacement** for `areal.infra.controller.RolloutController`. It must have the identical public API — same constructor signature `(inf_engine, config, scheduler)`, same `initialize()`, `destroy()`, `submit()`, `wait()`, `rollout_batch()`, `prepare_batch()`, `agenerate()`, and all version/weight/pause methods. Internally, instead of direct RPC to inference engines, it launches gateway microservices (Router, Data Proxy, Gateway) as a sidecar layer that routes requests to the engines managed by the scheduler.
>
> **Core Design Principle**: `GatewayController` **IS** a `RolloutController` subclass (or duck-type-compatible replacement). The existing test suite `tests/test_rollout_controller.py` MUST pass when `RolloutController` is swapped with `GatewayController` using the same `MockScheduler`.
>
> **Deliverables**:
> - `areal/experimental/gateway/controller/__init__.py`
> - `areal/experimental/gateway/controller/controller.py` — `GatewayController`
> - `tests/experimental/gateway/test_controller.py` — unit tests (can reuse MockScheduler from `tests/test_rollout_controller.py`)
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3

---

## Context

### The Original RolloutController API

`GatewayController` must match this exact public interface from `areal/infra/controller/rollout_controller.py`:

**Constructor:**
```python
class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    )
```

**Lifecycle:**
```python
def initialize(self, role: str, alloc_mode: AllocationMode, server_args=None, server_infos=None, *args, **kwargs)
def destroy(self)
```

**Task submission:**
```python
def submit(self, data, workflow, workflow_kwargs=None, should_accept_fn=None, task_id=None, is_eval=False, group_size=1, proxy_addr=None) -> int
def wait(self, count, timeout=None, raise_timeout=True) -> list[dict | None]
def rollout_batch(self, data, workflow, workflow_kwargs=None, should_accept_fn=None, group_size=1) -> dict
def prepare_batch(self, dataloader, workflow, workflow_kwargs=None, should_accept_fn=None, group_size=1, dynamic_bs=False) -> dict
```

**Inference:**
```python
async def agenerate(self, req: ModelRequest) -> ModelResponse
```

**Version management:**
```python
def set_version(self, version: int) -> None
def get_version(self) -> int
def get_capacity(self) -> int
```

**Weight updates:**
```python
async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None
async def update_weights_from_distributed(self, meta, param_specs) -> None
async def update_weights_from_disk(self, meta) -> None
```

**Control plane:**
```python
async def pause_generation(self)
async def continue_generation(self)
def pause(self)
def resume(self)
```

**Stats & properties:**
```python
def export_stats(self) -> dict[str, float]
@property staleness_manager
@property dispatcher
@property runner
@property callback_addr -> str
```

### What GatewayController Changes Internally

The key difference is that `GatewayController.initialize()` **additionally** spins up the gateway microservices (Router, Data Proxy, Gateway) as lightweight HTTP processes alongside the regular scheduler-managed inference engines. The microservices form a sidecar layer:

1. **Engine workers are still created via `scheduler.create_workers(Job(...))`** exactly like `RolloutController` — this is unchanged.
2. **After engines are ready**, `GatewayController` additionally starts:
   - One Router service (in-process or as a thread, NOT via scheduler — it's just a FastAPI app)
   - One Data Proxy per engine worker (each pointed at that worker's SGLang backend)
   - One Gateway service that knows about the Router
3. **Registers each data proxy as a worker in the Router**
4. For methods like `submit()`, `wait()`, `rollout_batch()` — the GatewayController delegates to the **same** `BatchTaskDispatcher` + callback mechanism as `RolloutController`. The gateway microservices provide an alternative HTTP-based access path for external callers.

### Key Decisions

- **Subclass approach**: `GatewayController` should **subclass** `RolloutController` and override `initialize()` and `destroy()` to add gateway service lifecycle. All other methods are inherited unchanged.
- **Same constructor signature**: `__init__(inf_engine, config, scheduler)` — identical to `RolloutController`.
- **Gateway services run in-process as background threads** (not via scheduler), since they're lightweight FastAPI apps. This avoids complex scheduler orchestration for non-GPU services.
- **The existing `tests/test_rollout_controller.py` serves as the conformance test**: swap `RolloutController` → `GatewayController` and every test must pass.

### Reference Test: `tests/test_rollout_controller.py`

This file (1395 lines) contains the conformance test suite. Key test classes:
- `TestRolloutControllerInitialization` — constructor, initialize, staleness manager
- `TestRolloutControllerDestroy` — cleanup, scheduler interaction
- `TestRolloutControllerCapacity` — get_capacity, version-based capacity
- `TestRolloutControllerWorkerSelection` — round-robin
- `TestRolloutControllerSubmitAndWait` — submit, wait, timeout
- `TestRolloutControllerBatchOperations` — rollout_batch
- `TestRolloutControllerVersionManagement` — set_version, get_version
- `TestRolloutControllerWeightUpdates` — init_weights_update_group, update_weights_from_distributed, update_weights_from_disk
- `TestRolloutControllerLifecycle` — pause, resume
- `TestRolloutControllerAgenerate` — async generation
- `TestRolloutControllerErrorHandling` — timeout, empty results
- `TestRolloutControllerIntegration` — end-to-end workflow
- Parametrized tests for worker counts and capacity settings

All these tests use `MockScheduler` and `MockInferenceEngine` — no GPU required.

---

## Work Objectives

### Core Objective
Create `GatewayController` that is API-identical to `RolloutController` and passes all existing tests from `tests/test_rollout_controller.py`.

### Must Have
- `GatewayController` has the EXACT same public API as `RolloutController`
- `GatewayController` can be used as a drop-in replacement in `tests/test_rollout_controller.py`
- All 40+ tests in `tests/test_rollout_controller.py` pass when `RolloutController` is replaced with `GatewayController`
- `GatewayController` additionally manages gateway microservice lifecycle (Router, Data Proxy, Gateway) in `initialize()`/`destroy()`
- Gateway services are optional — if they fail to start, the controller still works for scheduler-based rollouts
- `tests/experimental/gateway/test_controller.py` contains gateway-specific tests

### Must NOT Have
- NO changes to the existing `RolloutController` or `tests/test_rollout_controller.py`
- NO new constructor arguments that break the `(inf_engine, config, scheduler)` signature
- NO removal of any public methods from `RolloutController`
- NO changes to how `MockScheduler` works
- NO GPU requirements in unit tests

---

## Execution Strategy

```
Wave 1 (start immediately):
├── Task 1: GatewayController class (subclass of RolloutController) [unspecified-high]
└── Task 2: Package scaffold + __init__.py [quick]

Wave 2 (after Wave 1):
└── Task 3: Unit tests — conformance + gateway-specific [unspecified-high]
```

---

## TODOs

- [ ] 1. `GatewayController` — subclass with gateway service lifecycle

  **What to do**:
  - Create `areal/experimental/gateway/controller/__init__.py`:
    ```python
    from areal.experimental.gateway.controller.controller import GatewayController
    __all__ = ["GatewayController"]
    ```
  - Create `areal/experimental/gateway/controller/controller.py`:
    - `class GatewayController(RolloutController):` — subclass
    - **Constructor**: call `super().__init__(inf_engine, config, scheduler)`. Store extra gateway state:
      - `self._gateway_services_started = False`
      - `self._router_thread = None`
      - `self._data_proxy_threads = []`
      - `self._gateway_thread = None`
    - **Override `initialize()`**: call `super().initialize(role, alloc_mode, ...)` first, then launch gateway microservices:
      1. Start Router as a background thread (using `create_app` from `areal.experimental.gateway.router.app`)
      2. Start one Data Proxy per worker (using `create_app` from `areal.experimental.gateway.data_proxy.app`)
      3. Start Gateway (using `create_app` from `areal.experimental.gateway.gateway.app`)
      4. Register data proxies with Router
      5. If any gateway service fails, log warning but don't fail — the controller still works via direct scheduler RPC
    - **Override `destroy()`**: stop gateway services first, then call `super().destroy()`
    - **All other methods are INHERITED from RolloutController** — do NOT override them
    - Use lazy imports for all `areal.experimental.gateway.*` imports (inside methods) to avoid Python 3.10 compat issues

  **Critical Design Notes**:
  - The subclass approach means ALL of `submit()`, `wait()`, `rollout_batch()`, `prepare_batch()`, `agenerate()`, version/weight/pause methods work EXACTLY as in `RolloutController` — zero code duplication.
  - Gateway services are a SIDECAR — they provide an HTTP-based access path but don't change the core scheduler-based rollout flow.
  - If the user's `MockScheduler` doesn't have gateway-related methods, the gateway services simply won't start (graceful degradation).

  **Must NOT do**:
  - Do NOT override `submit()`, `wait()`, `rollout_batch()`, or any method that works fine via inheritance
  - Do NOT change the constructor signature
  - Do NOT import `areal.api.*` at module level (Python 3.10 compat)

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 1. Blocks Task 3.

  **References**:

  **Pattern References** (existing code to follow):
  - `areal/infra/controller/rollout_controller.py` — THE reference implementation. `GatewayController` subclasses this. Study every method.
  - `areal/infra/controller/rollout_controller.py:67-118` — Constructor: exact fields and their defaults
  - `areal/infra/controller/rollout_controller.py:150-220` — `initialize()`: worker creation, staleness manager setup, dispatcher creation, callback server start
  - `areal/infra/controller/rollout_controller.py:301-334` — `destroy()`: cleanup order

  **API/Type References**:
  - `areal/api/cli_args.py:InferenceEngineConfig` — config type for constructor
  - `areal/api/alloc_mode.py:AllocationMode` — `initialize()` parameter
  - `areal/api/scheduler_api.py:Scheduler` — scheduler interface
  - `areal/api/engine_api.py:InferenceEngine` — engine type for constructor

  **Gateway References** (for service lifecycle):
  - `areal/experimental/gateway/router/app.py` — `create_app(config)` factory
  - `areal/experimental/gateway/data_proxy/app.py` — `create_app(config)` factory
  - `areal/experimental/gateway/gateway/app.py` — `create_app(config)` factory
  - `areal/experimental/gateway/router/config.py` — `RouterConfig`
  - `areal/experimental/gateway/data_proxy/config.py` — `DataProxyConfig`
  - `areal/experimental/gateway/gateway/config.py` — `GatewayConfig`

  **Test References**:
  - `tests/test_rollout_controller.py` — THE conformance test suite. `GatewayController` must pass ALL of these tests.
  - `tests/test_rollout_controller.py:40-165` — `MockScheduler` class — understand what it provides
  - `tests/test_rollout_controller.py:167-175` — `MockInferenceEngine` class

  **Acceptance Criteria**:
  - [ ] `from areal.experimental.gateway.controller import GatewayController` works
  - [ ] `GatewayController(inf_engine, config, scheduler)` accepts same args as `RolloutController`
  - [ ] `GatewayController` is a subclass of `RolloutController` (or has identical API surface)
  - [ ] Gateway services start in `initialize()` and stop in `destroy()` (when available)
  - [ ] If gateway services fail to start, controller still works for scheduler-based operations

  **QA Scenarios**:
  ```
  Scenario: GatewayController imports and instantiates
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  from areal.experimental.gateway.controller import GatewayController
  print('import OK')
  "
    Expected Result: prints 'import OK'
    Evidence: .sisyphus/evidence/c1-import.txt

  Scenario: GatewayController has same methods as RolloutController
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  from areal.experimental.gateway.controller.controller import GatewayController
  methods = ['initialize', 'destroy', 'submit', 'wait', 'rollout_batch',
             'prepare_batch', 'agenerate', 'set_version', 'get_version',
             'get_capacity', 'pause', 'resume', 'export_stats',
             'init_weights_update_group', 'update_weights_from_distributed',
             'update_weights_from_disk', 'pause_generation', 'continue_generation']
  for m in methods:
      assert hasattr(GatewayController, m), f'Missing method: {m}'
  print('All methods present')
  "
    Expected Result: prints 'All methods present'
    Evidence: .sisyphus/evidence/c1-methods.txt
  ```

  **Commit**: YES
  - Message: `feat(gateway/controller): GatewayController as drop-in RolloutController replacement`

---

- [ ] 2. Package scaffold

  **What to do**:
  - Ensure `areal/experimental/gateway/controller/__init__.py` exists with proper exports
  - This is a trivial task, likely done as part of Task 1

  **Recommended Agent Profile**: `quick`
  **Parallelization**: Wave 1 (parallel with Task 1, or merged into Task 1).

  **Commit**: YES (group with Task 1)

---

- [ ] 3. Unit tests — conformance suite + gateway-specific tests

  **What to do**:
  - Create `tests/experimental/gateway/test_controller.py`:

  **Part A: Conformance Tests (CRITICAL)**
  - Import `MockScheduler`, `MockInferenceEngine`, `create_test_config` from `tests/test_rollout_controller.py` (or duplicate them)
  - Re-run ALL test scenarios from `tests/test_rollout_controller.py` but using `GatewayController` instead of `RolloutController`
  - The simplest approach: parametrize a fixture that yields both `RolloutController` and `GatewayController`, so both are tested with the same test logic
  - OR: import and reuse the helper functions, create parallel test classes

  **Minimum conformance tests to include** (matching `tests/test_rollout_controller.py`):
  - `TestGatewayControllerInitialization`:
    - `test_constructor` — same fields, same defaults
    - `test_initialize_creates_workers` — workers created via scheduler
    - `test_initialize_creates_staleness_manager` — staleness manager configured correctly
    - `test_initialize_uses_consumer_batch_size_as_fallback`
  - `TestGatewayControllerDestroy`:
    - `test_destroy_cleans_up_resources`
    - `test_destroy_deletes_workers_via_scheduler`
    - `test_destroy_handles_scheduler_error`
  - `TestGatewayControllerCapacity`:
    - `test_get_capacity_initial_state`
    - `test_get_capacity_uses_version`
  - `TestGatewayControllerWorkerSelection`:
    - `test_choose_worker_round_robin`
  - `TestGatewayControllerSubmitAndWait`:
    - `test_wait_returns_distributed_batch`
    - `test_submit_passes_is_eval_and_group_size`
  - `TestGatewayControllerBatchOperations`:
    - `test_rollout_batch_submits_all_data`
    - `test_rollout_batch_waits_for_all_results`
  - `TestGatewayControllerVersionManagement`:
    - `test_get_version_initial`
    - `test_set_version_updates_controller_version`
    - `test_set_version_calls_workers`
  - `TestGatewayControllerWeightUpdates`:
    - `test_init_weights_update_group_returns_future`
    - `test_update_weights_from_distributed_returns_future`
    - `test_update_weights_from_disk_returns_future`
  - `TestGatewayControllerLifecycle`:
    - `test_pause_calls_all_workers`
    - `test_resume_calls_all_workers`
  - `TestGatewayControllerAgenerate`:
    - `test_agenerate_chooses_worker`
    - `test_agenerate_round_robin`
  - `TestGatewayControllerIntegration`:
    - `test_end_to_end_workflow`
  - Parametrized tests for worker counts and capacity settings

  **Part B: Gateway-Specific Tests**
  - Test that `GatewayController` is a subclass of `RolloutController`
  - Test that gateway services start during `initialize()` (when mocks support it)
  - Test that gateway services stop during `destroy()`
  - Test graceful degradation when gateway services fail to start

  **Must NOT do**:
  - Do NOT modify `tests/test_rollout_controller.py`
  - Do NOT require GPU in any test
  - Do NOT require real network connections

  **Recommended Agent Profile**: `unspecified-high`
  **Parallelization**: Wave 2. Depends on Task 1.

  **References**:

  **Test References**:
  - `tests/test_rollout_controller.py` — THE source of truth. Copy test patterns exactly.
  - `tests/test_rollout_controller.py:40-165` — `MockScheduler` and `MockInferenceEngine` — reuse or duplicate
  - `tests/test_rollout_controller.py:25-37` — `create_test_config()` helper
  - `tests/experimental/gateway/conftest.py` — Python 3.10 compatibility shim (import this)

  **Acceptance Criteria**:
  - [ ] `"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_controller.py -v` — all pass
  - [ ] All conformance tests (matching `tests/test_rollout_controller.py` behavior) pass
  - [ ] Gateway-specific tests pass
  - [ ] At least 25 tests total

  **QA Scenarios**:
  ```
  Scenario: All controller tests pass
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_controller.py -v --tb=long
    Expected Result: All PASSED, 0 failures, at least 25 tests
    Evidence: .sisyphus/evidence/c3-tests.txt

  Scenario: GatewayController passes RolloutController conformance
    Tool: Bash
    Steps:
      "D:\Programs\Python\Python310\python.exe" -c "
  # Verify GatewayController can be substituted for RolloutController
  from areal.experimental.gateway.controller import GatewayController
  from areal.infra.controller.rollout_controller import RolloutController
  assert issubclass(GatewayController, RolloutController)
  print('Subclass check: PASS')
  "
    Expected Result: prints 'Subclass check: PASS'
    Evidence: .sisyphus/evidence/c3-subclass.txt
  ```

  **Commit**: YES
  - Message: `test(gateway/controller): conformance and gateway-specific tests for GatewayController`
  - Pre-commit: `"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_controller.py -v`

---

## Success Criteria

```bash
# Gateway controller tests pass
"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/test_controller.py -v

# All existing gateway tests still pass
"D:\Programs\Python\Python310\python.exe" -m pytest tests/experimental/gateway/ -v --tb=long --ignore=tests/experimental/gateway/test_data_proxy_integration.py --ignore=tests/experimental/gateway/test_gateway_integration.py

# GatewayController is a drop-in replacement
"D:\Programs\Python\Python310\python.exe" -c "
from areal.experimental.gateway.controller import GatewayController
from areal.infra.controller.rollout_controller import RolloutController
assert issubclass(GatewayController, RolloutController)
print('Drop-in replacement: CONFIRMED')
"
```
