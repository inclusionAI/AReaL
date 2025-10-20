# Single Controller Architecture for AReaL

**Status:** Draft Design **Authors:** AReaL Team **Last Updated:** 2025-10-20

## Table of Contents

- [Background](#background)
- [Motivation](#motivation)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Proposed Architecture](#proposed-architecture)
- [API Design](#api-design)
- [Implementation Details](#implementation-details)
- [Task Breakdown](#task-breakdown)

______________________________________________________________________

## Background

AReaL currently uses an **SPMD (Single Program, Multiple Data)** execution model
inherited from standard pre-training frameworks like PyTorch FSDP, Megatron-LM, and TRL.
In this model, training experiments spawn `N` processes that execute identical code,
with data and model parameters sharded across processes.

While SPMD works well for supervised pre-training, it introduces significant
inefficiencies in reinforcement learning (RL) training workflows.

### The Straggler Problem

Reinforcement learning training differs from supervised learning by introducing an
**inference phase** (rollout generation) before each training step. In the current SPMD
architecture:

1. **Prompts are partitioned** across training processes based on data parallelism
1. Each process submits its partition to inference engines and **waits for completion**
1. Generation lengths vary unpredictably across prompts
1. Training cannot proceed until **all processes** receive their completions

This creates a **straggler problem**: processes with shorter generations wait idle for
processes with longer generations, degrading GPU utilization and overall throughput.

### Why Not Centralized Loading?

A naive solution is to load and submit all data on rank 0. However, this creates a
**communication bottleneck**, especially problematic for vision-language models (VLMs)
that transmit large image tensors. Centralizing data movement creates a single point of
congestion that limits scalability.

______________________________________________________________________

## Motivation

We need an execution model that:

1. **Eliminates stragglers** by decoupling rollout generation from training
   synchronization
1. **Avoids communication bottlenecks** by distributing data loading across processes
1. **Separates control plane from data plane** to enable flexible scheduling

The **single controller** architecture addresses these requirements by introducing a
centralized control plane that orchestrates distributed engines while keeping data
movement decentralized. This architectural pattern has been successfully adopted by
[verl](https://github.com/volcengine/verl).

______________________________________________________________________

## Goals and Non-Goals

### Goals

- **Minimal API changes**: Users should migrate with \< 10 lines of code changes
- **Transparent data distribution**: Users manipulate `DistributedBatch` objects without
  manual data flow control
- **Fault tolerance**: Automatic engine restart and request retry on failures

### Non-Goals

- **Changing training algorithms**: PPO/GRPO logic remains unchanged
- **Breaking existing workflows**: SPMD mode remains supported for compatibility
- **Modifying engine internals**: TrainEngine/InferenceEngine APIs stay intact
- **Optimizing data transfer protocols**: Initial implementation uses HTTP RPC
  (optimization deferred to future work)

______________________________________________________________________

## Proposed Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Training Script                        │
│                      (Control Plane)                            │
│                                                                 │
│  ┌─────────────────┐              ┌──────────────────┐         │
│  │ TrainController │              │ RolloutController│         │
│  │                 │              │                  │         │
│  │  - Metadata     │◄────────────►│  - Metadata      │         │
│  │  - RPC Calls    │              │  - RPC Calls     │         │
│  └────────┬────────┘              └────────┬─────────┘         │
│           │                                │                   │
└───────────┼────────────────────────────────┼───────────────────┘
            │                                │
            │ RPC (metadata only)            │ RPC (metadata only)
            │                                │
    ┌───────▼──────────┐          ┌─────────▼────────────┐
    │  TrainEngine     │          │ InferenceEngine      │
    │  Workers (N)     │          │ Workers (M)          │
    │                  │          │                      │
    │  ┌────────────┐  │          │  ┌────────────┐     │
    │  │ FSDP/Megatron│──────────┼──>┤ SGLang/vLLM│     │
    │  │ Process 0    │  Weight   │  │ Server 0     │     │
    │  └────────────┘  │  Update   │  └────────────┘     │
    │  ┌────────────┐  │          │  ┌────────────┐     │
    │  │ FSDP/Megatron│  │          │  │ SGLang/vLLM│     │
    │  │ Process 1    │  │          │  │ Server 1     │     │
    │  └────────────┘  │          │  └────────────┘     │
    │       ...        │          │       ...          │
    └──────────────────┘          └────────────────────┘
         (Data Plane)                  (Data Plane)
```

### Key Components

#### 1. Controllers (Control Plane)

**TrainController** and **RolloutController** run in the user script as lightweight
orchestrators:

- **Launch/terminate engines** via scheduler APIs
- **Transmit metadata only** (tensor shapes, dtypes, storage locations)
- **Invoke engine methods** via RPC without blocking on data transfer
- **Aggregate results** from multiple workers

#### 2. Engines (Data Plane)

**TrainEngine** and **InferenceEngine** workers run on allocated GPU resources:

- **Store data locally** (no centralized data movement)
- **Execute compute** (training, inference)
- **Return metadata** to controllers
- **Expose RPC endpoints** for remote method invocation

#### 3. DistributedBatch

A metadata container that abstracts distributed tensor storage:

```python
class DistributedBatch:
    # Stores: tensor shapes, dtypes, worker addresses, unique IDs
    # Does NOT store: actual tensor data (stays on workers)

    def load_data(self) -> Dict[str, torch.Tensor]:
        """Fetch data from remote workers via RPC."""

    def chunk(self, dp_size: int) -> List[DistributedBatch]:
        """Partition metadata across data parallel workers."""
```

______________________________________________________________________

## API Design

### Before: SPMD Mode

```python
# Launch: python3 -m areal.launcher.local script.py --config xxx.yaml

def main(args):
    # Initialize train engine (SPMD: N identical processes)
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset
    train_dataloader = create_dataloader(...)

    # Initialize inference engine (remote client)
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    # Connect engines for weight synchronization
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    workflow = RLVRWorkflow(...)

    for global_step in range(start_step, max_steps):
        # Data-parallel head loads batch
        batch = None
        if actor.is_data_parallel_head():
            batch = rollout.prepare_batch(...)
            batch = tensor_container_to(batch, actor.device)

        # Broadcast batch to all DP ranks
        batch = broadcast_tensor_container(
            batch,
            src_rank=actor.current_data_parallel_head(),
            group=actor.context_and_model_parallel_group,
        )

        # Training step
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            logp = actor.compute_logp(batch)
            batch["prox_logp"] = logp

        actor.compute_advantages(batch)
        stats = actor.ppo_update(batch)
        actor.step_lr_scheduler()

        # Update inference engine weights
        actor.update_weights(weight_update_meta)
        actor.set_version(global_step + 1)
```

### After: Single Controller Mode

```python
# Launch: python3 script.py --config xxx.yaml
# No launcher needed!

def main(args):
    # Initialize train engine (wrapped in controller)
    actor = TrainController(
        engine=FSDPPPOActor(config=config.actor),
        scheduler=LocalScheduler(...),  # or RayScheduler, SlurmScheduler
    )
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset
    train_dataloader = create_dataloader(...)

    # Initialize inference engine (wrapped in controller)
    rollout = RolloutController(
        engine=RemoteSGLangEngine(config.rollout),
        scheduler=LocalScheduler(...),
    )
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    # Connect engines for weight synchronization
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    workflow = RLVRWorkflow(...)

    for global_step in range(start_step, max_steps):
        # Prepare batch (returns DistributedBatch with metadata)
        batch = rollout.prepare_batch(...)
        # No manual broadcasting needed!

        # Training step (controller handles data distribution)
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            logp = actor.compute_logp(batch)
            batch["prox_logp"] = logp

        actor.compute_advantages(batch)
        stats = actor.ppo_update(batch)
        actor.step_lr_scheduler()

        # Update inference engine weights
        actor.update_weights(weight_update_meta)
        actor.set_version(global_step + 1)
```

### Key Differences

| Aspect              | SPMD Mode                                  | Single Controller Mode                                     |
| ------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **Launch**          | `python -m areal.launcher.local script.py` | `python script.py`                                         |
| **Engine Creation** | `FSDPPPOActor(...)`                        | `TrainController(engine=FSDPPPOActor(...), scheduler=...)` |
| **Batch Type**      | `Dict[str, Tensor]`                        | `DistributedBatch` (metadata container)                    |
| **Data Loading**    | Manual head-rank loading + broadcast       | Automatic metadata loading as `DistributedBatch`           |
| **Data Movement**   | Explicit `broadcast_tensor_container()`    | Implicit via RPC                                           |
| **Training Loop**   | Unchanged                                  | Unchanged                                                  |

______________________________________________________________________

## Implementation Details

### TrainController

**Purpose:** Orchestrate N SPMD `TrainEngine` workers for distributed training.

**Design:**

- Maintains SPMD semantics internally (N identical workers with different ranks)
- Distributes `DistributedBatch` metadata according to parallel strategy
- Workers fetch their data partitions via RPC when needed
- Aggregates output metadata from all workers

### RolloutController

**Purpose:** Orchestrate M independent `InferenceEngine` workers for rollout generation.

**Design:**

- **Not SPMD**: Each worker is fully independent (no collective communication)
- Workers may use internal parallelism (TP/PP/EP) but are data-parallel across workers
- Request-level load balancing across M workers
- Integrated capacity and staleness control (from `WorkflowExecutor`)
- Fault tolerance via engine health tracking and request retry

### DistributedBatch

**Purpose:** Metadata container for distributed tensor storage.

**Design (Prototype: Lazy Loading):**

```python
@dataclass
class TensorMetadata:
    id: str                    # Unique identifier
    worker_addr: str           # RPC endpoint (http://ip:port)
    shape: Tuple[int, ...]     # Tensor shape
    dtype: torch.dtype         # Data type
    device: str                # Original device (for debugging)

class DistributedBatch:
    def __init__(self, metadata: Dict[str, TensorMetadata]):
        self._metadata = metadata
        self._cache: Dict[str, torch.Tensor] = {}

    def load_data(self) -> Dict[str, torch.Tensor]:
        """
        Fetch tensors from workers via RPC.

        Implementation:
        1. Group requests by worker_addr
        2. Batch RPC calls per worker
        3. Cache loaded data in self._cache
        4. Return materialized tensors
        """
        if self._cache:
            return self._cache

        # Group by worker
        requests_by_worker = defaultdict(list)
        for key, meta in self._metadata.items():
            requests_by_worker[meta.worker_addr].append((key, meta.id))

        # Parallel RPC fetch
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._fetch_from_worker, addr, ids): addr
                for addr, ids in requests_by_worker.items()
            }
            for future in as_completed(futures):
                results.update(future.result())

        self._cache = results
        return results

    def chunk(self, dp_size: int) -> List[DistributedBatch]:
        """
        Partition metadata across data parallel ranks.

        Uses FFD (First Fit Decreasing) algorithm to balance sequence lengths.
        """

    def __getitem__(self, key: str) -> TensorMetadata:
        """Access metadata for a specific key."""
        return self._metadata[key]

    def __setitem__(self, key: str, value: TensorMetadata):
        """Update metadata (e.g., after compute_logp)."""
        self._metadata[key] = value
        self._cache.pop(key, None)  # Invalidate cache
```

**Future Optimization (Eager Prefetch):**

```python
class DistributedBatch:
    def __init__(self, metadata, prefetch: bool = False):
        self._metadata = metadata
        self._cache = {}
        if prefetch:
            self._prefetch_future = executor.submit(self.load_data)

    def load_data(self):
        if hasattr(self, '_prefetch_future'):
            return self._prefetch_future.result()
        # ... lazy loading logic
```

### Scheduler

**Abstract API** (already implemented in `areal/api/scheduler_api.py`):

```python
class Scheduler(abc.ABC):
    @abc.abstractmethod
    def create_workers(self, config: SchedulingConfig) -> str:
        """Launch workers, return job ID."""

    @abc.abstractmethod
    def get_workers(self, job_id: str) -> List[Worker]:
        """Wait for workers, return endpoints."""

    @abc.abstractmethod
    def delete_workers(self, job_id: str):
        """Terminate all workers."""

    @abc.abstractmethod
    async def create_engine(self, worker_id: str, engine_obj, init_config):
        """Instantiate engine on worker via RPC."""

    @abc.abstractmethod
    async def call_engine(self, worker_id: str, method: str, *args, **kwargs):
        """Invoke engine method via RPC."""
```

**Concrete Implementations:**

- `LocalScheduler`: Uses `subprocess.Popen` with `CUDA_VISIBLE_DEVICES`
- `SlurmScheduler`: Submits jobs via `sbatch`, parses job IDs
- `RayScheduler`: Uses Ray actors for worker management

### RPC Protocol

**Current Implementation** (HTTP-based, in `areal/scheduler/rpc/`):

- **Transport:** HTTP POST with cloudpickle + gzip
- **Endpoints:**
  - `/create_engine`: Initialize engine on worker
  - `/call`: Invoke engine method
  - `/fetch_data` (not implemented): Fetch data produced in the server
- **Serialization:** Automatic `DistributedBatch` ↔ dict conversion

**Future Optimization:** Consider gRPC or ZeroMQ for lower latency.

______________________________________________________________________

## Task Breakdown

### Phase 1: Core Infrastructure (Weeks 1-2)

#### 1.1 DistributedBatch Refinement

**File:** `areal/controller/distributed_batch.py`

- [ ] **Implement `TensorMetadata` dataclass**

  - Fields: `id`, `worker_addr`, `shape`, `dtype`, `device`, `seqlen`
  - Serialization support (use json instead of python pickle for safety reasons)

- [ ] **Revise `DistributedBatch` class**

  - [ ] `__init__(metadata: Dict[str, TensorMetadata])`
  - [ ] `load_data() -> Dict[str, Tensor]` with RPC batching
  - [ ] `chunk(dp_size: int, method='ffd') -> List[DistributedBatch]` using FFD
  - [ ] `concat(batches: List[DistributedBatch]) -> DistributedBatch`
  - [ ] `__getitem__(key: str) -> TensorMetadata`
  - [ ] `__setitem__(key: str, value: TensorMetadata)`
  - [ ] `__delitem__(key: str)`

- [ ] **RPC data transfer helpers**

  - [ ] `_fetch_from_worker(addr, tensor_ids)` with retry logic
  - [ ] Connection pooling for HTTP clients

#### 1.2 Enhanced RPC Server for Data Storage

**File:** `areal/scheduler/rpc/rpc_server.py`

- [ ] **Add `/store_data` endpoint**

  - Accept: `{tensor_id: str, data: Dict[str, Tensor]}`
  - Store in memory with LRU eviction policy
  - Return: `{tensor_id: str, metadata: Dict[str, TensorMetadata]}`

- [ ] **Add `/fetch_data` endpoint**

  - Accept: `{tensor_ids: List[str]}`
  - Return: `{tensor_id: Dict[str, Tensor], ...}`
  - Support batch loading for efficiency

- [ ] **Add `/delete_data` endpoint**

  - Accept: `{tensor_ids: List[str]}`
  - Free memory for garbage collection

- [ ] **Worker metadata tracking**

  - Track total memory usage
  - Expose `/health` endpoint with capacity info

**Testing:**

- Concurrent store/load stress tests
- Memory leak verification
- Large tensor handling (> 1GB)

#### 1.3 Scheduler Enhancements

**File:** `areal/scheduler/local_scheduler.py`

- [ ] **Implement `LocalScheduler.restart_worker(worker_id)`**

  - Terminate crashed worker process
  - Restart with same config
  - Update worker registry

- [ ] **Add health monitoring**

  - Periodic heartbeat checks (every 30s)
  - Mark workers as `HEALTHY` / `UNHEALTHY` / `RESTARTING`
  - Callback hooks for state transitions

**File:** `areal/scheduler/slurm_scheduler.py`

- [ ] **Implement `SlurmScheduler.restart_worker(worker_id)`**
  - Cancel Slurm job step
  - Re-submit with `sbatch`

**File:** `areal/scheduler/ray_scheduler.py`

- [ ] **Implement `RayScheduler.restart_worker(worker_id)`**
  - Kill Ray actor
  - Spawn new actor with same resources

**Testing:**

- Worker crash simulation
- Restart latency measurement
- Multi-scheduler integration tests

______________________________________________________________________

### Phase 2: Controller Implementation (Weeks 3-4)

#### 2.1 TrainController

**File:** `areal/controller/train_controller.py`

- [ ] **Core initialization**

  - [ ] `__init__(engine: Type[TrainEngine], scheduler: Scheduler)`
  - [ ] `initialize(*args, **kwargs)`: Create workers
  - [ ] `destroy()`: Cleanup workers and RPC clients

- [ ] **Distributed batch handling**

  - [ ] `_partition_batch(batch: DistributedBatch, parallel_strategy)`
  - [ ] `_aggregate_stats(results: List[Dict])`

- [ ] **API forwarding**

  - [ ] `create_process_group(parallel_strategy)`
  - [ ] `train_batch(batch: DistributedBatch, loss_fn, loss_weight_fn)`
  - [ ] `eval_batch(batch: DistributedBatch, loss_fn, loss_weight_fn)`
  - [ ] `forward(batch: DistributedBatch)`
  - [ ] `update_weights(meta: WeightUpdateMeta)`
  - [ ] `set_version(version: int)`
  - [ ] `get_version() -> int`
  - [ ] `save(meta: SaveLoadMeta)`
  - [ ] `load(meta: SaveLoadMeta)`
  - [ ] `connect_engine(rollout_controller, meta)`
  - [ ] `step_lr_scheduler()`

**Testing:**

- FSDP training smoke test (2 GPUs, small model)
- Megatron training smoke test (4 GPUs, TP=2, DP=2)
- Weight synchronization test with RolloutController
- Checkpoint save/load test

#### 2.2 RolloutController

**File:** `areal/controller/rollout_controller.py`

- [ ] **Core initialization**

  - [ ] `__init__(engine: Type[InferenceEngine], scheduler: Scheduler)`
  - [ ] `initialize(train_data_parallel_size: int)`: Launch M workers
  - [ ] `destroy()`: Cleanup workers
  - [ ] `_discover_workers()`: Auto-detect worker endpoints via scheduler

- [ ] **Capacity & staleness control**

  - [ ] Integrate `StalenessManager` from `areal/core/staleness_manager.py`
  - [ ] `get_capacity() -> int`: Aggregate capacity across workers
  - [ ] Track per-worker running counts

- [ ] **Request scheduling**

  - [ ] `_schedule_request(data, workflow) -> str`: Return request ID
  - [ ] Round-robin scheduler with worker health check
  - [ ] Request retry queue for failed requests

- [ ] **Fault tolerance**

  - [ ] `_monitor_worker_health()`: Background thread with heartbeat
  - [ ] `_handle_worker_failure(worker_id)`: Auto-restart via scheduler
  - [ ] `_retry_failed_requests()`: Re-submit to healthy workers
  - [ ] Exponential backoff for repeated failures (max 3 retries)

- [ ] **API forwarding**

  - [ ] `submit(data, workflow, workflow_builder, should_accept)`
  - [ ] `wait(count: int, timeout: float) -> DistributedBatch`
  - [ ] `rollout_batch(data, workflow) -> DistributedBatch`
  - [ ] `prepare_batch(dataloader, workflow) -> DistributedBatch`
  - [ ] `init_weights_update_group(meta)`
  - [ ] `update_weights_from_distributed(meta, param_specs)`
  - [ ] `update_weights_from_disk(meta)`
  - [ ] `set_version(version: int)`
  - [ ] `get_version() -> int`
  - [ ] `pause_generation()`
  - [ ] `continue_generation()`
  - [ ] `pause()`
  - [ ] `resume()`

**Testing:**

- Single worker rollout test
- Multi-worker (M=4) rollout with load balancing
- Worker crash and recovery test
- Staleness control verification (`max_staleness` enforcement)
- Request retry test with injected failures

______________________________________________________________________

### Phase 3: Integration & Migration (Week 5)

#### 3.1 Example Migration

**File:** `examples/math/gsm8k_grpo_single_controller.py`

- [ ] **Create `examples/math/gsm8k_grpo_single_controller.py`**

  - [ ] Replace launcher with direct script execution
  - [ ] Wrap `FSDPPPOActor` with `TrainController`
  - [ ] Wrap `RemoteSGLangEngine` with `RolloutController`
  - [ ] Remove manual broadcasting logic
  - [ ] Update type hints for `DistributedBatch`

- [ ] **Create README with migration guide**

  - [ ] Side-by-side comparison
  - [ ] Performance benchmarks (straggler elimination)

______________________________________________________________________

### Phase 4: Testing & Validation (Week 6)

#### 4.1 Unit Tests

- [ ] **DistributedBatch tests** (`tests/test_distributed_batch.py`)

  - Chunking correctness (DP=1,2,4,8)
  - FFD algorithm validation
  - RPC fetch with mock servers
  - Serialization round-trip

- [ ] **TrainController tests** (`tests/test_train_controller.py`)

  - Process group creation
  - Batch partitioning by parallel strategy
  - Stats aggregation
  - RPC error handling

- [ ] **RolloutController tests** (`tests/test_rollout_controller.py`)

  - Worker discovery
  - Round-robin scheduling
  - Capacity calculation
  - Fault tolerance and retry

#### 4.2 Integration Tests

- [ ] **Local scheduler integration** (`tests/integration/test_local_scheduler.py`)

  - Launch 2 TrainEngine workers + 2 InferenceEngine workers
  - Run 5 training steps
  - Verify weight synchronization

- [ ] **Slurm scheduler integration** (requires cluster access)

  - Submit multi-node job
  - Verify worker discovery across nodes

- [ ] **Ray scheduler integration** (`tests/integration/test_ray_scheduler.py`)

  - Ray cluster setup
  - Actor-based worker management

#### 4.3 Performance Benchmarks

- [ ] **Straggler reduction measurement**

  - [ ] SPMD mode vs. single controller mode
  - [ ] Vary prompt length distribution (uniform, skewed, bimodal)
  - [ ] Metrics: average training time per step, GPU utilization

- [ ] **Scalability test**

  - [ ] Scale M (inference workers) from 1 → 16
  - [ ] Scale N (training workers) from 8 → 64
  - [ ] Metrics: throughput (samples/sec), latency (s/step)

- [ ] **Fault tolerance overhead**

  - [ ] Baseline: no crashes
  - [ ] Inject worker crashes at 10% rate
  - [ ] Measure recovery latency and throughput impact

______________________________________________________________________

### Phase 5: Production Readiness (Week 7-8)

#### 5.1 Optimization

- [ ] **Implement eager prefetch in `DistributedBatch`**

  - [ ] Add `prefetch=True` parameter
  - [ ] Background thread for data loading
  - [ ] Cache eviction policy (LRU)

- [ ] **Capacity-aware scheduling in `RolloutController`**

  - [ ] Query worker capacity via `/health` endpoint
  - [ ] Weighted round-robin based on available slots

- [ ] **RPC protocol optimization**

  - [ ] Benchmark HTTP vs. gRPC vs. ZeroMQ
  - [ ] Implement zero-copy tensor transfer (shared memory for local workers)

#### 5.2 Observability

- [ ] **Controller metrics** (`areal/controller/metrics.py`)

  - [ ] RPC latency histogram
  - [ ] Worker health status gauge
  - [ ] Request retry count
  - [ ] Data transfer bandwidth

- [ ] **Integration with `StatsLogger`**

  - [ ] Report controller metrics to W&B/SwanLab
  - [ ] Distributed tracing for request lifecycle

- [ ] **Debug utilities**

  - [ ] `controller.inspect_workers()`: List all workers and status
  - [ ] `controller.get_batch_provenance(batch)`: Show which workers hold data

#### 5.3 Hardening

- [ ] **Error handling**

  - [ ] Graceful degradation when all workers fail
  - [ ] Clear error messages for config mismatches
  - [ ] Validation of parallel strategy vs. worker count

- [ ] **Resource cleanup**

  - [ ] Ensure worker termination on Ctrl+C
  - [ ] Cleanup RPC connections on destroy
  - [ ] Delete remote data on batch garbage collection

- [ ] **Configuration validation**

  - [ ] Check `SchedulingConfig` compatibility with engine requirements
  - [ ] Validate GPU availability before launch
