# Plan: Rewrite Rollout Section with Accurate Async Details

## Background

The user wants to:

1. Remove lines 236-248 (redundant TrainController/RolloutController subsections under
   PPOTrainer)
1. Rewrite the "Rollout: Generating Training Data" section to accurately explain
   asynchronous generation

**Key insight from user**: "Asynchronous" here means:

- Python async/await semantics
- **Background thread** (`BatchTaskDispatcher`) submitting rollout requests concurrently
- **Separate GPU process** (SGLang/vLLM server) handling requests on GPU
- **Overlap generation with training**: Rollout and training happen simultaneously

## Current Issues

**Lines 236-248**: Redundant subsections after we already have H2 sections for Train and
Rollout

- "TrainController: RPC Mechanism"
- "RolloutController: Async Workflow Execution"

**Lines 287-406**: Current async explanation is too simplistic

- Doesn't explain the background thread architecture
- Doesn't explain how RPC server receives and queues requests
- Doesn't explain GPU process separation (SGLang/vLLM runs independently)
- Missing the overlap with training explanation

## Key Code Insights

From `areal/infra/controller/rollout_controller.py`:

1. **RolloutController** creates workers via scheduler
1. Each worker runs an RPC server (`areal/scheduler/rpc/rpc_server.py`)
1. RPC server launches SGLang/vLLM as separate subprocess
1. **BatchTaskDispatcher** (`_dispatcher`) runs in background thread
1. Dispatcher submits tasks via `_create_submit_callback()` which:
   - Chooses worker (round-robin)
   - Calls `scheduler.async_call_engine(worker_id, "submit", ...)`
   - Returns immediately (non-blocking)
   - Worker's `RemoteInfEngine.submit()` queues work to SGLang/vLLM
   - Callback received when trajectory completes

From `areal/infra/remote_inf_engine.py`:

1. **RemoteInfEngine** wraps SGLang/vLLM HTTP servers
1. `submit()` method:
   - Resolves workflow from string
   - Calls `workflow_executor.submit()` which queues the task
   - Returns task_id immediately (non-blocking)
1. **WorkflowExecutor** runs in background thread
1. When trajectory completes, sends HTTP callback to RolloutController

From `areal/scheduler/rpc/rpc_server.py`:

1. **Flask server** with `/create_engine`, `/call` endpoints
1. **Engine thread** (`_engine_thread`) runs all engine operations serially
1. SGLang/vLLM launched as subprocess via `backend.launch_server()`
1. Separate GPU process handles actual generation

## Architecture Diagram Needed

```
Controller Process                  Worker Process (RPC Server)           GPU Process
──────────────────                  ───────────────────────────           ───────────
RolloutController                   Flask HTTP Server (CPU)               SGLang/vLLM
    │                                   │                                     │
    └─> BatchTaskDispatcher         /call endpoint                       Inference Engine
        (background thread)             │                                     │
            │                           └─> Engine Thread                     │
            ├─ submit task 1                └─> RemoteInfEngine               │
            │  (HTTP POST)                       └─> workflow_executor        │
            │                                         └─> submit() ───────────>│
            ├─ submit task 2                                               Generate
            │  (HTTP POST)                                                  tokens
            │                                                                  │
            ├─ submit task 3                          HTTP Callback  <────────┘
            │                                         (trajectory)
            │                           ┌──────────────┘
            └─ collect results <────────┘

Meanwhile...
TrainController                     Training Worker (RPC Server)
    │                                   │
    └─> compute_logp(batch) ────────────>│ Forward pass
    └─> ppo_update(batch) ──────────────>│ Backward pass

Key: Generation and training happen SIMULTANEOUSLY on different GPUs
```

## Implementation Plan

### 1. Remove Lines 236-248

Delete the redundant subsections:

- "### TrainController: RPC Mechanism" (L236-246)
- "### RolloutController: Async Workflow Execution" (L248)

These are redundant since we have separate H2 sections for training and rollout.

### 2. Rewrite "Rollout: Generating Training Data" (L250-406)

**New structure:**

**L250: Keep H2 header**

```markdown
## Rollout: Generating Training Data
```

**L252-266: Keep Workflow Specification (minor edits)**

- Current content is good, just ensure it's concise

**L268-285: Keep RLVRWorkflow overview**

- Current 4-step description is good

**L287+: REWRITE "Asynchronous Rollout Collection"**

New content should explain:

1. **Architecture Overview** (with diagram):

   - Controller process: RolloutController + BatchTaskDispatcher (background thread)
   - Worker process: RPC server (Flask HTTP on CPU) + Engine thread
   - GPU process: SGLang/vLLM server (separate subprocess)
   - HTTP callbacks for result collection

1. **Three Levels of Concurrency**:

   - **Level 1 (Controller)**: BatchTaskDispatcher background thread submits requests
   - **Level 2 (Worker RPC)**: Flask server accepts concurrent HTTP requests, Engine
     thread processes serially
   - **Level 3 (GPU)**: SGLang/vLLM subprocess handles multiple concurrent generation
     requests

1. **Request Flow**:

   ```
   1. Controller: actor.prepare_batch() calls rollout.prepare_batch()
   2. RolloutController.prepare_batch():
      - Creates task_input_generator from dataloader
      - Dispatcher submits tasks to workers (round-robin)
   3. Worker RPC receives HTTP POST /call (method="submit"):
      - Deserializes workflow string + kwargs
      - Engine thread runs RemoteInfEngine.submit()
      - Queues work to SGLang/vLLM subprocess
      - Returns task_id (non-blocking)
   4. SGLang/vLLM subprocess:
      - Processes requests from queue
      - Generates tokens on GPU
      - Returns trajectory via HTTP callback
   5. Controller receives callback:
      - BatchTaskDispatcher collects results
      - Waits for batch_size accepted trajectories
      - Concatenates and returns
   ```

1. **Overlap with Training**:

   - While training workers compute gradients, rollout workers continue generating
   - BatchTaskDispatcher maintains 2+ batches of pending requests
   - Staleness control ensures generated data isn't too old

1. **Staleness Management**:

   - Brief explanation of version tracking
   - `max_head_offpolicyness` config
   - Pause/resume mechanism during weight sync

### 3. Sections After Rollout

Keep and verify:

- "Training: Distributed Computing with Controllers" (should focus on TrainController
  details)
- "Training Methods: What Happens on Workers"
- "Weight Synchronization"
- "Monitoring and Utilities"

## Files to Modify

1. **`docs/lite/gsm8k_grpo.md`**
   - Remove: Lines 236-248
   - Rewrite: Lines 287-406 (Asynchronous Rollout Collection section)

## Verification

After completion:

1. Run `mdformat docs/lite/gsm8k_grpo.md`
1. Run `pre-commit run --files docs/lite/gsm8k_grpo.md`
1. Verify all code references are accurate
1. Check that async architecture is correctly explained
1. Ensure diagram accurately reflects the three-process architecture
