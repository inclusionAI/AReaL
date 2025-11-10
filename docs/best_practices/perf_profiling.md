# Performance Profiling

AReaL provides a lightweight profiling infrastructure through `perf_tracer` to help you
identify performance bottlenecks in distributed training workflows. The tracer emits
Chrome Trace-compatible events that can be visualized in Perfetto or chrome://tracing,
making it easy to correlate computation, communication, and I/O across multiple ranks.

**Key capabilities**:

- Flexible tracing APIs: decorators (`@trace_perf`, `@trace_session`), context managers
  (`trace_scope`, `atrace_session_phase`), and markers (`instant`)
- **Per-session lifecycle tracking** (submission → queue → execution → consumption) with
  derived metrics (queue wait, execution time, phase breakdown)

## Why profile your training?

Common bottlenecks in distributed RL training:

- **Train**: Training step duration, checkpoint save/load, weight updates
- **Rollout**: Generation latency, long-tail sessions, batching efficiency
- **Coordination**: Queue saturation, weight staleness, phase imbalance (generate vs
  reward)

`perf_tracer` provides function-level and **per-session** tracing to answer:

- Where is my training loop spending time?
- Are rollout sessions queued, rejected, or accepted?
- How long does each phase (generate, reward, execution) take?

## Quick start

### 1. Enable tracing in your config

Add a [`PerfTracerConfig`](../cli_reference.md#section-perf-tracer) to your training
script's YAML config or CLI overrides:

```yaml
perf_tracer:
  enabled: true
  experiment_name: ${experiment_name}  # Reuse top-level metadata
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}        # Shared filesystem path
  save_interval: 1                     # Write traces every step
  session_tracer:
    enabled: true                      # Track per-session lifecycle
    flush_threshold: 100               # Buffer 100 sessions before flushing
```

See `examples/tracer/gsm8k_grpo.yaml` for a complete example.

### 2. Initialize the tracer

Call `perf_tracer.configure()` once per rank at startup:

```python
from areal.utils import perf_tracer

if config.perf_tracer is not None:
    perf_tracer.configure(config.perf_tracer, rank=rank)
```

The global tracer is now active for this process.

### 3. Run your training and collect traces

Execute your training script as usual. The tracer automatically writes events to
`fileroot/logs/.../perf_tracer/traces-r{rank}.jsonl`. For multi-rank jobs, each rank
produces its own file.

```bash
python examples/tracer/gsm8k_grpo.py --config examples/tracer/gsm8k_grpo.yaml
```

### 4. View traces in Perfetto

Convert JSONL to JSON and open in [Perfetto](https://ui.perfetto.dev/) or
chrome://tracing:

```bash
python -m areal.tools.perf_trace_converter logs/**/perf_tracer/traces-*.jsonl merged.json
```

## Profiling patterns and APIs

### Pattern 1: Trace entire functions with `@trace_perf`

**Use case**: Understand total time spent in key methods (train_batch, forward,
ppo_update, etc.).

**API**: `@trace_perf(name, category=...)`

- Decorator that wraps sync/async functions
- Automatically records start/end timestamps
- Handles exceptions gracefully

**Example** (from `areal/engine/fsdp_engine.py`):

```python
from areal.utils.perf_tracer import trace_perf

@trace_perf("fsdp_engine.train_batch", category="compute")
def train_batch(self, input_: dict[str, Any], loss_fn, loss_weight_fn):
    # Training logic here
    ...
```

This creates a "complete event" (duration span) named `fsdp_engine.train_batch` in the
Chrome Trace output. The `category="compute"` tag helps filter events in the viewer.

**Categories**:

- `compute`: CPU/GPU computation (forward, backward, loss)
- `io`: Disk I/O (checkpoint save/load)
- `comm`: Distributed communication (all-reduce, broadcast)
- `sync`: Synchronization barriers
- `scheduler`: Rollout scheduling/queueing

### Pattern 2: Trace code blocks with `trace_scope` / `atrace_scope`

**Use case**: Profile specific code sections without extracting them into methods.

**API**: Context managers for sync/async code

- `with trace_scope(name, category, args)`: Sync context
- `async with atrace_scope(name, category, args)`: Async context

**Example** (from `examples/tracer/gsm8k_grpo.py`):

```python
from areal.utils import perf_tracer
from areal.utils.perf_tracer import Category

with perf_tracer.trace_scope(
    "train.rollout",
    category=Category.COMPUTE,
    args={"global_step": global_step, "epoch_step": step},
):
    batch = actor.prepare_batch(dataloader, n_samples)
    # Rollout generation happens here
```

The `args` dict attaches metadata (step numbers, batch size, etc.) to the event, visible
in the trace viewer.

### Pattern 3: Track session lifecycles with `@trace_session`

**Use case**: Measure per-session timing for async rollout workflows (e.g., how long
does a single prompt take from submission to reward calculation?).

**API**: `@trace_session(phase)`

- Decorator for async methods that participate in session processing
- Automatically reads `session_id` from async context (set via
  `perf_tracer.set_session_id()`)
- Records `mark_{phase}_start` and `mark_{phase}_end` events

**Example** (from `areal/workflow/rlvr.py`):

```python
from areal.utils.perf_tracer import trace_perf, trace_session

class RLVRWorkflow(RolloutWorkflow):
    @trace_perf("rlvr_workflow.arun_episode", category="compute")
    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        # WorkflowExecutor automatically sets session_id before calling this method
        session_id = perf_tracer.get_session_id()

        # Generate responses
        async with perf_tracer.atrace_session_phase(session_id, "generate"):
            resps = await asyncio.gather(
                *[engine.agenerate(req) for _ in range(n_samples)]
            )

        # Compute rewards (decorated method below)
        rewards, completions = await self._compute_rewards(resps, prompt_str, data)
        ...

    @trace_session("reward")
    async def _compute_rewards(self, resps, prompt_str, task_data):
        # Reward calculation logic
        # The decorator automatically traces this as the "reward" phase
        rewards = []
        for resp in resps:
            completion = self.tokenizer.decode(resp.output_tokens)
            reward = await self.async_reward_fn(prompt_str, completion, ...)
            rewards.append(reward)
        return rewards, completions
```

**How it works**:

1. `WorkflowExecutor` calls `perf_tracer.set_session_id()` before invoking
   `arun_episode`
1. `get_session_id()` retrieves the session ID from the async context variable
1. Child async functions inherit this context automatically
1. `@trace_session("reward")` reads the session ID and logs phase start/end events
1. Session traces appear in `session_tracer/sessions-r{rank}.jsonl` with computed
   metrics like `reward_calc_s`, `generate_s`

### Pattern 4: Manual phase scopes with `atrace_session_phase`

**Use case**: Trace phases that aren't cleanly extractable into methods (e.g., inline
generation loops).

**API**:
`async with atrace_session_phase(session_id, phase, start_payload, end_payload)`

- Context manager for session-phase tracing
- Pairs `mark_{phase}_start` and `mark_{phase}_end` automatically

**Example** (shown in Pattern 3 above):

```python
async with perf_tracer.atrace_session_phase(session_id, "generate"):
    resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])
```

### Pattern 5: Add instant markers with `instant()`

**Use case**: Mark specific points in time (e.g., "batch prepared", "queue state
snapshot").

**API**: `perf_tracer.instant(name, category, args)`

- Creates a point-in-time marker (not a duration)
- Useful for events that don't have a meaningful duration

**Example** (from `areal/core/workflow_executor.py`):

```python
perf_tracer.instant(
    "workflow_executor.prepare_batch",
    category="scheduler",
    args={"data": len(data)}
)

perf_tracer.instant(
    "workflow_executor.wait",
    category="scheduler",
    args={
        "queue_size": runner.get_output_queue_size(),
        "pending_results": len(pending_results),
    }
)
```

### Pattern 6: Manual session lifecycle events with `trace_session_event`

**Use case**: Track session lifecycle at orchestration level (submission, execution,
consumption).

**API**: `perf_tracer.trace_session_event(session_id, event_name, **payload)`

- Manually record lifecycle events for session tracking
- Used by `WorkflowExecutor` to track full session lifecycle
- Events: `mark_enqueued`, `mark_execution_start`, `mark_execution_end`, `mark_consumed`

**Example** (from `areal/core/workflow_executor.py`):

```python
from areal.utils.perf_tracer import trace_session_event

# Mark execution start
trace_session_event(session_id, "mark_execution_start")

# Run workflow
traj = await workflow.arun_episode(engine, data)

# Mark execution end with status and context
if should_accept:
    trace_session_event(
        session_id,
        "mark_execution_end",
        status="accepted",
        rollout_stats=manager.get_stats()
    )
else:
    trace_session_event(
        session_id,
        "mark_execution_end",
        status="rejected",
        rejection_reason="stale_weight",
        rollout_stats=manager.get_stats()
    )

# Later, mark consumption
trace_session_event(session_id, "mark_consumed")
```

This enables per-session metrics like `queue_wait_s`, `execution_s`, `total_s` in
session traces.

## Session lifecycle tracking

Enable `perf_tracer.session_tracer.enabled=true` to track per-session metrics beyond
just performance spans. This is useful for diagnosing queueing issues and staleness.

### Choosing the right session tracing API

AReaL provides three complementary APIs for session-phase tracing. Choose based on your
code structure and automation needs:

| API                      | Use case                            | Automation level | Session ID source | Best for                                          |
| ------------------------ | ----------------------------------- | ---------------- | ----------------- | ------------------------------------------------- |
| `@trace_session(phase)`  | Trace async methods as named phases | High             | Auto from context | Extractable workflow methods (`_compute_rewards`) |
| `atrace_session_phase()` | Trace inline code blocks            | Medium           | Manual parameter  | Inline loops, non-extractable logic               |
| `trace_session_event()`  | Manual lifecycle event recording    | Low              | Manual parameter  | Orchestration layer (`WorkflowExecutor`)          |

**Guidelines**:

- Use `@trace_session` for workflow methods you can decorate—it's the cleanest and
  inherits `session_id` automatically
- Use `atrace_session_phase` when you can't extract a method (e.g., inline generation
  loops)
- Use `trace_session_event` only at the orchestration level (submission, execution,
  consumption) or when you need custom event names beyond standard phases

All three APIs write to the same `session_tracer/sessions-r{rank}.jsonl` output and
contribute to derived metrics like `generate_s`, `reward_calc_s`, `execution_s`.

### What gets tracked

Each session record includes:

- **Lifecycle timestamps**: `submit_ts`, `enqueue_ts`, `wait_return_ts`
- **Status**: `pending`, `accepted`, `rejected`, `failed`, `dropped`
- **Phases**: Multiple executions of `generate`, `reward`, `execution` with start/end
  times
- **Derived metrics**: `queue_wait_s`, `runner_wait_s`, `execution_s`, `generate_s`,
  `reward_calc_s`, `total_s`
- **Context**: `rejection_reason`, `rollout_stats` (snapshot of queue state)

### API usage

The `WorkflowExecutor` automatically instruments session lifecycle events. For custom
workflows, use:

```python
tracer = perf_tracer.get_session_tracer()

# Register a new session
session_id = tracer.register_submission()

# Mark lifecycle events
perf_tracer.trace_session_event(session_id, "mark_enqueued")
perf_tracer.trace_session_event(session_id, "mark_execution_start")
perf_tracer.trace_session_event(
    session_id,
    "mark_execution_end",
    status="accepted",
    rollout_stats=manager.get_stats()
)
perf_tracer.trace_session_event(session_id, "mark_consumed")
```

### Output format

Session traces are written to `session_tracer/sessions-r{rank}.jsonl`. Each line is a
JSON object:

```json
{
    "session_id": 0,
    "rank": 0,
    "status": "accepted",
    "submit_ts": 8738314.227833074,
    "enqueue_ts": 8738314.228516107,
    "wait_return_ts": 8738318.774206188,
    "rollout_stats": {
        "accepted": 8,
        "enqueued": 192,
        "rejected": 0,
        "running": 56
    },
    "queue_wait_s": 0.0006830338388681412,
    "runner_wait_s": 0.20169099420309067,
    "execution_s": 3.5818509366363287,
    "post_wait_s": 0.7621481493115425,
    "total_s": 4.54637311398983,
    "generate_s": 3.315287623554468,
    "reward_calc_s": 0.168320894241333,
    "phases": {
        "execution": [{"start_ts": 8738314.430207102, "end_ts": 8738318.012058038}],
        "generate": [{"start_ts": 8738314.44885158, "end_ts": 8738317.764139203}],
        "reward": [{"start_ts": 8738317.764251094, "end_ts": 8738317.9342864}]
    }
}
```

## Reference: Complete example

The `examples/tracer/` directory provides a fully instrumented GRPO training loop.

### `examples/tracer/gsm8k_grpo.yaml`

```yaml
experiment_name: gsm8k_grpo_trace_demo
trial_name: trial_0

cluster:
  fileroot: /shared/experiments

perf_tracer:
  enabled: true
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  save_interval: 1
  session_tracer:
    enabled: true
    flush_threshold: 100
```

Traces will land in `/shared/experiments/logs/<user>/gsm8k_grpo_trace_demo/trial_0/`.

### `examples/tracer/gsm8k_grpo.py`

Key instrumentation points:

```python
# Configure at startup
if config.perf_tracer is not None:
    perf_tracer.configure(config.perf_tracer, rank=rank)

# Main training loop
for epoch in range(num_epochs):
    for step, data in enumerate(dataloader):
        global_step = epoch * len(dataloader) + step

        # Trace rollout phase
        with perf_tracer.trace_scope(
            "train.rollout",
            category=Category.COMPUTE,
            args={"global_step": global_step, "epoch_step": step},
        ):
            batch = actor.prepare_batch(dataloader, n_samples)

        # Trace PPO update
        with perf_tracer.trace_scope(
            "train.ppo_update",
            category=Category.COMPUTE,
            args={"global_step": global_step},
        ):
            stats = actor.ppo_update(batch)

        # Trace evaluation
        if step % eval_interval == 0:
            with perf_tracer.trace_scope(
                "train.eval",
                category=Category.COMPUTE,
                args={"global_step": global_step},
            ):
                eval_metrics = evaluate(actor, eval_loader)

        # Flush traces (respects save_interval)
        perf_tracer.save(step=global_step)

# Final flush
perf_tracer.save(force=True)
```

### Workflow-level instrumentation

The `RLVRWorkflow` and `VisionRLVRWorkflow` classes demonstrate session-phase tracing:

**`areal/workflow/rlvr.py`**:

```python
class RLVRWorkflow(RolloutWorkflow):
    @trace_perf("rlvr_workflow.arun_episode", category="compute")
    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        # session_id is automatically set by WorkflowExecutor
        session_id = perf_tracer.get_session_id()

        # Trace generation phase
        async with perf_tracer.atrace_session_phase(session_id, "generate"):
            resps = await asyncio.gather(
                *[engine.agenerate(req) for _ in range(n_samples)]
            )

        # Trace reward phase (via decorator)
        rewards, completions = await self._compute_rewards(resps, prompt_str, data)

        # Build result tensors
        results = self._build_result_tensors(resps, rewards)
        return concat_padded_tensors(results)

    @trace_session("reward")
    async def _compute_rewards(self, resps, prompt_str, task_data):
        """Compute rewards with automatic phase tracing."""
        rewards = []
        for resp in resps:
            completion = self.tokenizer.decode(resp.output_tokens)
            reward = await self.async_reward_fn(
                prompt_str, completion,
                resp.input_tokens, resp.output_tokens,
                **task_data
            )
            rewards.append(reward)
        return rewards, completions
```

This setup creates:

- **Perf traces**: Duration spans for `arun_episode`, nested `generate` and `reward`
  phases
- **Session traces**: Per-session records with `generate_s`, `reward_calc_s` computed
  from phase timestamps

### Engine-level instrumentation

Training and inference engines use `@trace_perf` to profile core operations:

**`areal/engine/fsdp_engine.py`**:

```python
@trace_perf("fsdp_engine.train_batch", category="compute")
def train_batch(self, input_, loss_fn, loss_weight_fn):
    # Forward + backward + optimizer step
    ...

@trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
def _update_weights_from_distributed(self, meta):
    # Broadcast weights from training to inference
    ...

@trace_perf("fsdp_engine.update_weights_from_disk", category="io")
def _update_weights_from_disk(self, meta):
    # Load checkpoint from disk
    ...
```

**`areal/engine/ppo/actor.py`**:

```python
@trace_perf("ppo_actor.compute_advantages", category="compute")
def compute_advantages(self, data):
    # GAE computation
    ...

@trace_perf("ppo_actor.ppo_update", category="compute")
def ppo_update(self, data):
    # PPO loss + optimization
    ...
```

## Best practices

1. **Start with coarse-grained tracing**: Use `@trace_perf` on top-level methods first,
   then add finer-grained scopes as needed.

1. **Use meaningful names**: Prefix with module/class (`fsdp_engine.train_batch`, not
   just `train_batch`) so traces from different components are distinguishable.

1. **Don't over-instrument hot paths**: Tracing adds approximately **1-2µs overhead per
   event** on modern CPUs. While negligible for most operations, avoid instrumenting:

   - Inner loops executing >1000 times per second
   - Lock acquisitions in critical sections

1. **Correlate with session traces**: Enable `session_tracer` to connect high-level
   performance spans with per-session lifecycle data (queue waits, rejection reasons).

1. **Profile representative workloads**: Trace a few complete epochs, not just initial
   steps, to capture steady-state behavior and periodic spikes (checkpoint saves,
   evaluation).

## Troubleshooting

**Q: Traces are empty or missing events**

A: Ensure `perf_tracer.save(force=True)` runs before exit. Check that
`perf_tracer.configure()` was called with the correct rank.

**Q: Session traces show all `status: "pending"`**

A: Lifecycle events (`mark_execution_end`, `mark_consumed`) aren't being recorded.
Verify `WorkflowExecutor` is calling `trace_session_event()` or your custom workflow
implements the full lifecycle.

**Q: Perfetto can't open my trace**

A: JSONL format requires conversion. Use the provided converter tool or manually wrap in
a JSON array:

```bash
python -m areal.tools.perf_trace_converter traces.jsonl trace.json
```

## See also

- [CLI Reference: PerfTracerConfig](../cli_reference.md#section-perf-tracer)
- [Workflow customization guide](../customization/agent.md)
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
- [Perfetto UI](https://ui.perfetto.dev/)
