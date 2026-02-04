# Evaluation

AReaL provides distributed inference evaluation using the same RolloutController
infrastructure as training. This allows you to leverage existing workflows and
schedulers to scale evaluation across multiple GPUs and nodes.

## Quick Start

Run evaluation on GSM8K:

```bash
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    actor.path=/path/to/checkpoint
```

For distributed evaluation:

```bash
# With Ray (3 nodes, 12 GPUs)
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=ray \
    allocation_mode=sglang:d12p1t1 \
    cluster.n_nodes=3

# With Slurm (12 nodes, 96 GPUs)
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=slurm \
    allocation_mode=sglang:d96p1t1 \
    cluster.n_nodes=12
```

## Architecture

Evaluation uses a single-controller architecture without training workers:

```
Controller Process
    │
    └─> RolloutController
        ├─> Scheduler creates inference workers (SGLang/vLLM)
        ├─> BatchTaskDispatcher submits eval tasks
        └─> Collects results and computes metrics
```

The controller orchestrates evaluation from a CPU process while inference workers run on
GPUs.

## Implementation

See [`examples/math/gsm8k_eval.py`](../../examples/math/gsm8k_eval.py) for a complete
example. The key pattern:

```python
# Parse allocation mode and initialize scheduler
allocation_mode = AllocationMode.from_str(config.allocation_mode)
scheduler = LocalScheduler(exp_config=config)  # or Ray/Slurm

# Create RolloutController
if allocation_mode.gen_backend == "sglang":
    engine_cls = RemoteSGLangEngine
    server_args = SGLangConfig.build_args(...)
elif allocation_mode.gen_backend == "vllm":
    engine_cls = RemotevLLMEngine
    server_args = vLLMConfig.build_args(...)

eval_rollout = engine_cls.as_controller(config.rollout, scheduler)
eval_rollout.initialize(
    role="eval-rollout",
    alloc_mode=allocation_mode,
    server_args=server_args,
)

# Submit evaluation tasks
for data in dataloader:
    for item in data:
        eval_rollout.submit(item, workflow, group_size=config.gconfig.n_samples)
        cnt += 1

# Wait and collect results
eval_rollout.wait(cnt, timeout=None)
eval_stats = eval_rollout.export_stats()
```

This follows the same controller pattern as training (see
[`areal/experimental/trainer/rl.py:540-594`](../../areal/experimental/trainer/rl.py))
but without training components.

## Configuration

Evaluation configs use only inference components.

It is also valid to use the previous training config but with the evaluation script.

```yaml
experiment_name: gsm8k-eval
trial_name: eval0
seed: 1

allocation_mode: sglang:d4p1t1  # Inference-only

scheduler:
  type: local  # or 'ray', 'slurm'

rollout:
  max_concurrent_rollouts: 256
  max_head_offpolicyness: 1e12  # No staleness control

gconfig:
  n_samples: 8
  temperature: 1.0
  max_new_tokens: 1024

actor:
  path: Qwen/Qwen2.5-1.5B-Instruct
  dtype: bfloat16
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cmd: python3 -m areal.scheduler.rpc.rpc_server

valid_dataset:
  name: gsm8k
  split: test
  batch_size: 32
```

## Logging Metrics

Export stats and log to both console and wandb:

```python
eval_stats = eval_rollout.export_stats()

if rank == 0:
    print(tabulate_stats(eval_stats))

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"eval": eval_stats})
    except ImportError:
        pass
```

## Custom Workflows

Reuse training workflows or create custom ones. See
[agentic RL tutorial](../tutorial/agentic_rl.md) and
[Customization: Rollout Workflows](../customization/agent.md) for complete guides.

## Next Steps

- [Distributed Experiments](quickstart.md#distributed-experiments-with-ray-or-slurm)
- [Customization: Workflows](../customization/agent.md)
- [Agentic RL Tutorial](../tutorial/agentic_rl.md)
- [Large MoE Training](../tutorial/megatron.md)
