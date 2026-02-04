# Debugging Guide

This guide covers debugging AReaL training applications, including:

- Debugging your agent (a `RolloutWorkflow` implementation) with a persistent inference
  server
- Debugging custom RL algorithms
- Comparing rollout results between Transformers and inference engines

## Debugging `RolloutWorkflow` with a Persistent Inference Server

You can launch a **standalone, persistent inference server** for your agent's generation
logic, enabling repeated testing without server restarts.

**Benefits:**

- **Lightweight** — Your debug program only requires CPU while inference runs on GPU
- **IDE-friendly** — Works seamlessly with VS Code's Python debugger and other IDEs
- **Fast iterations** — No server restarts needed between debugging sessions

### 1. Launch the Standalone SGLang Server

Start your SGLang server with an inference-only `allocation_mode` such as
`sglang:d4p1t1` (omit the content after "+" in a real allocation mode):

```bash
nohup python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    allocation_mode=sglang:d4p1t1 > llm_server.log 2>&1 &
```

**Note:** For debugging purposes, only the `allocation_mode` and `sglang` configurations
are relevant—you can ignore other settings in the example YAML file. Review the
inference engine launch arguments based on your model type. For example, verify whether
`sglang.enable_multimodal` should be enabled, as multimodal support is disabled by
default in SGLang for models like Gemma3, Llama4, and Step3VL.

Once it's running, you'll find the server address in the log:

```
LLM inference server launched at: AREAL_LLM_SERVER_ADDRS=127.0.0.1:20082
```

### 2. Run Your Debug Program

Create a debug script (e.g., `agent_debug.py`) with your custom workflow implementation:

```python
# Create dataset and dataloaders
train_dataset = get_custom_dataset(...)
# Select a small subset of the dataset for debugging
train_dataset = train_dataset.select(range(config.train_dataset.batch_size))
train_dataloader = StatefulDataLoader(...)

# Initialize inference engine
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(addr="127.0.0.1:20082")  # the printed address above

# Create rollout workflow
workflow = MyWorkflow(...)

dump_dir = os.path.join(
    StatsLogger.get_log_path(config.stats_logger), "generated"
)

generated_data = rollout.prepare_batch(train_dataloader, workflow=workflow)

# Save generated data for later use
torch.save(generated_data, os.path.join(dump_dir, "batch_data.pt"))

rollout.destroy()
```

Now run your debug script:

```bash
python agent_debug.py --config agent_debug.yaml \
    rollout.enable_rollout_tracing=True
```

## Debugging Custom RL Algorithms

> **Note:** If you're using existing AReaL algorithms like GRPO, you can skip this
> section.

When debugging custom RL algorithms, you can treat them like offline training (e.g.,
SFT) by using pre-generated data instead of running live inference.

**Benefits:**

- **No inference servers** — Eliminate server management overhead
- **Faster iterations** — Skip the expensive data collection step
- **Reproducible** — Use identical data across debugging sessions
- **Isolated testing** — Focus exclusively on your RL logic

### 1. Configure Allocation Mode

Disable SGLang inference in your configuration (keep only the allocation after "+"):

```yaml
allocation_mode: fsdp:d4p1t1
```

### 2. Create Your RL Debug Script

Create your debug script that loads the pre-generated data:

```python
# Create dataset and dataloaders
train_dataset = get_custom_dataset(...)
train_dataloader = StatefulDataLoader(train_dataset, ...)

# Configure tokenizer stop tokens
if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
    config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
    config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

# Load previously generated data
dump_dir = os.path.join(
    StatsLogger.get_log_path(config.stats_logger), "generated"
)
batch = torch.load(os.path.join(dump_dir, "batch_data.pt"), weights_only=False)

# Prepare batch for training
batch = batch.to('cuda')
torch.cuda.synchronize()
dist.barrier(group=actor.cpu_group)

# Your custom algorithm logic here
...
```

## Rollout Consistency

Comparing rollout results between `transformers` and your inference engine helps verify
consistency and correctness. While most models produce nearly identical results, some
may exhibit significant differences due to the extensive optimizations that inference
backends (e.g., `sglang`, `vllm`) apply to accelerate the forward pass.

If you suspect discrepancies, or if you're working with models lacking first-class
support in Transformers or SGLang, compare outputs against a dataset using a simple
validation script. See `examples/docs/debug/cmp_rollout.py` for a complete example
comparing rollout results for `google/gemma3-4b-it` on the `BUAADreamer/clevr_count_70k`
dataset.
