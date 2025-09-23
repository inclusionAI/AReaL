# Debugging Guide

Here's how to debug AReaL training applications, including:

- Debugging `RolloutWorkflow` with a persistent inference server;
- Debugging custom RL algorithms;
- Comparing the rollout results between Transformers and inference engine.

## Debugging `RolloutWorkflow` with a Persistent Inference Server

The trick is to launch a **standalone, persistent inference server** for your agent's
generation logic. This way, you can test repeatedly without restarting the server each
time.

**Why this works well:**

- **Lightweight** - Your debug program only needs CPU while inference runs on GPU
- **IDE friendly** - Works perfectly with VS Code's Python debugger
- **Fast iterations** - No need to restart servers between debugging sessions

### 1. Launch the Standalone SGLang Server

First, start your SGLang server with an inference-only `allocation_mode` like
`sglang.d4p1t1`:

```bash
nohup python -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    allocation_mode=sglang.d4p1t1 > llm_server.log 2>&1 &
```

**Note:** For debugging purposes, only the `allocation_mode` and `sglang` configs
matter. You can ignore everything else in the example YAML file. In addition, it is
strongly recommended to examine the launch arguments related to the inference engine.
For example, you may need to check if `sglang.enable_multimodal` should be set based on
your model type since multimodal is disabled in SGLang by default in models such as
Gemma3, Llama4, and Step3VL.

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

# Initialize inference engine - reads server addresses from environment variable
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(...)

# Create rollout workflow
workflow = MyWorkflow(...)

dump_dir = os.path.join(
    StatsLogger.get_log_path(config.stats_logger), "generated"
)

data_generator = cycle_dataloader(train_dataloader)
generated_data = rollout.rollout_batch(next(data_generator), workflow=workflow)

# Save generated data for later use
torch.save(generated_data, os.path.join(dump_dir, "batch_data.pt"))

rollout.destroy()
```

Now run your debug script, passing the server address through the environment:

```bash
AREAL_LLM_SERVER_ADDRS=127.0.0.1:20082 \
    python agent_debug.py --config agent_debug.yaml \
    rollout.enable_rollout_tracing=True
```

## Debugging Custom RL Algorithms

> If you're using existing AReaL algorithms like GRPO, you can skip this section.

For custom RL algorithms, you can debug them just like offline training (think SFT) by
using pre-generated data instead of running inference.

**This approach is great because:**

- **No inference servers** - You don't need to manage any servers
- **Faster iterations** - Skip the expensive data collection step
- **Reproducible** - Use the same data across debugging sessions
- **Isolated testing** - Focus purely on your RL logic

### 1. Configure Allocation Mode

First, turn off SGLang inference in your config:

```yaml
allocation_mode: d4p1t1
```

### 2. Create Your RL Debug Script

Then create your debug script that loads the pre-generated data:

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
dist.barrier(device_ids=[actor.device.index])
torch.cuda.synchronize()

# Your custom algorithm logic here
...
```

## Comparing the rollout results between Transformers and inference engine

It is often useful to compare the rollout results between Transformers and the inference
engine to ensure consistency and correctness. Most models will yield nearly identical
results, but some models may have significant differences because the inference engine
does a lot of efforts in accelerating the forward process.

If you suspect any discrepancies, or if your workflow involves models that do not have
first-class support in Transformers/SGLang, it is recommended to use a simple script to
compare the outputs against a dataset. Please refer to
`examples/docs/debug/cmp_rollout.py` for a complete example, which compares the rollout
results of `google/gemma3-4b-it` on `BUAADreamer/clevr_count_70k` dataset.
