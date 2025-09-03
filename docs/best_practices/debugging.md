# Debugging Guide

This guide outlines best practices for debugging training applications based on AReaL.
We focus on two key components: the agent's generation logic (`RolloutWorkflow`) and
custom reinforcement learning (RL) algorithms.

## Debugging `RolloutWorkflow` with a Persistent Inference Server

To debug the agent's generation logic in isolation, you can launch an inference server
as a **standalone, persistent process**. This approach allows you to repeatedly test the
agent's behavior without the overhead of relaunching the server for each debug session.

### 1. Launch the Standalone SGLang Server

Start the SGLang server with `allocation_mode` configured for inference only (e.g.,
`sglang.d4p1t1`):

```bash
nohup python -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml allocation_mode=sglang.d4p1t1 > llm_server.log 2>&1 &
```

**Note:** Only the `allocation_mode` and inference server configurations (e.g.,
`sglang`) are relevant for debugging. The remaining configurations in
`examples/lite/configs/gsm8k_grpo.yaml` can be ignored, so you can use our provided
example scripts as-is.

Once launched, you can find the server addresses in the log output:

```
LLM inference server launched at: AREAL_LLM_SERVER_ADDRS=10.11.16.244:20082,10.11.16.244:24145,10.11.16.244:30422,10.11.16.244:40325
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
rollout.initialize(None, None)

# Create rollout workflow
workflow = MyWorkflow(...)

dump_dir = os.path.join(
    StatsLogger.get_log_path(config.stats_logger), "generated"
)

data_generator = itertools.cycle(train_dataloader)
generated_data = rollout.rollout_batch(next(data_generator), workflow=workflow)

# Save generated data for later use
torch.save(generated_data, os.path.join(dump_dir, "batch_data.pt"))

rollout.destroy()
```

Execute your debug script by passing the server addresses via the
`AREAL_LLM_SERVER_ADDRS` environment variable:

```bash
AREAL_LLM_SERVER_ADDRS=10.11.16.244:20082,10.11.16.244:24145,10.11.16.244:30422,10.11.16.244:40325 python agent_debug.py --config agent_debug.yaml rollout.enable_rollout_tracing=True
```

**Key Benefits:**

- While inference servers run on GPUs, this debugging program only requires CPU and a
  single process
- You can use the built-in Python debugger in VS Code
- If your debugging program encounters errors and exits, the inference servers remain
  running
- You can repeatedly execute the debug command until your workflow functions correctly
- No need to relaunch the inference server between debugging iterations

## Debugging Custom RL Algorithms

::::{note} If you're using an algorithm already implemented in AReaL (e.g., GRPO), you
can skip this section. ::::

### Steps for RL Algorithm Debugging

1. **Configure allocation mode** to exclude SGLang inference:

```yaml
allocation_mode: d4p1t1
```

2. **Create your RL debugging script** using the previously generated data:

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

### Benefits of This Approach

Debugging RL algorithms this way is similar to debugging offline training algorithms
(e.g., SFT):

- **No inference server required** - eliminates the overhead of launching and managing
  inference servers
- **Faster iteration cycles** - skip the expensive data collection phase during
  debugging
- **Reproducible debugging** - use the same generated data across multiple debugging
  sessions
- **Isolated testing** - debug your RL logic independently of the rollout workflow
