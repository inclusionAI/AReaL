# Debugging
This guide explains how to debug the two key components: the agent's generation logic and the reinforcement learning (RL) algorithm.

## Debugging Agent Generation

To debug the agent's generation logic in isolation, you can launch the sglang inference server as a standalone, persistent process. This allows you to repeatedly test the agent's behavior.

### 1. Launch the Standalone SGLang Server
First, start the sglang server with `allocation_mode` only for inference (e.g. `sglang.d4p1t1`).
```bash
python -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml allocation_mode=sglang.d4p1t1 
```
Once launched, you can find the server addresses in the log output, for example:
```
LLM inference server launched at: AREAL_LLM_SERVER_ADDRS=10.11.16.244:20082,10.11.16.244:24145,10.11.16.244:30422,10.11.16.244:40325
```

### 2. Run Debug Program
Next, run the agent debug script and pass the server addresses to it via the `AREAL_LLM_SERVER_ADDRS` environment variable.

```bash
AREAL_LLM_SERVER_ADDRS=10.11.16.244:20082,10.11.16.244:24145,10.11.16.244:30422,10.11.16.244:40325 python agent_debug.py --config scripts-su18/areal-lite/config/gsm8k_grpo.yaml trial_name=debug rollout.enable_rollout_tracing=True
```

The example debug script (`agent_debug.py`) below shows how to generate a batch of data and save it to a directory.

```python
# Create dataset and dataloaders
train_dataset = get_custom_dataset(...)
# Select a small subset of the dataset for debugging
train_dataset = train_dataset.select(range(config.train_dataset.batch_size)) 
train_dataloader = StatefulDataLoader(...)

# Initialize inference engine
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(None, None)

# Create rollout workflow
workflow = RLVRWorkflow(...)

dump_dir = os.path.join(
    StatsLogger.get_log_path(config.stats_logger), "generated"
)

data_generator = itertools.cycle(train_dataloader)
generated_data = rollout.rollout_batch(next(data_generator), workflow=workflow)

# Save generated data
torch.save(generated_data, os.path.join(dump_dir, "batch_data.pt"))

rollout.destroy()
```

::::{note}
This setup allows you to repeatedly run and debug your agent's code without relaunching the inference server.
::::

## Debugging RL Algorithm

Once you have a saved batch of generated data, you can debug your reinforcement learning (RL) algorithm independently. 

1. Modify `allocation_mode` to exclude `sglang`:
```yaml
allocation_mode: d4p1t1
```

2. Debug the RL algorithm using previously generated data:

```python
# Create dataset and dataloaders
train_dataset = get_custom_dataset(...)
train_dataloader = StatefulDataLoader(train_dataset, ...)

# Initialize train engine
actor = FSDPPPOActor(config=config.actor)
actor.initialize(None, ft_spec)

if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
    config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
    config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

# Run training.

# Load previously generated data
dump_dir = os.path.join(
    StatsLogger.get_log_path(config.stats_logger), "generated"
)
batch = torch.load(os.path.join(dump_dir, "batch_data.pt"), weights_only=False)
batch = batch.to(actor.device)

dist.barrier(device_ids=[actor.device.index])
torch.cuda.synchronize()

# Execute a single step of the RL Algorithm
if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
    logp = actor.compute_logp(batch)
    batch["prox_logp"] = logp
actor.compute_advantages(batch)
...
actor.ppo_update(batch)
actor.step_lr_scheduler()

dist.barrier(device_ids=[actor.device.index])
torch.cuda.synchronize()

actor.destroy()
```