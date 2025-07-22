# Running GRPO on GSM8K Dataset

This guide walks through the code for running the GRPO algorithm on the GSM8K dataset, using the training script [examples/arealite/gsm8k_grpo.py](../../examples/arealite/gsm8k_grpo.py) and configuration file [examples/arealite/configs/gsm8k_grpo.yaml](../../examples/arealite/configs/gsm8k_grpo.yaml). 

## Launching the Experiment

As shown in [Quickstart Guide](../tutorial/quickstart_arealite.md), experiments in AReaLite are launched using standalone launchers with the following commands:

```
# Local Launcher
python -m arealite.launcher.local <training script> --config <configuration file> <cli args>
# Ray Launcher
python -m arealite.launcher.ray <training script> --config <configuration file> <cli args>
# Slurm Launcher
python -m arealite.launcher.slurm <training script> --config <configuration file> <cli args>
```

In AReaLite:
- The **training script** is an SPMD python script that serves as the experiment entry point.
- The launcher runs the training script with its distributed backend (`subprocess` for `LocalLauncher`, `ray.remote` for `RayLauncher`, `srun` for `SlurmLauncher`).
- The launcher also manages inference servers (currently only supporting `SGLangServer`).
- For distributed launchers (`RayLauncher` and `SlurmLauncher`), inference servers run with a wrapper [arealite/launcher/sglang_server.py](../../arealite/launcher/sglang_server.py) to handle addresses and ports in distributed settings.

The **configuration file** is a YAML file that sets the options provided in [arealite/api/cli_args.py](../../arealite/api/cli_args.py).
It could be modified via CLI arguments such as `actor.path=Qwen/Qwen3-1.7B` and `+sglang.attention_backend=triton`.
The training scripts parse the config with CLI arguments into the config class defined in [arealite/api/cli_args.py](../../arealite/api/cli_args.py). 
```
config, _ = load_expr_config(args, GRPOConfig)
config: GRPOConfig
```

## Loading and Preprocessing Dataset

We use the `datasets` and `torchdata` packages to load and preprocess the dataset into our dataloader. First, we download `openai/gsm8k` from Huggingface and split it by data parallel ranks, then map it to our desired format:
```python
def process_gsm8k_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["question"]}]
        return {"messages": messages}
    dataset = dataset.map(process).remove_columns(["question"])
    return dataset

def get_gsm8k_dataset(split, rank, world_size):
    dataset = load_dataset(path="openai/gsm8k", name="main", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return process_gsm8k_rl_dataset(dataset)
```

We then prepare training and evaluation dataloaders with `torchdata.StatefulDataLoader`:

```python
train_dataloader = torchdata.StatefulDataLoader(
    get_gsm8k_dataset("train", rank, world_size),
    batch_size=config.train_dataset.batch_size // world_size,
    shuffle=config.train_dataset.shuffle,
    num_workers=config.train_dataset.num_workers,
    collate_fn=lambda x: x,
    drop_last=config.train_dataset.drop_last,
)
valid_dataloader = ...
```

If you wish to use your own huggingface datasets or datasets on your local storage, please refers to [Customization: Dataset](../customization/dataset.md) for further details.

## Rollout

The data lifecycle is controlled by an `RLVRWorkflow`, which defines how data progresses from prompt to complete rollout data with fields required for training. Our example shows a single-turn RLVR workflow with a math reward function.

<!-- 
Next, we prepare for data rollout. The life-cycle of a piece of data is controlled by a `RLVRWorkflow`, which defines how data is processed from a prompt to a complete rollout data with fields required for training. Note that the workflow can involve multiple turns of generation, tool calling and reward calculation. In our example here, we only show a single-turn RLVR workflow with a math reward function.
-->

First, we define a math reward function for GSM8K.

```python
from ... import extract_answer, extract_solution

def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    sol = extract_answer(completions, data_name="math")
    ans = extract_solution(solution_str=answer, method="strict")
    if sol is None:
        return 0
    if ans is None:
        return 0
    return int(sol.strip() == ans.strip())
```

Then we initialize `RLVRWorkflow` for reward computation.

```python
tokenizer = load_hf_tokenizer(config.tokenizer_path)
workflow = RLVRWorkflow(
    reward_fn=gsm8k_reward_fn, 
    gconfig=config.gconfig, 
    tokenizer=tokenizer,
)
```

For generation, we assume the launchers have started `SGLangServer` instances and set the `AREAL_LLM_SERVER_ADDRS` environment variable with their addresses and ports.

We initialize `RemoteSGLangEngine`, whose APIs fall into two categories:
-  Sending requests to remote inference servers. Related APIs include `agenerate` and `update_weights`.
-  Executing the rollout workflow, managing streaming data, and collating completed rollout data into batched training samples. Related APIs include `prepare_batch` and `rollout_batch`.

The following code shows how `RemoteSGLangEngine` generates data batches for RL training:

<!--
Note that whether asynchronous RL training is enabled is solely controlled by the API `RemoteSGLangEngine` use to rollout data batches. In `prepare_batch`, data is processed in a streaming style and only batched in output, while in `rollout_batch`, the engine submits the data in a batch and waits for the results in a synchronous style.
-->

```python
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize()
eval_rollout = ...

data_generator = iter(train_dataloader)
for global_step in range(max_steps):
    # rollout batched training data for current step
    if config.async_training:
        batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
    else:
        try:
            data = next(data_generator)
        except StopIteration:
            data_generator = iter(train_dataloader)
            data = next(data_generator)
        batch = rollout.rollout_batch(data, workflow=workflow)
```

If you want to use rollout workflows with custom reward functions or agentic tool calling, see [Customization: Rollout Workflows](../customization/agent.md).

## Training

After obtaining the training batch, we use `FSDPPPOActor` to calculate losses and update weights. Each train engine corresponds to one model, therefore we need an additional engine for reference model. 

```python
actor = FSDPPPOActor(config=config.actor)
actor.initialize(None, ft_spec)
ref = None
if config.actor.kl_ctl > 0 and config.ref is not None:
    ref = FSDPPPOActor(config=config.ref)
    ref.initialize(None, ft_spec)
```

The following code shows a GRPO training step:

```python
logp = actor.compute_logp(batch)
batch["prox_logp"] = logp
if ref is not None:
    batch["ref_logp"] = ref.compute_logp(batch)
    log_gpu_stats("ref logp")
actor.compute_advantages(batch)
stats = actor.ppo_update(batch)
actor.step_lr_scheduler()
```

`FSDPPPOActor` is a high-level engine with algorithm-specific APIs, such as `compute_logp`,`compute_advantages` and `ppo_update`.
`FSDPPPOActor` is powered by the lower-level train engine `FSDPEngine`, who only provides basic APIs for the model, such as `train_batch` and `forward`. 

## Transferring Weights to Inference Servers

After training, we transfer updated parameters to remote inference servers through cooperation between `FSDPPPOActor` and `RemoteSGLangEngine`.
In our example, we show a simple case in which parameters are transfered from disks:

```python
path = update_weight_path(global_step)
meta = WeightUpdateMeta(
    type="disk",
    path=path,
    model_version=global_step + 1
)
# send requests to remote servers, tell them to update weights
if dist.get_rank() == 0:
    future = rollout.update_weights(meta)
# actor save weights
actor.upload_weights(meta)
# remote servers returns after finishing updates
if dist.get_rank() == 0:
    future.result()
    shutil.rmtree(path, ignore_errors=True)
# synchronize rollout processes for model version update
dist.barrier()
torch.cuda.synchronize()
# update version for rollout engine
rollout.set_version(global_step + 1)
```

The core GRPO training logic in AReaLite can be summarized as:

```python
data_generator = iter(train_dataloader)
for global_step in range(max_steps):
    if config.async_training:
        batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
    else:
        try:
            data = next(data_generator)
        except StopIteration:
            data_generator = iter(train_dataloader)
            data = next(data_generator)
        batch = rollout.rollout_batch(data, workflow=workflow)

    batch = batch.to(actor.device)
    # Create barrier to synchronize all rollout processes.
    dist.barrier()
    torch.cuda.synchronize()
    
    logp = actor.compute_logp(batch)
    batch["prox_logp"] = logp
    if ref is not None:
        batch["ref_logp"] = ref.compute_logp(batch)
        log_gpu_stats("ref logp")
    actor.compute_advantages(batch)
    stats = actor.ppo_update(batch)
    actor.step_lr_scheduler()
    
    path = update_weight_path(global_step)
    meta = WeightUpdateMeta(
        type="disk",
        path=path,
        model_version=global_step + 1
    )
    # send requests to remote servers, tell them to update weights
    if dist.get_rank() == 0:
        future = rollout.update_weights(meta)
    # actor save weights
    actor.upload_weights(meta)
    # remote servers returns after finishing updates
    if dist.get_rank() == 0:
        future.result()
        shutil.rmtree(path, ignore_errors=True)
    # synchronize rollout processes for model version update
    dist.barrier()
    torch.cuda.synchronize()
    # update version for rollout engine
    rollout.set_version(global_step + 1)
```

## Utilities 

In AReaLite, we provide a wide range of utilities for basic functionalities required for observing and tuning your experiments, including:
- `Saver` ([arealite/utils/saver.py](../../arealite/utils/saver.py)): Saves the checkpoints in a frequency set by config.
- `Evaluator` ([arealite/utils/evaluator.py](../../arealite/utils/evaluator.py)): Evaluates the model in a frequency set by config.
- `StatsLogger` ([arealite/utils/stats_logger.py](../../arealite/utils/stats_logger.py)): Logs training data to backends like `wandb` and `tensorboard`. Also manages outputs to terminal or log files.
- `stats_tracker` ([realhf/base/stats_tracker.py](../../realhf/base/stats_tracker.py)): Gathers and manages training statistics.

