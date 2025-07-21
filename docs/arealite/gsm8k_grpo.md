# Code Walkthrough: Running GRPO on GSM8K Dataset

In this guide, we will walk you through the detailed code of an example that runs GRPO algorithm on GSM8K dataset, with training script [examples/arealite/gsm8k_grpo.py](../../examples/arealite/gsm8k_grpo.py) and configuration file [examples/arealite/configs/gsm8k_grpo.yaml](../../examples/arealite/configs/gsm8k_grpo.yaml). 

## Launching the Experiment

As shown in [Quickstart Guide](../tutorial/quickstart_arealite.md), an experiment of AReaLite is launched by standalone launchers with command:

```
# Local Launcher
python -m arealite.launcher.local <training script> --config <configuration file> <cli args>
# Ray Launcher
python -m arealite.launcher.ray <training script> --config <configuration file> <cli args>
# Slurm Launcher
python -m arealite.launcher.slurm <training script> --config <configuration file> <cli args>
```

In AReaLite, the **training script** is **an SPMD python script** that serves as an entry point to launch the experiment.
The launcher directly runs the training script with their distributed backend (`subprocess` for `LocalLauncher`, `ray.remote` for `RayLauncher`, `srun` for `SlurmLauncher`).
Except for the training script, the launcher also is responsible for running inference servers (currently only support `SGLangServer`). 
For distributed launchers (`RayLauncher` and `SlurmLauncher`), they run inference servers with a wrapper [arealite/launcher/sglang_server.py](../../arealite/launcher/sglang_server.py) for managing addresses and ports in the distributed settings.

The **configuration file**, which is a YAML file that sets the options provided in [arealite/api/cli_args.py](../../arealite/api/cli_args.py).
It could be changed by CLI arguments such as `actor.path=Qwen/Qwen3-1.7B` and `+sglang.attention_backend=triton`.
The training scripts uses [load_expr_config(args, config_cls)](../../arealite/api/cli_args.py#L886) to parse the config with CLI arguments to the config class defined in [arealite/api/cli_args.py](../../arealite/api/cli_args.py). 

In the example:
```
config, _ = load_expr_config(args, GRPOConfig)
config: GRPOConfig
```

## Loading and Pre-processing Dataset

In our example, we directly use tools from package `datasets` and `torchdata` to load and pre-process the dataset into our dataloader.  
we first download `openai/gsm8k` from huggingface and split them by data parallel ranks, and then map them to the format we want.

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

Then we prepare training and evaluation data loaders with `torchdata.StatefulDataLoader`, which will serve as an input for data rollout.

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

Next, we prepare for data rollout. The life-cycle of a piece of data is controlled by a `RLVRWorkflow`, which defines how data is processed from a prompt to a complete rollout data with fields required for training. Note that the workflow can involve multiple turns of generation, tool calling and reward calculation. In our example here, we only show a single-turn RLVR workflow with a math reward function.

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

As for generation, we assume that the launchers have already launched instances of `SGLangServer`, and passed in the environment variable `AREAL_LLM_SERVER_ADDRS` to tell us the addresses and ports of these inference servers to connect to. 

In the next step, we initialize `RemoteSGLangEngine` in the training script. Its APIs could be catagorized into two types:
-  Sending requests such as generation and update weights to remote inference servers and returns the replies. Related APIs include `agenerate` and `update_weights`.
-  Execute the rollout workflow. Manage the streaming data going through the rollout workflow to control parameter version differences between generation and training (data offpolicyness), and collate completed rollout data into a batched training sample. Related APIs include `prepare_batch` and `rollout_batch`.

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
If you want to customize your own rollout workflow with customized reward functions or agentic tool calling, please refer to [Customization: Rollout Workflows](agent.md).

## Training


## Utilities 





