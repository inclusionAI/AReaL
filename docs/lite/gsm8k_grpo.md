# Running GRPO on GSM8K Dataset

This guide introduces how AReaL-lite runs the GRPO algorithm on the GSM8K dataset, using
the training script
[examples/math/gsm8k_grpo.py](https://github.com/inclusionAI/AReaL/blob/main/examples/math/gsm8k_grpo.py)
and configuration file
[examples/math/gsm8k_grpo.yaml](https://github.com/inclusionAI/AReaL/blob/main/examples/math/gsm8k_grpo.yaml).

## How AReaL-lite Works

The following figure illustrates the launching and one asynchronous training step of the
GRPO algorithm on the GSM8K dataset on AReaL-lite. Compared with the old AReaL
implementation, AReaL-lite runs inference servers and a SPMD training script instead of
a bunch of various workers. In a training step, AReaL-lite:

1. Submits prompts from the dataset to `RemoteSGLangEngine`, who runs `RLVRWorkflow` in
   a streaming manner.
1. Completes `RLVRWorkflow` by interacting with remote `SGLangServer` instances to
   generate sequences, and computing rewards with the reward function.
1. Once there are enough outputs from `RLVRWorkflow`, aggregates them into a data batch
   for algorithm-specific training engine `FSDPPPOActor`.
1. Computes losses and update weights in `FSDPPPOActor`.
1. Transfers the updated weights to remote `SGLangServer` instances.

![AReaL-lite-gsm8k-example](gsm8k_grpo.png)

In the following sections, we will walk you through the code to explain concepts and
show you how these steps are done in details.

## Launching the Experiment

As shown in the [quickstart guide](../tutorial/quickstart.md), experiments in AReaL-lite
are launched using standalone launchers with the following commands:

```
# Local Launcher
python -m areal.launcher.local <training script> --config <configuration file> <cli args>
# Ray Launcher
python -m areal.launcher.ray <training script> --config <configuration file> <cli args>
# Slurm Launcher
python -m areal.launcher.slurm <training script> --config <configuration file> <cli args>
```

In AReaL-lite:

- The **training script** is an SPMD python script that serves as the experiment entry
  point.
- The launcher runs the training script with its distributed backend (`subprocess` for
  `LocalLauncher`, `ray.remote` for `RayLauncher`, `srun` for `SlurmLauncher`).
- The launcher also manages inference servers (currently only supporting
  `SGLangServer`). The number and parallelization strategies (e.g. tensor parallel) are
  determined by the option
  [allocation_mode](https://github.com/inclusionAI/AReaL/blob/main/areal/api/cli_args.py#L797).
- For distributed launchers (`RayLauncher` and `SlurmLauncher`), inference servers run
  with a wrapper
  [areal/launcher/sglang_server.py](https://github.com/inclusionAI/AReaL/blob/main/areal/launcher/sglang_server.py)
  to handle addresses and ports in distributed settings.
- After `SGLangServer` instances are started, launchers collect their addresses and
  ports to set the `AREAL_LLM_SERVER_ADDRS` environment variable for training scripts to
  access these inference servers.

The **configuration file** is a YAML file that sets the options provided in
[areal/api/cli_args.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/cli_args.py).
It could be modified via CLI arguments such as `actor.path=Qwen/Qwen3-1.7B` and
`+sglang.attention_backend=triton`. The training scripts parse the config with CLI
arguments into the config class defined in
[areal/api/cli_args.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/cli_args.py).

```
config, _ = load_expr_config(args, GRPOConfig)
config: GRPOConfig
```

## Loading and Preprocessing Dataset

We use the `datasets` and `torchdata` packages to load and preprocess the dataset into
our dataloader. First, we download `openai/gsm8k` from Hugging Face and split it by data
parallel ranks, then map it to our desired format:

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

If you wish to use your own huggingface datasets or datasets on your local storage,
please refers to [Customization: Dataset](../customization/dataset.md) for further
details.

## Rollout

### Inference Engine: `RemoteSGLangEngine`

In AReaL-lite, generation tasks are offloaded to remote inference servers, which operate
on separate GPUs from those used for training. The `RemoteSGLangEngine` acts as a client
that interacts with the servers. `RemoteSGLangEngine` runs in a SPMD manner on every
training process, without occupying any GPUs.

`RemoteSGLangEngine` provides two core APIs that access the remote servers, `agenerate`
and `update_weights_async`. It is worth mentioning that, in asynchronous RL experiment
in AReaL-lite, inference-side weight update could happen **in the middle of** generation
of one prompt. With that being said, one output sequence could be generated by multiple
versions of models. Let us glimpse into code of `agenerate` and `update_weights_async`
for a better understanding.

In `update_weights_async`, the engine first send `pause_generation` requests to all
inference servers, notifying them a weight update is about to happen. Upon receiveing
`pause_generation`, inference servers will immediately stop generating and respond with
already generated tokens. Then, the engine sends `update_weights_from_distributed` (for
NCCL update) or `update_weights_from_disk` (for disk update). After the update is
finished, the engine sends `continue_generation` to inference server telling them to
start working again.

```python
class RemoteSGLangEngine:
    ...
    def update_weights_async(self, meta: WeightUpdateMeta):
        # `update_weights_async` is completely async.
        # It submits task to a ProcessPoolExecutor and returns a future
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/pause_generation")
        if meta.type == "nccl":
            future = self.executor.submit(
                # a function that send `update_weights_from_distributed` request
                update_weights_from_distributed,
            )
        elif meta.type == "disk":
            ...

        def callback(future):
            for addr in self.addresses
                requests.post(f"http://{addr}/continue_generation")

        future.add_done_callback(callback)
        return future
```

`agenerate` takes an `ModelRequest` with `input_ids` of **a single prompt** and
generation hyperparameters, and returns the final generation result, an `ModelResponse`
with `output_tokens` and other outputs. Since the generation could be interrupted,
`agenerate` iteratively prepares payload, sends requests and receives responses until
the generation finishes.

```python
class RemoteSGLangEngine:
    ...
    async def agenerate(self, req: ModelRequest):
        payload = ... # prepare payload for request
        # If request is from the same workflow, choose old server
        # to allow KVCache reuse. Otherwise choose server in a round
        # robin manner.
        server_addr = self.choose_server(req)
        stop_reason = None
        # other outputs are omitted for simplicity
        output_tokens = []
        while (stop_reason != "stop" and len(output_tokens) < max_new_tokens):
            # Request is interrupted, wait to avoid contention
            if stop_reason is not None:
                await asyncio.sleep(0.5)
            # send request to remote sever
            result = await arequest_with_retry(
                addr=server_addr,
                endpoint="/generate",
                payload=payload,
                method="POST"
            )
            output_tokens.extend(result["output_ids"])
            # prepare payload for the next request
            payload["input_ids"] += results["output_ids"]
            payload["sample_params"]["max_new_tokens"] -= len(results["output_ids"])
        return ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=output_tokens,
            ...
        )

```

The `InferenceEngine` class is designed to be extensible, supporting not just SGLang but
also other backends like vLLM. While different inference engines may be used, the
rollout management logic remains consistent. This common functionality is abstracted
into the `WorkflowExecutor`, which will be introduced in the following section.

### `RLVRWorkflow` and `WorkflowExecutor`

The rollout data lifecycle is controlled by an `RLVRWorkflow`, which defines how data
progresses from prompts to complete rollout data containing all fields required for
training. Our example shows a single-turn RLVR workflow with a math reward function. The
core logic of the workflow is implemented in an async method `arun_episode`, which takes
a prompt, generate answers with `RemoteSGLangEngine`, computes rewards, and populates
additional fields to produce finalized training data.

```python
class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self, reward_fn, gconfig, tokenizer, ...
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer

    async def arun_episode(self, engine, data):
        # rollout data with inference engine
        input_ids = self.tokenizer.apply_chat_template(data["message"], ...)
        req = ModelRequest(rid=..., input_ids=input_ids, gconfig=self.gconfig.new(n_samples=1))
        resps = await asyncio.gather(
            *[engine.agenerate(req) for _ in range(self.gconfig.n_samples)]
        )
        # post process rollout responses
        results = []
        for resp in resps:
            reward = self.reward_fn(...)
            ... # other required fields for training
            res = dict(
                input_ids=...,
                rewards=...,
                ... # other required fields for training
            )
            results.append(res)
        # return padded `self.gconfig.n_samples` samples with prompt `data["message"]`
        return concat_padded_tensors(results)

def gsm8k_reward_fn(completions, answer):
    ...

tokenizer = load_hf_tokenizer(config.tokenizer_path)
workflow = RLVRWorkflow(
    reward_fn=gsm8k_reward_fn,
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    ...
)
```

`WorkflowExecutor` is responsible for managing the data streaming through rollout
workflows, and collates completed rollout data into batched training samples. When
initializing, it launches a rollout thread that runs rollout workflows as `asyncio`
tasks. The following code shows the simplified version of rollout thread implementation,
which iteratively:

- Checks available capacity. The capacity controls current number of rollout workflows
  to limit concurrency and **data off-policyness** (The difference between the model
  version used by generation and the model version updated by the trainer).
- If there is capacity left and rollout is not paused for weight update, continuously
  obtains data from `input_queue` and creates `asyncio` tasks to run the workflows.
- Waits for rollout workflows to finish.
- Gathers data from finished workflows and puts them into `output_queue`

```python
class WorkflowExecutor:
    ...
    async def _rollout_thread_async(self):
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create rollout tasks with data obtained from input_queue
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    data, workflow = self.input_queue.get_nowait()
                    task = asyncio.create_task(
                        workflow.arun_episode(self, data), name=str(rid)
                    )
                    rollout_tasks[str(rid)] = task
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    capacity -= 1
                    rid += 1
                # Wait for rollout completion
                tasks = list(rollout_tasks.values())
                completed_tasks = []
                if tasks:
                    completed_tasks, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results, put the results into output queue
                for task in completed_tasks:
                    traj = await task
                    task_rid = task.get_name()
                    rollout_tasks.pop(task_rid)
                    self.rollout_stat.accepted += 1
                    self.output_queue.put_nowait(traj)
                    self.rollout_stat.running -= 1
                await asyncio.sleep(1)
    ...
```

With this rollout thread running, the training script (the main thread) submits prompts
into `input_queue` and collects rollout data from `output_queue` into training batches
with `prepare_batch` (for asynchronous RL) or `rollout_batch` (for synchronous RL). The
following code shows the implementation of `prepare_batch`:

```python
def prepare_batch(
    self,
    dataloader: StatefulDataLoader,
    workflow: Optional["RolloutWorkflow"] = None,
    workflow_builder: Optional[Callable] = None,
    should_accept: Callable | None = None,
):
    if not hasattr(self, "data_generator"):
        self.data_generator = cycle_dataloader(dataloader)
    assert dataloader.batch_size is not None
    while True:
        # Submit at least two batches to allow maximum overlap
        if (
            self.get_capacity() + dataloader.batch_size > 0
            and self.input_queue.qsize() + dataloader.batch_size
            < self.input_queue.maxsize
        ):
            data = next(self.data_generator)
            for item in data:
                self.submit(
                    item,
                    workflow=workflow,
                    workflow_builder=workflow_builder,
                    should_accept=should_accept,
                )
        try:
            return self.wait(dataloader.batch_size, timeout=1)
        except TimeoutError:
            pass
```

The `RemoteSGLangEngine` exposes `rollout_batch` and `prepare_batch` by calling them in
the workflow executor:

```python
class RemoteSGLangEngine(InferenceEngine):
    ...
    def prepare_batch(self, *args, **kwargs):
        return self.workflow_executor.prepare_batch(*args, **kwargs)
```

The usage of `RemoteSGLangEngine` in the training script is simple:

```python
rollout = RemoteSGLangEngine(config.inf_engine)
rollout.initialize()
eval_rollout = ...

data_generator = cycle_dataloader(train_dataloader)
for global_step in range(max_steps):
    # rollout batched training data for current step
    if config.async_training:
        batch = rollout.prepare_batch(train_dataloader, workflow=workflow, should_accept=lambda sample: True)
    else:
        batch = rollout.rollout_batch(next(data_generator), workflow=workflow, should_accept=lambda sample: True)
```

You may notice that the above code creates a dummy lambda function for the
`should_accept` argument. This optional argument can be used for dynamic filtering ---
an important training technique used in many RL papers. With asynchronous rollout under
the hood, dynamic filtering is quite straight-forward: once a rollout completes, we run
this `should_accept` function on the collected sample to determine whether this rollout
is accepted or not.

For example, if we want to filter out samples that provide all-positive or all-negative
rewards, you should write:

```python
batch = rollout.prepare_batch(train_dataloader,
                              workflow=workflow,
                              should_accept=lambda sample: sample['rewards'].mean() > 0 and sample['rewards'].mean() < 1)
```

However, we note that AReaL's implementation has a subtle difference from DAPO: we
remain a constant batch size when dynamic filtering is enabled (i.e., we still wait
until `batch_size` samples are accepted), while DAPO filter samples after collecting a
complete batch, resulting variable batch sizes across training steps.

If you want to use rollout workflows with custom reward functions or agentic tool
calling, see [Customization: Rollout Workflows](../customization/agent.md) for more
details.

## Training

After obtaining the training batch, we use `FSDPPPOActor` to calculate losses and update
weights. Each train engine corresponds to one model, therefore we need an additional
engine for the reference model. Note that `torch.distributed` process groups will be
lazily initialized using `init_process_group` when the first train engine is
initialized. The initialization of train engine will also load model weights from paths
specified by the configuration.

```python
actor = FSDPPPOActor(config=config.actor)
actor.create_process_group()
actor.initialize(None, ft_spec)
actor.connect_engine(rollout, weight_update_meta)

ref = None
if config.actor.kl_ctl > 0 and config.ref is not None:
    ref = FSDPPPOActor(config=config.ref)
    ref.create_process_group()
    ref.initialize(None, ft_spec)
```

`FSDPPPOActor` is a high-level engine with algorithm-specific APIs, such as
`compute_logp`,`compute_advantages` and `ppo_update`. `FSDPPPOActor` is powered by the
lower-level train engine `FSDPEngine`, which use **pytorch FSDP2** to provide basic APIs
for the model such as `train_batch` and `forward`. The following code shows a GRPO
training step:

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

If you want to customize your own training algorithm, see
[Customize algorithms](https://inclusionai.github.io/AReaL/customization/algorithm.html)
for more details.

## Transferring Weights to Inference Servers

After training, we transfer updated model weights to remote inference servers through
cooperation between `FSDPPPOActor` and `RemoteSGLangEngine`. We provide options to
transfer model weights from shared storage or NCCL. In our example training script, we
first prepare `WeightUpdateMeta` for NCCL backend on all training processes.

```python
weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
    AllocationMode.from_str(config.allocation_mode)
)
```

If you wish to transfer model weights from shared storage, you can use:

```python
weight_update_meta = WeightUpdateMeta.from_disk(config.saver)
```

After a training step is finished, we transfer new weights from actor engine to remote
inference servers:

1. The rollout engine needs to stop sending generation requests to remote servers
   (`rollout.pause()`) before weight update to avoid server-side congestion.
1. Since we need to invoke weight update on the trainer engine and remote inference
   servers at the same time, in the training script, we asynchronously send requests to
   remote inference servers, and then immediately upload weights on the trainer engine.

```python
rollout.pause()
actor.update_weights(weight_update_meta)
rollout.resume()
actor.set_version(global_step + 1)
rollout.set_version(global_step + 1)
```

Now a complete GRPO training step in AReaL-lite is done! The core logic of our example
training script can be summarized as:

```python
data_generator = cycle_dataloader(train_dataloader)
for global_step in range(max_steps):
    if config.async_training:
        batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
    else:
        batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

    logp = actor.compute_logp(batch)
    batch["prox_logp"] = logp
    if ref is not None:
        batch["ref_logp"] = ref.compute_logp(batch)
        log_gpu_stats("ref logp")
    actor.compute_advantages(batch)
    stats = actor.ppo_update(batch)
    actor.step_lr_scheduler()

    rollout.pause()
    actor.update_weights(weight_update_meta)
    rollout.resume()
    actor.set_version(global_step + 1)
    rollout.set_version(global_step + 1)
```

## Utilities

In AReaL-lite, we provide a wide range of utilities for basic functionalities required
for observing and tuning your experiments.

### `Saver` and `Evaluator`

`Saver`
([areal/utils/saver.py](https://github.com/inclusionAI/AReaL/blob/main/areal/utils/saver.py))
and `Evaluator`
([areal/utils/evaluator.py](https://github.com/inclusionAI/AReaL/blob/main/areal/utils/evaluator.py))
manage the frequency to save and evaluate the model with the train engine.

In our example, we call `saver.save` and `evaluator.evaluate` after every training step.
these two methods will automatically check if it is time to save or evaluate the model,
according to the experiment configuration.

### `stats_tracker`

`stats_tracker`
([realhf/base/stats_tracker.py](https://github.com/inclusionAI/AReaL/blob/main/realhf/base/stats_tracker.py))
gathers training statistics across parallel ranks and reduce them.

1. **Scalar-type statistics** are recorded by `stats_tracker.scalar(key=value)` and will
   be averaged by the number of scalars with the same key when reduced.
1. **Tensor-type statistics** require `denominator` and `reduce_type` to decide how to
   reduce statistics under the same key.

- `denominator` is a bool tensor that masks the elements in the tensor that we do not
  want to record.
- `reduce_type` includes average, sum, min and max. By default, the average, min and max
  are all calculated.

For example, if we want to record the length of sequences with correct and incorrect
answers in a training batch:

```python
seqlens = ... # tensor of shape [#seqs,]
reward_score = ... # tensor of shape [#seqs,]

result_denominators = {
    "correct_n_seqs": (reward_score > 0).bool(),
    "incorrect_n_seqs": (reward_score <= 0).bool(),
}
# register the denominator
stats_tracker.denominator(**result_denominators)
# record the correct and incorrect sequence length
stats_tracker.stat(
    correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
)
stats_tracker.stat(
    incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
)
```

`stats_tracker` offers timer context to record time cost of a code block as a scalar.
And there is also a scope context to manage keys of statistics.

```python
with stats_tracker.record_timing("train_step"):
    # training step
    ...

with stats_tracker.scope("A"):
    stats_tracker.scalar(c=123) # key="A/c", value=123
    with stats_tracker.scope("B"):
        stats_tracker.scalar(c=234) # key="A/B/c", value=234
```

After recording sufficient data, e.g. after a `train_batch` is finished,
`stats_tracker.export` is called to aggregate all statistics and dump them into a
dictionary.

```python
stats = stats_tracker.export()
```

### `StatsLogger`

`StatsLogger`
([areal/utils/stats_logger.py](https://github.com/inclusionAI/AReaL/blob/main/areal/utils/stats_logger.py))
logs gathered training data to recorders like `wandb` and `tensorboard` on rank 0. In
our example script, after finishing a training step,
`logger.commit(epoch, step, global_step, stats)` is called to record all statistics from
`stats_tracker` to print them as well as log them into the recorders set by the
configuration.

## Next Steps

- [Customize dataset](../customization/dataset.md)
- [Customize Agentic/RVLR rollout workflows](../customization/agent.md)
- [Customize algorithms](../customization/algorithm.md)
