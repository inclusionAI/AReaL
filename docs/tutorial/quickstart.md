# Quickstart

This guide walks you through a simple example of training an LLM to solve math problems. Please ensure you have properly [installed dependencies and set up the runtime environment](installation.md) before proceeding.

## Dataset

Use `huggingface-cli` to download our open-source dataset:

```bash
huggingface-cli download --repo-type=dataset inclusionAI/AReaL-RL-Data
```

> **Note**: The command above will display the path of the downloaded dataset. You'll need to pass this path to the training command.

## Model

We train using open-source models available on Hugging Face Hub. You can either download the model in advance or use the model identifier when running the experiment.

```bash
# If you want to download it in advance
huggingface-cli download Qwen/Qwen3-1.7B
```

Refer to the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) for more information on using `huggingface-cli`.

## Training

From the repository directory, run:

```bash
# examples/run_async_ppo.sh
python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=/path/to/save/logs/checkpoints/ \
    actor.type._class=qwen3 \
    actor.path=Qwen/Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen3-1.7B \
    dataset.path=/path/to/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4
```

::::{important}
Running `main_async_ppo.py` with `ppo.recompute_logprob=False`, `ppo.use_decoupled_loss=False`, and `max_head_offpolicyness=0` will essentially replicate the behavior of synchronous PPO. Therefore, it's usually not recommended to run synchronous PPO directly (i.e., `main_sync_ppo.py`). The workflow of asynchronous RL is more stable and easier to customize.
::::

## Command Line Options

To view all available options:

```bash
python3 training/main_sync_ppo.py --help
```

### Configuration Parameters

- **`experiment_name`**: The name of your project.
- **`trial_name`**: The name of this trial in your project.
- **`{actor|ref}.path`**: The path to the model files.
- **`dataset.path`**: The path to the dataset JSONL file.
- **`cluster.fileroot`**: The root path for saving training outputs (logs and checkpoints).
- **`n_nodes`**: The number of nodes in the cluster.
- **`n_gpus_per_node`**: The number of GPUs per node.
- **`allocation_mode`**: The GPU allocation strategy and 3D parallelism configuration for the experiment. Format:
  - `sglang.d${DP1}m${TP1}p${PP1}+d${DP2}m${TP2}p${PP2}`: Configures parallel strategies for SGLang generation and training respectively. Generation and training use separate GPU sets, and the total GPU count must equal: DP1×TP1×PP1 + DP2×TP2×PP2 = #GPUs.

### Training Control

- **`exp_ctrl.total_train_epochs`**: Number of training epochs (complete dataset iterations).
- **`exp_ctrl.save_freq_{epochs|steps|secs}`**: Frequency for saving model parameters to persistent storage. Set to null to disable saving.
- **`exp_ctrl.ckpt_freq_{epochs|steps|secs}`**: Frequency for saving temporary parameters for restart capability.
- **`dataset.train_bs_n_seqs`**: Training batch size (number of prompts sampled per training iteration).
- **`group_size`**: Number of responses sampled per prompt.

### Memory and Performance

- **`{actor_train|ref_inf|actor_inf}.mb_spec.max_tokens_per_mb`**: Maximum tokens per mini-batch for forward/backward passes during reference model inference and actor model training. Reduce this value to avoid OOM errors.
- **`max_concurrent_rollouts`**: The maximum number of concurrent rollouts. SGLang will run out of memory if this value is too large. Defaults to `dataset.train_bs_n_seqs`.

### Algorithm Configuration

- **`max_head_offpolicyness`**: The allowed maximum data staleness. 0 recovers synchronous training. A large value will increase generation throughput but degrade final performance. We recommend keeping this value at 8 or below.
- **`ppo.recompute_logprob`**: Whether to compute proximal log probabilities for training. Defaults to True for asynchronous experiments and False for synchronous baselines.
- **`ppo.use_decoupled_loss`**: Use decoupled loss to stabilize asynchronous training. Defaults to True.
- **`ppo.gen.max_new_tokens`**: Maximum tokens to generate per prompt.
- **`ppo.ppo_n_minibatches`**: Number of mini-batches for dividing data during each PPO update.
- **`success_rate_ub`**: Upper bound of success rate. Prompts with a higher success rate will be filtered out.
- **`success_rate_lb`**: Lower bound of success rate. Prompts with a lower success rate will be filtered out.

## Monitoring the Training Process

+ We recommend using [Weights & Biases (wandb)](https://github.com/wandb/wandb)  or [SwanLab](https://github.com/SwanHubX/SwanLab)  for monitoring—run `wandb login` or `swanlab login`, or set the corresponding environment variable API key (`WANDB_API_KEY` or `SWANLAB_API_KEY`). Set `wandb.mode="online"` or `swanlab.mode="cloud"` in your configuration to upload training statistics. If you cannot connect to the server, you can also use `wandb.mode="offline"` or `swanlab.mode="local"` to save data locally without uploading.


You can also use TensorBoard by setting the `tensorboard.path` parameter.

The main log will be saved to `${fileroot}/logs/${USER}/${experiment_name}/${trial_name}/main.log` and contains the statistics uploaded to wandb.

If SwanLab is enabled, logs will be saved to the directory specified by `swanlab.logdir`.

### Key Training Statistics

- **`Epoch 1/5`**: Indicates the total epochs required and the current epoch being trained.
- **`step 6/19`**: Shows that the current epoch has 19 steps, with the 6th step just completed.
- **`global step 6`**: Step count across all epochs.
- **`ppo_actor/task_reward/avg`**: Average reward value of all sampled responses in this step. This should steadily increase during training and eventually stabilize.
- **`ppo_actor/importance_weight/avg`**: Average importance sampling ratio across all tokens in the PPO loss. This is typically close to 1.0.
- **`ppo_actor/actor_clip_ratio/avg`**: Ratio of clipped tokens in PPO loss to total tokens. This is usually less than 0.1.
- **`ppo_actor/actor_loss/avg`**: PPO loss value. **This does not show clear trends during training** and should not be used as a performance indicator.

## Next Steps

[Evaluate your model](eval.md) or check the [troubleshooting section](troubleshooting.md) if you encounter any issues.