# Quickstart

We will go over a simple example of training an LLM to solve math problems.

## Dataset

Use `huggingface-cli` to download our open-source dataset.

```bash
huggingface-cli download --repo-type=dataset inclusionAI/AReaL-RL-Data
```

```{note}
The above command will prompt you the path of the downloaded dataset.
The user should pass it to the training command.
```

## Model

We train using open-source models available on Hugging Face Hub. You can either download it in advance or use the model identifier when running the experiment.

```bash
# If you want to download it in advance
huggingface-cli download Qwen/Qwen3-1.7B
```

Refer to the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) for using `huggingface-cli`.

## Training

Under the repo directory, run

```bash
# examples/run_async_ppo.sh
python3 training/main_sync_ppo.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=/path/to/save/logs/checkpoints/ \
    actor.type._class=qwen3 \
    actor.path=/path/to/models/Qwen__Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=/path/to/models/Qwen__Qwen3-1.7B \
    dataset.path=/path/to/dataset/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=32768
```

## Command Line Options

To view all available options:

```bash
python3 training/main_sync_ppo.py --help
```

- **`experiment_name`**: The name of your project.
- **`trial_name`**: The name of this trial in your project.

- **`{actor|ref}.path`**: The path to the model files.
- **`dataset.path`**: The path to the dataset JSONL file.
- **`cluster.fileroot`**: The root path for saving training outputs (logs and checkpoints).
- **`n_nodes`**: The number of nodes in the cluster.
- **`n_gpus_per_node`**: The number of GPUs per node.
- **`allocation_mode`**: The GPU allocation strategy and 3D parallelism configuration for the experiment. Format:
  - `sglang.d${DP1}m${TP1}p${PP1}+d${DP2}m${TP2}p${PP2}`: Configures parallel strategies for SGLang generation and training respectively. Generation and training use separate GPU sets, and the total GPU count must equal: DP1×TP1×PP1 + DP2×TP2×PP2 = #GPUs.

- **`exp_ctrl.total_train_epochs`**: Number of training epochs (complete dataset iterations).
- **`exp_ctrl.save_freq_{epochs|steps|secs}`**: Frequency for saving model parameters to persistent storage. Set to null to disable saving.
- **`exp_ctrl.ckpt_freq_{epochs|steps|secs}`**: Frequency for saving temporary parameters for restart capability.
- **`dataset.train_bs_n_seqs`**: Training batch size (number of prompts sampled per training iteration).
- **`group_size`**: Number of responses sampled per prompt.
- **`{actor_train|ref_inf|actor_inf}.mb_spec.max_tokens_per_mb`**: Maximum tokens per mini-batch for forward/backward passes during reference model inference and actor model training. Reduce to avoid OOM errors.
- **`ppo.ppo_n_minibatches`**: Number of mini-batches for dividing data during each PPO update.
- **`ppo.recompute_logprob`**: Whether to compute proximal log probabilities for training. Defaults to True for asynchronous experiments and False for synchronous baselines.
- **`ppo.use_decoupled_loss`**: Use decoupled loss to stabilize asynchronous training. Defaults to True.
- **`ppo.gen.max_new_tokens`**: Maximum tokens to generate per prompt.

## Monitoring the Training Process

We recommend using Weights & Biases (wandb) for monitoring. Run `wandb login` or set the `WANDB_API_KEY` environment variable. Set `wandb.mode=True` in your configuration to upload training statistics.

You can also use tensorboard by passing `tensorboard.path`.

The main log will be saved to `${fileroot}/logs/${USER}/${experiment_name}/${trial_name}/main.log` and contains the statistics uploaded to wandb.

### Key Training Statistics

- **`Epoch 1/5`**: Indicates total epochs required and current epoch being trained.
- **`step 6/19`**: Shows current epoch has 19 steps, with the 6th step just completed.
- **`global step 6`**: Step count across all epochs.
- **`ppo_actor/task_reward/avg`**: Average reward value of all sampled responses in this step. Should steadily increase during training and eventually stabilize.
- **`ppo_actor/importance_weight/avg`**: Average importance sampling ratio across all tokens in the PPO loss. Typically close to 1.0.
- **`ppo_actor/actor_clip_ratio/avg`**: Ratio of clipped tokens in PPO loss to total tokens. Usually less than 0.1.
- **`ppo_actor/actor_loss/avg`**: PPO loss value. **Does not show clear trends during training** and should not be used as a performance indicator.

## Next Step

[Evaluate your model](eval.md) or check the [troubleshooting section](troubleshooting.md) if you encounter any issues.
