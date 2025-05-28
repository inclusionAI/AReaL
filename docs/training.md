# Training

## Launch the Ray Cluster

### Start the Ray Head Node

On the first node, start the Ray Head with the following command:

```bash
docker run -d --name r1-ray-head --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.3.0 /bin/bash -c "ray start --head --port=6379 && tail -f /dev/null"
```

### Start Ray Worker Nodes

On all other nodes, start the Ray Worker with the following command (skip this step for single-node setups):

```bash
# Replace with the actual IP address of the first node
RAY_HEAD_IP=xxx.xxx.xxx.xxx
docker run -d --name r1-ray-worker --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.3.0 /bin/bash -c "ray start --address=$RAY_HEAD_IP:6379 && tail -f /dev/null"
```

### Verify Cluster Status

Once all nodes are running, check the Ray cluster status by entering the container on the first node:

```bash
docker exec -it r1-ray-head bash
ray status
```

You should see the Ray resource status displayed.

## Launch an Experiment

On the first node (where the Ray Head is located), run the following to launch an asynchronous PPO experiment:

```bash
docker exec -it r1-ray-head bash
cd /storage/codes/AReaL
pip3 install -e .
python3 training/main_async_ppo.py --config-name=async-ppo-1.7b-gpu8
```

This command will locate the YAML configuration file `async-ppo-1.7b-gpu8.yaml` in the `training/configs/async-ppo` folder. The meaning of each configuration entry can be found in `realhf/api/cli_args.py`. You can run asynchronous PPO, synchronous PPO, or SFT depending on the script you execute.

After starting, you'll see training launch information like this:

```
20250528-17:12:16.804 quickstart INFO: Running async-ppo-math experiment.
20250528-17:12:16.804 quickstart INFO: Logs will be dumped to /storage/experiments/logs/admin/async-ppo-1.7b-gpu8/my-trial
20250528-17:12:16.804 quickstart INFO: Experiment configs will be dumped to /storage/experiments/logs/admin/async-ppo-1.7b-gpu8/my-trial/config.yaml
20250528-17:12:16.804 quickstart INFO: Model checkpoints will be saved to /storage/experiments/checkpoints/admin/async-ppo-1.7b-gpu8/my-trial
20250528-17:12:19.261 quickstart INFO: Launching experiments with RAY...
```

**Note**: The saved YAML configuration at `/storage/experiments/logs/admin/async-ppo-1.7b-gpu8/my-trial/config.yaml` can be used to reproduce previous experiments.

## Command Line Options

To view all available options:

```bash
python3 -m realhf.apps.quickstart async-ppo-math --help
```

### Important Parameters

- **`mode`**: Always set to `ray`. Do not change this value when following this tutorial.
- **`{actor|critic|ref}.path`**: The path to the model files.
- **`dataset.path`**: The path to the dataset JSONL file.
- **`cluster.fileroot`**: The root path for saving training outputs.
- **`n_nodes`**: The number of nodes in the cluster.
- **`n_gpus_per_node`**: The number of GPUs per node.
- **`allocation_mode`**: The GPU allocation strategy and 3D parallelism configuration for the experiment. Format:
  - `sglang.d${DP1}m${TP1}p${PP1}+d${DP2}m${TP2}p${PP2}`: Configures parallel strategies for SGLang generation and training respectively. Generation and training use separate GPU sets, and the total GPU count must equal: DP1×TP1×PP1 + DP2×TP2×PP2 = #GPUs.

### Training Control Parameters

- **`exp_ctrl.total_train_epochs`**: Number of training epochs (complete dataset iterations).
- **`exp_ctrl.save_freq_{epochs|steps|secs}`**: Frequency for saving model parameters to persistent storage. Set to null to disable saving.
- **`exp_ctrl.ckpt_freq_{epochs|steps|secs}`**: Frequency for saving temporary parameters for restart capability.
- **`dataset.train_bs_n_seqs`**: Training batch size (number of prompts sampled per training iteration).
- **`group_size`**: Number of responses sampled per prompt.
- **`{actor_train|ref_inf|actor_inf}.mb_spec.max_tokens_per_mb`**: Maximum tokens per mini-batch for forward/backward passes during reference model inference and actor model training. Reduce to avoid OOM errors.
- **`ppo.ppo_n_minibatches`**: Number of mini-batches for dividing data during each PPO update.
- **`ppo.recompute_logprob`**: Whether to compute proximal log probabilities for training.
- **`ppo.use_decoupled_loss`**: Use decoupled loss to stabilize asynchronous training.
- **`ppo.gen.max_new_tokens`**: Maximum tokens to generate per prompt (default: 16k).
- **`ppo.gen.min_new_tokens`**: Minimum tokens to generate per prompt (default: 0).

## Monitoring the Training Process

We recommend using Weights & Biases (wandb) for monitoring. Run `wandb login` or set the `WANDB_API_KEY` environment variable. Set `wandb.mode=True` in your configuration to upload training statistics.

The main log will be saved to `/storage/experiments/logs/admin/async-ppo-1.7b-gpu8/my-trial/main.log` and contains the statistics uploaded to wandb.

### Key Training Statistics

- **`Epoch 1/5`**: Indicates total epochs required and current epoch being trained.
- **`step 6/19`**: Shows current epoch has 19 steps, with the 6th step just completed.
- **`global step 6`**: Step count across all epochs.
- **`task_reward`**: Average reward value of all sampled responses in this step. Should steadily increase during training and eventually stabilize.
- **`importance_weight`**: Average importance sampling ratio across all tokens in the PPO loss. Typically close to 1.0.
- **`actor_clip_ratio`**: Ratio of clipped tokens in PPO loss to total tokens. Usually less than 0.1.
- **`actor_loss`**: PPO loss value. **Does not show clear trends during training** and should not be used as a performance indicator.
- **`avg_seq_len`**: Average length of all sequences (prompts with sampled responses) in this step.
- **`no_eos_ratio`**: Ratio of sampled responses truncated due to exceeding maximum generation length. An increase indicates longer average response lengths.