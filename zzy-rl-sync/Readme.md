# ThreadWeaver

ThreadWeaver is a reinforcement learning framework for training large language models to generate parallel reasoning paths when solving complex mathematical problems. Built on top of [VERL](https://github.com/volcengine/verl), it enables models to explore multiple solution strategies simultaneously, improving both solution quality and computational efficiency.

## Project Structure

```
.
├── launch_rl.sh                   # Main RL training launch script
├── threadweaver_rl/               # RL training (VERL-based)
│   ├── verl/trainer/main_ppo.py   # PPO training entry point
│   ├── verl/trainer/config/       # Hydra configuration files
│   │   ├── ppo_trainer.yaml       # Main PPO config
│   │   └── reward_model/
│   │       └── reward_model.yaml  # Reward configuration
│   ├── deepscaler/rewards/        # Reward function implementations
│   └── multinode_run.slurm        # Multi-node SLURM script
├── threadweaver_sft/              # Supervised fine-tuning
├── data_generation/               # Data generation pipeline
└── assets/                        # Documentation images
```

For installation and prerequisites, see [threadweaver_rl/README.md](threadweaver_rl/README.md).

## Running RL Training

### Quick Start (Single-Node)

Set your paths and run:
```bash
export VLLM_USE_V1=1
export MODEL_PATH=/path/to/your/sft/model
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/val.parquet
export CHECKPOINT_DIR=/path/to/save/checkpoints
export ROLLOUT_DATA_DIR=/path/to/save/rollout/data

bash launch_rl.sh
```

### Custom Launch

The training entry point is `python3 -m verl.trainer.main_ppo` with Hydra config overrides. A minimal example:

```bash
export VLLM_USE_V1=1
MODEL_PATH=/path/to/your/sft/model

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="/path/to/train.parquet" \
  data.val_files="/path/to/val.parquet" \
  data.max_prompt_length=9216 \
  data.max_response_length=8192 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.n=8 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=30 \
  trainer.logger='[console,tensorboard,wandb]'
```

### Multi-Node Training (Ray)

ThreadWeaver uses [Ray](https://docs.ray.io/) for distributed training across multiple nodes.

0. **Start 8 GPUs nodes**, each with 8 GPUs:

   Each set the env var
   ```bash
      export PYTHONPATH=/YOUR_PYTHONPATH
      export WANDB_BASE_URL=
      export WANDB_API_KEY=
      unset ROCR_VISIBLE_DEVICES
      export VLLM_USE_V1=1
      export HF_HUB_OFFLINE=1
      unset VERL_USE_MODELSCOPE
      export VLLM_USE_MODELSCOPE=False      
   ```
1. **Start the Ray head node** on your primary machine:
   ```bash
   ray start --head --dashboard-host=0.0.0.0
   ```

2. **Start Ray worker nodes** on each additional machine, pointing to the head node IP:
   ```bash
   ray start --address=<head-node-ip>:6379 # Follow the ip in head node
   ```

3. **Verify the cluster**:
   ```bash
   ray status
   ```
   THere should be 8 nodes

4. **Launch training** from the head node. Set `trainer.nnodes` to match your cluster size refer to launch_rl.sh:
   ```bash
   export VLLM_USE_V1=1
   MODEL_PATH=/path/to/your/sft/model

   python3 -m verl.trainer.main_ppo \
     algorithm.adv_estimator=grpo \
     data.train_files="/path/to/train.parquet" \
     data.val_files="/path/to/val.parquet" \
     data.max_prompt_length=9216 \
     data.max_response_length=8192 \
     actor_rollout_ref.model.path=$MODEL_PATH \
     actor_rollout_ref.actor.optim.lr=1e-6 \
     actor_rollout_ref.rollout.n=8 \
     trainer.n_gpus_per_node=8 \
     trainer.nnodes=4 \
     trainer.total_epochs=30 \
     trainer.logger='[console,tensorboard,wandb]'
   ```

<!-- 5. **Monitor**:
   ```bash
   # Ray dashboard (default http://<head-node-ip>:8265)
   # TensorBoard
   tensorboard --logdir=./tensorboard_log
   ``` -->

5. **Shut down the cluster** when done:
   ```bash
   ray stop  # run on every node
   ```

## Configuring the Reward Function

The reward is configured in `threadweaver_rl/verl/trainer/config/reward_model/reward_model.yaml` under the `config` section. You can either edit the YAML directly or pass overrides on the command line via `reward_model.config.<key>=<value>`.

### Reward Components

The total reward is: **`reward = r_correct + r_accel + r_parallel`**

#### 1. Correctness Reward

| Parameter | Default | Description |
|-----------|---------|-------------|
| `correct_reward` | `1.0` | Reward for a correct answer |
| `incorrect_reward` | `-1.0` | Reward for an incorrect answer |
| `format_error_reward` | `-1.0` | Reward when the output format is invalid |

#### 2. Acceleration Ratio Reward (`r_accel`)

Encourages the model to produce parallel reasoning that reduces critical-path latency.

```
r_accel = acceleration_ratio_reward
        * acceleration_ratio_reward_factor
        * min(acceleration_ratio, acceleration_ratio_clip_max)
```

Where `acceleration_ratio = 1 - critical_path_tokens / total_tokens`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `acceleration_ratio_reward` | `0.0` | Master switch (set >0 to enable) |
| `acceleration_ratio_reward_factor` | `1.0` | Scaling factor |
| `acceleration_ratio_clip_max` | `1.0` | Upper bound on the ratio |

#### 3. Parallel Bonus Reward (`r_parallel`, v2 only)

Group-normalized z-score bonuses applied only to correct samples. Set `version: "v2"` to enable.

```
r_parallel = subtask_beta  * z(subtask_ratio)
           + trial_beta    * z(trial_ratio)
           + parallel_ratio_beta * z(parallel_ratio)
           + latency_alpha * z(acceleration_ratio)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `version` | `"v1"` | Set to `"v2"` to enable parallel bonus |
| `subtask_beta` | `0.0` | Weight for subtask ratio z-score |
| `trial_beta` | `0.1` | Weight for trial ratio z-score |
| `parallel_ratio_beta` | `0.0` | Weight for parallel ratio z-score |
| `latency_alpha` | `0.0` | Weight for acceleration ratio z-score |
| `group_shaping_eps` | `1e-8` | Epsilon for z-score normalization |

#### 4. Length Penalty (optional)

Penalizes overly long responses when group accuracy is high enough.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length_penalty_enabled` | `False` | Enable length penalty |
| `length_penalty_tau` | `0.8` | Accuracy threshold to activate |
| `length_penalty_beta` | `0.2` | Maximum penalty scale |
| `length_penalty_window` | `10000.0` | Token window for overlong ratio |

### Example: Override Rewards from CLI

```bash
python3 -m verl.trainer.main_ppo \
  ... \
  reward_model.config.version=v2 \
  reward_model.config.correct_reward=2.0 \
  reward_model.config.incorrect_reward=-0.5 \
  reward_model.config.acceleration_ratio_reward=1.0 \
  reward_model.config.acceleration_ratio_reward_factor=0.5 \
  reward_model.config.acceleration_ratio_clip_max=0.2 \
  reward_model.config.trial_beta=0.1 \
  reward_model.reward_manager_type=reward_manager_with_server
```

### Other Reward Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `require_think_end` | `True` | Require `</think>` token in output |
| `strip_comma_from_answer` | `False` | Strip commas from parsed answers |
| `reward_manager_type` | `reward_manager` | `reward_manager` (local) or `reward_manager_with_server` (server-based) |
| `reward_manager_server_workers` | `64` | Number of gunicorn workers for server mode |
| `verbose` | `0` | Verbosity level for reward logging |

## Reference Results

Both models trained for 400 steps on math multiplication data:

| Setting | Num Tokens in Longest Thread | Accuracy |
|---------|:---:|:---|
| Sequential Baseline | 3322 | 99.0% |
| ThreadWeaver (Parallel) | 2632 | 99.0% |

ThreadWeaver achieves **1.26x speedup** in token latency vs. the sequential baseline.

For more details, see the [ThreadWeaver RL README](threadweaver_rl/README.md).

