# Configurations

This page provides a comprehensive reference for all configuration parameters available
in AReaL's command-line interface. These parameters are defined using dataclasses and
can be specified in YAML configuration files or overridden via command line arguments.

## Usage

Configuration files are specified using the `--config` parameter:

```bash
python -m areal.launcher --config path/to/config.yaml
```

You can override specific parameters from the command line:

```bash
python -m areal.launcher --config path/to/config.yaml actor.lr=1e-4 seed=42
```

For detailed examples, see the experiment configurations in the `examples/` directory.

## Table of Contents

### Core Experiment Configurations

- [Base Experiment Configuration](section-base-experiment)
- [SFT Configuration](section-sft)
- [GRPO Configuration](section-grpo)
- [Reward Model Configuration](section-reward-model)

### Training Configurations

- [Training Engine Configuration](section-train-engine)
- [PPO Actor Configuration](section-ppo-actor)
- [Optimizer Configuration](section-optimizer)
- [Micro-batch Specification](section-microbatch)
- [Normalization Configuration](section-normalization)

### Inference Configurations

- [Inference Engine Configuration](section-inference-engine)
- [SGLang Configuration](section-sglang)
- [Generation Hyperparameters](section-generation)

### Dataset

- [Dataset Configuration](section-dataset)

### System and Cluster Configurations

- [Cluster Specification](section-cluster)
- [Launcher Configuration](section-launcher)

### Logging and Monitoring

- [Statistics Logger Configuration](section-stats-logger)
- [Checkpoint Saver Configuration](section-saver)
- [Evaluator Configuration](section-evaluator)
- [Recovery Configuration](section-recovery)
- [Scheduler Configuration](section-scheduler)

______________________________________________________________________

(section-base-experiment)=

## Base Experiment Configuration

Base configuration shared by all experiment types (SFT, GRPO, etc.)

| Parameter            | Type                    | Default      | Description                                                                                                                |
| -------------------- | ----------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                  | `"???"`      | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                  | `"???"`      | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | `ClusterSpecConfig`     | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                  | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                 | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                 | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None         | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None         | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                  | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | `DatasetConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `valid_dataset`      | `DatasetConfig` \| None | `None`       | No description available. Please check the description of this dataclass.                                                  |
| `saver`              | `SaverConfig`           | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `evaluator`          | `EvaluatorConfig`       | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `stats_logger`       | `StatsLoggerConfig`     | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `recover`            | `RecoverConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `sglang`             | `SGLangConfig`          | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `launcher`           | `LauncherConfig`        | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `scheduler`          | `SchedulerConfig`       | **Required** | No description available. Please check the description of this dataclass.                                                  |

(section-sft)=

## SFT Configuration

Configuration specific to supervised fine-tuning experiments

| Parameter            | Type                    | Default      | Description                                                                                                                |
| -------------------- | ----------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                  | `"???"`      | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                  | `"???"`      | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | `ClusterSpecConfig`     | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                  | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                 | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                 | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None         | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None         | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                  | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | `DatasetConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `valid_dataset`      | `DatasetConfig` \| None | `None`       | No description available. Please check the description of this dataclass.                                                  |
| `saver`              | `SaverConfig`           | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `evaluator`          | `EvaluatorConfig`       | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `stats_logger`       | `StatsLoggerConfig`     | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `recover`            | `RecoverConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `sglang`             | `SGLangConfig`          | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `launcher`           | `LauncherConfig`        | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `scheduler`          | `SchedulerConfig`       | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `model`              | `TrainEngineConfig`     | **Required** | No description available. Please check the description of this dataclass.                                                  |

(section-grpo)=

## GRPO Configuration

Configuration for GRPO reinforcement learning experiments

| Parameter            | Type                        | Default      | Description                                                                                                                |
| -------------------- | --------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                      | `"???"`      | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                      | `"???"`      | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | `ClusterSpecConfig`         | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                      | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                     | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                     | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None             | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None             | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                      | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | `DatasetConfig`             | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `valid_dataset`      | `DatasetConfig` \| None     | `None`       | No description available. Please check the description of this dataclass.                                                  |
| `saver`              | `SaverConfig`               | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `evaluator`          | `EvaluatorConfig`           | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `stats_logger`       | `StatsLoggerConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `recover`            | `RecoverConfig`             | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `sglang`             | `SGLangConfig`              | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `launcher`           | `LauncherConfig`            | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `scheduler`          | `SchedulerConfig`           | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `async_training`     | boolean                     | `True`       | Enable asynchronous training between rollout and policy update.                                                            |
| `gconfig`            | `GenerationHyperparameters` | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `rollout`            | `InferenceEngineConfig`     | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `actor`              | `PPOActorConfig`            | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `ref`                | `PPOActorConfig`            | **Required** | No description available. Please check the description of this dataclass.                                                  |

(section-reward-model)=

## Reward Model Configuration

Configuration for training reward models

| Parameter            | Type                    | Default      | Description                                                                                                                |
| -------------------- | ----------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                  | `"???"`      | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                  | `"???"`      | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | `ClusterSpecConfig`     | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                  | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                 | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                 | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None         | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None         | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                  | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | `DatasetConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `valid_dataset`      | `DatasetConfig` \| None | `None`       | No description available. Please check the description of this dataclass.                                                  |
| `saver`              | `SaverConfig`           | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `evaluator`          | `EvaluatorConfig`       | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `stats_logger`       | `StatsLoggerConfig`     | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `recover`            | `RecoverConfig`         | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `sglang`             | `SGLangConfig`          | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `launcher`           | `LauncherConfig`        | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `scheduler`          | `SchedulerConfig`       | **Required** | No description available. Please check the description of this dataclass.                                                  |
| `model`              | `TrainEngineConfig`     | **Required** | No description available. Please check the description of this dataclass.                                                  |

(section-train-engine)=

## Training Engine Configuration

Core configuration for model training, including optimization and backend settings

| Parameter                | Type                          | Default               | Description                                                                                                                             |
| ------------------------ | ----------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`        | string                        | `"???"`               | No description available. Please check the description of this dataclass.                                                               |
| `trial_name`             | string                        | `"???"`               | No description available. Please check the description of this dataclass.                                                               |
| `path`                   | string                        | `""`                  | Path to HuggingFace checkpoint                                                                                                          |
| `attn_impl`              | string                        | `"flash_attention_2"` | Attention implementation for huggingface transformers model. **Choices:** `flash_attention_2`                                           |
| `init_from_scratch`      | boolean                       | `False`               | Initialize model weights randomly                                                                                                       |
| `is_critic`              | boolean                       | `False`               | Whether to use a critic/reward model                                                                                                    |
| `mb_spec`                | `MicroBatchSpec`              | **Required**          | No description available. Please check the description of this dataclass.                                                               |
| `pad_to_maximum`         | boolean                       | `False`               | Whether to pad each microbatch to the length upper bound specified by mb_spec. Can reduce memory fragmentation but slows down training. |
| `disable_dropout`        | boolean                       | `False`               | Disable dropout layers during training                                                                                                  |
| `gradient_checkpointing` | boolean                       | `True`                | Enable gradient checkpointing                                                                                                           |
| `dtype`                  | string                        | `"bfloat16"`          | Parameter data type.                                                                                                                    |
| `grad_reduce_dtype`      | string                        | `"float32"`           | Gradient reduction data type.                                                                                                           |
| `optimizer`              | `OptimizerConfig` \| None     | `None`                | Optimizer configuration                                                                                                                 |
| `backend`                | string                        | `""`                  | Training backend (refer to documentation)                                                                                               |
| `fsdp`                   | `FSDPEngineConfig`            | **Required**          | No description available. Please check the description of this dataclass.                                                               |
| `ds_auto_tp`             | `DeepSpeedAutoTPEngineConfig` | **Required**          | No description available. Please check the description of this dataclass.                                                               |

(section-ppo-actor)=

## PPO Actor Configuration

Configuration for PPO actor models in RL training

| Parameter                 | Type                          | Default               | Description                                                                                                                                                                                                                                                                                                                |
| ------------------------- | ----------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`         | string                        | `"???"`               | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                  |
| `trial_name`              | string                        | `"???"`               | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                  |
| `path`                    | string                        | `""`                  | Path to HuggingFace checkpoint                                                                                                                                                                                                                                                                                             |
| `attn_impl`               | string                        | `"flash_attention_2"` | Attention implementation for huggingface transformers model. **Choices:** `flash_attention_2`                                                                                                                                                                                                                              |
| `init_from_scratch`       | boolean                       | `False`               | Initialize model weights randomly                                                                                                                                                                                                                                                                                          |
| `is_critic`               | boolean                       | `False`               | Whether to use a critic/reward model                                                                                                                                                                                                                                                                                       |
| `mb_spec`                 | `MicroBatchSpec`              | **Required**          | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                  |
| `pad_to_maximum`          | boolean                       | `False`               | Whether to pad each microbatch to the length upper bound specified by mb_spec. Can reduce memory fragmentation but slows down training.                                                                                                                                                                                    |
| `disable_dropout`         | boolean                       | `False`               | Disable dropout layers during training                                                                                                                                                                                                                                                                                     |
| `gradient_checkpointing`  | boolean                       | `True`                | Enable gradient checkpointing                                                                                                                                                                                                                                                                                              |
| `dtype`                   | string                        | `"bfloat16"`          | Parameter data type.                                                                                                                                                                                                                                                                                                       |
| `grad_reduce_dtype`       | string                        | `"float32"`           | Gradient reduction data type.                                                                                                                                                                                                                                                                                              |
| `optimizer`               | `OptimizerConfig` \| None     | `None`                | Optimizer configuration                                                                                                                                                                                                                                                                                                    |
| `backend`                 | string                        | `""`                  | Training backend (refer to documentation)                                                                                                                                                                                                                                                                                  |
| `fsdp`                    | `FSDPEngineConfig`            | **Required**          | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                  |
| `ds_auto_tp`              | `DeepSpeedAutoTPEngineConfig` | **Required**          | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                  |
| `group_size`              | integer                       | `1`                   | Number of sequences in each group                                                                                                                                                                                                                                                                                          |
| `ppo_n_minibatches`       | integer                       | `4`                   | Number of minibatches for each PPO update                                                                                                                                                                                                                                                                                  |
| `eps_clip`                | float                         | `0.2`                 | Clipping factor for policy ratio                                                                                                                                                                                                                                                                                           |
| `eps_clip_higher`         | float \| None                 | `None`                | Clipping factor (higher value) for policy ratio. Default is None. When eps_clip_higher is set (decoupled), eps_clip will be used as the lower value.                                                                                                                                                                       |
| `c_clip`                  | float \| None                 | `None`                | Dual clipping factor for policy ratio, must be > 1.0. None disables dual clipping.                                                                                                                                                                                                                                         |
| `temperature`             | float                         | `1.0`                 | Temperature during generation.                                                                                                                                                                                                                                                                                             |
| `reward_norm`             | `NormConfig` \| None          | `None`                | Normalization configuration for rewards                                                                                                                                                                                                                                                                                    |
| `reward_scaling`          | float                         | `1.0`                 | Reward scaling factor                                                                                                                                                                                                                                                                                                      |
| `reward_bias`             | float                         | `0.0`                 | Reward bias                                                                                                                                                                                                                                                                                                                |
| `reward_clip`             | float                         | `20.0`                | Maximum absolute value for reward clipping                                                                                                                                                                                                                                                                                 |
| `overlong_reward_penalty` | boolean                       | `False`               | Penalty for overlong sequences. Used within DAPO.                                                                                                                                                                                                                                                                          |
| `overlong_tokens`         | integer \| None               | `None`                | Number of tokens in the tail that will receive a penalty                                                                                                                                                                                                                                                                   |
| `overlong_penalty_factor` | float \| None                 | `None`                | Penalty factor for tokens in the tail                                                                                                                                                                                                                                                                                      |
| `mask_no_eos_with_zero`   | boolean                       | `False`               | Mask truncated generations (no EOS token) and exclude from training                                                                                                                                                                                                                                                        |
| `discount`                | float                         | `1.0`                 | Discount factor for future rewards                                                                                                                                                                                                                                                                                         |
| `gae_lambda`              | float                         | `1.0`                 | Lambda parameter for GAE                                                                                                                                                                                                                                                                                                   |
| `adv_norm`                | `NormConfig` \| None          | `None`                | Normalization configuration for advantages.                                                                                                                                                                                                                                                                                |
| `kl_ctl`                  | float                         | `0.1`                 | KL divergence coefficient                                                                                                                                                                                                                                                                                                  |
| `recompute_logprob`       | boolean                       | `False`               | Recompute log probability and replace the log probability returned by inference.                                                                                                                                                                                                                                           |
| `use_decoupled_loss`      | boolean                       | `False`               | Use the decoupled loss. Implicitly enables recompute_logprob.                                                                                                                                                                                                                                                              |
| `behav_imp_weight_cap`    | float \| None                 | `None`                | Filter out tokens where behav_imp_weight exceeds behav_imp_weight_cap when computing loss. Must be > 1.0. use_decoupled_loss must be true.                                                                                                                                                                                 |
| `dynamic_sampling`        | boolean                       | `False`               | Enable dynamic sampling (within DAPO). If enabled, groups with the same reward will be masked out. Note that enabling this option will lead to variable batch sizes. If you want to use a constant batch size with dynamic filtering, you should use the `should_accept` parameter in `rollout_batch` and `prepare_batch`. |
| `log_agent_stats`         | boolean                       | `False`               | Log statistics for agent trajectories                                                                                                                                                                                                                                                                                      |
| `log_agent_stats_keys`    | list of string                | **Required**          | Keys for logging agent trajectory statistics                                                                                                                                                                                                                                                                               |
| `max_new_tokens`          | integer                       | `1024`                | Maximum number of new tokens to generate                                                                                                                                                                                                                                                                                   |

(section-optimizer)=

## Optimizer Configuration

Settings for model optimization during training

Configuration for model optimization during training. Note: Set type to "empty" for
models that won't be trained.

| Parameter                 | Type    | Default      | Description                                                              |
| ------------------------- | ------- | ------------ | ------------------------------------------------------------------------ |
| `type`                    | string  | `"adam"`     | Optimizer type **Choices:** `adam`, `empty`                              |
| `lr`                      | float   | `2e-05`      | Learning rate                                                            |
| `weight_decay`            | float   | `0.05`       | Weight decay                                                             |
| `beta1`                   | float   | `0.9`        | Adam beta1 parameter                                                     |
| `beta2`                   | float   | `0.95`       | Adam beta2 parameter                                                     |
| `eps`                     | float   | `1e-05`      | Adam epsilon parameter                                                   |
| `min_lr_ratio`            | float   | `0.0`        | Minimum learning rate ratio after annealing                              |
| `lr_scheduler_type`       | string  | `"constant"` | Learning rate scheduler type **Choices:** `linear`, `cosine`, `constant` |
| `warmup_steps_proportion` | float   | `0.001`      | Proportion of training steps for warmup                                  |
| `offload`                 | boolean | `False`      | Enable optimizer state offloading                                        |
| `initial_loss_scale`      | float   | `4294967296` | Initial loss scaling factor                                              |
| `min_loss_scale`          | float   | `1.0`        | Minimum loss scaling factor                                              |
| `loss_scale_window`       | float   | `5`          | Window size for loss scaling adjustment                                  |
| `hysteresis`              | integer | `2`          | Hysteresis (scaling factor) for loss scaling                             |
| `gradient_clipping`       | float   | `1.0`        | Gradient clipping threshold                                              |

(section-microbatch)=

## Micro-batch Specification

Configuration for splitting data into micro-batches during training

Specification for splitting micro-batches during training.

| Parameter           | Type            | Default | Description                                                                                                                      |
| ------------------- | --------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `n_mbs`             | integer \| None | `1`     | Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count |
| `granularity`       | integer         | `1`     | Granularity of each micro-batch. Adjacent sequences are grouped by this size when dividing microbatches.                         |
| `max_tokens_per_mb` | integer \| None | `None`  | Maximum tokens per micro-batch for each forward pass. When set, n_mbs becomes the minimum number of micro-batches.               |

(section-inference-engine)=

## Inference Engine Configuration

Configuration for model inference and rollout generation

| Parameter                 | Type            | Default         | Description                                                                                                                                                         |
| ------------------------- | --------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`         | string \| None  | `None`          | No description available. Please check the description of this dataclass.                                                                                           |
| `trial_name`              | string \| None  | `None`          | No description available. Please check the description of this dataclass.                                                                                           |
| `max_concurrent_rollouts` | integer \| None | `None`          | Maximum number of concurrent rollouts to the inference engine. Defaults to consumer_batch_size.                                                                     |
| `queue_size`              | integer \| None | `None`          | Input/Output queue size for async rollout.                                                                                                                          |
| `consumer_batch_size`     | integer         | `1`             | Batch size for consuming rollouts from the queue.                                                                                                                   |
| `max_head_offpolicyness`  | integer         | `0`             | Maximum off-policyness for the head. If the current version is more than this many versions behind, the request will not be accepted.                               |
| `enable_rollout_tracing`  | boolean         | `False`         | Whether to output verbose tracing messages for each generation request.                                                                                             |
| `check_trajectory_format` | boolean         | `False`         | Whether to check the format of produced trajectories of a customized workflow. Useful when debugging the workflow in isolation. Should be False during RL training. |
| `schedule_policy`         | string          | `"round_robin"` | Request scheduling policy **Choices:** `round_robin`                                                                                                                |
| `setup_timeout`           | float           | `120.0`         | Timeout in seconds of connecting to remote servers or launching local servers.                                                                                      |
| `request_timeout`         | float           | `3600`          | Timeout for HTTP requests.                                                                                                                                          |
| `request_retries`         | integer         | `3`             | Number of retries for failed requests.                                                                                                                              |

(section-sglang)=

## SGLang Configuration

Configuration for SGLang inference runtime

Configuration for SGLang runtime. Refer to: https://github.com/sgl-project/sglang for
detailed documentation.

| Parameter                         | Type                    | Default      | Description                                                               |
| --------------------------------- | ----------------------- | ------------ | ------------------------------------------------------------------------- |
| `model_path`                      | string                  | `""`         | No description available. Please check the description of this dataclass. |
| `random_seed`                     | integer                 | `1`          | No description available. Please check the description of this dataclass. |
| `skip_tokenizer_init`             | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `disable_cuda_graph`              | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `disable_radix_cache`             | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `disable_cuda_graph_padding`      | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_nccl_nvls`                | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `disable_outlines_disk_cache`     | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `disable_custom_all_reduce`       | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `disable_overlap_schedule`        | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_mixed_chunk`              | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_dp_attention`             | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_ep_moe`                   | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_torch_compile`            | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `torch_compile_max_bs`            | integer                 | `32`         | No description available. Please check the description of this dataclass. |
| `cuda_graph_max_bs`               | integer \| None         | `None`       | No description available. Please check the description of this dataclass. |
| `cuda_graph_bs`                   | list of integer \| None | `None`       | No description available. Please check the description of this dataclass. |
| `torchao_config`                  | string                  | `""`         | No description available. Please check the description of this dataclass. |
| `enable_nan_detection`            | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_p2p_check`                | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `triton_attention_reduce_in_fp32` | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `triton_attention_num_kv_splits`  | integer                 | `8`          | No description available. Please check the description of this dataclass. |
| `num_continuous_decode_steps`     | integer                 | `1`          | No description available. Please check the description of this dataclass. |
| `enable_memory_saver`             | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `allow_auto_truncate`             | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `attention_backend`               | string \| None          | `"fa3"`      | No description available. Please check the description of this dataclass. |
| `enable_multimodal`               | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `sampling_backend`                | string \| None          | `None`       | No description available. Please check the description of this dataclass. |
| `context_length`                  | integer \| None         | `32768`      | No description available. Please check the description of this dataclass. |
| `mem_fraction_static`             | float \| None           | `0.9`        | No description available. Please check the description of this dataclass. |
| `max_running_requests`            | integer \| None         | `None`       | No description available. Please check the description of this dataclass. |
| `chunked_prefill_size`            | integer \| None         | `-1`         | No description available. Please check the description of this dataclass. |
| `max_prefill_tokens`              | integer                 | `32768`      | No description available. Please check the description of this dataclass. |
| `schedule_policy`                 | string                  | `"lpm"`      | No description available. Please check the description of this dataclass. |
| `schedule_conservativeness`       | float                   | `1.0`        | No description available. Please check the description of this dataclass. |
| `cpu_offload_gb`                  | integer                 | `0`          | No description available. Please check the description of this dataclass. |
| `dtype`                           | string                  | `"bfloat16"` | No description available. Please check the description of this dataclass. |
| `kv_cache_dtype`                  | string                  | `"auto"`     | No description available. Please check the description of this dataclass. |
| `dp_size`                         | integer                 | `1`          | No description available. Please check the description of this dataclass. |
| `ep_size`                         | integer                 | `1`          | No description available. Please check the description of this dataclass. |
| `log_level`                       | string                  | `"warning"`  | No description available. Please check the description of this dataclass. |
| `log_level_http`                  | string \| None          | `"warning"`  | No description available. Please check the description of this dataclass. |
| `log_requests`                    | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `log_requests_level`              | integer                 | `0`          | No description available. Please check the description of this dataclass. |
| `show_time_cost`                  | boolean                 | `False`      | No description available. Please check the description of this dataclass. |
| `enable_metrics`                  | boolean                 | `True`       | No description available. Please check the description of this dataclass. |
| `decode_log_interval`             | integer                 | `1`          | No description available. Please check the description of this dataclass. |

(section-generation)=

## Generation Hyperparameters

Parameters controlling text generation behavior during RL training

Controls text generation behavior for RL training.

| Parameter           | Type                   | Default      | Description                                                                                                                           |
| ------------------- | ---------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `n_samples`         | integer                | `1`          | Number of sequences to generate per prompt.                                                                                           |
| `max_new_tokens`    | integer                | `16384`      | Maximum number of tokens to generate.                                                                                                 |
| `min_new_tokens`    | integer                | `0`          | Minimum number of tokens to generate.                                                                                                 |
| `max_tokens`        | integer                | `65536`      | Maximum number of tokens including prompt and generated tokens.                                                                       |
| `greedy`            | boolean                | `False`      | Whether to use greedy decoding (max probability).                                                                                     |
| `top_p`             | float                  | `1.0`        | Nucleus sampling probability threshold (0.0, 1.0\].                                                                                   |
| `top_k`             | integer                | `100000000`  | Number of highest probability tokens to consider.                                                                                     |
| `temperature`       | float                  | `1.0`        | Sampling temperature. Higher values increase diversity.                                                                               |
| `stop_token_ids`    | list of integer        | **Required** | Stop generation when encountering these token IDs.                                                                                    |
| `stop`              | list of string \| None | `None`       | One or multiple stop words. Generation will stop if one of these words is sampled.                                                    |
| `frequency_penalty` | float                  | `0.0`        | Penalizes tokens based on their frequency in generation so far. Must be between -2 and 2 where negative numbers encourage repetition. |

(section-dataset)=

## Dataset Configuration

Configuration for training and validation datasets

| Parameter     | Type            | Default | Description                                                                      |
| ------------- | --------------- | ------- | -------------------------------------------------------------------------------- |
| `path`        | string          | `"???"` | Path to the dataset. Can be a local path or a HuggingFace dataset name.          |
| `type`        | string          | `"???"` | Type of training method, e.g., 'sft', 'rl', etc.                                 |
| `batch_size`  | integer         | `1`     | Batch size for the dataloader                                                    |
| `shuffle`     | boolean         | `True`  | Whether to shuffle the dataset                                                   |
| `pin_memory`  | boolean         | `False` | Pin memory for faster data loading (set True for GPU training)                   |
| `num_workers` | integer         | `0`     | Number of worker processes for data loading                                      |
| `drop_last`   | boolean         | `True`  | Drop the last incomplete batch                                                   |
| `max_length`  | integer \| None | `None`  | Maximum token length of sequences in dataset. Longer sequences are filtered out. |

(section-normalization)=

## Normalization Configuration

Settings for data normalization (rewards, advantages, etc.)

Configuration for normalization.

| Parameter    | Type           | Default   | Description                                                                                       |
| ------------ | -------------- | --------- | ------------------------------------------------------------------------------------------------- |
| `mean_level` | string \| None | `"batch"` | Mean level for normalization. Choices: batch, group. Omit for no mean normalization.              |
| `std_level`  | string \| None | `"batch"` | Standard deviation level for normalization. Choices: batch, group. Omit for no std normalization. |
| `group_size` | integer        | `1`       | Group size for group-level normalization                                                          |

(section-cluster)=

## Cluster Specification

Configuration for distributed training cluster setup

| Parameter         | Type                | Default                   | Description                                                      |
| ----------------- | ------------------- | ------------------------- | ---------------------------------------------------------------- |
| `name_resolve`    | `NameResolveConfig` | **Required**              | Name resolving configuration.                                    |
| `cluster_name`    | string              | `"local"`                 | Name of the cluster. Used to set specific environs.              |
| `fileroot`        | string              | `"/home/fw/.cache/areal"` | Root for logs and checkpoints. Should be available on all nodes. |
| `n_nodes`         | integer             | `32`                      | The size of the cluster. Used to decide slurm hostname suffix.   |
| `n_gpus_per_node` | integer             | `8`                       | Number of GPUs per node (physical).                              |

(section-launcher)=

## Launcher Configuration

Settings for launching training and inference processes

Configuration for launching the SGLang server.

| Parameter                       | Type                  | Default      | Description                                                                                      |
| ------------------------------- | --------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `inference_server_cpus_per_gpu` | integer               | `4`          | Number of CPUs allocated per GPU for inference server.                                           |
| `inference_server_mem_per_gpu`  | integer               | `32768`      | Memory allocated per GPU for inference server in MB.                                             |
| `trainer_cpus_per_gpu`          | integer               | `4`          | Number of CPUs allocated per GPU for training.                                                   |
| `trainer_mem_per_gpu`           | integer               | `32768`      | Memory allocated per GPU for training in MB.                                                     |
| `inference_server_env_vars`     | string                | `""`         | Environment variables for inference server, separated by commas. Example: 'ENV1=val1,ENV2=val2'. |
| `trainer_env_vars`              | string                | `""`         | Environment variables for training, separated by commas. Example: 'ENV1=val1,ENV2=val2'.         |
| `slurm`                         | `SlurmLauncherConfig` | **Required** | Slurm launcher configuration.                                                                    |

(section-stats-logger)=

## Statistics Logger Configuration

Configuration for experiment logging and monitoring

| Parameter         | Type                | Default      | Description                                                               |
| ----------------- | ------------------- | ------------ | ------------------------------------------------------------------------- |
| `experiment_name` | string              | `"???"`      | No description available. Please check the description of this dataclass. |
| `trial_name`      | string              | `"???"`      | No description available. Please check the description of this dataclass. |
| `fileroot`        | string              | `"???"`      | No description available. Please check the description of this dataclass. |
| `wandb`           | `WandBConfig`       | **Required** | Weights & Biases configuration.                                           |
| `swanlab`         | `SwanlabConfig`     | **Required** | SwanLab configuration.                                                    |
| `tensorboard`     | `TensorBoardConfig` | **Required** | TensorBoard configuration. Only 'path' field required.                    |

(section-saver)=

## Checkpoint Saver Configuration

Settings for saving model checkpoints

| Parameter         | Type            | Default | Description                                                               |
| ----------------- | --------------- | ------- | ------------------------------------------------------------------------- |
| `experiment_name` | string          | `"???"` | No description available. Please check the description of this dataclass. |
| `trial_name`      | string          | `"???"` | No description available. Please check the description of this dataclass. |
| `fileroot`        | string          | `"???"` | No description available. Please check the description of this dataclass. |
| `freq_epochs`     | integer \| None | `None`  | Trigger frequency in epochs. None disables epoch-based saving.            |
| `freq_steps`      | integer \| None | `None`  | Trigger frequency in steps. None disables step-based saving.              |
| `freq_secs`       | integer \| None | `None`  | Trigger frequency in seconds. None disables time-based saving.            |

(section-evaluator)=

## Evaluator Configuration

Configuration for model evaluation during training

| Parameter         | Type            | Default | Description                                                               |
| ----------------- | --------------- | ------- | ------------------------------------------------------------------------- |
| `experiment_name` | string          | `"???"` | No description available. Please check the description of this dataclass. |
| `trial_name`      | string          | `"???"` | No description available. Please check the description of this dataclass. |
| `fileroot`        | string          | `"???"` | No description available. Please check the description of this dataclass. |
| `freq_epochs`     | integer \| None | `None`  | Trigger frequency in epochs. None disables epoch-based saving.            |
| `freq_steps`      | integer \| None | `None`  | Trigger frequency in steps. None disables step-based saving.              |
| `freq_secs`       | integer \| None | `None`  | Trigger frequency in seconds. None disables time-based saving.            |

(section-recovery)=

## Recovery Configuration

Settings for experiment recovery and fault tolerance

| Parameter         | Type            | Default      | Description                                                                                                                                                                                                                                                                                                                                                 |
| ----------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name` | string          | `"???"`      | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                                                   |
| `trial_name`      | string          | `"???"`      | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                                                   |
| `fileroot`        | string          | `"???"`      | No description available. Please check the description of this dataclass.                                                                                                                                                                                                                                                                                   |
| `freq_epochs`     | integer \| None | `None`       | Trigger frequency in epochs. None disables epoch-based saving.                                                                                                                                                                                                                                                                                              |
| `freq_steps`      | integer \| None | `None`       | Trigger frequency in steps. None disables step-based saving.                                                                                                                                                                                                                                                                                                |
| `freq_secs`       | integer \| None | `None`       | Trigger frequency in seconds. None disables time-based saving.                                                                                                                                                                                                                                                                                              |
| `mode`            | string          | `"disabled"` | Recovery mode for the launcher. Options: 'disabled': Never recover from previous runs. 'auto': Automatically recover from previous runs if recover info and checkpoints are available. 'fault': Only recover from previous runs if the new run fails. 'resume': Force to resume, raise an error if no recover info was found. Never resume if failed again. |
| `retries`         | integer         | `3`          | Number of recovery retries (auto/fault modes only).                                                                                                                                                                                                                                                                                                         |

(section-scheduler)=

## Scheduler Configuration

Configuration for the AReaL scheduler service

| Parameter                     | Type   | Default                             | Description                                                               |
| ----------------------------- | ------ | ----------------------------------- | ------------------------------------------------------------------------- |
| `endpoint`                    | string | `"http://localhost:8081"`           | No description available. Please check the description of this dataclass. |
| `deploy_mode`                 | string | `"separation"`                      | No description available. Please check the description of this dataclass. |
| `functioncall_service_domain` | string | `"http://localhost:8080"`           | No description available. Please check the description of this dataclass. |
| `reward_functioncall_config`  | `Dict` | **Required**                        | No description available. Please check the description of this dataclass. |
| `reward_model_path`           | string | `""`                                | No description available. Please check the description of this dataclass. |
| `reward_model_service_url`    | string | `"http://localhost:30000/classify"` | No description available. Please check the description of this dataclass. |
