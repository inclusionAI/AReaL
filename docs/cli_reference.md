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

- [BaseExperiment Configuration](section-base-experiment)
- [GRPO Configuration](section-grpo)
- [PPO Configuration](section-ppo)
- [RW Configuration](section-rw)
- [SFT Configuration](section-sft)

### Training Configurations

- [FSDPEngine Configuration](section-fsdp-engine)
- [FSDPWrapPolicy](section-fsdp-wrap-policy)
- [MicroBatch Specification](section-micro-batch)
- [Norm Configuration](section-norm)
- [Optimizer Configuration](section-optimizer)
- [PPOActor Configuration](section-ppo-actor)
- [PPOCritic Configuration](section-ppo-critic)
- [TrainEngine Configuration](section-train-engine)

### Inference Configurations

- [GenerationHyperparameters](section-generation-hyperparameters)
- [InferenceEngine Configuration](section-inference-engine)
- [SGLang Configuration](section-sg-lang)
- [vLLM Configuration](section-v-llm)

### Dataset

- [Dataset Configuration](section-dataset)

### System and Cluster Configurations

- [Cluster Specification Configuration](section-cluster)
- [Launcher Configuration](section-launcher)
- [NameResolve Configuration](section-name-resolve)
- [SlurmLauncher Configuration](section-slurm-launcher)

### Logging and Monitoring

- [Evaluator Configuration](section-evaluator)
- [Recover Configuration](section-recover)
- [Saver Configuration](section-saver)
- [StatsLogger Configuration](section-stats-logger)
- [Swanlab Configuration](section-swanlab)
- [TensorBoard Configuration](section-tensor-board)
- [WandB Configuration](section-wand-b)

### Others

- [Scheduler Configuration](section-scheduler)

______________________________________________________________________

(section-base-experiment)=

## BaseExperiment Configuration

Base configuration class for all experiment types with common settings.

| Parameter            | Type                                        | Default      | Description                                                                                                                |
| -------------------- | ------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                                      | **Required** | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                                      | **Required** | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | [`ClusterSpecConfig`](section-cluster)      | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                                      | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                                     | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                                     | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None                             | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None                             | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                                      | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | [`DatasetConfig`](section-dataset)          | **Required** | -                                                                                                                          |
| `valid_dataset`      | [`DatasetConfig`](section-dataset) \| None  | `None`       | -                                                                                                                          |
| `saver`              | [`SaverConfig`](section-saver)              | **Required** | -                                                                                                                          |
| `evaluator`          | [`EvaluatorConfig`](section-evaluator)      | **Required** | -                                                                                                                          |
| `stats_logger`       | [`StatsLoggerConfig`](section-stats-logger) | **Required** | -                                                                                                                          |
| `recover`            | [`RecoverConfig`](section-recover)          | **Required** | -                                                                                                                          |
| `sglang`             | [`SGLangConfig`](section-sg-lang)           | **Required** | -                                                                                                                          |
| `vllm`               | [`vLLMConfig`](section-v-llm)               | **Required** | -                                                                                                                          |
| `launcher`           | [`LauncherConfig`](section-launcher)        | **Required** | -                                                                                                                          |
| `scheduler`          | [`SchedulerConfig`](section-scheduler)      | **Required** | -                                                                                                                          |

(section-grpo)=

## GRPO Configuration

Configuration for Group Relative Policy Optimization (GRPO) reinforcement learning
experiments.

| Parameter            | Type                                                              | Default      | Description                                                                                                                |
| -------------------- | ----------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                                                            | **Required** | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                                                            | **Required** | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | [`ClusterSpecConfig`](section-cluster)                            | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                                                            | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                                                           | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                                                           | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None                                                   | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None                                                   | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                                                            | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | [`DatasetConfig`](section-dataset)                                | **Required** | -                                                                                                                          |
| `valid_dataset`      | [`DatasetConfig`](section-dataset) \| None                        | `None`       | -                                                                                                                          |
| `saver`              | [`SaverConfig`](section-saver)                                    | **Required** | -                                                                                                                          |
| `evaluator`          | [`EvaluatorConfig`](section-evaluator)                            | **Required** | -                                                                                                                          |
| `stats_logger`       | [`StatsLoggerConfig`](section-stats-logger)                       | **Required** | -                                                                                                                          |
| `recover`            | [`RecoverConfig`](section-recover)                                | **Required** | -                                                                                                                          |
| `sglang`             | [`SGLangConfig`](section-sg-lang)                                 | **Required** | -                                                                                                                          |
| `vllm`               | [`vLLMConfig`](section-v-llm)                                     | **Required** | -                                                                                                                          |
| `launcher`           | [`LauncherConfig`](section-launcher)                              | **Required** | -                                                                                                                          |
| `scheduler`          | [`SchedulerConfig`](section-scheduler)                            | **Required** | -                                                                                                                          |
| `async_training`     | boolean                                                           | `True`       | Enable asynchronous training between rollout and policy update.                                                            |
| `gconfig`            | [`GenerationHyperparameters`](section-generation-hyperparameters) | **Required** | -                                                                                                                          |
| `rollout`            | [`InferenceEngineConfig`](section-inference-engine)               | **Required** | -                                                                                                                          |
| `actor`              | [`PPOActorConfig`](section-ppo-actor)                             | **Required** | -                                                                                                                          |
| `ref`                | [`PPOActorConfig`](section-ppo-actor)                             | **Required** | -                                                                                                                          |

(section-ppo)=

## PPO Configuration

Configuration for Proximal Policy Optimization (PPO) reinforcement learning experiments.

| Parameter            | Type                                                              | Default      | Description                                                                                                                |
| -------------------- | ----------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                                                            | **Required** | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                                                            | **Required** | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | [`ClusterSpecConfig`](section-cluster)                            | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                                                            | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                                                           | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                                                           | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None                                                   | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None                                                   | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                                                            | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | [`DatasetConfig`](section-dataset)                                | **Required** | -                                                                                                                          |
| `valid_dataset`      | [`DatasetConfig`](section-dataset) \| None                        | `None`       | -                                                                                                                          |
| `saver`              | [`SaverConfig`](section-saver)                                    | **Required** | -                                                                                                                          |
| `evaluator`          | [`EvaluatorConfig`](section-evaluator)                            | **Required** | -                                                                                                                          |
| `stats_logger`       | [`StatsLoggerConfig`](section-stats-logger)                       | **Required** | -                                                                                                                          |
| `recover`            | [`RecoverConfig`](section-recover)                                | **Required** | -                                                                                                                          |
| `sglang`             | [`SGLangConfig`](section-sg-lang)                                 | **Required** | -                                                                                                                          |
| `vllm`               | [`vLLMConfig`](section-v-llm)                                     | **Required** | -                                                                                                                          |
| `launcher`           | [`LauncherConfig`](section-launcher)                              | **Required** | -                                                                                                                          |
| `scheduler`          | [`SchedulerConfig`](section-scheduler)                            | **Required** | -                                                                                                                          |
| `async_training`     | boolean                                                           | `True`       | Enable asynchronous training between rollout and policy update.                                                            |
| `gconfig`            | [`GenerationHyperparameters`](section-generation-hyperparameters) | **Required** | -                                                                                                                          |
| `rollout`            | [`InferenceEngineConfig`](section-inference-engine)               | **Required** | -                                                                                                                          |
| `actor`              | [`PPOActorConfig`](section-ppo-actor)                             | **Required** | -                                                                                                                          |
| `ref`                | [`PPOActorConfig`](section-ppo-actor)                             | **Required** | -                                                                                                                          |
| `critic`             | [`PPOCriticConfig`](section-ppo-critic)                           | **Required** | -                                                                                                                          |

(section-rw)=

## RW Configuration

Configuration for Reward Model (RW) training experiments.

| Parameter            | Type                                        | Default      | Description                                                                                                                |
| -------------------- | ------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                                      | **Required** | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                                      | **Required** | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | [`ClusterSpecConfig`](section-cluster)      | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                                      | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                                     | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                                     | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None                             | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None                             | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                                      | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | [`DatasetConfig`](section-dataset)          | **Required** | -                                                                                                                          |
| `valid_dataset`      | [`DatasetConfig`](section-dataset) \| None  | `None`       | -                                                                                                                          |
| `saver`              | [`SaverConfig`](section-saver)              | **Required** | -                                                                                                                          |
| `evaluator`          | [`EvaluatorConfig`](section-evaluator)      | **Required** | -                                                                                                                          |
| `stats_logger`       | [`StatsLoggerConfig`](section-stats-logger) | **Required** | -                                                                                                                          |
| `recover`            | [`RecoverConfig`](section-recover)          | **Required** | -                                                                                                                          |
| `sglang`             | [`SGLangConfig`](section-sg-lang)           | **Required** | -                                                                                                                          |
| `vllm`               | [`vLLMConfig`](section-v-llm)               | **Required** | -                                                                                                                          |
| `launcher`           | [`LauncherConfig`](section-launcher)        | **Required** | -                                                                                                                          |
| `scheduler`          | [`SchedulerConfig`](section-scheduler)      | **Required** | -                                                                                                                          |
| `model`              | [`TrainEngineConfig`](section-train-engine) | **Required** | -                                                                                                                          |

(section-sft)=

## SFT Configuration

Configuration for Supervised Fine-Tuning (SFT) experiments.

| Parameter            | Type                                        | Default      | Description                                                                                                                |
| -------------------- | ------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`    | string                                      | **Required** | Name of the experiment (no '\_' or '/'). Required.                                                                         |
| `trial_name`         | string                                      | **Required** | Name of the trial (no '-' or '/'). Required.                                                                               |
| `cluster`            | [`ClusterSpecConfig`](section-cluster)      | **Required** | Cluster specification. Mainly used by slurm.                                                                               |
| `allocation_mode`    | string                                      | `""`         | GPU parallel strategy allocation mode. Options: manual/heuristic or pattern-based.                                         |
| `seed`               | integer                                     | `1`          | Random seed for reproducibility.                                                                                           |
| `total_train_epochs` | integer                                     | `1`          | Total number of epochs to train the model.                                                                                 |
| `total_train_steps`  | integer \| None                             | `None`       | Terminate training after this number of steps. For benchmarking purposes only. None indicates normal training.             |
| `total_train_n_seqs` | integer \| None                             | `None`       | Terminate training after consuming this number of samples. For benchmarking purposes only. None indicates normal training. |
| `tokenizer_path`     | string                                      | `""`         | Path to the tokenizer.                                                                                                     |
| `train_dataset`      | [`DatasetConfig`](section-dataset)          | **Required** | -                                                                                                                          |
| `valid_dataset`      | [`DatasetConfig`](section-dataset) \| None  | `None`       | -                                                                                                                          |
| `saver`              | [`SaverConfig`](section-saver)              | **Required** | -                                                                                                                          |
| `evaluator`          | [`EvaluatorConfig`](section-evaluator)      | **Required** | -                                                                                                                          |
| `stats_logger`       | [`StatsLoggerConfig`](section-stats-logger) | **Required** | -                                                                                                                          |
| `recover`            | [`RecoverConfig`](section-recover)          | **Required** | -                                                                                                                          |
| `sglang`             | [`SGLangConfig`](section-sg-lang)           | **Required** | -                                                                                                                          |
| `vllm`               | [`vLLMConfig`](section-v-llm)               | **Required** | -                                                                                                                          |
| `launcher`           | [`LauncherConfig`](section-launcher)        | **Required** | -                                                                                                                          |
| `scheduler`          | [`SchedulerConfig`](section-scheduler)      | **Required** | -                                                                                                                          |
| `model`              | [`TrainEngineConfig`](section-train-engine) | **Required** | -                                                                                                                          |

(section-fsdp-engine)=

## FSDPEngine Configuration

Configuration for Fully Sharded Data Parallel (FSDP) training backend.

| Parameter        | Type                                                 | Default | Description                                        |
| ---------------- | ---------------------------------------------------- | ------- | -------------------------------------------------- |
| `wrap_policy`    | [`FSDPWrapPolicy`](section-fsdp-wrap-policy) \| None | `None`  | FSDP wrap policy, specifying model layers to wrap. |
| `offload_params` | boolean                                              | `False` | Whether to offload FSDP parameters to CPU.         |

(section-fsdp-wrap-policy)=

## FSDPWrapPolicy

Policy configuration for FSDP model layer wrapping. None defaults to wrapping
transformer decoder layers defined by transformers.

| Parameter                       | Type                   | Default | Description                                         |
| ------------------------------- | ---------------------- | ------- | --------------------------------------------------- |
| `transformer_layer_cls_to_wrap` | list of string \| None | `None`  | A list of transformer layer names for FSDP to wrap. |

(section-micro-batch)=

## MicroBatch Specification

Specification for splitting micro-batches during training.

| Parameter           | Type            | Default | Description                                                                                                                      |
| ------------------- | --------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `n_mbs`             | integer \| None | `1`     | Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count |
| `granularity`       | integer         | `1`     | Granularity of each micro-batch. Adjacent sequences are grouped by this size when dividing microbatches.                         |
| `max_tokens_per_mb` | integer \| None | `None`  | Maximum tokens per micro-batch for each forward pass. When set, n_mbs becomes the minimum number of micro-batches.               |

(section-norm)=

## Norm Configuration

Configuration for reward/advantage normalization.

| Parameter        | Type           | Default   | Description                                                                                                      |
| ---------------- | -------------- | --------- | ---------------------------------------------------------------------------------------------------------------- |
| `mean_level`     | string \| None | `"batch"` | Mean level for normalization. None for no mean normalization. **Choices:** `batch`, `group`, `None`              |
| `mean_leave1out` | boolean        | `False`   | Whether to use leave-one-out average.                                                                            |
| `std_level`      | string \| None | `"batch"` | Standard deviation level for normalization. None for no std normalization. **Choices:** `batch`, `group`, `None` |
| `std_unbiased`   | boolean        | `True`    | Whether to use unbiased standard deviation computation. Defaults to True (changed from False in v0.3.4).         |
| `eps`            | float          | `1e-05`   | The eps when dividing by standard deviation to avoid numerical issues.                                           |
| `group_size`     | integer        | `1`       | Group size for group-level normalization                                                                         |

(section-optimizer)=

## Optimizer Configuration

Configuration for model optimization during training.

| Parameter                 | Type    | Default      | Description                                                                                             |
| ------------------------- | ------- | ------------ | ------------------------------------------------------------------------------------------------------- |
| `type`                    | string  | `"adam"`     | Optimizer type. Adam_bf16 currently only supported FSDP Engine. **Choices:** `adam`, `sgd`, `adam_bf16` |
| `lr`                      | float   | `2e-05`      | Learning rate                                                                                           |
| `weight_decay`            | float   | `0.05`       | Weight decay                                                                                            |
| `beta1`                   | float   | `0.9`        | Adam beta1 parameter. Only effective when optimizer_type is adam/adam_bf16                              |
| `beta2`                   | float   | `0.95`       | Adam beta2 parameter. Only effective when optimizer_type is adam/adam_bf16                              |
| `eps`                     | float   | `1e-05`      | Adam epsilon parameter. Only effective when optimizer_type is adam/adam_bf16                            |
| `min_lr_ratio`            | float   | `0.0`        | Minimum learning rate ratio after annealing                                                             |
| `lr_scheduler_type`       | string  | `"constant"` | Learning rate scheduler type **Choices:** `linear`, `cosine`, `constant`                                |
| `warmup_steps_proportion` | float   | `0.001`      | Proportion of training steps for warmup                                                                 |
| `offload`                 | boolean | `False`      | Enable optimizer state offloading                                                                       |
| `initial_loss_scale`      | float   | `4294967296` | Initial loss scaling factor                                                                             |
| `min_loss_scale`          | float   | `1.0`        | Minimum loss scaling factor                                                                             |
| `loss_scale_window`       | float   | `5`          | Window size for loss scaling adjustment                                                                 |
| `hysteresis`              | integer | `2`          | Hysteresis (scaling factor) for loss scaling                                                            |
| `gradient_clipping`       | float   | `1.0`        | Gradient clipping threshold                                                                             |

(section-ppo-actor)=

## PPOActor Configuration

Configuration for PPO actor model, a subclass of a TrainEngine.

| Parameter                 | Type                                           | Default               | Description                                                                                                                                                                                                                                                                                                                |
| ------------------------- | ---------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`         | string                                         | **Required**          | -                                                                                                                                                                                                                                                                                                                          |
| `trial_name`              | string                                         | **Required**          | -                                                                                                                                                                                                                                                                                                                          |
| `path`                    | string                                         | `""`                  | Path to HuggingFace checkpoint                                                                                                                                                                                                                                                                                             |
| `attn_impl`               | string                                         | `"flash_attention_2"` | Attention implementation for huggingface transformers model. **Choices:** `flash_attention_2`                                                                                                                                                                                                                              |
| `init_from_scratch`       | boolean                                        | `False`               | Initialize model weights randomly                                                                                                                                                                                                                                                                                          |
| `is_critic`               | boolean                                        | `False`               | Whether to use a critic/reward model                                                                                                                                                                                                                                                                                       |
| `mb_spec`                 | [`MicroBatchSpec`](section-micro-batch)        | **Required**          | -                                                                                                                                                                                                                                                                                                                          |
| `pad_to_maximum`          | boolean                                        | `False`               | Whether to pad each microbatch to the length upper bound specified by mb_spec. Can reduce memory fragmentation but slows down training.                                                                                                                                                                                    |
| `disable_dropout`         | boolean                                        | `False`               | Disable dropout layers during training                                                                                                                                                                                                                                                                                     |
| `gradient_checkpointing`  | boolean                                        | `True`                | Enable gradient checkpointing                                                                                                                                                                                                                                                                                              |
| `dtype`                   | string                                         | `"bfloat16"`          | Parameter data type.                                                                                                                                                                                                                                                                                                       |
| `grad_reduce_dtype`       | string                                         | `"float32"`           | Gradient reduction data type.                                                                                                                                                                                                                                                                                              |
| `optimizer`               | [`OptimizerConfig`](section-optimizer) \| None | `None`                | Optimizer configuration. None means no training.                                                                                                                                                                                                                                                                           |
| `weight_update_mode`      | string                                         | `"disk"`              | -                                                                                                                                                                                                                                                                                                                          |
| `backend`                 | string                                         | `""`                  | Training backend (refer to documentation)                                                                                                                                                                                                                                                                                  |
| `fsdp`                    | [`FSDPEngineConfig`](section-fsdp-engine)      | **Required**          | -                                                                                                                                                                                                                                                                                                                          |
| `use_lora`                | boolean                                        | `False`               | Whether to use LoRA. Only support FSDP. Note that should be enabled together with vLLM/SGLang.                                                                                                                                                                                                                             |
| `lora_rank`               | integer                                        | `32`                  | lora rank                                                                                                                                                                                                                                                                                                                  |
| `lora_alpha`              | integer                                        | `16`                  | lora alpha                                                                                                                                                                                                                                                                                                                 |
| `target_modules`          | list of string                                 | **Required**          | lora target_modules.                                                                                                                                                                                                                                                                                                       |
| `peft_type`               | string                                         | `"lora"`              | peft method type. Only LoRA is supported for now.                                                                                                                                                                                                                                                                          |
| `group_size`              | integer                                        | `1`                   | Number of sequences in each group                                                                                                                                                                                                                                                                                          |
| `ppo_n_minibatches`       | integer                                        | `4`                   | Number of minibatches for each PPO update                                                                                                                                                                                                                                                                                  |
| `eps_clip`                | float                                          | `0.2`                 | Clipping factor for policy ratio                                                                                                                                                                                                                                                                                           |
| `eps_clip_higher`         | float \| None                                  | `None`                | Clipping factor (higher value) for policy ratio. Default is None. When eps_clip_higher is set (decoupled), eps_clip will be used as the lower value.                                                                                                                                                                       |
| `c_clip`                  | float \| None                                  | `None`                | Dual clipping factor for policy ratio, must be > 1.0. None disables dual clipping.                                                                                                                                                                                                                                         |
| `temperature`             | float                                          | `1.0`                 | Temperature during generation.                                                                                                                                                                                                                                                                                             |
| `reward_norm`             | [`NormConfig`](section-norm) \| None           | `None`                | Normalization configuration for rewards                                                                                                                                                                                                                                                                                    |
| `reward_scaling`          | float                                          | `1.0`                 | Reward scaling factor                                                                                                                                                                                                                                                                                                      |
| `reward_bias`             | float                                          | `0.0`                 | Reward bias                                                                                                                                                                                                                                                                                                                |
| `reward_clip`             | float                                          | `20.0`                | Maximum absolute value for reward clipping                                                                                                                                                                                                                                                                                 |
| `overlong_reward_penalty` | boolean                                        | `False`               | Penalty for overlong sequences. Used within DAPO.                                                                                                                                                                                                                                                                          |
| `overlong_tokens`         | integer \| None                                | `None`                | Number of tokens in the tail that will receive a penalty                                                                                                                                                                                                                                                                   |
| `overlong_penalty_factor` | float \| None                                  | `None`                | Penalty factor for tokens in the tail                                                                                                                                                                                                                                                                                      |
| `mask_no_eos_with_zero`   | boolean                                        | `False`               | Mask truncated generations (no EOS token) and exclude from training                                                                                                                                                                                                                                                        |
| `discount`                | float                                          | `1.0`                 | Discount factor for future rewards                                                                                                                                                                                                                                                                                         |
| `gae_lambda`              | float                                          | `1.0`                 | Lambda parameter for GAE                                                                                                                                                                                                                                                                                                   |
| `adv_norm`                | [`NormConfig`](section-norm) \| None           | `None`                | Normalization configuration for advantages.                                                                                                                                                                                                                                                                                |
| `kl_ctl`                  | float                                          | `0.1`                 | KL divergence coefficient                                                                                                                                                                                                                                                                                                  |
| `kl_estimator`            | string                                         | `"k1"`                | KL divergence estimator **Choices:** `k1`, `k2`, `k3`                                                                                                                                                                                                                                                                      |
| `recompute_logprob`       | boolean                                        | `False`               | Recompute log probability and replace the log probability returned by inference.                                                                                                                                                                                                                                           |
| `use_decoupled_loss`      | boolean                                        | `False`               | Use the decoupled loss. Implicitly enables recompute_logprob.                                                                                                                                                                                                                                                              |
| `behav_imp_weight_cap`    | float \| None                                  | `None`                | Filter out tokens where behav_imp_weight exceeds behav_imp_weight_cap when computing loss. Must be > 1.0. use_decoupled_loss must be true.                                                                                                                                                                                 |
| `dynamic_sampling`        | boolean                                        | `False`               | Enable dynamic sampling (within DAPO). If enabled, groups with the same reward will be masked out. Note that enabling this option will lead to variable batch sizes. If you want to use a constant batch size with dynamic filtering, you should use the `should_accept` parameter in `rollout_batch` and `prepare_batch`. |
| `log_agent_stats`         | boolean                                        | `False`               | Log statistics for agent trajectories                                                                                                                                                                                                                                                                                      |
| `log_agent_stats_keys`    | list of string                                 | **Required**          | Keys for logging agent trajectory statistics                                                                                                                                                                                                                                                                               |
| `max_new_tokens`          | integer                                        | `1024`                | Maximum number of new tokens to generate                                                                                                                                                                                                                                                                                   |

(section-ppo-critic)=

## PPOCritic Configuration

Configuration for PPO critic model, a subclass of a TrainEngine.

| Parameter                | Type                                           | Default               | Description                                                                                                                             |
| ------------------------ | ---------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`        | string                                         | **Required**          | -                                                                                                                                       |
| `trial_name`             | string                                         | **Required**          | -                                                                                                                                       |
| `path`                   | string                                         | `""`                  | Path to HuggingFace checkpoint                                                                                                          |
| `attn_impl`              | string                                         | `"flash_attention_2"` | Attention implementation for huggingface transformers model. **Choices:** `flash_attention_2`                                           |
| `init_from_scratch`      | boolean                                        | `False`               | Initialize model weights randomly                                                                                                       |
| `is_critic`              | boolean                                        | `False`               | Whether to use a critic/reward model                                                                                                    |
| `mb_spec`                | [`MicroBatchSpec`](section-micro-batch)        | **Required**          | -                                                                                                                                       |
| `pad_to_maximum`         | boolean                                        | `False`               | Whether to pad each microbatch to the length upper bound specified by mb_spec. Can reduce memory fragmentation but slows down training. |
| `disable_dropout`        | boolean                                        | `False`               | Disable dropout layers during training                                                                                                  |
| `gradient_checkpointing` | boolean                                        | `True`                | Enable gradient checkpointing                                                                                                           |
| `dtype`                  | string                                         | `"bfloat16"`          | Parameter data type.                                                                                                                    |
| `grad_reduce_dtype`      | string                                         | `"float32"`           | Gradient reduction data type.                                                                                                           |
| `optimizer`              | [`OptimizerConfig`](section-optimizer) \| None | `None`                | Optimizer configuration. None means no training.                                                                                        |
| `weight_update_mode`     | string                                         | `"disk"`              | -                                                                                                                                       |
| `backend`                | string                                         | `""`                  | Training backend (refer to documentation)                                                                                               |
| `fsdp`                   | [`FSDPEngineConfig`](section-fsdp-engine)      | **Required**          | -                                                                                                                                       |
| `use_lora`               | boolean                                        | `False`               | Whether to use LoRA. Only support FSDP. Note that should be enabled together with vLLM/SGLang.                                          |
| `lora_rank`              | integer                                        | `32`                  | lora rank                                                                                                                               |
| `lora_alpha`             | integer                                        | `16`                  | lora alpha                                                                                                                              |
| `target_modules`         | list of string                                 | **Required**          | lora target_modules.                                                                                                                    |
| `peft_type`              | string                                         | `"lora"`              | peft method type. Only LoRA is supported for now.                                                                                       |
| `ppo_n_minibatches`      | integer                                        | `4`                   | Number of minibatches for each PPO update                                                                                               |
| `eps_clip`               | float                                          | `0.5`                 | Clipping factor for value loss                                                                                                          |
| `mask_no_eos_with_zero`  | boolean                                        | `False`               | Mask truncated generations (no EOS token) and exclude from training                                                                     |

(section-train-engine)=

## TrainEngine Configuration

Core configuration for model training, including optimization and backend settings.

| Parameter                | Type                                           | Default               | Description                                                                                                                             |
| ------------------------ | ---------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`        | string                                         | **Required**          | -                                                                                                                                       |
| `trial_name`             | string                                         | **Required**          | -                                                                                                                                       |
| `path`                   | string                                         | `""`                  | Path to HuggingFace checkpoint                                                                                                          |
| `attn_impl`              | string                                         | `"flash_attention_2"` | Attention implementation for huggingface transformers model. **Choices:** `flash_attention_2`                                           |
| `init_from_scratch`      | boolean                                        | `False`               | Initialize model weights randomly                                                                                                       |
| `is_critic`              | boolean                                        | `False`               | Whether to use a critic/reward model                                                                                                    |
| `mb_spec`                | [`MicroBatchSpec`](section-micro-batch)        | **Required**          | -                                                                                                                                       |
| `pad_to_maximum`         | boolean                                        | `False`               | Whether to pad each microbatch to the length upper bound specified by mb_spec. Can reduce memory fragmentation but slows down training. |
| `disable_dropout`        | boolean                                        | `False`               | Disable dropout layers during training                                                                                                  |
| `gradient_checkpointing` | boolean                                        | `True`                | Enable gradient checkpointing                                                                                                           |
| `dtype`                  | string                                         | `"bfloat16"`          | Parameter data type.                                                                                                                    |
| `grad_reduce_dtype`      | string                                         | `"float32"`           | Gradient reduction data type.                                                                                                           |
| `optimizer`              | [`OptimizerConfig`](section-optimizer) \| None | `None`                | Optimizer configuration. None means no training.                                                                                        |
| `weight_update_mode`     | string                                         | `"disk"`              | -                                                                                                                                       |
| `backend`                | string                                         | `""`                  | Training backend (refer to documentation)                                                                                               |
| `fsdp`                   | [`FSDPEngineConfig`](section-fsdp-engine)      | **Required**          | -                                                                                                                                       |
| `use_lora`               | boolean                                        | `False`               | Whether to use LoRA. Only support FSDP. Note that should be enabled together with vLLM/SGLang.                                          |
| `lora_rank`              | integer                                        | `32`                  | lora rank                                                                                                                               |
| `lora_alpha`             | integer                                        | `16`                  | lora alpha                                                                                                                              |
| `target_modules`         | list of string                                 | **Required**          | lora target_modules.                                                                                                                    |
| `peft_type`              | string                                         | `"lora"`              | peft method type. Only LoRA is supported for now.                                                                                       |

(section-generation-hyperparameters)=

## GenerationHyperparameters

Controls text generation behavior for rollout.

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

(section-inference-engine)=

## InferenceEngine Configuration

Configuration for inference servers, including offpolicyness control.

| Parameter                 | Type            | Default         | Description                                                                                                                                                         |
| ------------------------- | --------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name`         | string \| None  | `None`          | -                                                                                                                                                                   |
| `trial_name`              | string \| None  | `None`          | -                                                                                                                                                                   |
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
| `pause_grace_period`      | float           | `0.0`           | The grace period after calling /pause_generation. Wait until all requests have been dropped.                                                                        |

(section-sg-lang)=

## SGLang Configuration

Configuration for SGLang runtime. Refer to:

https://github.com/sgl-project/sglang for detailed documentation.

| Parameter                         | Type                    | Default      | Description |
| --------------------------------- | ----------------------- | ------------ | ----------- |
| `model_path`                      | string                  | `""`         | -           |
| `random_seed`                     | integer                 | `1`          | -           |
| `skip_tokenizer_init`             | boolean                 | `False`      | -           |
| `disable_cuda_graph`              | boolean                 | `False`      | -           |
| `disable_radix_cache`             | boolean                 | `True`       | -           |
| `disable_cuda_graph_padding`      | boolean                 | `False`      | -           |
| `enable_nccl_nvls`                | boolean                 | `False`      | -           |
| `disable_outlines_disk_cache`     | boolean                 | `False`      | -           |
| `disable_custom_all_reduce`       | boolean                 | `False`      | -           |
| `disable_overlap_schedule`        | boolean                 | `False`      | -           |
| `enable_mixed_chunk`              | boolean                 | `False`      | -           |
| `enable_dp_attention`             | boolean                 | `False`      | -           |
| `enable_ep_moe`                   | boolean                 | `False`      | -           |
| `enable_torch_compile`            | boolean                 | `False`      | -           |
| `torch_compile_max_bs`            | integer                 | `32`         | -           |
| `cuda_graph_max_bs`               | integer \| None         | `None`       | -           |
| `cuda_graph_bs`                   | list of integer \| None | `None`       | -           |
| `torchao_config`                  | string                  | `""`         | -           |
| `enable_nan_detection`            | boolean                 | `False`      | -           |
| `enable_p2p_check`                | boolean                 | `False`      | -           |
| `triton_attention_reduce_in_fp32` | boolean                 | `False`      | -           |
| `triton_attention_num_kv_splits`  | integer                 | `8`          | -           |
| `num_continuous_decode_steps`     | integer                 | `1`          | -           |
| `enable_memory_saver`             | boolean                 | `False`      | -           |
| `allow_auto_truncate`             | boolean                 | `False`      | -           |
| `attention_backend`               | string \| None          | `"fa3"`      | -           |
| `enable_multimodal`               | boolean                 | `False`      | -           |
| `sampling_backend`                | string \| None          | `None`       | -           |
| `context_length`                  | integer \| None         | `32768`      | -           |
| `mem_fraction_static`             | float \| None           | `0.9`        | -           |
| `max_running_requests`            | integer \| None         | `None`       | -           |
| `chunked_prefill_size`            | integer \| None         | `-1`         | -           |
| `max_prefill_tokens`              | integer                 | `32768`      | -           |
| `schedule_policy`                 | string                  | `"lpm"`      | -           |
| `schedule_conservativeness`       | float                   | `1.0`        | -           |
| `cpu_offload_gb`                  | integer                 | `0`          | -           |
| `dtype`                           | string                  | `"bfloat16"` | -           |
| `kv_cache_dtype`                  | string                  | `"auto"`     | -           |
| `dp_size`                         | integer                 | `1`          | -           |
| `ep_size`                         | integer                 | `1`          | -           |
| `enable_lora`                     | boolean \| None         | `None`       | -           |
| `max_lora_rank`                   | integer \| None         | `None`       | -           |
| `lora_target_modules`             | list of string \| None  | `None`       | -           |
| `lora_paths`                      | list of string \| None  | `None`       | -           |
| `max_loaded_loras`                | integer                 | `1`          | -           |
| `max_loras_per_batch`             | integer                 | `1`          | -           |
| `lora_backend`                    | string                  | `"triton"`   | -           |
| `log_level`                       | string                  | `"warning"`  | -           |
| `log_level_http`                  | string \| None          | `"warning"`  | -           |
| `log_requests`                    | boolean                 | `False`      | -           |
| `log_requests_level`              | integer                 | `0`          | -           |
| `show_time_cost`                  | boolean                 | `False`      | -           |
| `enable_metrics`                  | boolean                 | `True`       | -           |
| `decode_log_interval`             | integer                 | `1`          | -           |
| `enable_multithread_load`         | boolean                 | `False`      | -           |
| `enable_fast_load`                | boolean                 | `False`      | -           |

(section-v-llm)=

## vLLM Configuration

Configuration for vLLM runtime. Refer to:

https://docs.vllm.ai/en/stable/api/index.html for detailed documentation.

| Parameter                      | Type            | Default                                                             | Description |
| ------------------------------ | --------------- | ------------------------------------------------------------------- | ----------- |
| `model`                        | string          | `""`                                                                | -           |
| `seed`                         | integer         | `1`                                                                 | -           |
| `skip_tokenizer_init`          | boolean         | `False`                                                             | -           |
| `enforce_eager`                | boolean         | `True`                                                              | -           |
| `dtype`                        | string          | `"bfloat16"`                                                        | -           |
| `distributed_executor_backend` | string          | `"mp"`                                                              | -           |
| `max_num_seqs`                 | integer         | `256`                                                               | -           |
| `block_size`                   | integer         | `16`                                                                | -           |
| `swap_space`                   | integer         | `4`                                                                 | -           |
| `cpu_offload_gb`               | float           | `0`                                                                 | -           |
| `max_seq_len_to_capture`       | integer         | `32768`                                                             | -           |
| `disable_sliding_window`       | boolean         | `True`                                                              | -           |
| `max_model_len`                | integer \| None | `32768`                                                             | -           |
| `enable_chunked_prefill`       | boolean         | `False`                                                             | -           |
| `enable_prefix_caching`        | boolean         | `False`                                                             | -           |
| `gpu_memory_utilization`       | float           | `0.9`                                                               | -           |
| `worker_extension_cls`         | string          | `"areal.thirdparty.vllm.vllm_worker_extension.VLLMWorkerExtension"` | -           |
| `enable_sleep_mode`            | boolean         | `False`                                                             | -           |
| `uvicorn_log_level`            | string          | `"warning"`                                                         | -           |

(section-dataset)=

## Dataset Configuration

Configuration for dataset loading and preprocessing.

| Parameter     | Type            | Default      | Description                                                                      |
| ------------- | --------------- | ------------ | -------------------------------------------------------------------------------- |
| `path`        | string          | **Required** | Path to the dataset. Can be a local path or a HuggingFace dataset name.          |
| `type`        | string          | **Required** | Type of training method, e.g., 'sft', 'rl', etc.                                 |
| `batch_size`  | integer         | `1`          | Batch size for the dataloader                                                    |
| `shuffle`     | boolean         | `True`       | Whether to shuffle the dataset                                                   |
| `pin_memory`  | boolean         | `False`      | Pin memory for faster data loading (set True for GPU training)                   |
| `num_workers` | integer         | `0`          | Number of worker processes for data loading                                      |
| `drop_last`   | boolean         | `True`       | Drop the last incomplete batch                                                   |
| `max_length`  | integer \| None | `None`       | Maximum token length of sequences in dataset. Longer sequences are filtered out. |

(section-cluster)=

## Cluster Specification Configuration

Configuration for cluster specification and distributed computing setup.

| Parameter         | Type                                        | Default         | Description                                                      |
| ----------------- | ------------------------------------------- | --------------- | ---------------------------------------------------------------- |
| `name_resolve`    | [`NameResolveConfig`](section-name-resolve) | **Required**    | Name resolving configuration.                                    |
| `cluster_name`    | string                                      | `"local"`       | Name of the cluster. Used to set specific environs.              |
| `fileroot`        | string                                      | `"/tmp/areal/"` | Root for logs and checkpoints. Should be available on all nodes. |
| `n_nodes`         | integer                                     | `32`            | The size of the cluster. Used to decide slurm hostname suffix.   |
| `n_gpus_per_node` | integer                                     | `8`             | Number of GPUs per node (physical).                              |

(section-launcher)=

## Launcher Configuration

Configuration for launching the LLM server and trainer processes.

| Parameter                       | Type                                            | Default      | Description                                                                                      |
| ------------------------------- | ----------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `inference_server_cpus_per_gpu` | integer                                         | `4`          | Number of CPUs allocated per GPU for inference server.                                           |
| `inference_server_mem_per_gpu`  | integer                                         | `32768`      | Memory allocated per GPU for inference server in MB.                                             |
| `trainer_cpus_per_gpu`          | integer                                         | `4`          | Number of CPUs allocated per GPU for training.                                                   |
| `trainer_mem_per_gpu`           | integer                                         | `32768`      | Memory allocated per GPU for training in MB.                                                     |
| `inference_server_env_vars`     | string                                          | `""`         | Environment variables for inference server, separated by commas. Example: 'ENV1=val1,ENV2=val2'. |
| `trainer_env_vars`              | string                                          | `""`         | Environment variables for training, separated by commas. Example: 'ENV1=val1,ENV2=val2'.         |
| `slurm`                         | [`SlurmLauncherConfig`](section-slurm-launcher) | **Required** | Slurm launcher configuration.                                                                    |

(section-name-resolve)=

## NameResolve Configuration

Configuration for distributed name resolution and service discovery.

| Parameter         | Type   | Default                     | Description                                                                             |
| ----------------- | ------ | --------------------------- | --------------------------------------------------------------------------------------- |
| `type`            | string | `"nfs"`                     | Type of the distributed KV store for name resolving. **Choices:** `nfs`, `etcd3`, `ray` |
| `nfs_record_root` | string | `"/tmp/areal/name_resolve"` | Record root for NFS name resolving. Should be available on all nodes.                   |
| `etcd3_addr`      | string | `"localhost:2379"`          | Address of the ETCD3 server.                                                            |
| `ray_actor_name`  | string | `"ray_kv_store"`            | Name of the distributed Ray KV store.                                                   |

(section-slurm-launcher)=

## SlurmLauncher Configuration

Configuration for launching the training jobs with Slurm.

| Parameter                | Type           | Default                        | Description                                                       |
| ------------------------ | -------------- | ------------------------------ | ----------------------------------------------------------------- |
| `srun_additional_args`   | string         | `"--mpi=pmi2 -K --chdir $PWD"` | Additional arguments to pass to the srun command.                 |
| `container_type`         | string         | `"apptainer"`                  | Type of containers used in slurm **Choices:** `apptainer`, `none` |
| `mount`                  | string         | `"/storage:/storage"`          | Mount path for slurm.                                             |
| `trainer_image`          | string \| None | `None`                         | slurm image for trainers.                                         |
| `inference_server_image` | string \| None | `None`                         | slurm image for LLM inference.                                    |

(section-evaluator)=

## Evaluator Configuration

Configuration for model evaluation scheduling and timing.

| Parameter         | Type            | Default      | Description                                                    |
| ----------------- | --------------- | ------------ | -------------------------------------------------------------- |
| `experiment_name` | string          | **Required** | -                                                              |
| `trial_name`      | string          | **Required** | -                                                              |
| `fileroot`        | string          | **Required** | -                                                              |
| `freq_epochs`     | integer \| None | `None`       | Trigger frequency in epochs. None disables epoch-based saving. |
| `freq_steps`      | integer \| None | `None`       | Trigger frequency in steps. None disables step-based saving.   |
| `freq_secs`       | integer \| None | `None`       | Trigger frequency in seconds. None disables time-based saving. |

(section-recover)=

## Recover Configuration

Configuration for experiment recovery and fault tolerance.

| Parameter         | Type            | Default      | Description                                                                                                                                                                                                                                                                                                                                                 |
| ----------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_name` | string          | **Required** | -                                                                                                                                                                                                                                                                                                                                                           |
| `trial_name`      | string          | **Required** | -                                                                                                                                                                                                                                                                                                                                                           |
| `fileroot`        | string          | **Required** | -                                                                                                                                                                                                                                                                                                                                                           |
| `freq_epochs`     | integer \| None | `None`       | Trigger frequency in epochs. None disables epoch-based saving.                                                                                                                                                                                                                                                                                              |
| `freq_steps`      | integer \| None | `None`       | Trigger frequency in steps. None disables step-based saving.                                                                                                                                                                                                                                                                                                |
| `freq_secs`       | integer \| None | `None`       | Trigger frequency in seconds. None disables time-based saving.                                                                                                                                                                                                                                                                                              |
| `mode`            | string          | `"disabled"` | Recovery mode for the launcher. Options: 'disabled': Never recover from previous runs. 'auto': Automatically recover from previous runs if recover info and checkpoints are available. 'fault': Only recover from previous runs if the new run fails. 'resume': Force to resume, raise an error if no recover info was found. Never resume if failed again. |
| `retries`         | integer         | `3`          | Number of recovery retries (auto/fault modes only).                                                                                                                                                                                                                                                                                                         |

(section-saver)=

## Saver Configuration

Configuration for model checkpoint saving scheduling and timing.

| Parameter         | Type            | Default      | Description                                                    |
| ----------------- | --------------- | ------------ | -------------------------------------------------------------- |
| `experiment_name` | string          | **Required** | -                                                              |
| `trial_name`      | string          | **Required** | -                                                              |
| `fileroot`        | string          | **Required** | -                                                              |
| `freq_epochs`     | integer \| None | `None`       | Trigger frequency in epochs. None disables epoch-based saving. |
| `freq_steps`      | integer \| None | `None`       | Trigger frequency in steps. None disables step-based saving.   |
| `freq_secs`       | integer \| None | `None`       | Trigger frequency in seconds. None disables time-based saving. |

(section-stats-logger)=

## StatsLogger Configuration

Configuration for experiment statistics logging and tracking services.

| Parameter         | Type                                        | Default      | Description                                            |
| ----------------- | ------------------------------------------- | ------------ | ------------------------------------------------------ |
| `experiment_name` | string                                      | **Required** | -                                                      |
| `trial_name`      | string                                      | **Required** | -                                                      |
| `fileroot`        | string                                      | **Required** | -                                                      |
| `wandb`           | [`WandBConfig`](section-wand-b)             | **Required** | Weights & Biases configuration.                        |
| `swanlab`         | [`SwanlabConfig`](section-swanlab)          | **Required** | SwanLab configuration.                                 |
| `tensorboard`     | [`TensorBoardConfig`](section-tensor-board) | **Required** | TensorBoard configuration. Only 'path' field required. |

(section-swanlab)=

## Swanlab Configuration

Configuration for SwanLab experiment tracking and monitoring.

| Parameter | Type           | Default      | Description |
| --------- | -------------- | ------------ | ----------- |
| `project` | string \| None | `None`       | -           |
| `name`    | string \| None | `None`       | -           |
| `config`  | `Dict` \| None | `None`       | -           |
| `logdir`  | string \| None | `None`       | -           |
| `mode`    | string \| None | `"disabled"` | -           |
| `api_key` | string \| None | `None`       | -           |

(section-tensor-board)=

## TensorBoard Configuration

Configuration for TensorBoard logging and visualization.

| Parameter | Type           | Default | Description |
| --------- | -------------- | ------- | ----------- |
| `path`    | string \| None | `None`  | -           |

(section-wand-b)=

## WandB Configuration

Configuration for Weights & Biases experiment tracking.

| Parameter        | Type                   | Default      | Description |
| ---------------- | ---------------------- | ------------ | ----------- |
| `mode`           | string                 | `"disabled"` | -           |
| `wandb_base_url` | string                 | `""`         | -           |
| `wandb_api_key`  | string                 | `""`         | -           |
| `entity`         | string \| None         | `None`       | -           |
| `project`        | string \| None         | `None`       | -           |
| `name`           | string \| None         | `None`       | -           |
| `job_type`       | string \| None         | `None`       | -           |
| `group`          | string \| None         | `None`       | -           |
| `notes`          | string \| None         | `None`       | -           |
| `tags`           | list of string \| None | `None`       | -           |
| `config`         | `Dict` \| None         | `None`       | -           |
| `id_suffix`      | string \| None         | `"train"`    | -           |

(section-scheduler)=

## Scheduler Configuration

Configuration for worker scheduling. Used in the single-controller mode. Experimental.

| Parameter                     | Type   | Default                             | Description |
| ----------------------------- | ------ | ----------------------------------- | ----------- |
| `endpoint`                    | string | `"http://localhost:8081"`           | -           |
| `deploy_mode`                 | string | `"separation"`                      | -           |
| `functioncall_service_domain` | string | `"http://localhost:8080"`           | -           |
| `reward_functioncall_config`  | `Dict` | **Required**                        | -           |
| `reward_model_path`           | string | `""`                                | -           |
| `reward_model_service_url`    | string | `"http://localhost:30000/classify"` | -           |
