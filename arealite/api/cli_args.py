# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from realhf.api.cli_args import (
    ClusterSpecConfig,
    ExperimentSaveEvalControl,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
    ParallelismConfig,
    SGLangConfig,
    TensorBoardConfig,
    WandBConfig,
)


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for PPO training."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding (max probability)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )
    top_k: int = field(
        default=int(1e8),
        metadata={"help": "Number of highest probability tokens to consider."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )
    stop_token_ids: List[int] = field(
        default_factory=list,
        metadata={"help": "Stop generation when encoutering these token ids."},
    )

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)


## Inference config for clients and servers. ##
@dataclass
class LLMServiceConfig:
    experiment_name: str = field(
        default=MISSING, metadata={"help": "Name of the experiment. Required."}
    )
    trial_name: str = field(
        default=MISSING, metadata={"help": "Name of the trial. Required."}
    )
    served_model_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the served model"}
    )
    seed: int = field(default=1, metadata={"help": "Random seed"})
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification configuration"},
    )
    server_backend: str = field(
        default="sglang",
        metadata={"help": "Backend for serving", "choices": ["sglang", "vllm"]},
    )
    model_path: str = field(default="", metadata={"help": "Path to model"})
    parallel: ParallelismConfig = field(
        default_factory=ParallelismConfig,
        metadata={"help": "Model parallelism configuration"},
    )
    health_check_interval: int = field(
        default=5, metadata={"help": "Health check interval in seconds"}
    )
    startup_timeout: int = field(
        default=90, metadata={"help": "Startup timeout in seconds"}
    )
    max_unhealth_count: int = field(
        default=3, metadata={"help": "Max unhealthy count before restart"}
    )
    graceful_shutdown_on_unhealthy: bool = field(
        default=True, metadata={"help": "Enable graceful shutdown when unhealthy"}
    )
    sglang: Optional[SGLangConfig] = field(
        default=None,
        metadata={"help": "SGLang configuration (if using SGLang backend)"},
    )


@dataclass
class LLMClientConfig:
    server_backend: str = field(
        default="sglang",
        metadata={"help": "Backend for client", "choices": ["sglang"]},
    )
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    tokenizer_path: str = field(default="", metadata={"help": "Path to tokenizer"})
    request_timeout: int = field(
        default=3600, metadata={"help": "Request timeout in seconds"}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for each request"}
    )


## Dataset configs. ##
@dataclass
class DatasetConfig:
    path: str = field(
        default="", metadata={"help": "Path or HuggingFace identifier to the dataset"}
    )
    name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name (for HuggingFace datasets)"}
    )
    split: Optional[str] = field(
        default=None, metadata={"help": "Dataset split to use (e.g., 'train', 'test')"}
    )
    data_files: Optional[str] = field(
        default=None, metadata={"help": "Specific data files to load"}
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size of the dataloader"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the dataset"}
    )
    pin_memory: bool = field(
        default=False,
        metadata={
            "help": "Pin memory for faster data loading (set True for GPU training)"
        },
    )
    num_workers: int = field(
        default=0, metadata={"help": "Number of worker processes for data loading"}
    )


## Training backend configs. ##


@dataclass
class FSDPWrapPolicy:
    transformer_layer_cls_to_wrap: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of transformer layer names for FSDP to wrap."},
    )


@dataclass
class FSDPConfig:
    wrap_policy: Optional[FSDPWrapPolicy] = field(
        default=None,
        metadata={"help": "FSDP wrap policy, specifying model layers to wrap."},
    )
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )


@dataclass
class EngineBackendConfig:
    type: str = field(
        default="hf",
        metadata={"help": "Training backend", "choices": ["fsdp", "hf"]},
    )
    fsdp: Optional[FSDPConfig] = field(
        default=None, metadata={"help": "FSDP configuration (if using FSDP backend)"}
    )


@dataclass
class EngineConfig:
    # Model Architecture Configuration
    type: ModelFamily = field(
        default_factory=lambda: ModelFamily("llama", False),
        metadata={"help": "Model family specification"},
    )
    path: str = field(default="", metadata={"help": "Path to HuggingFace checkpoint"})
    init_from_scratch: bool = field(
        default=False, metadata={"help": "Initialize model weights randomly"}
    )
    init_critic_from_actor: bool = field(
        default=False,
        metadata={"help": "Initialize critic/reward model from LM checkpoint"},
    )

    # Training Backend Configuration
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    bf16: bool = field(default=False, metadata={"help": "Use bf16 precision"})
    optimizer: Optional[OptimizerConfig] = field(
        default=None, metadata={"help": "Optimizer configuration"}
    )
    backend: EngineBackendConfig = field(
        default_factory=EngineBackendConfig,
        metadata={"help": "Training backend configuration"},
    )


## Agent configurations. ##


@dataclass
class MathCodeSingleStepConfig:
    solution_path: str = field(default="", metadata={"help": "Path to solutions"})


@dataclass
class RLVRConfig:
    reward_type: str = "math"
    solution_path: str = field(default="", metadata={"help": "Path to solutions"})


@dataclass
class RolloutWorkflowConfig:
    type: str = field(
        default="rlvr",
        metadata={
            "help": "Rollout workflow type",
            "choices": ["rlvr", "math_code_single_step"],
        },
    )
    rlvr: Optional[RLVRConfig] = field(
        default=None,
        metadata={"help": "The configuration for the single-step math/code workflow"},
    )
    math_code_single_step: Optional[MathCodeSingleStepConfig] = field(
        default=None,
        metadata={"help": "The configuration for the single-step math/code workflow"},
    )


## RolloutController configurations. ##


@dataclass
class RolloutControllerConfig:
    workflow: RolloutWorkflowConfig = field(
        default_factory=RolloutWorkflowConfig,
        metadata={"help": "Rollout workflow configuration."},
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of rollout worker processes"}
    )
    max_concurrent_rollouts: int = field(
        default=1, metadata={"help": "Maximum number of concurrent rollouts"}
    )
    max_head_offpolicyness: int = field(
        default=0,
        metadata={"help": "Maximum off-policyness tolerance for the first token"},
    )
    filter_reward_lb: float = field(
        default=-float("inf"), metadata={"help": "Lower bound for reward filtering"}
    )
    filter_reward_ub: float = field(
        default=float("inf"), metadata={"help": "Upper bound for reward filtering"}
    )
    llm_client: LLMClientConfig = field(
        default_factory=LLMClientConfig,
        metadata={"help": "LLM client configuration for rollouts"},
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters,
        metadata={"help": "Generation hyperparameters for rollouts"},
    )


## Trainer configurations. ##


@dataclass
class SFTTrainerConfig:
    model: EngineConfig = field(
        default_factory=EngineConfig,
        metadata={"help": "Model configuration for SFT training"},
    )
    mb_spec: MicroBatchSpec = field(
        default_factory=MicroBatchSpec,
        metadata={"help": "Micro-batch specification for SFT training"},
    )


@dataclass
class PPOTrainerConfig:
    async_training: bool = field(
        default=True, metadata={"help": "Enable asynchronous training mode"}
    )
    actor: EngineConfig = field(
        default_factory=EngineConfig,
        metadata={"help": "Actor model configuration for PPO training"},
    )
    ref: Optional[EngineConfig] = field(
        default=None, metadata={"help": "Reference model configuration"}
    )
    mb_spec: MicroBatchSpec = field(
        default_factory=MicroBatchSpec,
        metadata={"help": "Micro-batch specification for PPO training"},
    )

    # Core PPO/GRPO Parameters
    group_adv_norm: bool = field(
        default=True,
        metadata={
            "help": "Normalize advantages within each prompt group rather than globally"
        },
    )
    ppo_n_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches for each PPO update"}
    )
    eps_clip: float = field(
        default=0.2, metadata={"help": "Clipping factor for policy ratio"}
    )
    c_clip: Optional[float] = field(
        default=None,
        metadata={
            "help": "Dual clipping factor for policy ratio, must > 1.0. None disables dual clipping."
        },
    )
    actor_sample_reuse: int = field(
        default=1, metadata={"help": "The data reuse (aka PPO epoch) for actor."}
    )

    # Reward
    group_reward_norm: bool = field(
        default=False,
        metadata={
            "help": "Normalize final reward of each sequence (GRPO-style) to reduce length bias"
        },
    )
    reward_scaling: float = field(
        default=1.0, metadata={"help": "Reward scaling factor"}
    )
    reward_bias: float = field(default=0.0, metadata={"help": "Reward bias"})
    mask_no_eos_with_zero: bool = field(
        default=False,
        metadata={
            "help": "Mask truncated generations (no EOS token) and exclude from training"
        },
    )

    # Advantage Estimation
    discount: float = field(
        default=1.0, metadata={"help": "Discount factor for future rewards"}
    )
    gae_lambda: float = field(
        default=1.0, metadata={"help": "Lambda parameter for GAE"}
    )
    adv_norm: bool = field(
        default=True, metadata={"help": "Enable advantage normalization"}
    )

    # KL Control
    kl_ctl: float = field(default=0.1, metadata={"help": "KL divergence coefficient"})

    # Reward clipping
    max_reward_clip: float = field(
        default=20.0, metadata={"help": "Maximum absolute value for reward clipping"}
    )
    recompute_logprob: bool = field(
        default=False,
        metadata={"help": "Recompute logp and replace the logp returned by inference."},
    )
    use_decoupled_loss: bool = field(
        default=False,
        metadata={"help": "Use the decoupled loss. recompute_logprob must be True."},
    )
    behav_imp_weight_cap: Optional[float] = field(
        default=None,
        metadata={
            "help": "We filter out the tokens where behav_imp_weight exceeds behav_imp_weight_cap when computing the loss, must be > 1.0, use_decoupled_loss must be true"
        },
    )


@dataclass
class TrainerConfig:
    type: str = field(
        default="ppo",
        metadata={"help": "Trainer type", "choices": ["ppo", "sft", "null"]},
    )
    ppo: Optional[PPOTrainerConfig] = field(
        default=None, metadata={"help": "PPO trainer configuration (if using PPO)"}
    )
    sft: Optional[SFTTrainerConfig] = field(
        default=None, metadata={"help": "SFT trainer configuration (if using SFT)"}
    )


## Entrypoint. ##


@dataclass
class TrainingArgs:
    experiment_name: str = field(
        default=MISSING,
        metadata={"help": "Name of the experiment (no '_' or '/'). Required."},
    )
    trial_name: str = field(
        default=MISSING,
        metadata={"help": "Name of the trial (no '-' or '/'). Required."},
    )
    mode: str = field(
        default="slurm",
        metadata={
            "help": "Experiment launching mode.",
            "choices": ["slurm", "local", "ray"],
        },
    )
    wandb: WandBConfig = field(
        default_factory=WandBConfig,
        metadata={"help": "Weights & Biases configuration."},
    )
    tensorboard: TensorBoardConfig = field(
        default_factory=TensorBoardConfig,
        metadata={"help": "TensorBoard configuration. Only 'path' field required."},
    )
    allocation_mode: str = field(
        default="",
        metadata={
            "help": "GPU parallel strategy allocation mode. "
            "Options: manual/heuristic or pattern-based."
        },
    )
    ray_temp_path: str = field(
        default="/tmp/ray", metadata={"help": "Absolute path for Ray's log."}
    )
    n_nodes: int = field(
        default=1, metadata={"help": "Number of nodes for experiment."}
    )
    n_gpus_per_node: int = field(
        default=8, metadata={"help": "Number of GPUs per node for this experiment."}
    )
    seed: int = field(default=1, metadata={"help": "Random seed for reproducibility."})
    exp_ctrl: ExperimentSaveEvalControl = field(
        default_factory=ExperimentSaveEvalControl,
        metadata={"help": "Experiment save/evaluation control configuration."},
    )
    min_required_servers: int = field(
        default=0,
        metadata={"help": "The minimum LLM server instances for RL training."},
    )
    shutdown_server_on_exit: bool = field(
        default=False,
        metadata={"help": "Whether to shut down the LLM generation server on exit."},
    )
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification. Mainly used by slurm."},
    )

    # RL workflow configuration
    train_dataset: DatasetConfig = field(
        default_factory=DatasetConfig, metadata={"help": "Train dataset configuration"}
    )
    valid_dataset: Optional[DatasetConfig] = field(
        default=None, metadata={"help": "Validation dataset configuration"}
    )
    rollout: Optional[RolloutControllerConfig] = field(
        default_factory=RolloutControllerConfig,
        metadata={"help": "Rollout controller configuration for RL training"},
    )
    trainer: Optional[TrainerConfig] = field(
        default=None, metadata={"help": "Trainer configuration"}
    )
