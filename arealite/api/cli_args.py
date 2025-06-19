from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from realhf.api.cli_args import (
    ClusterSpecConfig,
    ExperimentSaveEvalControl,
    GenerationHyperparameters,
    MicroBatchSpec,
    ModelFamily,
)
from realhf.api.cli_args import (
    OptimizerConfig,
    ParallelismConfig,
    SGLangConfig,
    TensorBoardConfig,
    WandBConfig,
    vLLMConfig,
)


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
    cluster: ClusterSpecConfig = field(default_factory=ClusterSpecConfig)
    server_backend: str = field(
        default="sglang",
        metadata={"help": "Backend for serving", "choices": ["sglang", "vllm"]},
    )
    model_path: str = field(default="", metadata={"help": "Path to model"})
    parallel: ParallelismConfig = field(default_factory=ParallelismConfig)
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
    vllm: Optional[vLLMConfig] = field(
        default=None, metadata={"help": "vLLM configuration (if using vLLM backend)"}
    )


@dataclass
class LLMClientConfig:
    server_backend: str = field(
        default="sglang",
        metadata={"help": "Backend for client", "choices": ["sglang", "vllm"]},
    )
    schedule_policy: str = "round_robin"
    tokenizer_path: str = field(default="", metadata={"help": "Path to tokenizer"})
    gen_timeout: int = field(
        default=1800, metadata={"help": "Generation timeout in seconds"}
    )
    update_weights_timeout: int = field(
        default=300, metadata={"help": "Weight update timeout in seconds"}
    )
    update_weights_retries: int = field(
        default=3, metadata={"help": "Number of weight update retries"}
    )


## Dataset configs. ##
@dataclass
class DatasetConfig:
    path: str = ""
    name: Optional[str] = None
    split: Optional[str] = None
    data_files: Optional[str] = None
    batch_size: int = field(default=1, metadata={"help": "Training batch size"})
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the dataset"}
    )
    pin_memory: bool = False
    num_workers: int = 0


## Training backend configs. ##


@dataclass
class FSDPConfig:
    sync_module_states: bool = field(
        default=True, metadata={"help": "Synchronize module states across processes"}
    )
    use_orig_params: bool = field(
        default=False,
        metadata={"help": "Use original parameters instead of flattened ones"},
    )


@dataclass
class EngineBackendConfig:
    type: str = field(
        default="fsdp",
        metadata={"help": "Training backend", "choices": ["fsdp"]},
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
    )


## Agent configurations. ##


@dataclass
class MathCodeSingleStepConfig:
    solution_path: str = field(default="", metadata={"help": "Path to solutions"})


@dataclass
class AgenticWorkflowConfig:
    type: str = "default"
    math_code_single_step: Optional[MathCodeSingleStepConfig] = field(
        default_factory=MathCodeSingleStepConfig
    )


## RolloutController configurations. ##


@dataclass
class RolloutControllerConfig:
    workflow: Optional[AgenticWorkflowConfig] = field(
        default_factory=AgenticWorkflowConfig,
        metadata={
            "help": "Agentic workflow configuration. If None, degenerate to the RLVR pipeline."
        },
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
    llm_client: LLMClientConfig = field(default_factory=LLMClientConfig)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )


## Trainer configurations. ##


@dataclass
class SFTTrainerConfig:
    model: EngineConfig = field(default_factory=EngineConfig)
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)


@dataclass
class PPOTrainerConfig:
    async_training: bool = True
    actor: EngineConfig = field(default_factory=EngineConfig)
    ref: Optional[EngineConfig] = field(
        default=None, metadata={"help": "Reference model configuration"}
    )
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)

    # Core PPO/GRPO Parameters
    group_size: int = field(
        default=16,
        metadata={"help": "Number of trajectories to sample for each prompt."},
    )
    group_adv_norm: bool = True
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
    group_reward_norm: bool = False
    reward_scaling: float = field(
        default=1.0, metadata={"help": "Reward scaling factor"}
    )
    reward_bias: float = field(default=0.0, metadata={"help": "Reward bias"})
    mask_no_eos_with_zero: bool = False

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
    max_reward_clip: float = 100.0
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
    rollout: Optional[RolloutControllerConfig] = None
    trainer: Optional[TrainerConfig] = field(
        default=None, metadata={"help": "Trainer configuration"}
    )
