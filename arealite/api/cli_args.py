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
from realhf.api.cli_args import ModelTrainEvalConfig as EngineConfig
from realhf.api.cli_args import (
    OptimizerConfig,
    ParallelismConfig,
    PromptAnswerDatasetConfig,
    PromptOnlyDatasetConfig,
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
    served_model_name: Optional[str] = None
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
    sglang: Optional[SGLangConfig] = None
    vllm: Optional[vLLMConfig] = None


@dataclass
class LLMClientConfig:
    server_backend: str = field(
        default="sglang",
        metadata={"help": "Backend for client", "choices": ["sglang", "vllm"]},
    )
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


## Training backend configs. ##


@dataclass
class FSDPConfig:
    sync_module_states: bool = True
    use_orig_params: bool = False


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
    backend: str = field(
        default="megatron",
        metadata={"help": "Training backend", "choices": ["megatron"]},
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    bf16: bool = field(default=False, metadata={"help": "Use bf16 precision"})

    # Backend-Specific Configurations
    optimizer: Optional[OptimizerConfig] = field(
        default_factory=OptimizerConfig, metadata={"help": "Optimizer configuration"}
    )
    fsdp: Optional[FSDPConfig] = None
    vllm: Optional[vLLMConfig] = None
    sglang: Optional[SGLangConfig] = None


## Agent configurations. ##
@dataclass
class MathCodeSingleStepAgentConfig:
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    tokenizer_path: str = field(default="", metadata={"help": "Path to tokenizer"})


@dataclass
class AgentConfig:
    type: str = field(default="", metadata={"help": "Agent type"})
    math_code_single_step: Optional[MathCodeSingleStepAgentConfig] = None


## Environment configurations. ##
@dataclass
class MathCodeSingleStepEnvConfig:
    dataset_path: str = field(default="", metadata={"help": "Path to dataset"})


@dataclass
class EnvConfig:
    type: str = field(default="", metadata={"help": "Environment type"})
    reward_scaling: float = field(
        default=1.0, metadata={"help": "Reward scaling factor"}
    )
    reward_bias: float = field(default=0.0, metadata={"help": "Reward bias"})
    math_code_single_step: Optional[MathCodeSingleStepEnvConfig] = None


## TrajCollector configurations. ##


@dataclass
class TrajCollectorConfig:
    type: str = field(default="llm", metadata={"help": "Trajectory collector type"})
    llm_client: LLMClientConfig = field(default_factory=LLMClientConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)


## Trainer configurations. ##


@dataclass
class SFTTrainerConfig:
    model: EngineConfig = field(default_factory=EngineConfig)
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)
    dataset: PromptAnswerDatasetConfig = field(
        default_factory=PromptAnswerDatasetConfig
    )


@dataclass
class PPOTrainerConfig:
    actor: EngineConfig = field(default_factory=EngineConfig)
    critic: Optional[EngineConfig] = None
    ref: Optional[EngineConfig] = None
    rew: Optional[EngineConfig] = None
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)

    dataset: PromptOnlyDatasetConfig = field(default_factory=PromptOnlyDatasetConfig)

    collector: Optional[TrajCollectorConfig] = field(
        default_factory=TrajCollectorConfig,
    )

    # Core PPO Parameters
    group_size: int = field(
        default=16,
        metadata={"help": "Number of trajectories to sample for each prompt."},
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
    value_eps_clip: float = field(
        default=0.2, metadata={"help": "Clipping factor for value updates"}
    )
    early_stop_imp_ratio: float = field(
        default=5.0, metadata={"help": "Early stop threshold for importance ratio"}
    )
    actor_sample_reuse: int = field(
        default=1, metadata={"help": "The data reuse (aka PPO epoch) for actor."}
    )
    critic_sample_reuse: int = field(
        default=1, metadata={"help": "The data reuse (aka PPO epoch) for critic."}
    )

    # Reward Processing
    max_reward_clip: float = field(
        default=20.0, metadata={"help": "Maximum absolute value for clipped rewards"}
    )
    reward_output_scaling: float = field(
        default=1.0, metadata={"help": "Scaling factor for reward model outputs"}
    )
    reward_output_bias: float = field(
        default=0.0, metadata={"help": "Bias term for reward model outputs"}
    )
    fuse_rew_ref: bool = field(
        default=True,
        metadata={"help": "Whether to fuse reward and reference model computations"},
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
    use_adaptive_kl_ctl: bool = field(
        default=False, metadata={"help": "Use adaptive KL coefficient control"}
    )

    # Value Function Configuration
    disable_value: bool = field(
        default=False, metadata={"help": "Disable value/critic model"}
    )
    value_norm: bool = field(
        default=True, metadata={"help": "Enable value normalization"}
    )
    value_norm_type: str = field(
        default="exp",
        metadata={"help": "Type of value normalization", "choices": ["exp", "ma"]},
    )
    value_norm_beta: float = field(
        default=0.99995,
        metadata={"help": "Decay factor for exponential moving average"},
    )
    value_norm_eps: float = field(
        default=1e-5, metadata={"help": "Epsilon term for numerical stability"}
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
        default="ppo", metadata={"help": "Trainer type", "choices": ["ppo", "sft"]}
    )
    ppo: Optional[PPOTrainerConfig] = None
    sft: Optional[SFTTrainerConfig] = None


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
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification. Mainly used by slurm."},
    )
    trainer: TrainerConfig = field(
        default_factory=TrainerConfig,
    )
