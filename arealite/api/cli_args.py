from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from realhf.api.cli_args import (
    ClusterSpecConfig,
    ExperimentSaveEvalControl,
    GenerationHyperparameters,
    MicroBatchSpec,
)
from realhf.api.cli_args import ModelTrainEvalConfig as EngineConfig
from realhf.api.cli_args import (
    PPOHyperparameters,
    PromptAnswerDatasetConfig,
    PromptOnlyDatasetConfig,
    SGLangConfig,
    TensorBoardConfig,
    WandBConfig,
    vLLMConfig,
)


@dataclass
class LLMServiceConfig:
    server_backend: str
    health_check_interval: int = 5
    startup_timeout: int = 90
    max_unhealth_count: int = 3
    graceful_shutdown_on_unhealthy: bool = True
    sglang: SGLangConfig | None = None
    vllm: vLLMConfig | None = None


@dataclass
class LLMClientConfig:
    server_backend: str


## Agent configurations. ##
@dataclass
class MathCodeSingleStepAgentConfig:
    gconfig: GenerationHyperparameters
    tokenizer_path: str
    success_rate_lb: float
    success_rate_ub: float
    reward_scaling: float
    reward_bias: float


@dataclass
class AgentConfig:
    type: str
    math_code_single_step: Optional[MathCodeSingleStepAgentConfig]


## Environment configurations. ##
@dataclass
class MathCodeSingleStepEnvConfig:
    dataset_path: str


@dataclass
class EnvConfig:
    type: str
    math_code_single_step: Optional[MathCodeSingleStepEnvConfig]


## TrajCollector configurations. ##


@dataclass
class TrajCollectorConfig:
    type: str
    agent: AgentConfig
    env: EnvConfig


## Trainer configurations. ##


@dataclass
class SFTTrainerConfig:
    model: EngineConfig
    mb_spec: MicroBatchSpec
    dataset: PromptAnswerDatasetConfig


@dataclass
class PPOTrainerConfig:
    actor: EngineConfig
    critic: Optional[EngineConfig]
    ref: Optional[EngineConfig]
    rew: Optional[EngineConfig]
    mb_spec: MicroBatchSpec

    dataset: PromptOnlyDatasetConfig

    inf_service: LLMServiceConfig
    collector: Optional[TrajCollectorConfig] = field(
        default_factory=TrajCollectorConfig,
    )

    # Core PPO Parameters
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
    type: str
    ppo: Optional[PPOTrainerConfig]
    sft: Optional[SFTTrainerConfig]


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
