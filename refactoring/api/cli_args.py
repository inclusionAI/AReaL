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
    inf_service: LLMServiceConfig
    math_code_single_step: Optional[MathCodeSingleStepAgentConfig]


## Environment configurations. ##
@dataclass
class MathCodeSingleStepEnvConfig:
    dataset_path: str


@dataclass
class EnvConfig:
    type: str
    math_code_single_step: Optional[MathCodeSingleStepEnvConfig]


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
    actor_train_mb_spec: MicroBatchSpec
    actor_inf_mb_spec: Optional[MicroBatchSpec]
    critic_train_mb_spec: Optional[MicroBatchSpec]
    critic_inf_mb_spec: Optional[MicroBatchSpec]
    rew_inf_mb_spec: Optional[MicroBatchSpec]
    ref_inf_mb_spec: Optional[MicroBatchSpec]
    dataset: PromptOnlyDatasetConfig
    ppo: PPOHyperparameters


@dataclass
class TrainerConfig:
    type: str
    ppo: Optional[PPOTrainerConfig]
    sft: Optional[SFTTrainerConfig]


## TrajCollector configurations. ##


@dataclass
class TrajCollectorConfig:
    type: str
    agent: AgentConfig
    env: EnvConfig


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
    collector: Optional[TrajCollectorConfig] = field(
        default_factory=TrajCollectorConfig,
    )
