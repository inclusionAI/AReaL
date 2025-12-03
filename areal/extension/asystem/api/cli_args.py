from dataclasses import dataclass, field

from omegaconf import MISSING

from areal.api.cli_args import (
    BaseExperimentConfig,
    EvaluatorConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    SaverConfig,
    SchedulerConfig,
)
from areal.api.cli_args import TrainEngineConfig as BaseTrainEngineConfig


@dataclass
class RemoteHybridInferenceConfig(InferenceEngineConfig):
    model_path: str = field(
        default=MISSING,
        metadata={"help": "model path"},
    )
    storage_path: str = field(
        default=MISSING,
        metadata={"help": "storage path"},
    )
    random_seed: int = field(
        default=0,
        metadata={"help": "random seed"},
    )
    engine_config: dict = field(default_factory=dict)
    dp_size: int = field(
        default=1,
        metadata={"help": "dp size"},
    )
    pp_size: int = field(
        default=1,
        metadata={"help": "pp size"},
    )
    tp_size: int = field(
        default=1,
        metadata={"help": "tp size"},
    )
    seed: int = field(
        default=1,
        metadata={"help": "seed"},
    )
    batch_requests: bool = field(
        default=False,
        metadata={"help": "batch requests"},
    )


@dataclass
class AdvantageNormalizationConfig:
    mean_level: str = field(
        default="batch", metadata={"choices": ["none", "batch", "group"]}
    )
    std_level: str = field(
        default="batch", metadata={"choices": ["none", "batch", "group"]}
    )

@dataclass
class RemoteMegatronWrapPolicy:
    n_minibatches: int = 1
    kl_ctl: float = 0.0
    adv_norm: AdvantageNormalizationConfig = field(
        default_factory=AdvantageNormalizationConfig,
        metadata={"help": "Advantage normalization configuration"},
    )
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    c_clip: float | None = None
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    disable_value: bool = True
    early_stop_kl: float | None = None
    early_stop_imp_ratio: float | None = None
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: float | None = 6
    adaptive_kl_horizon: float | None = 10000
    enable_save: bool = True
    value_norm: bool = True
    value_norm_type: str = field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    group_size: int = 8
    generation_size: int | None = None
    mask_no_eos_with_zero: bool = False
    mask_too_long: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True
    token_normalize_scope: str = field(
        default="global", metadata={"choices": ["global", "dp"]}
    )
    sample_reuse: int = 1
    temperature: float = 1.0  # GenerationHyperparameters
    reward_output_scaling: float = field(
        default=1.0, metadata={"help": "Reward scaling factor"}
    )
    reward_output_bias: float = field(default=0.0, metadata={"help": "Reward bias"})
    recompute_logp: bool = False


@dataclass
class RemoteMegatronEngineConfig:
    wrap_policy: RemoteMegatronWrapPolicy | None = field(
        default_factory=RemoteMegatronWrapPolicy,
        metadata={"help": "RemoteMegatron wrap policy."},
    )
    remote_megatron_config: dict = field(default_factory=dict)
    loss_configs: dict = field(default_factory=dict)
    recover_dir: str = field(default="")

    @staticmethod
    def assign_wrap_policy(policy_dict: dict) -> RemoteMegatronWrapPolicy:
        """Assign values from dictionary to RemoteMegatronWrapPolicy fields.

        Args:
            policy_dict: Dictionary containing wrap policy configuration

        Returns:
            Configured RemoteMegatronWrapPolicy instance
        """
        policy = RemoteMegatronWrapPolicy()
        for field_name, field_value in policy_dict.items():
            if hasattr(policy, field_name):
                setattr(policy, field_name, field_value)
        return policy

    experiment_name: str = field(
        default="test-exp",
        metadata={"help": "Name of the experiment (no '_' or '/'). Required."},
    )
    trial_name: str = field(
        default="test-trial",
        metadata={"help": "Name of the trial (no '-' or '/'). Required."},
    )
    group_size: int = field(
        default=8,
        metadata={"help": "Number of answers retained per prompt (best-of-n)."},
    )
    train_bs_n_seqs: int = field(
        default=32, metadata={"help": "Training batch size in number of sequences"}
    )
    n_mbs: int = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
        },
    )
    max_tokens_per_mb: int = field(
        default=16384,
        metadata={
            "help": "Maximum tokens per micro-batch. When set, n_mbs becomes the minimum number of micro-batches",
        },
    )
    global_step: int = field(
        default=0,
        metadata={
            "help": "global step for recover",
        },
    )


@dataclass
class TrainEngineConfig(BaseTrainEngineConfig):
    hybrid_engine: RemoteMegatronEngineConfig = field(
        default_factory=RemoteMegatronEngineConfig
    )


@dataclass
class RecoverConfig:
    experiment_name: str = field(default="default-experiment")
    trial_name: str = field(default="trial0")
    fileroot: str = field(default="")
    recover_meta_info_path: str = field(default="")
    enable_recover: bool = field(default=False)
    latest_disable_save_hf: bool = field(
        default=True, metadata={"help": "Disable saving latest huggingFace"}
    )
    periodic_disable_save_hf: bool = field(
        default=False, metadata={"help": "Disable saving periodic huggingFace"}
    )
    periodic_save_interval: int | None = field(
        default=None, metadata={"help": "Periodic save steps"}
    )
    latest_save_interval: int | None = field(
        default=None, metadata={"help": "Latest save steps"}
    )


@dataclass
class BaseExperimentConfigExtension(BaseExperimentConfig):
    enable_colocate_mode: bool = field(
        default=False, metadata={"help": "Enable colocate mode."}
    )
    storage_prefix: str = field(
        default="", metadata={"help": "Storage prefix for colocate mode."}
    )
    weight_update_type: str = field(default="nccl", metadata={"help": "nccl/disk"})

    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig, metadata={"help": "Scheduler config."}
    )

    # 删除 saver 属性
    saver: SaverConfig | None = field(
        default=None, metadata={"help": "Saver configuration (disabled in ASystem)"}
    )

    # 删除 evaluator 属性
    evaluator: EvaluatorConfig | None = field(
        default=None, metadata={"help": "Evaluator configuration (disabled in ASystem)"}
    )


@dataclass
class GRPOConfig(BaseExperimentConfigExtension):
    async_training: bool = field(
        default=True,
        metadata={
            "help": "Enable asynchronous training between rollout and policy update."
        },
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: RemoteHybridInferenceConfig = field(
        default_factory=RemoteHybridInferenceConfig
    )
    actor: TrainEngineConfig = field(default_factory=TrainEngineConfig)
    ref: TrainEngineConfig = field(default_factory=TrainEngineConfig)
    recover: RecoverConfig = field(default_factory=RecoverConfig)
