from areal.api.cli_args import *


@dataclass
class DistributedDataParallelConfig:
    """Configuration for Megatron's DistributedDataParallel.
    Refer to Megatron-LM documentation for details.
    """

    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = True
    check_for_nan_in_grad: bool = False
    bucket_size: Optional[int] = None
    average_in_collective: bool = False
    fp8_param_gather: bool = False


@dataclass
class MegatronEngineConfig:
    """Configuration for Megatron-LM training framework.
    Refer to Megatron-LM documentation for implementation details.
    """

    # Distributed Training Configuration
    wrap_with_ddp: bool = True
    use_torch_fsdp2: bool = False  # TODO: pending test
    use_custom_fsdp: bool = False  # TODO: pending test
    ddp: DistributedDataParallelConfig = field(
        default_factory=DistributedDataParallelConfig
    )
    # Don't use MegatronOptimizerConfig here because OmegaConf
    # does not recognize the annotation "torch.dtype"
    overlap_param_gather_with_optimizer_step: bool = False

    # Precision Configuration
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: str = "float32"
    main_params_dtype: str = "float32"
    exp_avg_dtype: str = "float32"
    exp_avg_sq_dtype: str = "float32"

    # Checkpointing Configuration
    async_save: bool = False
    use_checkpoint_opt_param_scheduler: bool = True


@dataclass
class ExperimentalTrainEngineConfig(TrainEngineConfig):
    megatron: MegatronEngineConfig = field(default_factory=MegatronEngineConfig)


@dataclass
class ExperimentalPPOActorConfig(PPOActorConfig, ExperimentalTrainEngineConfig):
    pass


@dataclass
class ExperimentalSFTConfig(BaseExperimentConfig):
    model: ExperimentalTrainEngineConfig = field(
        default_factory=ExperimentalTrainEngineConfig
    )


@dataclass
class ExperimentalGRPOConfig(BaseExperimentConfig):
    async_training: bool = field(default=True)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    actor: ExperimentalPPOActorConfig = field(
        default_factory=ExperimentalPPOActorConfig
    )
    ref: ExperimentalPPOActorConfig = field(default_factory=ExperimentalPPOActorConfig)
