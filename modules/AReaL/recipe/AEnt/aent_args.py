from dataclasses import dataclass, field

from areal.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    PPOActorConfig,
)


@dataclass
class AEntConfig:
    # clamped entropy regularization configs
    entropy_coeff: float = field(
        default=0.001, metadata={"help": "Entropy regularization coefficient."}
    )
    entropy_clamp: float = field(
        default=0.2,
        metadata={"help": "Token space clamping percentage in entropy regularization."},
    )

    adaptive_coeff: bool = field(
        default=False,
        metadata={"help": "Whether to use an adaptive entropy coefficient."},
    )
    # following params are disabled if adaptive_coeff==False
    entropy_high: float = field(
        default=0.5,
        metadata={"help": "Coeff will decrease if entropy is above this value."},
    )
    entropy_low: float = field(
        default=0.1,
        metadata={"help": "Coeff will increase if entropy drops below this value."},
    )
    coeff_lr: float = field(
        default=0.001, metadata={"help": "Step size for coefficient update."}
    )
    coeff_box_high: float = field(
        default=0.01,
        metadata={"help": "The coeff will be bounded smaller than this value."},
    )
    coeff_box_low: float = field(
        default=1e-5,
        metadata={"help": "The coeff will be bounded larger than this value."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Coeff update will start after this step."}
    )


# PPOActorConfig + AEnt regularization config
@dataclass
class AEntPPOActorConfig(PPOActorConfig):
    aent: AEntConfig = field(default_factory=AEntConfig)


# GRPO Config + AEnt regularization config
@dataclass
class AEntGRPOConfig(BaseExperimentConfig):
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    actor: AEntPPOActorConfig = field(default_factory=AEntPPOActorConfig)
    ref: PPOActorConfig = field(default_factory=PPOActorConfig)
