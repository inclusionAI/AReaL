from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict
from omegaconf import MISSING

from realhf.api.cli_args import OptimizerConfig


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when "
                    "max_tokens_per_mb is None or as minimum count",
        },
    )
    max_tokens_per_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum tokens per micro-batch. When set, n_mbs becomes the minimum number of micro-batches",
        },
    )

    @classmethod
    def new(cls, mb_spec: "MicroBatchSpec", **kwargs):
        """Create new spec with updated fields while maintaining Omegaconf compatibility."""
        fields = dict(
            n_mbs=mb_spec.n_mbs,
            max_tokens_per_mb=mb_spec.max_tokens_per_mb,
        )
        fields.update(kwargs)
        return cls(**fields)


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for RL training."""

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


# Train Engine Configs
@dataclass
class TrainEngineConfig:
    path: str = field(default="", metadata={"help": "Path to HuggingFace checkpoint"})
    init_from_scratch: bool = field(
        default=False, metadata={"help": "Initialize model weights randomly"}
    )
    init_critic_from_actor: bool = field(
        default=False,
        metadata={"help": "Initialize critic/reward model from LM checkpoint"},
    )

    # Training Backend Configuration
    pad_mbs_to_max_tokens: bool = field(
        default=True,
        metadata={
            "help": "Pad micro-batches to configured max tokens per micro-batch"
            "when running train_batch/forward/eval_batch."
        },
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    bf16: bool = field(default=False, metadata={"help": "Use bf16 precision"})
    optimizer: Optional[OptimizerConfig] = field(
        default=None, metadata={"help": "Optimizer configuration"}
    )


# FSDP
@dataclass
class FSDPWrapPolicy:
    transformer_layer_cls_to_wrap: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of transformer layer names for FSDP to wrap."},
    )


@dataclass
class FSDPEngineConfig(TrainEngineConfig):
    wrap_policy: Optional[FSDPWrapPolicy] = field(
        default=None,
        metadata={"help": "FSDP wrap policy, specifying model layers to wrap."},
    )
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )


# remote megatron
@dataclass
class RemoteMegatronWrapPolicy:
    n_minibatches: int = 4
    kl_ctl: float = 0.1
    adv_norm: bool = True
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    c_clip: Optional[float] = None
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    disable_value: bool = False
    early_stop_kl: Optional[float] = None
    early_stop_imp_ratio: Optional[float] = None
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000
    enable_save: bool = True
    value_norm: bool = False
    value_norm_type: str = field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    group_size: int = 1
    generation_size: Optional[int] = None
    mask_no_eos_with_zero: bool = False
    group_adv_norm: bool = False
    mask_too_long: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True
    token_normalize_scope: str = field(default="global", metadata={"choices": ["global", "dp"]})
    sample_reuse: int = 1
    temperature: float = 1.0  # GenerationHyperparameters


@dataclass
class RemoteMegatronEngineConfig(TrainEngineConfig):
    wrap_policy: Optional[RemoteMegatronWrapPolicy] = field(
        default_factory=RemoteMegatronWrapPolicy,
        metadata={"help": "RemoteMegatron wrap policy."},
    )

    remote_megatron_config: Dict = field(default_factory=dict)
    loss_configs: Dict = field(default_factory=dict)
    experiment_name: str = field(
        default="test-exp",
        metadata={"help": "Name of the experiment (no '_' or '/'). Required."},
    )
    trial_name: str = field(
        default="test-trial",
        metadata={"help": "Name of the trial (no '-' or '/'). Required."},
    )
    group_size: int = field(
        default=1,
        metadata={"help": "Number of answers retained per prompt (best-of-n)."},
    )
    train_bs_n_seqs: int = field(
        default=256, metadata={"help": "Training batch size in number of sequences"}
    )
    n_mbs: int = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
        },
    )
    max_tokens_per_mb: int = field(
        default=int(1e12),
        metadata={
            "help": "Maximum tokens per micro-batch. When set, n_mbs becomes the minimum number of micro-batches",
        },
    )

@dataclass
class InferenceEngineConfig:
    # Used by remote inference engines.
    server_addrs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of server addresses for inference."},
    )
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    request_timeout: float = field(
        default=30.0, metadata={"help": "Timeout for HTTP requests."}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for failed requests."}
    )


@dataclass
class SGLangEngineConfig:
    pass
