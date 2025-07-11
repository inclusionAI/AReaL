import getpass
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

from omegaconf import MISSING

from realhf.base import logging, pkg_version

logger = logging.getLogger("CLI args")

## Data and datasets. ##


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

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
class PromptAnswerDatasetConfig:
    """Configuration for Supervised Fine-Tuning (SFT) datasets.

    Dataset format requirements:
    - JSON/JSONL files
    - Each entry: {"prompt": str, "answer": str}
    """

    train_path: str = field(default="", metadata={"help": "Path to training dataset"})
    valid_path: str = field(default="", metadata={"help": "Path to validation dataset"})
    max_seqlen: int = field(
        default=1024, metadata={"help": "Maximum sequence length (prompt + answer)"}
    )
    train_bs_n_seqs: int = field(
        default=256, metadata={"help": "Training batch size in number of sequences"}
    )
    valid_bs_n_seqs: int = field(
        default=256, metadata={"help": "Validation batch size in number of sequences"}
    )
    fill_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Pad sequences to max length. For testing only - left-fills with non-pad tokens",
        },
    )


@dataclass
class PromptOnlyDatasetConfig:
    """Configuration for PPO RLHF datasets.

    Dataset format requirements:
    - JSON/JSONL files
    - Each entry: {"prompt": str}
    """

    path: str = field(default="", metadata={"help": "Path to dataset"})
    max_prompt_len: int = field(
        default=256, metadata={"help": "Maximum prompt length (truncated if longer)"}
    )
    train_bs_n_seqs: int = field(
        default=256, metadata={"help": "Batch size in number of prompts"}
    )
    fill_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Pad sequences to max length. For testing only - left-fills with non-pad tokens",
        },
    )


## Model, optimizer, and backends. ##


@dataclass(unsafe_hash=True)
class ModelFamily:
    """Identifier for HuggingFace model types (e.g., llama, gpt2).

    Used for model registration and allocation.
    """

    _class: str = field(
        metadata={
            "help": "Model class name (e.g., 'llama'). Must be registered in `register_hf_family`. See "
            "`realhf/api/from_hf` for supported models.",
        }
    )
    is_critic: bool = field(
        default=False,
        metadata={
            "help": "Whether this is a critic/reward model. False indicates a standard LLM",
        },
    )

    def __repr__(self):
        """Returns formatted string representation: '{class}[-critic]'."""
        s = f"{self._class}"
        if self.is_critic:
            s += "-critic"
        return s


@dataclass(unsafe_hash=True)
class ParallelismConfig:
    """Configuration for 3D parallelism (tensor, pipeline, and data parallelism).

    Note:
        Sequence parallelism is only used in combination with tensor-model parallelism.
    """

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Size of tensor-model parallelism"}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "Number of pipeline parallel stages"}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Data parallelism size for ZeRO optimization"}
    )
    use_sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Enable sequence parallelism. Only used with tensor-model parallelism in Megatron",
        },
    )

    def __str__(self):
        """Returns compact string representation: 'Parallel(mp=X,pp=Y,dp=Z)'."""
        return (
            f"Parallel(mp={self.tensor_parallel_size},"
            f"pp={self.pipeline_parallel_size},"
            f"dp={self.data_parallel_size})"
        )

    @staticmethod
    def parallelism_eq(this, other):
        """Compare parallelism configurations (excluding sequence parallelism).

        Note:
            Implemented as static method to avoid OmegaConf compatibility issues.
        """
        return (
            (this.tensor_parallel_size == other.tensor_parallel_size)
            and (this.pipeline_parallel_size == other.pipeline_parallel_size)
            and (this.data_parallel_size == other.data_parallel_size)
        )


@dataclass
class OptimizerConfig:
    """Configuration for model optimization during training.

    Note:
        Set type to "empty" for models that won't be trained.
    """

    type: str = field(
        default="adam",
        metadata={"help": "Optimizer type", "choices": ["adam", "empty"]},
    )
    lr: float = field(default=2e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay"})
    beta1: float = field(default=0.9, metadata={"help": "Adam beta1 parameter"})
    beta2: float = field(default=0.95, metadata={"help": "Adam beta2 parameter"})
    eps: float = field(default=1e-5, metadata={"help": "Adam epsilon parameter"})
    min_lr_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Minimum learning rate ratio after annealing",
        },
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate scheduler type",
            "choices": ["linear", "cosine", "constant"],
        },
    )
    warmup_steps_proportion: float = field(
        default=0.001,
        metadata={
            "help": "Proportion of training steps for warmup",
        },
    )
    offload: bool = field(
        default=False, metadata={"help": "Enable optimizer state offloading"}
    )
    initial_loss_scale: float = field(
        default=2**32, metadata={"help": "Initial loss scaling factor"}
    )
    min_loss_scale: float = field(
        default=1.0, metadata={"help": "Minimum loss scaling factor"}
    )
    loss_scale_window: float = field(
        default=5, metadata={"help": "Window size for loss scaling adjustment"}
    )
    hysteresis: int = field(
        default=2, metadata={"help": "Hysteresis (scaling factor) for loss scaling"}
    )
    gradient_clipping: float = field(
        default=1.0, metadata={"help": "Gradient clipping threshold"}
    )


@dataclass
class vLLMConfig:
    """Configuration for vLLM inference engine. Refer to:
    https://github.com/vllm-project/vllm for detailed documentation.
    """

    max_num_seqs: int = 256
    dtype: str = "float16"
    kv_cache_type: str = "auto"
    num_scheduler_steps: int = 1
    multi_step_stream_outputs: bool = True
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: float = 0
    max_seq_len_to_capture: int = 32768

    disable_sliding_window: bool = True

    # NOTE: Defaults max_model_len to 32k because a larger value
    # will enable chunked prefill in vLLM, which will cause
    # evalution performance degeneration.
    max_model_len: Optional[int] = 32768
    enable_chunked_prefill: bool = False

    # NOTE: Setting enable_prefix_caching to False
    # because it will reuse the block after
    # model weights are updated. Using v0.7.2 reset_prefix_cache
    # will fix this issue.
    enable_prefix_caching: bool = False

    gpu_memory_utilization: float = 0.9

    enforce_eager: bool = False
    hybrid_train: bool = False
    additional_engine_args: Dict = field(default_factory=dict)


@dataclass
class SGLangConfig:
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    disable_cuda_graph: bool = False
    disable_radix_cache: bool = False
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    # NOTE: to avoid the illegal memory access error
    attention_backend: Optional[str] = "flashinfer"
    sampling_backend: Optional[str] = None
    context_length: Optional[int] = 32768
    mem_fraction_static: Optional[float] = 0.9
    max_running_requests: Optional[int] = None
    # NOTE: chunked_prefill_size is by default 8192 on GPUs with 80GB mem in SGLang,
    # but we disable it to avoid precision issues
    chunked_prefill_size: Optional[int] = -1
    max_prefill_tokens: int = 32768
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    dtype: str = "float16"
    kv_cache_dtype: str = "auto"

    # logging
    log_level: str = "warning"
    log_level_http: Optional[str] = "warning"
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = True  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: int = 1

    # Not used.
    hybrid_train: bool = False

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        sglang_config: "SGLangConfig",
        model_path,
        tp_size,
        base_gpu_id,
        dist_init_addr: Optional[str] = None,
        served_model_name: Optional[str] = None,
        skip_tokenizer_init: bool = True,
    ):
        from realhf.base import constants, network, pkg_version, seeding
        from realhf.experiments.common.utils import asdict as conf_as_dict

        args: Dict = conf_as_dict(sglang_config)
        args.pop("hybrid_train")
        args["random_seed"] = seeding.get_seed()

        if served_model_name is None:
            served_model_name = model_path
        host_ip = network.gethostip()
        host = "localhost" if not sglang_config.enable_metrics else host_ip
        args = dict(
            host=host,
            model_path=model_path,
            # Model and tokenizer
            tokenizer_path=model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            device="cuda",
            served_model_name=served_model_name,
            is_embedding=False,
            skip_tokenizer_init=skip_tokenizer_init,
            # Other runtime options
            tp_size=tp_size,
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=base_gpu_id,
            # Data parallelism
            dp_size=1,  # TODO: check whether we require SGLang dp
            load_balance_method="round_robin",
            # Expert parallelism
            ep_size=1,  # TODO: check
            nnodes=1,
            node_rank=0,
            dist_init_addr=dist_init_addr,
            **args,
        )

        if pkg_version.is_version_less("sglang", "0.4.4"):
            args.pop("log_requests_level")
        if pkg_version.is_version_less("sglang", "0.4.3"):
            args.pop("enable_nccl_nvls")
            args.pop("triton_attention_num_kv_splits")
            args.pop("cuda_graph_bs")
            args.pop("enable_memory_saver")
            args.pop("allow_auto_truncate")
            args.pop("file_storage_path")

        flags = []
        for k, v in args.items():
            if v is None or v is False or v == "":
                continue
            if v is True:
                flags.append(f"--{k.replace('_','-')} ")
                continue
            if isinstance(v, list):
                values = " ".join(map(str, v))
                flags.append(f"--{k.replace('_','-')} {values}")
                continue
            flags.append(f"--{k.replace('_','-')} {v}")
        flags = " ".join(flags)
        return f"python3 -m sglang.launch_server {flags}"


@dataclass
class DistributedDataParallelConfig:
    """Configuration for Megatron's DistributedDataParallel.
    Refer to Megatron-LM documentation for details.
    """

    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = True
    check_for_nan_in_grad: bool = False
    bucket_size: Optional[int] = None
    average_in_collective: bool = False
    fp8_param_gather: bool = False


@dataclass
class MegatronConfig:
    """Configuration for Megatron-LM training framework.
    Refer to Megatron-LM documentation for implementation details.
    """

    # Distributed Training Configuration
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


@dataclass
class ModelTrainEvalConfig:
    """Runtime configuration for LLMs in ReaL framework.

    Uses a custom model implementation supporting:
    - 3D and sequence parallelism
    - Flash attention for training/generation
    - Packed 1D tensor inputs for memory efficiency

    Note: Requires manual conversion from HuggingFace models.
    Implemented conversions are in `realhf/api/from_hf/`.
    """

    # Model Architecture Configuration
    type: ModelFamily = field(
        default=ModelFamily("llama", False),
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
        default=True, metadata={"help": "Enable memory-saving gradient checkpointing"}
    )
    bf16: bool = field(
        default=False, metadata={"help": "Use bf16 precision (otherwise fp16)"}
    )

    # Backend-Specific Configurations
    optimizer: Optional[OptimizerConfig] = field(
        default_factory=OptimizerConfig, metadata={"help": "Optimizer configuration"}
    )
    megatron: MegatronConfig = field(
        default_factory=MegatronConfig,
        metadata={
            "help": "Megatron-specific configuration. Can be ignored if this model is not trained."
        },
    )
    vllm: vLLMConfig = field(
        default_factory=vLLMConfig,
        metadata={
            "help": "vLLM inference configuration. Can be ignored if this model doesn't use vLLM."
        },
    )
    sglang: SGLangConfig = field(
        default_factory=SGLangConfig,
        metadata={
            "help": "SGLang runtime configuration.  Can be ignored if this model doesn't use SGLang."
        },
    )


@dataclass
class MFCConfig:
    """Configuration for a single Micro-Function Chain (MFC).

    Contains specifications for micro-batch splitting and parallel execution.

    device_mesh format depends on scope:
    - Multi-node: SLURM nodelist (e.g., 'node[01-02]' or 'node01,node02')
    - Single-node: MPI-style hostfile format (e.g., 'node01:0,1,2,3' for first 4 GPUs)
    Must use 1, 2, 4, or 8 contiguous GPUs on single node
    """

    mb_spec: MicroBatchSpec = field(
        default_factory=MicroBatchSpec,
        metadata={
            "help": "Micro-batch splitting specification",
        },
    )
    parallel: ParallelismConfig = field(
        default_factory=ParallelismConfig,
        metadata={
            "help": "Parallelism strategy.",
        },
    )
    device_mesh: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device mesh specification for manual allocation",
        },
    )


## RL related. ##


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for PPO training."""

    n: int = field(
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

    # Deprecated parameters
    use_cuda_graph: bool = field(
        default=True,
        metadata={"help": "[Deprecated] Whether to use CUDA graph optimization."},
    )
    force_cudagraph_recapture: bool = field(
        default=True,
        metadata={"help": "[Deprecated] Force CUDA graph recapture to release memory."},
    )
    force_no_logits_mask: bool = field(
        default=True,
        metadata={
            "help": "[Deprecated] Disable logits masking (reduces stability but saves memory)."
        },
    )

    def __post_init__(self):
        if self.temperature == 0.0:
            self.greedy = True
            self.temperature = 1.0
        if self.top_p <= 0.0 or self.top_p > 1:
            raise ValueError("top_p must be in (0.0, 1.0].")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.use_cuda_graph and pkg_version.is_version_less("torch", "2.3.0"):
            raise ValueError(
                f"To use CUDAGraph, ReaL's PyTorch version should be at least 2.3.0."
            )

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)


@dataclass
class PPOHyperparameters:
    """Configuration for Proximal Policy Optimization (PPO) training parameters."""

    # Generation Configuration
    gen: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters,
        metadata={"help": "Text generation hyperparameters"},
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


## Experiment utilities. ##


@dataclass
class ExperimentSaveEvalControl:
    """Controls the frequency of model saving and evaluation during training.

    Manages independent counters for epochs, steps, and seconds. The model will be saved
    or evaluated when any specified frequency condition is met.

    Note:
        - Epoch: Number of full passes through the training dataset
        - Step: Number of individual training iterations
        - Seconds: Wall-clock time duration
    """

    total_train_epochs: int = field(
        default=1, metadata={"help": "Total number of epochs to train the model."}
    )
    # Save control
    save_freq_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Save frequency in epochs. None disables epoch-based saving."
        },
    )
    save_freq_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Save frequency in steps. None disables step-based saving."},
    )
    save_freq_secs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Save frequency in seconds. None disables time-based saving."
        },
    )
    # Checkpointing control
    ckpt_freq_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Checkpoint frequency in epochs. None uses save_freq_epochs. "
            "Checkpointing is used for recover. Previous checkpoint is overwritten to save space."
        },
    )
    ckpt_freq_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Checkpoint frequency in steps. None disables step-based checkpointing."
        },
    )
    ckpt_freq_secs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Checkpoint frequency in seconds. None disables time-based checkpointing."
        },
    )
    # Evaluation control
    eval_freq_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Evaluation frequency in epochs. None disables epoch-based evaluation."
        },
    )
    eval_freq_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Evaluation frequency in steps. None disables step-based evaluation."
        },
    )
    eval_freq_secs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Evaluation frequency in seconds. None disables time-based evaluation."
        },
    )
    # Benchmark control
    benchmark_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Terminate training after this number of steps. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    benchmark_n_seqs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Terminate training after consuming this number of samples. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )


@dataclass
class AutomaticEvaluator:
    """Configuration for automatic model evaluation during training.

    Controls how and when evaluation jobs are launched to assess model performance
    on specified datasets.
    """

    data_names: str = field(
        default="aime24",
        metadata={
            "help": "Comma-separated dataset names for evaluation. "
            "Supported datasets: 'aime24', 'amc23', 'math_500'."
        },
    )
    max_gen_tokens: int = field(
        default=32768,
        metadata={"help": "Maximum number of tokens to generate during evaluation."},
    )
    max_concurrent_jobs: int = field(
        default=3,
        metadata={
            "help": "Maximum number of concurrent evaluation jobs. "
            "New jobs wait when this limit is reached."
        },
    )
    eval_job_image: Optional[str] = field(
        default=None,
        metadata={
            "help": "Container image for evaluation jobs. "
            "None uses the training GPU image."
        },
    )
    initial_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Initial checkpoint to evaluate. "
            "Results stored as global_step=0 if specified."
        },
    )
    prompt_type: str = field(
        default="deepscaler",
        metadata={"help": "Prompt format to use during evaluation."},
    )


@dataclass
class WandBConfig:
    mode: str = "disabled"
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    job_type: Optional[str] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict] = None


@dataclass
class SwanlabConfig:
    project: Optional[str] = None
    name: Optional[str] = None
    config: Optional[Dict] = None
    logdir: Optional[str] = None
    mode: Optional[str] = "local"
    api_key: Optional[str] = os.getenv("SWANLAB_API_KEY", None)


@dataclass
class TensorBoardConfig:
    path: Optional[str] = None


def get_user_tmp():
    user = getpass.getuser()
    user_tmp = os.path.join("/home", user, ".cache", "realhf")
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


@dataclass
class NameResolveConfig:
    type: str = field(
        default="nfs",
        metadata={
            "help": "Type of the distributed KV store for name resolving.",
            "choices": ["nfs", "etcd3", "ray"],
        },
    )
    nfs_record_root: str = field(
        default="/tmp/areal/name_resolve",
        metadata={
            "help": "Record root for NFS name resolving. Should be available in all nodes."
        },
    )
    etcd3_addr: str = field(
        default="localhost:2379", metadata={"help": "Address of the ETCD3 server."}
    )
    ray_actor_name: str = field(
        default="ray_kv_store",
        metadata={"help": "Name of the distributed Ray KV store."},
    )


@dataclass
class ClusterSpecConfig:
    config_path: str = field(
        default="",
        metadata={
            "help": "JSON config path. If not given, use the following CLI args."
        },
    )
    name_resolve: NameResolveConfig = field(
        default_factory=NameResolveConfig,
        metadata={"help": "Name resolving configuration."},
    )
    cluster_name: str = field(
        default="local",
        metadata={"help": "Name of the cluster. Used to set specific environs."},
    )
    fileroot: str = field(
        default=get_user_tmp(),
        metadata={
            "help": "Root for logs and checkpoints. Should be available to all nodes."
        },
    )
    gpu_type: str = field(
        default="tesla", metadata={"help": "GPU type of the cluster. Used by slurm."}
    )
    mount: str = field(
        default="/storage:/storage", metadata={"help": "Mount path for slurm."}
    )
    gpu_image: str = field(default="", metadata={"help": "slurm image for trainers."})
    cpu_image: str = field(default="", metadata={"help": "slurm image for CPU jobs."})
    gpu_infer_image: str = field(
        default="", metadata={"help": "slurm image for LLM inference."}
    )
    node_name_prefix: str = field(
        default="slurmd-", metadata={"help": "Node prefix for a slurm cluster."}
    )
    n_nodes: int = field(
        default=32,
        metadata={
            "help": "The size of the cluster. Used to decide slurm hostname suffix."
        },
    )
    n_gpus_per_node: int = field(
        default=8,
        metadata={"help": "GPUs per node (physically)."},
    )


@dataclass
class BaseExperimentConfig:
    """Configuration for quickstart experiments.

    All parameters can be modified via command line arguments. Supports various
    recovery modes and parallelization strategies.

    Note:
        - Recovery modes: auto, fault, resume, disabled
        - Allocation modes: manual, heuristic, or pattern-based
    """

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
    debug: bool = field(
        default=True,
        metadata={
            "help": "Debug mode. False disables assertions for better performance."
        },
    )
    metric_discovery_port: int = field(
        default=0,
        metadata={"help": "Discovery port for prometheus metrics service discovery."},
    )
    partition: str = field(
        default="dev", metadata={"help": "SLURM partition for running the experiment."}
    )
    schedule_strategy: str = field(
        default="empty_first", metadata={"help": "Resource scheduling strategy."}
    )
    wandb: WandBConfig = field(
        default_factory=WandBConfig,
        metadata={"help": "Weights & Biases configuration."},
    )
    swanlab: SwanlabConfig = field(
        default_factory=SwanlabConfig,
        metadata={"help": "SwanLab configuration."},
    )
    tensorboard: TensorBoardConfig = field(
        default_factory=TensorBoardConfig,
        metadata={"help": "TensorBoard configuration. Only 'path' field required."},
    )
    recover_mode: str = field(
        default="disabled",
        metadata={
            "help": "Recovery mode (auto/fault/resume/disabled). "
            "Use 'disabled' if unfamiliar with recovery mechanism."
        },
    )
    recover_retries: int = field(
        default=1,
        metadata={"help": "Number of recovery retries (auto/fault modes only)."},
    )
    recover_after: int = field(
        default=10,
        metadata={"help": "Recovery interval in seconds (auto/fault modes only)."},
    )
    ignore_worker_error: bool = field(
        default=False,
        metadata={
            "help": "Ignore worker runtime errors (disabled mode only). "
            "Only enable if certain errors can be safely ignored."
        },
    )
    allocation_mode: str = field(
        default="",
        metadata={
            "help": "GPU parallel strategy allocation mode. "
            "Options: manual/heuristic or pattern-based."
        },
    )
    n_nodes: int = field(
        default=1, metadata={"help": "Number of nodes for experiment."}
    )
    n_gpus_per_node: int = field(
        default=8, metadata={"help": "Number of GPUs per node for this experiment."}
    )
    nodelist: Optional[str] = field(
        default=None,
        metadata={
            "help": "SLURM nodelist for manual allocation. "
            "Format: 'slurmd-01:0,1,2,3' or 'slurmd-[01-02,03,07],COM08'."
        },
    )
    exclude: Optional[str] = field(
        default=None,
        metadata={
            "help": "SLURM nodelist to exclude from allocation. "
            "Format: 'slurmd-01:0,1,2,3' or 'slurmd-[01-02,03,07],COM08'."
        },
    )
    seed: int = field(default=1, metadata={"help": "Random seed for reproducibility."})
    cache_clear_freq: Optional[int] = field(
        default=10,
        metadata={
            "help": "Clear data transfer cache every N steps. "
            "Set lower if OOM occurs. None disables clearing."
        },
    )
    exp_ctrl: ExperimentSaveEvalControl = field(
        default_factory=ExperimentSaveEvalControl,
        metadata={"help": "Experiment save/evaluation control configuration."},
    )
    torch_cache_mysophobia: bool = field(
        default=True,
        metadata={
            "help": "Clear torch cache before each RPC (~0.1s overhead per RPC)."
        },
    )
    auto_eval: bool = field(
        default=False,
        metadata={
            "help": "Enable automatic evaluation during training. "
            "Results logged to disk and WandB or Swanlab(if active)."
        },
    )
    auto_eval_config: AutomaticEvaluator = field(
        default_factory=AutomaticEvaluator,
        metadata={"help": "Automatic evaluation configuration."},
    )
    cpus_per_master_worker: int = field(
        default=4, metadata={"help": "CPU cores per master worker."}
    )
    mem_per_master_worker: int = field(
        default=20000, metadata={"help": "Memory per master worker (MB)."}
    )
    cpus_per_model_worker: int = field(
        default=4, metadata={"help": "CPU cores per model worker."}
    )
    mem_per_model_worker: int = field(
        default=90000, metadata={"help": "Memory per model worker (MB)."}
    )
    shuffle_dataset: bool = field(
        default=True, metadata={"help": "Shuffle in each epoch."}
    )
    ray_temp_path: str = field(
        default="/tmp/ray", metadata={"help": "Absolute path for Ray's log."}
    )
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification. Mainly used by slurm."},
    )


## Configuration options of asynchronous experiments. ##


@dataclass
class AsyncRLOptions:
    schedule_policy: str = field(
        default="round_robin",
        metadata={
            "help": "The request schedule policy during generation. Available options: [round_robin]."
        },
    )
    new_tokens_per_chunk: int = field(
        default=int(1e10),
        metadata={
            "help": "The length of chunked generation. Only valid if inference can't be interrupted."
        },
    )
    max_head_offpolicyness: int = field(
        default=0,
        metadata={"help": "Maximum off-policyness tolerance for the first token."},
    )

    n_rollout_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of rollout workers. None defaults to train world size."
        },
    )
    max_concurrent_rollouts: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max concurrent rollouts globally. Defaults to train batch size."
        },
    )
    flush_request_timeout: int = field(
        default=300,
        metadata={"help": "The timeout of flushing requests upon weight update."},
    )

    cpus_per_generation_server: int = field(
        default=4, metadata={"help": "Generation server CPUs."}
    )
    mem_per_generation_server: int = field(
        default=60 * 1024, metadata={"help": "Generation server CPU memory in MB."}
    )
    cpus_per_gserver_manager: int = field(
        default=4, metadata={"help": "Generation manager CPUs."}
    )
    mem_per_gserver_manager: int = field(
        default=10 * 1024, metadata={"help": "Generation manager CPU memory in MB."}
    )
    cpus_per_rollout_worker: int = field(
        default=4, metadata={"help": "Rollout worker CPUs."}
    )
    mem_per_rollout_worker: int = field(
        default=20 * 1024, metadata={"help": "Rollout worker CPU memory in MB."}
    )


## Configurations for practical experiments. ##


@dataclass
class NullPPOExperimentOptions:
    """Configuration for a null PPO experiment (testing purposes only)."""

    model: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Model configuration for testing."},
    )
    inf: MFCConfig = field(
        default_factory=MFCConfig,
        metadata={"help": "Inference model function call configuration."},
    )
    train: MFCConfig = field(
        default_factory=MFCConfig,
        metadata={"help": "Training model function call configuration."},
    )
    dataset: PromptOnlyDatasetConfig = field(
        default_factory=PromptOnlyDatasetConfig,
        metadata={"help": "Dataset configuration for testing."},
    )
    dataset_filter_threshold: float = field(
        default=0.2,
        metadata={"help": "Threshold value for dataset filtering in tests."},
    )
    dataset_max_filter_percentage: float = field(
        default=0.1,
        metadata={"help": "Maximum percentage of dataset to filter in tests."},
    )


@dataclass
class SFTExperimentOptions:
    """Configuration for supervised fine-tuning (SFT) experiments."""

    model: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Model runtime configuration."},
    )
    allocation: MFCConfig = field(
        default_factory=MFCConfig,
        metadata={"help": "Device allocation and parallelism configuration."},
    )
    dataset: PromptAnswerDatasetConfig = field(
        default_factory=PromptAnswerDatasetConfig,
        metadata={"help": "Dataset configuration."},
    )


@dataclass
class PPOMATHExperimentOptions:
    """Configuration for PPO (Proximal Policy Optimization) experiments.

    Manages four distinct models and their interactions through model function calls.

    Note:
        Models:
        - Actor: Primary LLM for text generation
        - Critic: Value function estimator
        - Ref: Reference model for KL regularization
        - Rew: Reward model (or function) for reward signals
    """

    # Model configurations
    actor: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Primary LLM configuration."},
    )
    critic: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Critic model configuration."},
    )
    ref: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Reference model configuration."},
    )
    rew: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Reward model configuration."},
    )

    # Model function call configurations
    actor_train: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "TrainActor MFC configuration."}
    )
    critic_train: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "TrainCritic MFC configuration."}
    )
    actor_gen: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "Rollout MFC configuration."}
    )
    critic_inf: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "InfValues MFC configuration."}
    )
    rew_inf: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "InfReward MFC configuration."}
    )
    ref_inf: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "InfRef MFC configuration."}
    )
    actor_inf: MFCConfig = field(
        default_factory=MFCConfig,
        metadata={"help": "Actor inference MFC configuration."},
    )

    # Dataset and algorithm configurations
    dataset: PromptOnlyDatasetConfig = field(
        default_factory=PromptOnlyDatasetConfig,
        metadata={"help": "Dataset configuration."},
    )
    ppo: PPOHyperparameters = field(
        default_factory=PPOHyperparameters,
        metadata={"help": "PPO algorithm hyperparameters."},
    )

    # Sampling and reward processing
    group_size: int = field(
        default=1,
        metadata={"help": "Number of answers retained per prompt (best-of-n)."},
    )
    generation_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of answers sampled per prompt. None uses group_size."
        },
    )
    mask_no_eos_with_zero: bool = field(
        default=False,
        metadata={"help": "Mask reward for truncated answers (no EOS token)."},
    )
    mask_too_long: bool = field(
        default=False, metadata={"help": "Mask PPO loss for length-truncated answers."}
    )
    check_verifier_status: bool = field(
        default=False,
        metadata={"help": "Raise error if reward is all-zero (verifier bug check)."},
    )
    group_adv_norm: bool = field(
        default=False, metadata={"help": "Use grouped advantage normalization in GRPO."}
    )
    ref_ema_eta: Optional[float] = field(
        default=None,
        metadata={
            "help": "EMA decay rate for reference model updates. 1.0 means full update."
        },
    )
    rw_type: Optional[str] = field(
        default="sparse",
        metadata={
            "help": "Type of reward processing. Only `sparse` is valid for now.",
            "choices": ["sparse"],
        },
    )
    check_xml_format: bool = field(
        default=False, metadata={"help": "Validate XML format in generated responses."}
    )

    # Dataset filtering
    dataset_filter_threshold: float = field(
        default=100.0,
        metadata={
            "help": "Rewards higher than this value will be filtered out after each epoch's training."
        },
    )
    dataset_max_filter_percentage: float = field(
        default=0.0, metadata={"help": "Maximum percentage of dataset to each filter."}
    )

    success_rate_ub: float = field(
        default=1.0,
        metadata={
            "help": "Success rate higher than this value will be filtered out after generation. Valid for async training."
        },
    )
    success_rate_lb: float = field(
        default=0.0,
        metadata={
            "help": "Success rate lower than this value will be filtered out after generation. Valid for async training."
        },
    )

    # testing only
    no_training: bool = field(
        default=False,
        metadata={"help": "Run without training. Test-only."},
    )


@dataclass
class MathCodeEvalOptions:
    gen_config: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )

    actor: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Primary LLM configuration."},
    )
    rew: ModelTrainEvalConfig = field(
        default_factory=ModelTrainEvalConfig,
        metadata={"help": "Reward model configuration."},
    )

    actor_gen: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "Rollout MFC configuration."}
    )
    rew_inf: MFCConfig = field(
        default_factory=MFCConfig, metadata={"help": "InfReward MFC configuration."}
    )

    dataset: PromptOnlyDatasetConfig = field(
        default_factory=PromptOnlyDatasetConfig,
        metadata={"help": "Dataset configuration."},
    )

    group_size: int = field(
        default=1,
        metadata={"help": "Number of answers retained per prompt (best-of-n)."},
    )
    rw_type: Optional[str] = field(
        default="sparse",
        metadata={
            "help": "Type of reward processing. Only `sparse` is valid for now.",
            "choices": ["sparse"],
        },
    )
    check_xml_format: bool = field(
        default=False, metadata={"help": "Validate XML format in generated responses."}
    )

    check_verifier_status: bool = field(
        default=False,
        metadata={"help": "Raise error if reward is all-zero (verifier bug check)."},
    )


## A helper function to visualize the helper messages. ##
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.panel import Panel
from rich.rule import Rule
from rich.theme import Theme

# Custom theme for colors
help_theme = Theme(
    {
        "title": "bold cyan",
        "header": "bold green",
        "field": "bold yellow",
        "type": "italic blue",
        "help": "dim white",
        "default": "dim green",
        "example": "italic cyan",
        "border": "dim blue",
    }
)

console = Console(theme=help_theme)


class CliHighlighter(RegexHighlighter):
    base_style = "example."
    highlights = [r"(python -m .+?)(?=\s|$)"]


highlighter = CliHighlighter()


def print_config_help(
    config, prefix: str = "", parent_name: str = "", indent: int = 0
) -> None:
    """Prints help for a structured config with proper indentation and smart default display"""
    if not is_dataclass(config):
        return

    for field in fields(config):
        value = getattr(config, field.name)
        full_name = f"{parent_name}.{field.name}" if parent_name else field.name
        indent_space = "    " * indent

        # Field type handling
        type_name = (
            field.type.__name__ if isinstance(field.type, Type) else str(field.type)
        )

        # Create help text components
        help_parts = []
        if "help" in field.metadata:
            help_parts.append(field.metadata["help"])

        # Only show default for leaf nodes (non-dataclass fields)
        if not is_dataclass(value):
            default_value = field.default if hasattr(field, "default") else MISSING
            help_parts.append(f"[default]Default: {default_value}[/default]")

        # Print the field info
        console.print(f"{indent_space}[field]{full_name}[/field]", end=" ")
        console.print(f"[type]({type_name})[/type]", end=" ")
        if help_parts:
            console.print("- " + " ".join(help_parts))
        else:
            console.print()

        # Handle nested dataclasses with increased indentation
        if is_dataclass(value):
            print_config_help(value, prefix, full_name, indent + 1)


def print_config_values(
    config,
    prefix: str = "",
    parent_name: str = "",
    indent: int = 0,
    show_types: bool = True,
) -> None:
    """Prints current values with clean indentation and subtle separation"""
    console.print()  # Add space before

    top_rule = Rule("Current Configuration Begin", style="bold cyan", align="center")
    bottom_rule = Rule("Current Configuration End", style="bold cyan", align="center")

    # Title with subtle underline
    console.print(top_rule)

    # Print config directly to main console
    _print_config_values_internal(
        console, config, prefix, parent_name, indent, show_types
    )

    # Closing rule
    console.print(bottom_rule)
    console.print()  # Add space after


def _print_config_values_internal(
    console: Console,
    config,
    prefix: str,
    parent_name: str,
    indent: int,
    show_types: bool,
) -> None:
    """Internal recursive function that does the actual printing"""
    if not is_dataclass(config):
        return

    for field in fields(config):
        value = getattr(config, field.name)
        full_name = f"{parent_name}.{field.name}" if parent_name else field.name
        indent_space = "    " * indent

        # Field type handling
        type_name = (
            field.type.__name__ if isinstance(field.type, Type) else str(field.type)
        )

        # Create help text components
        help_parts = []

        # Print the field info
        console.print(f"{indent_space}[field]{full_name}[/field]", end=" ")
        if show_types:
            console.print(f"[type]({type_name})[/type]", end=" ")

        # Always show current value
        value_str = str(value)
        if isinstance(value, (list, dict)):
            value_str = f"{type(value).__name__}(len={len(value)})"
        if not is_dataclass(value):
            help_parts.append(f"[value]{value_str}[/value]")

        if help_parts:
            console.print("- " + " ".join(help_parts))
        else:
            console.print()

        # Handle nested dataclasses
        if is_dataclass(value):
            _print_config_values_internal(
                console, value, prefix, full_name, indent + 1, show_types
            )


def print_runtime_helper(args):
    """Print comprehensive help with rich formatting"""

    exp_type = args.__class__.__name__
    # Main help panel
    console.print(
        Panel.fit(
            f"[header]Setting {exp_type} with the Following Values[/header]",
            border_style="border",
        ),
        justify="center",
    )

    # Configuration options section
    print_config_values(args)
