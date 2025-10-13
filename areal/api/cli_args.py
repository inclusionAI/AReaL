import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import uvloop
import yaml

from areal.utils.pkg_version import is_version_less

uvloop.install()
from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from hydra.core.global_hydra import GlobalHydra
from omegaconf import MISSING, DictConfig, OmegaConf

from areal.platforms import current_platform
from areal.utils import name_resolve, pkg_version


@dataclass
class NormConfig:
    """Configuration for reward/advantage normalization."""

    mean_level: str | None = field(
        default="batch",
        metadata={
            "help": "Mean level for normalization. None for no mean normalization.",
            "choices": ["batch", "group", None],
        },
    )
    mean_leave1out: bool = field(
        default=False,
        metadata={"help": "Whether to use leave-one-out average."},
    )
    std_level: str | None = field(
        default="batch",
        metadata={
            "help": "Standard deviation level for normalization. None for no std normalization.",
            "choices": ["batch", "group", None],
        },
    )
    std_unbiased: bool = field(
        default=True,
        metadata={
            "help": "Whether to use unbiased standard deviation computation. Defaults to True (changed from False in v0.3.4)."
        },
    )
    eps: float = field(
        default=1e-5,
        metadata={
            "help": "The eps when dividing by standard deviation to avoid numerical issues."
        },
    )
    group_size: int = field(
        default=1, metadata={"help": "Group size for group-level normalization"}
    )


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int | None = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
        },
    )
    granularity: int = field(
        default=1,
        metadata={
            "help": "Granularity of each micro-batch. Adjacent sequences are grouped by this size when dividing microbatches.",
        },
    )
    max_tokens_per_mb: int | None = field(
        default=None,
        metadata={
            "help": "Maximum tokens per micro-batch for each forward pass. When set, n_mbs becomes the minimum number of micro-batches.",
        },
    )

    @classmethod
    def new(cls, mb_spec: "MicroBatchSpec", **kwargs):
        """Create new spec with updated fields while maintaining Omegaconf compatibility."""
        fields = dict(
            n_mbs=mb_spec.n_mbs,
            granularity=mb_spec.granularity,
            max_tokens_per_mb=mb_spec.max_tokens_per_mb,
        )
        fields.update(kwargs)
        return cls(**fields)


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for rollout."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    max_tokens: int = field(
        default=65536,
        metadata={
            "help": "Maximum number of tokens including prompt and generated tokens."
        },
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
        metadata={"help": "Stop generation when encountering these token IDs."},
    )
    stop: List[str] | None = field(
        default=None,
        metadata={
            "help": "One or multiple stop words. Generation will stop if one of these words is sampled."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Penalizes tokens based on their frequency in generation so far. "
                "Must be between -2 and 2 where negative numbers encourage repetition."
            )
        },
    )

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)


# Train Engine Configs


@dataclass
class OptimizerConfig:
    """Configuration for model optimization during training."""

    type: str = field(
        default="adam",
        metadata={
            "help": "Optimizer type. Adam_bf16 currently only supported FSDP Engine.",
            "choices": ["adam", "sgd", "adam_bf16"],
        },
    )
    lr: float = field(default=2e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay"})
    beta1: float = field(
        default=0.9,
        metadata={
            "help": "Adam beta1 parameter. Only effective when optimizer_type is adam/adam_bf16"
        },
    )
    beta2: float = field(
        default=0.95,
        metadata={
            "help": "Adam beta2 parameter. Only effective when optimizer_type is adam/adam_bf16"
        },
    )
    eps: float = field(
        default=1e-5,
        metadata={
            "help": "Adam epsilon parameter. Only effective when optimizer_type is adam/adam_bf16"
        },
    )
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
class FSDPWrapPolicy:
    """Policy configuration for FSDP model layer wrapping. None defaults to wrapping transformer decoder layers defined by transformers."""

    transformer_layer_cls_to_wrap: List[str] | None = field(
        default=None,
        metadata={"help": "A list of transformer layer names for FSDP to wrap."},
    )


@dataclass
class FSDPEngineConfig:
    """Configuration for Fully Sharded Data Parallel (FSDP) training backend."""

    wrap_policy: FSDPWrapPolicy | None = field(
        default=None,
        metadata={"help": "FSDP wrap policy, specifying model layers to wrap."},
    )
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )


@dataclass
class TrainEngineConfig:
    """Core configuration for model training, including optimization and backend settings."""

    experiment_name: str = MISSING
    trial_name: str = MISSING
    path: str = field(default="", metadata={"help": "Path to HuggingFace checkpoint"})
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": "Attention implementation for huggingface transformers model.",
            "choices": ["flash_attention_2"],
        },
    )
    init_from_scratch: bool = field(
        default=False, metadata={"help": "Initialize model weights randomly"}
    )
    is_critic: bool = field(
        default=False,
        metadata={"help": "Whether to use a critic/reward model"},
    )
    # Runtime microbatch limit
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)
    pad_to_maximum: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad each microbatch to the length upper bound specified by mb_spec. "
                "Can reduce memory fragmentation but slows down training."
            )
        },
    )

    # Training Backend Configuration
    disable_dropout: bool = field(
        default=False, metadata={"help": "Disable dropout layers during training"}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    dtype: str = field(default="bfloat16", metadata={"help": "Parameter data type."})
    grad_reduce_dtype: str = field(
        default="float32", metadata={"help": "Gradient reduction data type."}
    )
    optimizer: OptimizerConfig | None = field(
        default=None,
        metadata={"help": "Optimizer configuration. None means no training."},
    )

    weight_update_mode: str = field(default="disk")
    backend: str = field(
        default="", metadata={"help": "Training backend (refer to documentation)"}
    )
    fsdp: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)

    # Lora
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use LoRA. Only support FSDP. Note that should be enabled together with vLLM/SGLang."
        },
    )
    lora_rank: int = field(default=32, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=16, metadata={"help": "lora alpha"})
    target_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "lora target_modules."},
    )
    peft_type: str = field(
        default="lora",
        metadata={"help": "peft method type. Only LoRA is supported for now."},
    )


@dataclass
class PPOActorConfig(TrainEngineConfig):
    """Configuration for PPO actor model, a subclass of a TrainEngine."""

    # Core PPO/GRPO Parameters
    group_size: int = field(
        default=1, metadata={"help": "Number of sequences in each group"}
    )
    ppo_n_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches for each PPO update"}
    )
    eps_clip: float = field(
        default=0.2, metadata={"help": "Clipping factor for policy ratio"}
    )
    eps_clip_higher: float | None = field(
        default=None,
        metadata={
            "help": "Clipping factor (higher value) for policy ratio. Default is None. When eps_clip_higher is set (decoupled), eps_clip will be used as the lower value."
        },
    )
    c_clip: float | None = field(
        default=None,
        metadata={
            "help": "Dual clipping factor for policy ratio, must be > 1.0. None disables dual clipping."
        },
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Temperature during generation."}
    )
    # Reward
    reward_norm: NormConfig | None = field(
        default=None,
        metadata={"help": "Normalization configuration for rewards"},
    )
    reward_scaling: float = field(
        default=1.0, metadata={"help": "Reward scaling factor"}
    )
    reward_bias: float = field(default=0.0, metadata={"help": "Reward bias"})
    reward_clip: float = field(
        default=20.0, metadata={"help": "Maximum absolute value for reward clipping"}
    )
    overlong_reward_penalty: bool = field(
        default=False,
        metadata={"help": "Penalty for overlong sequences. Used within DAPO."},
    )
    overlong_tokens: int | None = field(
        default=None,
        metadata={"help": "Number of tokens in the tail that will receive a penalty"},
    )
    overlong_penalty_factor: float | None = field(
        default=None,
        metadata={"help": "Penalty factor for tokens in the tail"},
    )
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
    adv_norm: NormConfig | None = field(
        default=None, metadata={"help": "Normalization configuration for advantages."}
    )

    # KL Control
    kl_ctl: float = field(default=0.1, metadata={"help": "KL divergence coefficient"})
    kl_estimator: str = field(
        default="k1",
        metadata={"help": "KL divergence estimator", "choices": ["k1", "k2", "k3"]},
    )

    # Asynchronous RL
    recompute_logprob: bool = field(
        default=False,
        metadata={
            "help": "Recompute log probability and replace the log probability returned by inference."
        },
    )
    use_decoupled_loss: bool = field(
        default=False,
        metadata={
            "help": "Use the decoupled loss. Implicitly enables recompute_logprob."
        },
    )
    behav_imp_weight_cap: float | None = field(
        default=None,
        metadata={
            "help": "Filter out tokens where behav_imp_weight exceeds behav_imp_weight_cap when computing loss. Must be > 1.0. use_decoupled_loss must be true."
        },
    )
    # Advanced Options
    dynamic_sampling: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic sampling (within DAPO). If enabled, groups with the same reward will be masked out. "
            "Note that enabling this option will lead to variable batch sizes. If you want to use a constant batch size with dynamic filtering, "
            "you should use the `should_accept` parameter in `rollout_batch` and `prepare_batch`."
        },
    )

    # Logging Agent Trajectories
    log_agent_stats: bool = field(
        default=False,
        metadata={"help": "Log statistics for agent trajectories"},
    )
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys for logging agent trajectory statistics"},
    )
    # Others
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate"},
    )


@dataclass
class PPOCriticConfig(TrainEngineConfig):
    """Configuration for PPO critic model, a subclass of a TrainEngine."""

    ppo_n_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches for each PPO update"}
    )
    eps_clip: float = field(
        default=0.5, metadata={"help": "Clipping factor for value loss"}
    )
    mask_no_eos_with_zero: bool = field(
        default=False,
        metadata={
            "help": "Mask truncated generations (no EOS token) and exclude from training"
        },
    )


@dataclass
class vLLMConfig:
    """Configuration for vLLM runtime. Refer to:
    https://docs.vllm.ai/en/stable/api/index.html for detailed documentation.
    """

    model: str = ""
    seed: int = 1
    skip_tokenizer_init: bool = False
    enforce_eager: bool = True
    dtype: str = "bfloat16"
    distributed_executor_backend: str = "mp"
    # original
    max_num_seqs: int = 256
    # kv_cache_type: str = "auto"
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: float = 0
    max_seq_len_to_capture: int = 32768
    disable_sliding_window: bool = True
    # NOTE: Defaults max_model_len to 32k because a larger value
    # will enable chunked prefill in vLLM, which will cause
    # evalution performance degeneration.
    max_model_len: int | None = 32768
    enable_chunked_prefill: bool = False
    # NOTE: Setting enable_prefix_caching to False
    # because it will reuse the block after
    # model weights are updated. Using v0.7.2 reset_prefix_cache
    # will fix this issue.
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.9
    worker_extension_cls: str = (
        "areal.thirdparty.vllm.vllm_worker_extension.VLLMWorkerExtension"
    )
    enable_sleep_mode: bool = False
    uvicorn_log_level: str = "warning"

    @staticmethod
    def build_args(
        vllm_config: "vLLMConfig",
        tp_size,
        host,
        port,
        dist_init_addr: str | None = None,
    ):
        args: Dict = conf_as_dict(vllm_config)
        args = dict(
            host=host,
            port=port,
            # Model and tokenizer
            tokenizer=vllm_config.model,
            load_format="auto",
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            **args,
        )
        return args

    @staticmethod
    def build_cmd(
        vllm_config: "vLLMConfig",
        tp_size,
        host,
        port,
        dist_init_addr: str | None = None,
    ):
        args = vLLMConfig.build_args(
            vllm_config=vllm_config,
            tp_size=tp_size,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
        )
        # convert to flags
        flags = []
        for k, v in args.items():
            if v is None or v is False or v == "":
                continue
            if v is True:
                flags.append(f"--{k.replace('_','-')}")
            elif isinstance(v, list):
                flags.append(f"--{k.replace('_','-')} {' '.join(map(str, v))}")
            else:
                flags.append(f"--{k.replace('_','-')} {v}")
        return f"python3 -m areal.thirdparty.vllm.areal_vllm_server {' '.join(flags)}"


@dataclass
class SGLangConfig:
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    model_path: str = ""
    random_seed: int = 1
    skip_tokenizer_init: bool = False
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = True
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
    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: List[int] | None = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    attention_backend: str | None = "fa3"
    enable_multimodal: bool = False
    sampling_backend: str | None = None
    context_length: int | None = 32768
    mem_fraction_static: float | None = 0.9
    max_running_requests: int | None = None
    # NOTE: chunked_prefill_size is by default 8192 on GPUs with 80GB mem in SGLang,
    # but we disable it to avoid precision issues
    chunked_prefill_size: int | None = -1
    max_prefill_tokens: int = 32768
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    dp_size: int = 1  # only used for dp attention
    ep_size: int = 1
    # lora
    enable_lora: bool | None = None
    max_lora_rank: int | None = None
    lora_target_modules: List[str] | None = None
    lora_paths: List[str] | None = None
    max_loaded_loras: int = 1
    max_loras_per_batch: int = 1
    lora_backend: str = "triton"
    # logging
    log_level: str = "warning"
    log_level_http: str | None = "warning"
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = True  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: int = 1
    # Extra loader arguments
    # NOTE: These arguments will be parsed into a dict json-string
    # and passed as `model_loader_extra_config` to SGLang.
    enable_multithread_load: bool = False
    enable_fast_load: bool = False

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host,
        port,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args = SGLangConfig.build_args(
            sglang_config=sglang_config,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
            n_nodes=n_nodes,
            node_rank=node_rank,
        )

        # convert to flags
        flags = []
        for k, v in args.items():
            if is_version_less("sglang", "0.4.10.post2") and "max_loaded_loras" in k:
                continue
            if v is None or v is False or v == "":
                continue
            if v is True:
                flags.append(f"--{k.replace('_','-')}")
            elif isinstance(v, list):
                flags.append(f"--{k.replace('_','-')} {' '.join(map(str, v))}")
            else:
                flags.append(f"--{k.replace('_','-')} {v}")
        return f"python3 -m sglang.launch_server {' '.join(flags)}"

    @staticmethod
    def build_args(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host,
        port,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        # Map "all-linear" to "all"
        args: Dict = conf_as_dict(sglang_config)
        if sglang_config.enable_multithread_load or sglang_config.enable_fast_load:
            assert pkg_version.is_version_equal(
                "sglang", "0.5.2"
            ), f"Customized model loading requires exact SGLang version 0.5.2"
            model_loader_extra_config = dict(
                enable_multithread_load=sglang_config.enable_multithread_load,
                enable_fast_load=sglang_config.enable_fast_load,
            )
            args.pop("enable_multithread_load", None)
            args.pop("enable_fast_load", None)
            args["model_loader_extra_config"] = json.dumps(
                model_loader_extra_config, separators=(",", ":")
            )
        # Map "all-linear" to "all"
        if "lora_target_modules" in args and args["lora_target_modules"]:
            args["lora_target_modules"] = [
                x.replace("-linear", "") for x in args["lora_target_modules"]
            ]
        args = dict(
            host=host,
            port=port,
            # Model and tokenizer
            tokenizer_path=sglang_config.model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            device=current_platform.device_type,
            is_embedding=False,
            # Other runtime options
            tp_size=tp_size,
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=base_gpu_id,
            nnodes=n_nodes,
            node_rank=node_rank,
            # initialization addresses and ports
            dist_init_addr=dist_init_addr,
            **args,
        )
        if not pkg_version.is_version_greater_or_equal("sglang", "0.4.9.post2"):
            raise RuntimeError("Needs sglang>=0.4.9.post2 to run the code.")
        return args


@dataclass
class InferenceEngineConfig:
    """Configuration for inference servers, including offpolicyness control."""

    experiment_name: str | None = None
    trial_name: str | None = None
    max_concurrent_rollouts: None | int = field(
        default=None,
        metadata={
            "help": "Maximum number of concurrent rollouts to the inference engine. Defaults to consumer_batch_size."
        },
    )
    queue_size: None | int = field(
        default=None,
        metadata={"help": "Input/Output queue size for async rollout."},
    )
    consumer_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for consuming rollouts from the queue."},
    )
    max_head_offpolicyness: int = field(
        default=0,
        metadata={
            "help": "Maximum off-policyness for the head. "
            "If the current version is more than this many versions behind, "
            "the request will not be accepted.",
        },
    )
    enable_rollout_tracing: bool = field(
        default=False,
        metadata={
            "help": "Whether to output verbose tracing messages for each generation request."
        },
    )
    check_trajectory_format: bool = field(
        default=False,
        metadata={
            "help": "Whether to check the format of produced trajectories of a customized workflow. Useful when debugging the workflow in isolation. Should be False during RL training."
        },
    )
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    setup_timeout: float = field(
        default=120.0,
        metadata={
            "help": "Timeout in seconds of connecting to remote servers or launching local servers."
        },
    )
    request_timeout: float = field(
        default=3600, metadata={"help": "Timeout for HTTP requests."}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for failed requests."}
    )
    pause_grace_period: float = field(
        default=0.0,
        metadata={
            "help": "The grace period after calling /pause_generation. Wait until all requests have been dropped."
        },
    )


@dataclass
class _Timer:
    experiment_name: str = MISSING
    trial_name: str = MISSING
    fileroot: str = MISSING
    freq_epochs: int | None = field(
        default=None,
        metadata={
            "help": "Trigger frequency in epochs. None disables epoch-based saving."
        },
    )
    freq_steps: int | None = field(
        default=None,
        metadata={
            "help": "Trigger frequency in steps. None disables step-based saving."
        },
    )
    freq_secs: int | None = field(
        default=None,
        metadata={
            "help": "Trigger frequency in seconds. None disables time-based saving."
        },
    )


@dataclass
class EvaluatorConfig(_Timer):
    """Configuration for model evaluation scheduling and timing."""


@dataclass
class SaverConfig(_Timer):
    """Configuration for model checkpoint saving scheduling and timing."""


@dataclass
class RecoverConfig(_Timer):
    """Configuration for experiment recovery and fault tolerance."""

    mode: str = field(
        default="disabled",
        metadata={
            "help": "Recovery mode for the launcher. "
            "Options: "
            "'disabled': Never recover from previous runs. "
            "'auto': Automatically recover from previous runs if recover info and checkpoints are available. "
            "'fault': Only recover from previous runs if the new run fails. "
            "'resume': Force to resume, raise an error if no recover info was found. Never resume if failed again."
        },
    )
    retries: int = field(
        default=3,
        metadata={"help": "Number of recovery retries (auto/fault modes only)."},
    )


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases experiment tracking."""

    mode: str = "disabled"
    wandb_base_url: str = ""
    wandb_api_key: str = ""
    entity: str | None = None
    project: str | None = None
    name: str | None = None
    job_type: str | None = None
    group: str | None = None
    notes: str | None = None
    tags: List[str] | None = None
    config: Dict | None = None
    id_suffix: str | None = "train"


@dataclass
class SwanlabConfig:
    """Configuration for SwanLab experiment tracking and monitoring."""

    project: str | None = None
    name: str | None = None
    config: Dict | None = None
    logdir: str | None = None
    mode: str | None = "disabled"
    api_key: str | None = os.getenv("SWANLAB_API_KEY", None)


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard logging and visualization."""

    path: str | None = None


@dataclass
class StatsLoggerConfig:
    """Configuration for experiment statistics logging and tracking services."""

    experiment_name: str = MISSING
    trial_name: str = MISSING
    fileroot: str = MISSING
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


@dataclass
class NameResolveConfig:
    """Configuration for distributed name resolution and service discovery."""

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
            "help": "Record root for NFS name resolving. Should be available on all nodes."
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
    """Configuration for cluster specification and distributed computing setup."""

    name_resolve: NameResolveConfig = field(
        default_factory=NameResolveConfig,
        metadata={"help": "Name resolving configuration."},
    )
    cluster_name: str = field(
        default="local",
        metadata={"help": "Name of the cluster. Used to set specific environs."},
    )
    fileroot: str = field(
        default="/tmp/areal/",
        metadata={
            "help": "Root for logs and checkpoints. Should be available on all nodes."
        },
    )
    n_nodes: int = field(
        default=32,
        metadata={
            "help": "The size of the cluster. Used to decide slurm hostname suffix."
        },
    )
    n_gpus_per_node: int = field(
        default=8,
        metadata={"help": "Number of GPUs per node (physical)."},
    )


@dataclass
class SchedulerConfig:
    """Configuration for worker scheduling. Used in the single-controller mode. Experimental."""

    endpoint: str = field(default="http://localhost:8081")
    deploy_mode: str = field(default="separation")
    functioncall_service_domain: str = field(default="http://localhost:8080")
    reward_functioncall_config: Dict = field(default_factory=dict)
    reward_model_path: str = field(default="")
    reward_model_service_url: str = field(default="http://localhost:30000/classify")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    path: str = field(
        default=MISSING,
        metadata={
            "help": "Path to the dataset. Can be a local path or a HuggingFace dataset name."
        },
    )
    type: str = field(
        default=MISSING,
        metadata={"help": "Type of training method, e.g., 'sft', 'rl', etc."},
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size for the dataloader"}
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
    drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch"}
    )
    max_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum token length of sequences in dataset. Longer sequences are filtered out."
        },
    )


@dataclass
class SlurmLauncherConfig:
    """Configuration for launching the training jobs with Slurm."""

    srun_additional_args: str = field(
        default="--mpi=pmi2 -K --chdir $PWD",
        metadata={"help": "Additional arguments to pass to the srun command."},
    )
    container_type: str = field(
        default="apptainer",
        metadata={
            "help": "Type of containers used in slurm",
            "choices": ["apptainer", "none"],
        },
    )
    mount: str = field(
        default="/storage:/storage", metadata={"help": "Mount path for slurm."}
    )
    trainer_image: str | None = field(
        default=None, metadata={"help": "slurm image for trainers."}
    )
    inference_server_image: str | None = field(
        default=None, metadata={"help": "slurm image for LLM inference."}
    )


@dataclass
class LauncherConfig:
    """Configuration for launching the LLM server and trainer processes."""

    inference_server_cpus_per_gpu: int = field(
        default=4,
        metadata={"help": "Number of CPUs allocated per GPU for inference server."},
    )
    inference_server_mem_per_gpu: int = field(
        default=32 * 1024,
        metadata={"help": "Memory allocated per GPU for inference server in MB."},
    )
    trainer_cpus_per_gpu: int = field(
        default=4,
        metadata={"help": "Number of CPUs allocated per GPU for training."},
    )
    trainer_mem_per_gpu: int = field(
        default=32 * 1024,
        metadata={"help": "Memory allocated per GPU for training in MB."},
    )
    inference_server_env_vars: str = field(
        default="",
        metadata={
            "help": "Environment variables for inference server, separated by commas. "
            "Example: 'ENV1=val1,ENV2=val2'."
        },
    )
    trainer_env_vars: str = field(
        default="",
        metadata={
            "help": "Environment variables for training, separated by commas. "
            "Example: 'ENV1=val1,ENV2=val2'."
        },
    )
    slurm: SlurmLauncherConfig = field(
        default_factory=SlurmLauncherConfig,
        metadata={"help": "Slurm launcher configuration."},
    )


@dataclass
class BaseExperimentConfig:
    """Base configuration class for all experiment types with common settings."""

    # NOTE: we need this unified config class because different experiments
    # have different config structures, e.g., GRPO has two engine configs,
    # but SFT only has a single one. We use subclasses to represent these structures.
    experiment_name: str = field(
        default=MISSING,
        metadata={"help": "Name of the experiment (no '_' or '/'). Required."},
    )
    trial_name: str = field(
        default=MISSING,
        metadata={"help": "Name of the trial (no '-' or '/'). Required."},
    )
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification. Mainly used by slurm."},
    )
    allocation_mode: str = field(
        default="",
        metadata={
            "help": "GPU parallel strategy allocation mode. "
            "Options: manual/heuristic or pattern-based."
        },
    )
    seed: int = field(default=1, metadata={"help": "Random seed for reproducibility."})
    total_train_epochs: int = field(
        default=1, metadata={"help": "Total number of epochs to train the model."}
    )
    total_train_steps: int | None = field(
        default=None,
        metadata={
            "help": "Terminate training after this number of steps. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    total_train_n_seqs: int | None = field(
        default=None,
        metadata={
            "help": "Terminate training after consuming this number of samples. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    tokenizer_path: str = field(
        default="",
        metadata={"help": "Path to the tokenizer."},
    )

    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    valid_dataset: DatasetConfig | None = field(default=None)

    saver: SaverConfig = field(default_factory=SaverConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    stats_logger: StatsLoggerConfig = field(default_factory=StatsLoggerConfig)
    recover: RecoverConfig = field(default_factory=RecoverConfig)

    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    vllm: vLLMConfig = field(default_factory=vLLMConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class SFTConfig(BaseExperimentConfig):
    """Configuration for Supervised Fine-Tuning (SFT) experiments."""

    model: TrainEngineConfig = field(default_factory=TrainEngineConfig)


@dataclass
class RWConfig(BaseExperimentConfig):
    """Configuration for Reward Model (RW) training experiments."""

    model: TrainEngineConfig = field(default_factory=TrainEngineConfig)


@dataclass
class GRPOConfig(BaseExperimentConfig):
    """Configuration for Group Relative Policy Optimization (GRPO) reinforcement learning experiments."""

    async_training: bool = field(
        default=True,
        metadata={
            "help": "Enable asynchronous training between rollout and policy update."
        },
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    ref: PPOActorConfig = field(default_factory=PPOActorConfig)


@dataclass
class PPOConfig(GRPOConfig):
    """Configuration for Proximal Policy Optimization (PPO) reinforcement learning experiments."""

    critic: PPOCriticConfig = field(default_factory=PPOCriticConfig)


def parse_cli_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the main configuration file", required=True
    )
    # The first argument might be the path to a training script,
    # which should be ignored by the argument parser.
    if argv and argv[0].endswith(".py"):
        argv = argv[1:]
    args, overrides = parser.parse_known_args(argv)
    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists(), f"Config file {config_file} does not exist."
    # hydra only recognize relative paths
    relpath = Path(os.path.relpath(str(config_file), Path(__file__).parent.absolute()))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).split(".yaml")[0],
        overrides=overrides,
    )
    return cfg, config_file


def to_structured_cfg(cfg, config_cls):
    # Merge with the default configuration.
    # The yaml and commandline can omit some default values defined in python dataclasses.
    default_cfg = OmegaConf.structured(config_cls)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


def load_expr_config(argv: List[str], config_cls):
    cfg, config_file = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=config_cls)
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, config_cls)
    # Setup environment

    name_resolve.reconfigure(cfg.cluster.name_resolve)

    from areal.utils.stats_logger import StatsLogger

    # Save configuration as yaml
    if os.getenv("RANK", "0") == "0":
        save_config(cfg, StatsLogger.get_log_path(cfg.stats_logger))

    return cfg, str(config_file)


def conf_as_dict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)


def save_config(cfg, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    config_save_path = os.path.join(log_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        config_dict: Dict = asdict(cfg)
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
        )
