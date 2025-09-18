import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import uvloop

uvloop.install()
from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from omegaconf import MISSING, DictConfig, OmegaConf

from areal.platforms import current_platform
from areal.utils import name_resolve, pkg_version
from areal.utils.fs import get_user_tmp


@dataclass
class NormConfig:
    """Configuration for advantage normalization."""

    # TODO: add common fields of adv_norm and reward_norm


@dataclass
class AdvNormConfig(NormConfig):
    """Advanced configuration for advantage normalization."""

    mean_level: str = field(
        default="batch",
        metadata={
            "help": "mean_level for advantage normalization. options: batch, group, none"
        },
    )
    std_level: str = field(
        default="batch",
        metadata={
            "help": "std_level for advantage normalization. options: batch, group, none"
        },
    )
    group_size: int = field(
        default=1, metadata={"help": "group_size for advantage normalization"}
    )


@dataclass
class RewardNormConfig(NormConfig):
    # TODO: implement reward normalization
    pass


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
        },
    )
    max_tokens_per_mb: Optional[int] = field(
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
    stop: Optional[List[str]] = field(
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
                "Must be between -2 and 2 where negative numbers encourage repeatment."
            )
        },
    )
    max_workers: Optional[int] = field(
        default=32,
        metadata={"help": "Number of workers processing reward."}
    )
    interruptable_processpool: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Using an interruptable process pool to process reward function or not."
        },
    )

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)


# Train Engine Configs


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
class FSDPWrapPolicy:
    transformer_layer_cls_to_wrap: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of transformer layer names for FSDP to wrap."},
    )


@dataclass
class FSDPEngineConfig:
    wrap_policy: Optional[FSDPWrapPolicy] = field(
        default=None,
        metadata={"help": "FSDP wrap policy, specifying model layers to wrap."},
    )
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )


@dataclass
class DeepSpeedAutoTPEngineConfig:
    autotp_size: Optional[int] = field(
        default=1,
        metadata={"help": "DeepSpeed AutoTP size"},
    )


@dataclass
class TrainEngineConfig:
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
    init_critic_from_actor: bool = field(
        default=False,
        metadata={"help": "Initialize critic/reward model from LM checkpoint"},
    )
    # Runtime microbatch limit
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)
    pad_to_maximum: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad each microbatch to the length upper bound specified by mb_spec. "
                "Can reduce memory fragmentation but slow down training."
            )
        },
    )

    # Training Backend Configuration
    disable_dropout: bool = field(default=False)
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    dtype: str = field(default="bfloat16", metadata={"help": "Parameter dtype."})
    grad_reduce_dtype: str = field(
        default="float32", metadata={"help": "Gradient reduce dtype."}
    )
    optimizer: Optional[OptimizerConfig] = field(
        default=None, metadata={"help": "Optimizer configuration"}
    )

    backend: str = ""
    fsdp: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    ds_auto_tp: DeepSpeedAutoTPEngineConfig = field(
        default_factory=DeepSpeedAutoTPEngineConfig
    )


@dataclass
class PPOActorConfig(TrainEngineConfig):
    # Core PPO/GRPO Parameters
    group_size: int = field(
        default=1, metadata={"help": "Number of sequences in each group"}
    )
    group_adv_norm: bool = field(
        default=False,
        metadata={
            "help": "Normalize advantages within each prompt group rather than globally"
        },
    )
    ppo_n_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches for each PPO update"}
    )
    eps_clip: float = field(
        default=0.2, metadata={"help": "Clipping factor for policy ratio"}
    )
    eps_clip_higher: Optional[float] = field(
        default=None,
        metadata={
            "help": "Clipping factor (higher value) for policy ratio. Defaults is None. When eps_clip_higher is setted (decouppled), eps_clip will be used as the lower value."
        },
    )
    c_clip: Optional[float] = field(
        default=None,
        metadata={
            "help": "Dual clipping factor for policy ratio, must > 1.0. None disables dual clipping."
        },
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Temperature during generation."}
    )
    # Reward
    group_reward_norm: bool = field(
        default=False,
        metadata={
            "help": "Normalize final reward of each sequence (GRPO-style) to reduce length bias"
        },
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
        metadata={"help": "penalty for overlong sequences. used within DAPO."},
    )
    overlong_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "The numbers of token in the tail will receive a penalty"},
    )
    overlong_penalty_factor: Optional[float] = field(
        default=None,
        metadata={"help": "The numbers of token in the tail will receive a penalty"},
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

    # KL Control
    kl_ctl: float = field(default=0.1, metadata={"help": "KL divergence coefficient"})

    # Asynchronous RL
    recompute_logprob: bool = field(
        default=False,
        metadata={"help": "Recompute logp and replace the logp returned by inference."},
    )
    use_decoupled_loss: bool = field(
        default=False,
        metadata={
            "help": "Use the decoupled loss. Implicitly enable recompute_logprob."
        },
    )
    behav_imp_weight_cap: Optional[float] = field(
        default=None,
        metadata={
            "help": "We filter out the tokens where behav_imp_weight exceeds behav_imp_weight_cap when computing the loss, must be > 1.0, use_decoupled_loss must be true"
        },
    )
    # Advanced Options
    dynamic_sampling: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic sampling (within DAPO). If enabled, the group with same reward will be filtered out."
        },
    )

    # Logging Agent Trajectories
    log_agent_stats: bool = field(
        default=False,
        metadata={"help": "Log stats for agent trajectories"},
    )
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys of log stats for agent trajectories"},
    )

    # Logging Agent Trajectories
    log_agent_stats: bool = field(
        default=False,
        metadata={"help": "Log stats for agent trajectories"},
    )
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys of log stats for agent trajectories"},
    )
    # Others
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    adv_norm: Optional[AdvNormConfig] = field(
        default=None, metadata={"help": "Optimizer configuration, default is None"}
    )


@dataclass
class vLLMConfig:
    """Configuration for vLLM runtime.
    """
    model: str = ""
    seed: int = 1
    skip_tokenizer_init: bool = False
    enforce_eager: bool = True
    dtype: str = "bfloat16"
    distributed_executor_backend = "mp"
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
    max_model_len: Optional[int] = 32768
    enable_chunked_prefill: bool = False
    # NOTE: Setting enable_prefix_caching to False
    # because it will reuse the block after
    # model weights are updated. Using v0.7.2 reset_prefix_cache
    # will fix this issue.
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.9
    worker_extension_cls: str = "areal.launcher.vllm_worker_extension.VLLMWorkerExtension"
    enable_sleep_mode: bool = False
    # additional_engine_args: Dict = field(default_factory=dict)
    @staticmethod
    def build_args(
        vllm_config: "vLLMConfig",
        tp_size,
        host,
        port,
        dist_init_addr: Optional[str] = None,
    ):
        from realhf.experiments.common.utils import asdict as conf_as_dict
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
        base_gpu_id,
        host,
        port,
        dist_init_addr: Optional[str] = None,
    ):
        # TODO base_gpu_id ?
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
        return f"python3 -m areal.launcher.areal_vllm_server {' '.join(flags)}"


@dataclass
class SGLangConfig:
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    model_path: str = ""
    random_seed: int = 1
    skip_tokenizer_init: bool = False
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
    attention_backend: Optional[str] = "fa3"
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
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    dp_size: int = 1  # only used for dp attention
    ep_size: int = 1
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

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host,
        port,
        dist_init_addr: Optional[str] = None,
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
        dist_init_addr: Optional[str] = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):

        args: Dict = conf_as_dict(sglang_config)
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
    experiment_name: Optional[str] = None
    trial_name: Optional[str] = None
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
    enable_rollout_tracing: bool = field(default=False)
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    setup_timeout: float = field(default=120.0)
    request_timeout: float = field(
        default=3600, metadata={"help": "Timeout for HTTP requests."}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for failed requests."}
    )


@dataclass
class _Timer:
    experiment_name: str = MISSING
    trial_name: str = MISSING
    fileroot: str = MISSING
    freq_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Trigger frequency in epochs. None disables epoch-based saving."
        },
    )
    freq_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Trigger frequency in steps. None disables step-based saving."
        },
    )
    freq_secs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Trigger frequency in seconds. None disables time-based saving."
        },
    )


@dataclass
class EvaluatorConfig(_Timer):
    pass


@dataclass
class SaverConfig(_Timer):
    pass


@dataclass
class RecoverConfig(_Timer):
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


@dataclass
class StatsLoggerConfig:
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
class DatasetConfig:
    path: str = field(
        default=MISSING,
        metadata={
            "help": "Path to the dataset. Can be a local path or a HuggingFace dataset name."
        },
    )
    type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of training method.e.g., 'sft', 'rl', etc."},
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size of the dataloader"}
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
    drop_last: bool = field(default=True)
    reward_fn: Optional[str] = field(
        default=None,
    )
    max_length: Optional[int] = None


@dataclass
class SlurmLauncherConfig:
    """Configuration for launching the SGLang server with Slurm."""

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
    trainer_image: Optional[str] = field(
        default=None, metadata={"help": "slurm image for trainers."}
    )
    inference_server_image: Optional[str] = field(
        default=None, metadata={"help": "slurm image for LLM inference."}
    )


@dataclass
class LauncherConfig:
    """Configuration for launching the SGLang server."""

    inference_server_cpus_per_gpu: int = field(
        default=4,
        metadata={"help": "Number of CPUs allocated per GPU for inference server. "},
    )
    inference_server_mem_per_gpu: int = field(
        default=32 * 1024,
        metadata={"help": "Memory allocated per GPU for inference server in MB. "},
    )
    trainer_cpus_per_gpu: int = field(
        default=4,
        metadata={"help": "Number of CPUs allocated per GPU for training. "},
    )
    trainer_mem_per_gpu: int = field(
        default=32 * 1024,
        metadata={"help": "Memory allocated per GPU for training in MB. "},
    )
    inference_server_env_vars: str = field(
        default="",
        metadata={
            "help": "Environment variables for inference server, seperated by commas. "
            "Example: 'ENV1=val1,ENV2=val2'. "
        },
    )
    trainer_env_vars: str = field(
        default="",
        metadata={
            "help": "Environment variables for training, seperated by commas. "
            "Example: 'ENV1=val1,ENV2=val2'. "
        },
    )
    slurm: SlurmLauncherConfig = field(
        default_factory=SlurmLauncherConfig,
        metadata={"help": "Slurm launcher configuration."},
    )


@dataclass
class BaseExperimentConfig:
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
    total_train_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Terminate training after this number of steps. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    total_train_n_seqs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Terminate training after consuming this number of samples. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    tokenizer_path: str = field(default="")
    weight_update_mode: str = field(default="disk")

    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    valid_dataset: Optional[DatasetConfig] = field(default=None)

    saver: SaverConfig = field(default_factory=SaverConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    stats_logger: StatsLoggerConfig = field(default_factory=StatsLoggerConfig)
    recover: RecoverConfig = field(default_factory=RecoverConfig)

    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    vllm: vLLMConfig = field(default_factory=vLLMConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)


@dataclass
class SFTConfig(BaseExperimentConfig):
    model: TrainEngineConfig = field(default_factory=TrainEngineConfig)


@dataclass
class GRPOConfig(BaseExperimentConfig):
    async_training: bool = field(default=True)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    ref: PPOActorConfig = field(default_factory=PPOActorConfig)


def parse_cli_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="The path of the main configuration file", required=True
    )
    args, overrides = parser.parse_known_args(argv)
    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists(), f"Config file {config_file} does not exist."
    # hydra only recognize relative paths
    relpath = Path(os.path.relpath(str(config_file), Path(__file__).parent.absolute()))
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
    assert isinstance(cfg, BaseExperimentConfig)
    # Setup environment

    name_resolve.reconfigure(cfg.cluster.name_resolve)
    return cfg, str(config_file)


def conf_as_dict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)
