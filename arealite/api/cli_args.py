import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import uvloop

uvloop.install()
from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from omegaconf import MISSING, OmegaConf

from arealite.utils.fs import get_user_tmp
from realhf.api.cli_args import OptimizerConfig


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
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

    # Training Backend Configuration
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    bf16: bool = field(default=False, metadata={"help": "Use bf16 precision"})
    optimizer: Optional[OptimizerConfig] = field(
        default=None, metadata={"help": "Optimizer configuration"}
    )
    backend: str = ""
    fsdp: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)


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
        from realhf.base import network, pkg_version, seeding
        from realhf.experiments.common.utils import asdict as conf_as_dict

        args: Dict = conf_as_dict(sglang_config)
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
class InferenceEngineConfig:
    experiment_name: str
    trial_name: str
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
class DatasetConfig:
    type: Optional[str] = field(
        default=None, metadata={"help": "Type of implemented dataset"}
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
    n_nodes: int = field(
        default=1, metadata={"help": "Number of nodes for experiment."}
    )
    n_gpus_per_node: int = field(
        default=8, metadata={"help": "Number of GPUs per node for this experiment."}
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

    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    valid_dataset: DatasetConfig = field(default_factory=DatasetConfig)

    saver: SaverConfig = field(default_factory=SaverConfig)
    checkpointer: SaverConfig = field(default_factory=SaverConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    stats_logger: StatsLoggerConfig = field(default_factory=StatsLoggerConfig)


@dataclass
class SFTConfig(BaseExperimentConfig):
    model: TrainEngineConfig = field(default_factory=TrainEngineConfig)


def load_expr_config(argv: List[str], config_cls) -> Tuple[BaseExperimentConfig, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="The path of the main configuration file", required=True
    )
    args, overrides = parser.parse_known_args(argv)

    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists()
    # hydra only recognize relative paths
    relpath = Path(
        os.path.relpath(str(config_file), (Path(__file__).parent).absolute())
    )
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).rstrip(".yaml"),
        overrides=overrides,
    )

    # Merge with the default configuration.
    # The yaml and commandline can omit some default values defined in python dataclasses.
    default_cfg = OmegaConf.structured(config_cls)
    cfg = OmegaConf.merge(default_cfg, cfg)
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, BaseExperimentConfig)

    # Setup environment
    from realhf.base import constants, name_resolve

    constants.set_experiment_trial_names(cfg.experiment_name, cfg.trial_name)
    name_resolve.reconfigure(cfg.cluster.name_resolve)
    return cfg, str(config_file)
