from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict


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
            "help": "Maximum number of concurrent requests to the inference engine."
        },
    )
    consumer_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size for consuming requests from the queue. "
            "If None, it will be set to max_concurrent_requests.",
        },
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
