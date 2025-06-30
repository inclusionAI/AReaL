# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0
import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from omegaconf import MISSING, OmegaConf

from realhf.api.cli_args import (
    ClusterSpecConfig,
    ExperimentSaveEvalControl,
    MicroBatchSpec,
    OptimizerConfig,
    TensorBoardConfig,
    WandBConfig,
)


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


## Inference config for clients and servers. ##


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
class LLMServiceConfig:
    served_model_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the served model"}
    )
    health_check_interval: int = field(
        default=5, metadata={"help": "Health check interval in seconds"}
    )
    startup_timeout: int = field(
        default=90, metadata={"help": "Startup timeout in seconds"}
    )
    max_unhealth_count: int = field(
        default=3, metadata={"help": "Max unhealthy count before restart"}
    )
    graceful_shutdown_on_unhealthy: bool = field(
        default=True, metadata={"help": "Enable graceful shutdown when unhealthy"}
    )


@dataclass
class LLMClientConfig:
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    request_timeout: int = field(
        default=3600, metadata={"help": "Request timeout in seconds"}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for each request"}
    )


## Dataset configs. ##


@dataclass
class GSM8KPreprocessor:
    reward_mode: str = "strict"


@dataclass
class DatasetPreprocessor:
    type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Number of retries for each request",
        },
    )
    gsm8k: Optional[GSM8KPreprocessor] = None


@dataclass
class DatasetConfig:
    path: str = field(
        default="", metadata={"help": "Path or HuggingFace identifier to the dataset"}
    )
    name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name (for HuggingFace datasets)"}
    )
    split: Optional[str] = field(
        default=None, metadata={"help": "Dataset split to use (e.g., 'train', 'test')"}
    )
    data_files: Optional[str] = field(
        default=None, metadata={"help": "Specific data files to load"}
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
    preprocessor: Optional[DatasetPreprocessor] = field(
        default=None,
        metadata={"help": "Dataset preprocessor config. None means no preprocessing."},
    )


## Training backend configs. ##


@dataclass
class FSDPWrapPolicy:
    transformer_layer_cls_to_wrap: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of transformer layer names for FSDP to wrap."},
    )


@dataclass
class FSDPConfig:
    wrap_policy: Optional[FSDPWrapPolicy] = field(
        default=None,
        metadata={"help": "FSDP wrap policy, specifying model layers to wrap."},
    )
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )


@dataclass
class EngineBackendConfig:
    type: str = field(
        default="hf",
        metadata={"help": "Training backend", "choices": ["fsdp", "hf"]},
    )
    fsdp: Optional[FSDPConfig] = field(
        default=None, metadata={"help": "FSDP configuration (if using FSDP backend)"}
    )


@dataclass
class EngineConfig:
    # Model Architecture Configuration
    path: str = field(default="", metadata={"help": "Path to HuggingFace checkpoint"})
    init_from_scratch: bool = field(
        default=False, metadata={"help": "Initialize model weights randomly"}
    )
    init_critic_from_actor: bool = field(
        default=False,
        metadata={"help": "Initialize critic/reward model from LM checkpoint"},
    )

    # Training Backend Configuration
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    bf16: bool = field(default=False, metadata={"help": "Use bf16 precision"})
    optimizer: Optional[OptimizerConfig] = field(
        default=None, metadata={"help": "Optimizer configuration"}
    )
    backend: EngineBackendConfig = field(
        default_factory=EngineBackendConfig,
        metadata={"help": "Training backend configuration"},
    )


## Agent configurations. ##


@dataclass
class MathCodeSingleStepConfig:
    solution_path: str = field(default="", metadata={"help": "Path to solutions"})


@dataclass
class RLVRConfig:
    reward_type: str = field(
        default="areal-math",
        metadata={
            "help": "The type of the reward function",
            "choices": ["areal-math", "areal-code", "gsm8k"],
        },
    )
    solution_path: str = field(
        default="", metadata={"help": "Path to solutions. Required by areal-math/code."}
    )


@dataclass
class RolloutCollectorConfig:
    type: str = field(
        default="rlvr",
        metadata={
            "help": "Rollout collector type",
            "choices": ["rlvr", "math_code_single_step"],
        },
    )
    rlvr: Optional[RLVRConfig] = field(
        default=None,
        metadata={"help": "The configuration for the RLVR collector"},
    )
    math_code_single_step: Optional[MathCodeSingleStepConfig] = field(
        default=None,
        metadata={"help": "The configuration for the single-step math/code collector"},
    )


## Rollout configurations. ##


@dataclass
class RolloutConfig:
    server_backend: str = field(
        default="sglang",
        metadata={"help": "Backend for serving", "choices": ["sglang", "vllm"]},
    )
    model_path: str = field(default="", metadata={"help": "Path to the rollout model"})
    collector: RolloutCollectorConfig = field(
        default_factory=RolloutCollectorConfig,
        metadata={"help": "Rollout collector configuration."},
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of rollout worker processes"}
    )
    max_concurrent_rollouts: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of concurrent rollouts. Defaults to train batch size."
        },
    )
    max_head_offpolicyness: int = field(
        default=0,
        metadata={"help": "Maximum off-policyness tolerance for the first token"},
    )
    filter_reward_lb: float = field(
        default=-float("inf"), metadata={"help": "Lower bound for reward filtering"}
    )
    filter_reward_ub: float = field(
        default=float("inf"), metadata={"help": "Upper bound for reward filtering"}
    )
    llm_client: LLMClientConfig = field(
        default_factory=LLMClientConfig,
        metadata={"help": "LLM client configuration for rollouts"},
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters,
        metadata={"help": "Generation hyperparameters for rollouts"},
    )
    llm_service: LLMServiceConfig = field(
        default_factory=LLMServiceConfig, metadata={"help": "LLM server configuration"}
    )
    sglang: Optional[SGLangConfig] = field(
        default_factory=SGLangConfig,
        metadata={"help": "SGLang configuration (if using SGLang backend)"},
    )


## Trainer configurations. ##


@dataclass
class SFTTrainerConfig:
    model: EngineConfig = field(
        default_factory=EngineConfig,
        metadata={"help": "Model configuration for SFT training"},
    )
    mb_spec: MicroBatchSpec = field(
        default_factory=MicroBatchSpec,
        metadata={"help": "Micro-batch specification for SFT training"},
    )


@dataclass
class GRPOTrainerConfig:
    async_training: bool = field(
        default=True, metadata={"help": "Enable asynchronous training mode"}
    )
    actor: EngineConfig = field(
        default_factory=EngineConfig,
        metadata={"help": "Actor model configuration"},
    )
    ref: Optional[EngineConfig] = field(
        default=None, metadata={"help": "Reference model configuration"}
    )
    mb_spec: MicroBatchSpec = field(
        default_factory=MicroBatchSpec,
        metadata={"help": "Micro-batch specification"},
    )

    # Core PPO/GRPO Parameters
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
    c_clip: Optional[float] = field(
        default=None,
        metadata={
            "help": "Dual clipping factor for policy ratio, must > 1.0. None disables dual clipping."
        },
    )
    actor_sample_reuse: int = field(
        default=1, metadata={"help": "The data reuse (aka PPO epoch) for actor."}
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
    max_reward_clip: float = field(
        default=20.0, metadata={"help": "Maximum absolute value for reward clipping"}
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
    adv_norm: bool = field(
        default=True, metadata={"help": "Enable advantage normalization"}
    )

    # KL Control
    kl_ctl: float = field(default=0.1, metadata={"help": "KL divergence coefficient"})

    # Asynchronous PPO
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


@dataclass
class TrainerConfig:
    type: str = field(
        default="grpo",
        metadata={"help": "Trainer type", "choices": ["grpo", "sft", "null"]},
    )
    grpo: Optional[GRPOTrainerConfig] = field(
        default=None, metadata={"help": "GRPO trainer configuration (if using GRPO)"}
    )
    sft: Optional[SFTTrainerConfig] = field(
        default=None, metadata={"help": "SFT trainer configuration (if using SFT)"}
    )


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
    mode: str = field(
        default="slurm",
        metadata={
            "help": "Experiment launching mode.",
            "choices": ["slurm", "local", "ray"],
        },
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
        default="sglang.d1p1t1+d1p1t1",
        metadata={
            "help": "GPU parallel strategy allocation mode. "
            "Options: manual/heuristic or pattern-based."
        },
    )
    ray_temp_path: str = field(
        default="/tmp/ray", metadata={"help": "Absolute path for Ray's log."}
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
    exp_ctrl: ExperimentSaveEvalControl = field(
        default_factory=ExperimentSaveEvalControl,
        metadata={"help": "Experiment save/evaluation control configuration."},
    )
    shutdown_server_on_exit: bool = field(
        default=False,
        metadata={"help": "Whether to shut down the LLM generation server on exit."},
    )
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification. Mainly used by slurm."},
    )
    train_dataset: DatasetConfig = field(
        default_factory=DatasetConfig, metadata={"help": "Train dataset configuration"}
    )
    valid_dataset: Optional[DatasetConfig] = field(
        default=None, metadata={"help": "Validation dataset configuration"}
    )
    rollout: Optional[RolloutConfig] = field(
        default_factory=RolloutConfig,
        metadata={"help": "Rollout controller configuration for RL training"},
    )
    trainer: Optional[TrainerConfig] = field(
        default=None, metadata={"help": "Trainer configuration"}
    )
    cpu_per_inf_proc: int = 16
    mem_per_inf_proc: int = 100000
    cpu_per_train_proc: int = 16
    mem_per_train_proc: int = 100000


def prepare_training_args(argv: List[str]) -> Tuple[TrainingArgs, str]:
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
        os.path.relpath(
            str(config_file), (Path(__file__).parent.parent / "cli").absolute()
        )
    )
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).rstrip(".yaml"),
        overrides=overrides,
    )

    # Merge with the default configuration
    default_cfg = OmegaConf.structured(TrainingArgs)
    cfg = OmegaConf.merge(default_cfg, cfg)
    cfg: TrainingArgs = OmegaConf.to_object(cfg)

    # Setup environment
    from realhf.base import constants, name_resolve

    constants.set_experiment_trial_names(cfg.experiment_name, cfg.trial_name)
    name_resolve.reconfigure(cfg.cluster.name_resolve)
    return cfg, str(config_file)
