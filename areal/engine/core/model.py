import torch

from areal.api import AllocationMode, WeightUpdateMeta
from areal.api.cli_args import BaseExperimentConfig

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "gemma3",
]
# This registry is used to check if a model is a vision model that we have checked it works with AReaL.
# As different vision models vary in their image processing, special tokens and keys, etc.
# We will add models to this registry as we test them.
# If you want to add a new vision model, please make sure it works with AReaL.


def is_valid_vision_model(model_type: str) -> bool:
    return model_type in VALID_VISION_MODELS


def is_qwen2_vl_model(model_type: str) -> bool:
    return model_type in ["qwen2_vl", "qwen2_5_vl"]


def is_qwen3_vl_model(model_type: str) -> bool:
    return model_type in ["qwen3_vl"]


def is_qwen_vl_model(model_type: str) -> bool:
    return is_qwen2_vl_model(model_type) or is_qwen3_vl_model(model_type)


def is_gemma3_model(model_type: str) -> bool:
    return model_type in ["gemma3"]


VALID_MOE_MODELS = [
    "qwen3_moe",
]
# This registry is used to check if a model is a MoE model that we have checked it works with AReaL.


def is_moe_model(model_type: str) -> bool:
    return model_type in VALID_MOE_MODELS


def is_qwen3_moe_model(model_type: str) -> bool:
    return model_type in ["qwen3_moe"]


# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def get_model_update_meta(config: BaseExperimentConfig) -> WeightUpdateMeta:
    """Get weight update metadata based on configuration.

    Args:
        config: BaseExperimentConfig (e.g., GRPOConfig, PPOConfig, SFTConfig, RWConfig)
            The function will extract the appropriate engine config (actor/model)
            to determine the weight update mode.

    Returns:
        WeightUpdateMeta: Metadata for weight updates
    """
    if not isinstance(config, BaseExperimentConfig):
        raise TypeError(
            f"config must be BaseExperimentConfig (e.g., GRPOConfig, PPOConfig, SFTConfig), "
            f"got {type(config).__name__}"
        )

    # For experiment configs, try to get actor config first (for GRPO/PPO),
    # otherwise use model config (for SFT/RW)
    if hasattr(config, "actor"):
        engine_config = config.actor
    elif hasattr(config, "model"):
        engine_config = config.model
    else:
        raise ValueError(
            f"Config {type(config).__name__} must have either 'actor' or 'model' attribute"
        )

    weight_update_mode = engine_config.weight_update_mode

    if weight_update_mode == "disk":
        return WeightUpdateMeta.from_disk(
            config.experiment_name, config.trial_name, config.cluster.fileroot
        )
    if weight_update_mode == "awex":
        awex_cfg = getattr(config, "awex", None)
        if awex_cfg is None:
            raise ValueError("Awex config is required when weight_update_mode is 'awex'.")
        if not awex_cfg.meta_server_addr:
            raise ValueError("awex.meta_server_addr must be set when using awex.")
        comm_backend = awex_cfg.comm_backend
        ipc_backend = awex_cfg.weights_exchange_ipc_backend
        use_mindspeed = awex_cfg.use_mindspeed or awex_cfg.device_backend == "npu"
        if awex_cfg.device_backend == "npu":
            # Default to HCCL for NPU when NCCL is requested.
            if comm_backend == "nccl":
                comm_backend = "hccl"
            # CUDA IPC is not available on NPU; fall back to CPU.
            if ipc_backend == "cuda":
                ipc_backend = "cpu"
        meta = WeightUpdateMeta.from_awex(
            meta_server_addr=awex_cfg.meta_server_addr,
            comm_backend=comm_backend,
            weights_exchange_ipc_backend=ipc_backend,
            weights_comm_nccl_group_size=awex_cfg.weights_comm_nccl_group_size,
            enable_debug_mode=awex_cfg.enable_debug_mode,
            debug_mode_config=awex_cfg.debug_mode_config,
            disable_weights_exchange_pipeline=awex_cfg.disable_weights_exchange_pipeline,
            enable_colocate_mode=awex_cfg.enable_colocate_mode,
            weights_validation_steps=awex_cfg.weights_validation_steps,
            validate_weights_every_n_steps=awex_cfg.validate_weights_every_n_steps,
            dump_weights_list_for_validation=awex_cfg.dump_weights_list_for_validation,
            dump_weights_dir_for_validation=awex_cfg.dump_weights_dir_for_validation,
            nnodes=awex_cfg.nnodes,
            node_rank=awex_cfg.node_rank,
            use_mindspeed=use_mindspeed,
        )
        if awex_cfg.comm_backend == "file":
            disk_meta = WeightUpdateMeta.from_disk(
                config.experiment_name, config.trial_name, config.cluster.fileroot
            )
            meta.path = disk_meta.path
        return meta
    return WeightUpdateMeta.from_fsdp_xccl(
        AllocationMode.from_str(config.allocation_mode)
    )
