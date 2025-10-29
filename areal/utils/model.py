import torch

from areal.api.cli_args import BaseExperimentConfig
from areal.api.io_struct import AllocationMode, WeightUpdateMeta

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
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
    else:
        return WeightUpdateMeta.from_fsdp_xccl(
            AllocationMode.from_str(config.allocation_mode)
        )
