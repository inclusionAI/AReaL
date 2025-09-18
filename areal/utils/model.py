import torch

from areal.api.io_struct import AllocationMode, WeightUpdateMeta
from areal.platforms import is_npu_available

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
]
# This registry is used to check if a model is a vision model that we have checked it works with AReaL.
# As different vision models vary in their image processing, special tokens and keys, etc.
# We will add models to this registry as we test them.
# If you want to add a new vision model, please make sure it works with AReaL.


def is_qwen2_vl_model(model_type):
    return model_type in ["qwen2_vl", "qwen2_5_vl"]


def is_qwen3_moe_model(model_type):
    return model_type in ["qwen3_moe"]


# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def get_model_update_meta(config, actor):
    if config.weight_update_mode == "disk":
        weight_update_meta = [
            WeightUpdateMeta.from_disk(
                config.experiment_name, config.trial_name, config.cluster.fileroot
            )
        ]
    else:
        weight_update_meta = [
            WeightUpdateMeta.from_fsdp_xccl(
                'hccl' if is_npu_available else 'nccl', AllocationMode.from_str(config.allocation_mode), actor
            )
        ]
    return weight_update_meta