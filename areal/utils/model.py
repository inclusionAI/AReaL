import torch

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
]
# This registry is used to check if a model is a vision model that we have checked it works with AReaL.
# As different vision models vary in their image processing, special tokens and keys, etc. We will add models to this registry as we test them.
# If you want to add a new vision model, please make sure it works with AReaL.


# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
