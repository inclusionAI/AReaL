import torch

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
]

# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
