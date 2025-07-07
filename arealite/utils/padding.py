import torch
from typing import List, Dict, Any
from tensordict import TensorDict

def concat_padded_tensors(
    tensor_dicts: List[TensorDict], pad_value: float = 0.0
) -> TensorDict:
    """Concatenate and pad tensors from multiple padded tensor dictionaries."""
    if not tensor_dicts:
        return TensorDict()

    # Find max sequence length across all dictionaries
    lens = []
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            if key != "attention_mask" and len(tensor.shape) == 2:
                lens.append(tensor.shape[1])
                break
    max_length = max(lens)
    attn_mask = torch.arange(max_length).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)

    result = {}
    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
            # Skip 1D tensors like rewards
            if len(tensor.shape) == 1:
                tensors_to_concat.append(tensor)
                continue
            current_length = tensor.shape[1]
            if current_length < max_length:
                # Pad tensor to max_length
                pad_width = max_length - current_length
                if key == "attention_mask":
                    # Pad attention mask with 0s
                    padding = torch.zeros(
                        (tensor.shape[0], pad_width), dtype=tensor.dtype
                    )
                else:
                    # Pad feature tensors with pad_value
                    padding = torch.full(
                        (tensor.shape[0], pad_width), pad_value, dtype=tensor.dtype
                    )
                tensor = torch.cat([tensor, padding], dim=1)
            tensors_to_concat.append(tensor)

        result[key] = torch.cat(tensors_to_concat, dim=0)
    if "attention_mask" not in result:
        result["attention_mask"] = attn_mask
    return TensorDict(result)