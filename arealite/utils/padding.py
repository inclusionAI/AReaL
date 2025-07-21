from typing import List

import torch
from tensordict import TensorDict


def concat_padded_tensors(
    tensor_dicts: List[TensorDict], pad_value: float = 0.0
) -> TensorDict:
    """Concatenate and pad tensors from multiple padded tensor dictionaries."""
    if not tensor_dicts:
        return TensorDict()

    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    new_batch_size = [sum(x[0] for x in batch_sizes), *batch_sizes[0][1:]]

    # Find max sequence length across all dictionaries
    assert all("attention_mask" in td for td in tensor_dicts)
    max_length = max([x["attention_mask"].shape[1] for x in tensor_dicts])
    result = {}
    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
            if key == "pixle_values" or key == "image_grid_thw":
                tensors_to_concat.append(tensor)
                continue
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

    return TensorDict(result, batch_size=new_batch_size)
