from typing import Dict, List

import torch


def pad_sequences_to_tensors(
    sequence_list: List[Dict[str, List[float]]], pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    """Convert list of dict[str, List[float]] to padded tensors with attention mask."""
    if not sequence_list:
        return {}

    # Find max length across all sequences
    max_length = max(len(seq) for item in sequence_list for seq in item.values())

    result = {}

    # Create padded tensors for each key
    for key in sequence_list[0].keys():
        padded = [
            item[key] + [pad_value] * (max_length - len(item[key]))
            for item in sequence_list
        ]
        result[key] = torch.tensor(padded, dtype=torch.float32)

    # Create attention mask
    attention_mask = [
        [1] * len(next(iter(item.values())))
        + [0] * (max_length - len(next(iter(item.values()))))
        for item in sequence_list
    ]

    result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
    return result


def concat_padded_tensors(
    tensor_dicts: List[Dict[str, torch.Tensor]], pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    """Concatenate and pad tensors from multiple padded tensor dictionaries."""
    if not tensor_dicts:
        return {}

    # Find max sequence length across all dictionaries
    max_length = max(
        tensor.shape[1]
        for tensor_dict in tensor_dicts
        for key, tensor in tensor_dict.items()
        if key != "attention_mask"
    )

    result = {}

    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []

        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
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

    return result
