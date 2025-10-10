import torch

from areal.utils.errors import FrameworkError


def convert_list_to_dict(list_dataset):
    """convert list[Dict] to Dict[str, Union[Tensor, scalar, list]]"""
    if not list_dataset:
        return {}
    keys = list_dataset[0].keys()
    dict_dataset = {}
    for key in keys:
        values = [sample[key] for sample in list_dataset]

        # Handle different types
        if isinstance(values[0], torch.Tensor):
            # Handle tensor types
            if values[0].dim() == 0:
                dict_dataset[key] = torch.stack(values, dim=0)
            else:
                dict_dataset[key] = torch.cat(values, dim=0)
        else:
            dict_dataset[key] = values
    return dict_dataset


def convert_dict_to_list(dict_dataset):
    """Convert dictionary format dataset to list format (used for deletion by index)"""
    if not dict_dataset:
        return []

    # Get batch size
    first_value = next(iter(dict_dataset.values()))
    if isinstance(first_value, torch.Tensor):
        batch_size = first_value.shape[0]
    elif isinstance(first_value, list):
        batch_size = len(first_value)
    else:
        # For scalar values, batch size is 1
        batch_size = 1

    list_dataset = []
    for i in range(batch_size):
        sample = {}
        for k, v in dict_dataset.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v[i]
            elif isinstance(v, list):
                sample[k] = v[i]
            else:
                # For scalar values, return the same value for all indices
                sample[k] = v
        list_dataset.append(sample)
    return list_dataset


def validate_dict_dataset(dataset):
    """Validate the integrity of Dict[str, Union[Tensor, scalar, list]] format"""
    if not dataset:
        return

    # Get total size from first value
    first_value = next(iter(dataset.values()))
    if isinstance(first_value, torch.Tensor):
        total_size = first_value.shape[0]
    elif isinstance(first_value, list):
        total_size = len(first_value)
    else:
        # For scalar values, total size is 1
        total_size = 1

    # Validate all values have consistent total size
    for key, value in dataset.items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] != total_size:
                raise FrameworkError(
                    "FrameworkError",
                    "DistributedBatchMemoryError",
                    f"Total size mismatch for key '{key}': expected {total_size}, got {value.shape[0]}",
                )
        elif isinstance(value, list):
            if len(value) != total_size:
                raise FrameworkError(
                    "FrameworkError",
                    "DistributedBatchMemoryError",
                    f"Total size mismatch for key '{key}': expected {total_size}, got {len(value)}",
                )
