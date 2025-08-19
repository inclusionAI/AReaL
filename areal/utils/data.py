# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec
from realhf.base import datapack, logging

logger = logging.getLogger("data utils")


def reorder_list(xs: List, indices: List[int]) -> List:
    assert len(set(indices)) == len(xs)
    return [xs[i] for i in indices]


def dict_map(x: Dict, fn: Callable) -> Dict:
    return {k: fn(v) for k, v in x.items()}


def dict_of_list2list_of_dict(
    dict_of_lists: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    if not dict_of_lists:
        return []
    keys = list(dict_of_lists.keys())
    length = len(dict_of_lists[keys[0]])
    for key, value_list in dict_of_lists.items():
        if len(value_list) != length:
            raise ValueError(
                f"All lists must have the same length. Key '{key}' has length {len(value_list)}, expected {length}"
            )
    return [{key: dict_of_lists[key][i] for key in keys} for i in range(length)]


def list_of_dict2dict_of_list(
    list_of_dicts: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    if not list_of_dicts:
        return {}
    keys = list(list_of_dicts[0].keys())
    for i, dict_item in enumerate(list_of_dicts):
        if set(dict_item.keys()) != set(keys):
            raise ValueError(
                f"All dictionaries must have the same keys. Dictionary at index {i} has keys {set(dict_item.keys())}, expected {set(keys)}"
            )
    return {key: [dict_item[key] for dict_item in list_of_dicts] for key in keys}


def pad_sequences_to_tensors(
    sequence_list: List[TensorDict], pad_value: float = 0.0
) -> TensorDict:
    if not sequence_list:
        return TensorDict()
    skip_keys = {"pixel_values", "image_grid_thw"}
    max_length = max(
        len(seq)
        for item in sequence_list
        for key, seq in item.items()
        if key not in skip_keys
    )
    result = {}
    for key in sequence_list[0].keys():
        padded = []
        if key in skip_keys:
            result[key] = [sequence_list[i][key] for i in range(len(sequence_list))]
            continue
        for item in sequence_list:
            x = item[key]
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            padded_x = torch.nn.functional.pad(
                x, (0, max_length - len(item[key])), value=pad_value
            )
            padded.append(padded_x)
        result[key] = torch.stack(padded)
    attention_mask = [
        [1] * len(next(iter(item[key] for key in item.keys() if key not in skip_keys)))
        + [0]
        * (
            max_length
            - len(next(iter(item[key] for key in item.keys() if key not in skip_keys)))
        )
        for item in sequence_list
    ]
    result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
    return TensorDict(result, batch_size=[result["attention_mask"].shape[0]])


def unpad_input(
    hidden_states, attention_mask
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = hidden_states.new_zeros(batch * seqlen)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


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

    has_any_multi_modal = any("multi_modal_input" in td for td in tensor_dicts)

    merged_multi_modal = None

    if has_any_multi_modal:
        merged_multi_modal = []

        # Merge multi-modal data maintaining per-dp correspondence
        for tensor_dict in tensor_dicts:
            td_batch_size = tensor_dict.batch_size[0]

            if "multi_modal_input" in tensor_dict:
                # Has multi_modal_input - extend the lists
                multi_modal = tensor_dict["multi_modal_input"]
            else:
                multi_modal = [{} for _ in range(td_batch_size)]

            merged_multi_modal.extend(multi_modal)

        result["multi_modal_input"] = merged_multi_modal

    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []
        if key == "multi_modal_input":
            continue
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
    return TensorDict(result, batch_size=new_batch_size)


def to_device(data: Dict[str, torch.Tensor | Any], device) -> Dict[str, torch.Tensor]:
    """Move tensors in a dictionary to the specified device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    lens: Optional[List[int]] = None,
    dim: int = 0,
):
    """Unpack a sequence tensor into a list of tensors based on cumulative sequence lengths."""
    if lens is not None:
        return torch.split(x, lens, dim=dim)
    if cu_seqlens is not None:
        return torch.split(
            x, (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist(), dim=dim
        )
    raise ValueError("Either cu_seqlens or input_lens must be provided.")


def allocate_balanced_mbs(mb_spec: MicroBatchSpec, lens: List[int]) -> List[List[int]]:
    group_indices = datapack.ffd_allocate(
        lens, mb_spec.max_tokens_per_mb, min_groups=mb_spec.n_mbs
    )
    group_indices = sorted([sorted(g) for g in group_indices])
    return group_indices


def allocate_balanced_mbs_synced(
    mb_spec: MicroBatchSpec,
    lens: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> List[List[int]]:
    group_indices = allocate_balanced_mbs(mb_spec, lens)
    if not dist.is_initialized():
        return group_indices

    all_n_mbs = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(all_n_mbs, len(group_indices), group=group)
    if all(mbs == len(group_indices) for mbs in all_n_mbs):
        return group_indices
    return allocate_balanced_mbs_synced(
        MicroBatchSpec.new(mb_spec, n_mbs=max(all_n_mbs)), lens
    )


def pack_tensor_dict(data: TensorDict):
    """Pack a tensordict of shape [B, S, ...] into [total_length, ...], leaving other keys unchanged.

    Args:
        data (Dict[str, Any]): Dictionary containing tensors to be packed. Should contain key "attention_mask" with shape [B, S].

    Returns:
        Dict[str, Any]: Dictionary with packed tensors. The "attention_mask" key will be replaced by "cu_seqlens" with shape [B+1].
    """

    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."
    attention_mask = data["attention_mask"]
    assert attention_mask.ndim == 2, "Attention mask must be a 2D tensor."
    bs = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]

    # Calculate cumulative sequence lengths
    lens = attention_mask.sum(dim=1, dtype=torch.int32)
    max_seqlen = lens.max().item()
    cu_seqlens = torch.cumsum(lens, dim=0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    total_length = int(cu_seqlens[-1].item())
    # Pack tensors
    packed_data = {}
    for key, value in data.items():
        if key == "attention_mask":
            packed_data["cu_seqlens"] = cu_seqlens
            packed_data["max_seqlen"] = max_seqlen
            continue
        # tensor and of shape [B, S, ...]
        elif key == "pixel_values":
            assert value.dim() == 3, f"pixel_values must be [B,S,D], got {tuple(value.shape)}"
            packed_data[key] = value
        elif (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[0] == bs
            and value.shape[1] == seq_len
        ):
            packed_tensor = torch.empty(
                (total_length, *value.shape[2:]), dtype=value.dtype, device=value.device
            )
            # Fill the packed tensor with values from the original tensor
            for i in range(bs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                packed_tensor[start:end] = value[i][: end - start]
            packed_data[key] = packed_tensor
        else:
            packed_data[key] = value

    return TensorDict(**packed_data)


def pad_and_stack_tensors_along_first_dim(tensor_list: List[torch.Tensor]):
    max_length = max(tensor.shape[0] for tensor in tensor_list)
    n_dim = tensor_list[0].ndim
    assert all(
        tensor.ndim == n_dim for tensor in tensor_list
    ), "All tensors must have the same number of dimensions."

    padded_tensors = []
    for tensor in tensor_list:
        pad_mode = (0,) * (2 * (n_dim - 1)) + (0, max_length - tensor.shape[0])
        padded_tensor = F.pad(tensor, pad_mode, value=0.0)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim=0)


@dataclass
class MicroBatchList:
    data: TensorDict
    mb_spec: MicroBatchSpec
    mbs: List[TensorDict]
    forward_indices: List[int]
    backward_indices: List[int]
    group_lens: List[int]
    padded_mbs: Optional[List[TensorDict]] = None
    padding_lengths: Optional[List[int]] = None


DEFAULT_MAX_TOKENS_PER_MB = int(1e12)


def split_padded_tensor_dict_into_mb_list(
    data: TensorDict, mb_spec: MicroBatchSpec, group: Optional[dist.ProcessGroup] = None
) -> MicroBatchList:
    """Split a padded tensordict into micro-batches based on the attention mask.

    Args:
        data (TensorDict): Dictionary containing padded tensors.
        mb_spec (MicroBatchSpec): Specification for micro-batch splitting.
        group (Optional[dist.ProcessGroup]): Process group for distributed synchronization.

    Returns:
        MicroBatchList: A structure containing the split micro-batches and metadata.
    """
    assert (
        "attention_mask" in data
    ), "Input data must be padded and contain 'attention_mask' key."
    if mb_spec.max_tokens_per_mb is None:
        mb_spec = MicroBatchSpec.new(
            mb_spec, max_tokens_per_mb=DEFAULT_MAX_TOKENS_PER_MB
        )
    bs = data["attention_mask"].shape[0]
    max_seqlen = data["attention_mask"].shape[1]
    input_lens = data["attention_mask"].sum(1).long().cpu().numpy()

    # check tensor shape, split only 1d tensors with length "total_lens"
    to_split = {}
    not_to_split = {}
    for key, value in data.items():
        if key == "multi_modal_input":
            continue
        if key == "position_ids" or (
            torch.is_tensor(value) and value.numel() == bs * max_seqlen
        ):
            # NOTE: qwen2.5-vl position_ids.numel() == bs * max_seqlen * 3
            to_split[key] = value
        else:
            not_to_split[key] = value

    # split
    group_indices = allocate_balanced_mbs_synced(mb_spec, input_lens, group=group)
    splitted_lens = [
        [input_lens[i] for i in group_index] for group_index in group_indices
    ]
    group_n_seqs = [len(x) for x in splitted_lens]
    group_lens = [sum(x) for x in splitted_lens]

    forward_indices = datapack.flat2d(group_indices)
    backward_indices = np.zeros(bs, dtype=np.int64)
    backward_indices[forward_indices] = np.arange(bs)

    def _split(tensor):
        """Split and pad a tensor based on forward indices and lens."""
        # Unpack the sequence
        unpacked = [tensor[i] for i in range(bs)]
        # Reorder according to forward indices
        reordered = reorder_list(unpacked, forward_indices)
        reordered = torch.stack(reordered)
        # Unpack again according to split lens
        splitted = []
        offset = 0
        for _n_seqs in group_n_seqs:
            splitted.append(reordered[offset : offset + _n_seqs])
            offset += _n_seqs
        return splitted

    to_split = dict_map(to_split, lambda x: _split(x))

    if "multi_modal_input" in data:
        multi_modal_input = data["multi_modal_input"]

        # Prepare the pixel_values and image_grid_thw for each group
        multi_modal_input_split = []

        for group_index in group_indices:
            group_pixel_multi_modal_input = [multi_modal_input[i] for i in group_index]
            # Stack pixel_values for each group (assuming pixel_values is a list of tensors)
            multi_modal_input_split.append(group_pixel_multi_modal_input)
        # Pack the split pixel_values and image_grid_thw back into the data
        to_split["multi_modal_input"] = multi_modal_input_split
    mbs = dict_of_list2list_of_dict(to_split)

    results = []
    # organize splitted micro batches
    assert len(mbs) == len(splitted_lens), (len(mbs), len(splitted_lens))
    for i, (mb, lens) in enumerate(zip(mbs, splitted_lens)):
        results.append(TensorDict(**mb, **not_to_split))
    return MicroBatchList(
        data=data,
        mbs=results,
        mb_spec=mb_spec,
        forward_indices=forward_indices,
        backward_indices=backward_indices.tolist(),
        group_lens=group_lens,
    )


def pad_packed_tensor_dict(
    data: TensorDict,
    pad_to_length: int,
    pad_value: float = 0.0,
) -> Tuple[TensorDict, int]:
    """Pad a packed tensor dict to a specified length.
    This function assumes that the input data contains "cu_seqlens" and "max_seqlen" key,
    and all other tensors of shape [total_length, ] will be padded to `pad_to_length`.
    This function will pad a new sequence filled with `pad_value` to the end of each tensor,
    and update the "cu_seqlens" and "max_seqlen" keys accordingly.

    Args:
        data (TensorDict): Dictionary containing tensors to be packed.
        pad_to_length (int): The length to pad the tensors to. All tensors

    Returns:
        TensorDict: Dictionary with padded tensors and modified "cu_seqlens" and
            "max_seqlen".
        int: The pad length.
    """
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."
    total_length = data["cu_seqlens"][-1].item()
    pad_length = pad_to_length - total_length
    assert (
        pad_length >= 0
    ), f"pad_to_length {pad_to_length} must be greater than or equal to total length {total_length}."
    cu_seqlens = data["cu_seqlens"]
    max_seqlen = data["max_seqlen"]
    new_cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_to_length)
    new_max_seqlen = max(max_seqlen, pad_length)
    padded_data = {}
    for key, value in data.items():
        if key == "cu_seqlens":
            padded_data[key] = new_cu_seqlens
        elif key == "max_seqlen":
            padded_data[key] = new_max_seqlen
        elif key == "position_ids":
            # [bs*seqlen, channel] for qwen2.5 vl, channel==3 for t,h,w
            if len(value.shape) == 2 and value.shape[1] == 3:
                pad = (
                    torch.arange(pad_length, dtype=torch.long, device=value.device)
                    .unsqueeze(1)
                    .expand(-1, 3)
                )
                padded_tensor = torch.cat([value, pad])
            else:
                pad = torch.arange(pad_length, dtype=torch.long, device=value.device)
                padded_tensor = torch.cat([value, pad])
            padded_data[key] = padded_tensor
        elif key == "pixel_values":
            assert value.dim() == 3, f"pixel_values must be [M,S,D], got {tuple(value.shape)}"
            padded_data[key] = value
        elif torch.is_tensor(value) and value.numel() == total_length:
            # Pad the tensor to the new total length
            padded_tensor = torch.nn.functional.pad(
                value, (0, pad_length), value=pad_value
            )
            padded_data[key] = padded_tensor
        else:
            padded_data[key] = value
    return TensorDict(padded_data, batch_size=data.batch_size), pad_length


def pad_mb_list(
    mb_list: MicroBatchList,
    pad_value: float = 0.0,
    pad_to_maximum: bool = False,
) -> MicroBatchList:
    padded_mb_inputs, pad_lengths = [], []
    pad_to_lengths = []
    if pad_to_maximum and (
        mb_list.mb_spec.max_tokens_per_mb is None
        or mb_list.mb_spec.max_tokens_per_mb == DEFAULT_MAX_TOKENS_PER_MB
    ):
        logger.warning(
            f"Unable to pad to maximum because max_tokens_per_mb is not properly set."
        )
        pad_to_maximum = False
    for mb, l in zip(mb_list.mbs, mb_list.group_lens):
        if pad_to_maximum:
            pad_to_length = mb_list.mb_spec.max_tokens_per_mb
        else:
            # NOTE: GPU page size is 2MB
            # Take hidden size 4096 with bf16 dtype as an example,
            # the batch size of a page is 256
            pad_to_length = (int(l) + 255) // 256 * 256
        padded_mb, pad_len = pad_packed_tensor_dict(
            mb, pad_to_length, pad_value=pad_value
        )
        padded_mb_inputs.append(padded_mb)
        pad_lengths.append(pad_len)
        pad_to_lengths.append(pad_to_length)
    mb_list.padded_mbs = padded_mb_inputs
    mb_list.padding_lengths = pad_lengths
    mb_list.padded_to_lengths = pad_to_lengths
    return mb_list


def unsqueeze_packed_tensor_dict(data: TensorDict) -> TensorDict:
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."

    total_length = data["cu_seqlens"][-1].item()
    new_data = {}
    for key, value in data.items():
        if key == "position_ids" or (
            key
            not in [
                "cu_seqlens",
                "max_seqlen",
            ]
            and torch.is_tensor(value)
            and value.numel() == total_length
        ):
            new_data[key] = value.unsqueeze(dim=0)
        elif key == "position_ids":
            value = value.unsqueeze(dim=0)
            if len(value.shape) == 3 and value.shape[2] == 3:
                value = torch.einsum("ijk->kij", value)

            new_data[key] = value
        else:
            new_data[key] = value
    return TensorDict(new_data, batch_size=data.batch_size)


def unsqueeze_mb_list(
    mb_list: MicroBatchList,
) -> MicroBatchList:
    """Unsqueeze the packed tensordict in the micro-batch list."""
    new_padded_mbs = []
    for i, mb in enumerate(mb_list.mbs):
        if mb_list.padded_mbs is not None:
            new_padded_mbs.append(unsqueeze_packed_tensor_dict(mb_list.padded_mbs[i]))
    mb_list.padded_mbs = new_padded_mbs if mb_list.padded_mbs is not None else None
    return mb_list


def amend_position_ids(data: TensorDict) -> TensorDict:
    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."

    attn_mask = data["attention_mask"]
    bs, seqlen = attn_mask.shape[:2]
    position_ids = (
        torch.arange(0, seqlen, dtype=torch.long, device=attn_mask.device)
        .unsqueeze(0)
        .expand(bs, -1)
    )
    position_ids.masked_fill(~attn_mask.bool(), 0)
    data["position_ids"] = position_ids
    return data


def amend_position_ids_3d(data: TensorDict, rope_fn) -> TensorDict:
    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."
    torch.set_printoptions(threshold=float("inf"))
    attn_mask = data["attention_mask"]
    input_ids = data["input_ids"]
    image_grid_thw = data.get("image_grid_thw", None)
    video_grid_thw = data.get("video_grid_thw", None)

    if image_grid_thw != None:
        image_grid_thw = image_grid_thw.squeeze(1)
    position_ids, rope_deltas = rope_fn(
        input_ids, image_grid_thw, video_grid_thw, attn_mask
    )  # [channel=3,bs,seqlen]

    position_ids = torch.einsum("ijk->jki", position_ids)  # [bs,seqlen,channel=3]
    data["position_ids"] = position_ids
    return data
