# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import MicroBatchSpec, NormConfig
from areal.platforms import current_platform
from areal.utils import datapack, logging

logger = logging.getLogger("data utils")


def get_batch_size(data: Dict[str, Any]) -> int:
    if not data:
        return 0

    am = data.get("attention_mask")
    if torch.is_tensor(am) and am.ndim >= 1:
        return int(am.shape[0])

    cu = data.get("cu_seqlens")
    if torch.is_tensor(cu) and cu.ndim >= 1 and cu.numel() >= 1:
        return max(int(cu.shape[0]) - 1, 0)

    mmi = data.get("multi_modal_input")
    if isinstance(mmi, list):
        return len(mmi)

    for v in data.values():
        if torch.is_tensor(v) and v.ndim >= 1:
            return int(v.shape[0])

    return 0


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
    sequence_list: List[Dict[str, Any]], pad_value: float = 0.0
) -> Dict[str, Any]:
    if not sequence_list:
        return {}
    skip_keys = {"multi_modal_input"}
    max_length = max(
        len(seq)
        for item in sequence_list
        for key, seq in item.items()
        if key not in skip_keys
    )
    result = {}
    for key in sequence_list[0].keys():
        padded = []
        if key == "multi_modal_input":
            for i in range(len(sequence_list)):
                if sequence_list[i][key]:
                    item = sequence_list[i][key][0]
                    for k, v in item.items():
                        if not torch.is_tensor(v):
                            item[k] = torch.tensor(v)
            # list concat
            result[key] = sum(
                [sequence_list[i][key] for i in range(len(sequence_list))], []
            )
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
    return result


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
    tensor_dicts: List[Dict[str, Any]], pad_value: float = 0.0
) -> Dict[str, Any]:
    """Concatenate and pad tensors from multiple dictionaries of padded tensors."""
    if not tensor_dicts:
        return {}

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
            td_batch_size = get_batch_size(tensor_dict)

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
                        (tensor.shape[0], pad_width),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                else:
                    # Pad feature tensors with pad_value
                    padding = torch.full(
                        (tensor.shape[0], pad_width),
                        pad_value,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                tensor = torch.cat([tensor, padding], dim=1)
            tensors_to_concat.append(tensor)

        result[key] = torch.cat(tensors_to_concat, dim=0)
    return result


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
    assert mb_spec.max_tokens_per_mb is not None
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
        MicroBatchSpec.new(mb_spec, n_mbs=max(all_n_mbs)), lens, group=group
    )


def pack_tensor_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pack a dict of tensors of shape [B, S, ...] into [total_length, ...], leaving other keys unchanged.

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
    cu_seqlens = torch.cumsum(lens, dim=0, dtype=torch.int32)
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
        if (
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

    return packed_data


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


def tensor_container_to(
    d: Dict[str, Any] | torch.Tensor | List[torch.Tensor], *args, **kwargs
):
    """Apply `t.to(*args, **kwargs)` to all tensors in the dictionary.
    Support nested dictionaries.
    """
    new_dict = {}
    if torch.is_tensor(d):
        return d.to(*args, **kwargs)
    elif isinstance(d, list):
        return [tensor_container_to(v, *args, **kwargs) for v in d]
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_dict[key] = tensor_container_to(value, *args, **kwargs)
            elif torch.is_tensor(value):
                new_dict[key] = value.to(*args, **kwargs)
            else:
                new_dict[key] = value
        return new_dict
    else:
        raise ValueError(f"Unsupported type: {type(d)}")


@dataclass
class MicroBatchList:
    data: Dict[str, Any]
    mb_spec: MicroBatchSpec
    mbs: List[Dict[str, Any]]
    forward_indices: List[int]
    backward_indices: List[int]
    group_lens: List[int]
    padded_mbs: List[Dict[str, Any]] | None = None
    # Batch-level padding information
    padding_lengths: List[int] | None = None
    padded_to_lengths: List[int] | None = None
    # sequence-level padding information
    align_to_lengths: List[int] | None = None
    old_cu_seqlens_list: List[torch.Tensor] | None = None

    def to(self, *args, **kwargs):
        mbs = [tensor_container_to(mb, *args, **kwargs) for mb in self.mbs]
        data = tensor_container_to(self.data, *args, **kwargs)
        padded_mbs = None
        if self.padded_mbs is not None:
            padded_mbs = [
                tensor_container_to(mb, *args, **kwargs) for mb in self.padded_mbs
            ]
        old_cu_seqlens_list = None
        if self.old_cu_seqlens_list is not None:
            old_cu_seqlens_list = [
                t.to(*args, **kwargs) for t in self.old_cu_seqlens_list
            ]
        return MicroBatchList(
            data=data,
            mb_spec=self.mb_spec,
            mbs=mbs,
            forward_indices=self.forward_indices,
            backward_indices=self.backward_indices,
            group_lens=self.group_lens,
            padded_mbs=padded_mbs,
            padding_lengths=self.padding_lengths,
            padded_to_lengths=self.padded_to_lengths,
            old_cu_seqlens_list=old_cu_seqlens_list,
            align_to_lengths=self.align_to_lengths,
        )


DEFAULT_MAX_TOKENS_PER_MB = int(1e12)


def split_padded_tensor_dict_into_mb_list(
    data: Dict[str, Any],
    mb_spec: MicroBatchSpec,
    group: Optional[dist.ProcessGroup] = None,
) -> MicroBatchList:
    """Split a padded dict of tensors into micro-batches based on the attention mask.

    Args:
        data (Dict): Dictionary containing padded tensors.
        mb_spec (MicroBatchSpec): Specification for micro-batch splitting.
        group (Optional[dist.ProcessGroup]): Process group for distributed synchronization.

    Returns:
        MicroBatchList: A structure containing the split micro-batches and metadata.
    """
    # TODO: should align sequences first and then split, needs refactor
    assert (
        "attention_mask" in data
    ), "Input data must be padded and contain 'attention_mask' key."
    if mb_spec.max_tokens_per_mb is None:
        mb_spec = MicroBatchSpec.new(
            mb_spec, max_tokens_per_mb=DEFAULT_MAX_TOKENS_PER_MB
        )
    granularity = mb_spec.granularity
    bs = data["attention_mask"].shape[0]
    if bs % granularity != 0:
        raise RuntimeError(f"Batch size {bs} cannot divide granularity {granularity}.")
    max_seqlen = data["attention_mask"].shape[1]
    seq_lens = data["attention_mask"].sum(1).long().cpu().numpy().tolist()
    input_lens = (
        data["attention_mask"]
        .view(bs // granularity, granularity, -1)
        .sum(dim=(1, 2))
        .long()
        .cpu()
        .numpy()
    )

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
    group_indices = [
        datapack.flat2d(
            [list(range(i * granularity, (i + 1) * granularity)) for i in group_index]
        )
        for group_index in group_indices
    ]
    splitted_lens = [
        [seq_lens[i] for i in group_index] for group_index in group_indices
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
        results.append({**mb, **not_to_split})

    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=results,
        forward_indices=forward_indices,
        backward_indices=backward_indices.tolist(),
        group_lens=group_lens,
    )


N_TOKENS_PER_PAGE = 256


def pad_packed_tensor_dict(
    data: Dict[str, Any],
    pad_to_length: int,
    pad_value: float = 0.0,
    align_sequences: bool = False,
    align_to_multiple_of: Optional[int] = None,
) -> Tuple[Dict[str, Any], int, torch.Tensor, int]:
    """Pad a packed dict of tensors to a specified length.
    This function assumes that the input data contains "cu_seqlens" and "max_seqlen" key,
    and all other tensors of shape [total_length, ] will be padded to `pad_to_length`.
    This function will pad a new sequence filled with `pad_value` to the end of each tensor,
    and update the "cu_seqlens" and "max_seqlen" keys accordingly.

    Args:
        data (Dict): Dictionary containing tensors to be packed.
        pad_to_length (int): The length to pad the tensors to. All tensors

    Returns:
        Dict: Dictionary with padded tensors and modified "cu_seqlens" and
            "max_seqlen".
        int: The pad length.
    """
    assert "cu_seqlens" in data, "Input data must contain 'cu_seqlens' key."
    assert "max_seqlen" in data, "Input data must contain 'max_seqlen' key."
    cu_seqlens = data["cu_seqlens"]
    max_seqlen = data["max_seqlen"]
    old_cu_seqlens = cu_seqlens.clone()
    total_length = data["cu_seqlens"][-1].item()
    # First pad sequences
    sequence_padded_data = {}
    align_to_length = None
    if align_sequences:
        assert (
            align_to_multiple_of is not None
        ), "align_to_multiple_of must be specified when align_sequences is True."
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = input_lens.shape[0]
        # Align sequences to an integer multiple of align_to_multiple_of
        pad_size = (
            align_to_multiple_of - input_lens % align_to_multiple_of
        ) % align_to_multiple_of
        input_lens_padded = input_lens + pad_size
        cu_seqlens_padded = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=cu_seqlens.device
        )
        cu_seqlens_padded[1:] = torch.cumsum(input_lens_padded, dim=0)
        max_seqlens_padded = input_lens_padded.max().item()
        padded_shape = (input_lens_padded.sum().item(),)
        for key, value in data.items():
            if key == "cu_seqlens":
                sequence_padded_data["cu_seqlens"] = cu_seqlens_padded
            elif key == "max_seqlen":
                sequence_padded_data["max_seqlen"] = max_seqlens_padded
            elif key == "position_ids":
                if len(value.shape) == 2 and value.shape[1] == 3:
                    # [total_seq_len, channel] for qwen2.5 vl, channel==3 for t,h,w
                    new_value = torch.zeros(
                        (padded_shape[0], 3), dtype=value.dtype, device=value.device
                    )
                    for i in range(batch_size):
                        new_start = cu_seqlens_padded[i]
                        new_end = cu_seqlens_padded[i + 1]
                        old_start = cu_seqlens[i]
                        old_end = cu_seqlens[i + 1]
                        length = old_end - old_start
                        pad_length = new_end - new_start - length
                        new_value[new_start : new_start + length] = value[
                            old_start:old_end
                        ]
                        new_value[new_start + length : new_end] = (
                            torch.arange(
                                pad_length, dtype=torch.long, device=value.device
                            )
                            .unsqueeze(1)
                            .expand(-1, 3)
                        )
                else:
                    new_value = torch.zeros(
                        padded_shape, dtype=value.dtype, device=value.device
                    )
                    for i in range(batch_size):
                        new_start = cu_seqlens_padded[i]
                        new_end = cu_seqlens_padded[i + 1]
                        new_value[new_start:new_end] = torch.arange(
                            new_end - new_start, dtype=value.dtype, device=value.device
                        )
                sequence_padded_data[key] = new_value
            elif torch.is_tensor(value) and value.numel() == total_length:
                new_value = torch.full(
                    padded_shape,
                    fill_value=pad_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                for i in range(batch_size):
                    new_start = cu_seqlens_padded[i]
                    start = cu_seqlens[i]
                    end = cu_seqlens[i + 1]
                    length = end - start
                    new_value[new_start : new_start + length] = value[start:end]
                sequence_padded_data[key] = new_value
            else:
                sequence_padded_data[key] = value

        data = sequence_padded_data
        align_to_length = cu_seqlens_padded[-1].item()
        # ensure pad_to_length is a integer multiple of both align_to_multiple_of and N_TOKENS_PER_PAGE
        lcm = np.lcm(align_to_multiple_of, N_TOKENS_PER_PAGE).item()
        pad_to_length = (pad_to_length + lcm - 1) // lcm * lcm

        cu_seqlens = data["cu_seqlens"]
        max_seqlen = data["max_seqlen"]
        total_length = data["cu_seqlens"][-1].item()
        if pad_to_length < total_length:
            # NOTE: In some occasion where sequence lengths, sequence padding will make total length
            # exceed expected `pad_to_length`. This happens more often when sequence lengths are small.
            # In this case, we increase pad_to_length.
            pad_to_length = (total_length + lcm - 1) // lcm * lcm

    # Pad batch
    pad_length = pad_to_length - total_length
    assert (
        pad_length >= 0
    ), f"pad_to_length {pad_to_length} must be greater than or equal to total length {total_length}."
    new_cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_to_length)
    new_max_seqlen = max(max_seqlen, pad_length)
    padded_data = {}
    for key, value in data.items():
        if key == "cu_seqlens":
            padded_data[key] = new_cu_seqlens
        elif key == "max_seqlen":
            padded_data[key] = new_max_seqlen
        elif key == "position_ids":
            # [total_seqlen, channel] for qwen2.5 vl, channel==3 for t,h,w
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
        elif torch.is_tensor(value) and value.numel() == total_length:
            # Pad the tensor to the new total length
            padded_tensor = torch.nn.functional.pad(
                value, (0, pad_length), value=pad_value
            )
            padded_data[key] = padded_tensor
        else:
            padded_data[key] = value
    return (
        padded_data,
        pad_length,
        old_cu_seqlens,
        align_to_length,
    )


def pad_mb_list(
    mb_list: MicroBatchList,
    pad_value: float = 0.0,
    pad_to_maximum: bool = False,
    align_sequences: bool = False,
    align_to_multiple_of: Optional[int] = None,
) -> MicroBatchList:
    """Pad the micro-batch list to the maximum length or to a specific size to:
        1. Reduce memory fragmentation.
        2. Align sequences to an integer multiple of `align_to_multiple_of`
        to be equally sliced into context and sequence parallel ranks.

    Args:
        mb_list (MicroBatchList): The micro-batch list to pad.
        pad_value (float, optional): The value to pad the tensors with. Defaults to 0.0.
        pad_to_maximum (bool, optional): Whether to pad to the maximum length specified in `mb_spec`. Defaults to False.
        align_sequences (bool, optional): Whether to align sequences to an integer multiple of `align_to_multiple_of`. Defaults to False.
        align_to_multiple_of (int, optional): The size to align sequences to. Defaults to None.

    Returns:
        MicroBatchList: The padded micro-batch list.
    """
    if align_sequences:
        assert (
            align_to_multiple_of is not None
        ), "align_to_multiple_of must be specified when align_sequences is True."
    padded_mb_inputs, pad_lengths = [], []
    pad_to_lengths = []
    old_cu_seqlens_list = []
    align_to_lengths = []
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
            pad_to_length = (
                (int(l) + N_TOKENS_PER_PAGE - 1)
                // N_TOKENS_PER_PAGE
                * N_TOKENS_PER_PAGE
            )
        padded_mb, pad_len, old_cu_seqlens, align_to_length = pad_packed_tensor_dict(
            mb,
            pad_to_length,
            pad_value=pad_value,
            align_sequences=align_sequences,
            align_to_multiple_of=align_to_multiple_of,
        )
        padded_mb_inputs.append(padded_mb)
        pad_lengths.append(pad_len)
        pad_to_lengths.append(pad_to_length)
        old_cu_seqlens_list.append(old_cu_seqlens)
        align_to_lengths.append(align_to_length)
    mb_list.padded_mbs = padded_mb_inputs
    mb_list.padding_lengths = pad_lengths
    mb_list.padded_to_lengths = pad_to_lengths
    if align_sequences:
        mb_list.old_cu_seqlens_list = old_cu_seqlens_list
        mb_list.align_to_lengths = align_to_lengths
    return mb_list


def unpad_logits(
    logits: torch.Tensor,
    padding_length: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    old_cu_seqlens: Optional[torch.Tensor] = None,
):
    # TODO: when using megatron, logits are in fp32,
    # create new logits in bucket to reduce peak memory usage
    # First unpad batch
    if padding_length > 0:
        logits = logits[:-padding_length]

    # Then unpad according to old_cu_seqlens
    if old_cu_seqlens is not None:
        new_logits = torch.empty(
            (old_cu_seqlens[-1].item(), *logits.shape[1:]),
            dtype=logits.dtype,
            device=logits.device,
        )
        batch_size = old_cu_seqlens.shape[0] - 1
        for i in range(batch_size):
            old_start = old_cu_seqlens[i].item()
            old_end = old_cu_seqlens[i + 1].item()
            start = cu_seqlens[i].item()
            length = old_end - old_start
            new_logits[old_start:old_end] = logits[start : start + length]
        return new_logits

    return logits


def unsqueeze_packed_tensor_dict(
    data: Dict[str, Any],
) -> Dict[str, Any]:
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
        else:
            new_data[key] = value
    return new_data


def unsqueeze_mb_list(
    mb_list: MicroBatchList,
) -> MicroBatchList:
    """Unsqueeze the packed dict of tensors in the micro-batch list."""
    new_padded_mbs = []
    for i, mb in enumerate(mb_list.mbs):
        if mb_list.padded_mbs is not None:
            new_padded_mbs.append(unsqueeze_packed_tensor_dict(mb_list.padded_mbs[i]))
    mb_list.padded_mbs = new_padded_mbs if mb_list.padded_mbs is not None else None
    return mb_list


def amend_position_ids(data: Dict) -> Dict:
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


def broadcast_tensor(tensor: torch.Tensor | None, src_rank=0, group=None):
    """
    Broadcast a tensor from source rank to all other ranks in the process group.

    Args:
        tensor: Tensor on source rank, None on non-source ranks
        src_rank: The rank that holds the tensor to broadcast (default: 0)
        group: The process group to use for broadcasting (default: None, uses the default group)
        device: The device of the output tensor.

    Returns:
        Tensor: The broadcasted tensor on all ranks
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")

    current_rank = dist.get_rank()

    # On source rank, prepare the tensor for broadcasting
    if current_rank == src_rank:
        if tensor is None:
            raise ValueError(f"Tensor cannot be None on source rank {src_rank}")

        tensor = tensor.contiguous()
        device = tensor.device
        # Prepare metadata as Python objects
        metadata = {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "device_type": device.type,
        }

        # Broadcast metadata using broadcast_object_list
        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src=src_rank, group=group)

        # Broadcast the actual tensor
        tensor = tensor.contiguous()
        dist.broadcast(tensor, src=src_rank, group=group)

        return tensor
    else:
        # On non-source ranks, receive metadata
        metadata_list = [None]
        dist.broadcast_object_list(metadata_list, src=src_rank, group=group)

        metadata = metadata_list[0]
        tensor_shape = metadata["shape"]
        dtype = metadata["dtype"]
        device_type = metadata["device_type"]
        device = (
            torch.device("cpu")
            if device_type == "cpu"
            else current_platform.current_device()
        )
        # Create tensor with the received shape and dtype
        tensor = torch.empty(tensor_shape, dtype=dtype, device=device)

        # Receive the actual tensor data
        dist.broadcast(tensor, src=src_rank, group=group)

        return tensor


def _unpad_unflatten(x, shape):
    assert len(x.shape) == 1
    pad_size = x.numel() - np.prod(shape)
    assert pad_size >= 0, pad_size
    return x[: x.numel() - pad_size].view(*shape)


def _flatten_pad_to_max_numel(x, shapes):
    pad_size = max(np.prod(shape) for shape in shapes) - x.numel()
    assert pad_size >= 0, pad_size
    return torch.nn.functional.pad(x.view(-1), (0, pad_size), value=0)


def all_gather_tensor_container(data, group=None) -> List:
    if torch.is_tensor(data):

        local_shape = list(data.shape)
        shapes = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(shapes, local_shape, group=group)

        y = _flatten_pad_to_max_numel(data, shapes)

        ys = [torch.empty_like(y) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(ys, y, group=group)

        return [_unpad_unflatten(y, shape) for y, shape in zip(ys, shapes)]

    if isinstance(data, list):
        data = [all_gather_tensor_container(d, group=group) for d in data]
        return list(zip(*data))

    if isinstance(data, dict):
        results = {
            k: all_gather_tensor_container(v, group=group) for k, v in data.items()
        }
        results = [
            {k: v[i] for k, v in results.items()}
            for i in range(dist.get_world_size(group))
        ]
        return results

    results = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(results, data, group=group)
    return results


def broadcast_tensor_container(data, src_rank=0, group=None):
    if dist.get_rank() != src_rank:
        metadata = [None]
        dist.broadcast_object_list(metadata, src=src_rank, group=group)
        data_type, info = metadata[0]
        if data_type == "none":
            return None
        if data_type == "tensor":
            return broadcast_tensor(data, src_rank=src_rank, group=group)
        elif data_type == "list":
            length = info
            return [
                broadcast_tensor_container(None, src_rank=src_rank, group=group)
                for _ in range(length)
            ]
        elif data_type == "dict":
            keys = info
            return {
                k: broadcast_tensor_container(None, src_rank=src_rank, group=group)
                for k in keys
            }
        elif data_type == "object":
            to_broadcast = [None]
            dist.broadcast_object_list(to_broadcast, src=src_rank, group=group)
            return to_broadcast[0]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    else:
        if data is None:
            metadata = [("none", None)]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return None
        elif torch.is_tensor(data):
            metadata = [("tensor", None)]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return broadcast_tensor(data, src_rank=src_rank, group=group)
        elif isinstance(data, list):
            metadata = [("list", len(data))]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return [
                broadcast_tensor_container(d, src_rank=src_rank, group=group)
                for d in data
            ]
        elif isinstance(data, dict):
            metadata = [("dict", list(data.keys()))]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            return {
                k: broadcast_tensor_container(v, src_rank=src_rank, group=group)
                for k, v in data.items()
            }
        else:
            metadata = [("object", None)]
            dist.broadcast_object_list(metadata, src=src_rank, group=group)
            to_broadcast = [data]
            dist.broadcast_object_list(to_broadcast, src=src_rank, group=group)
            return to_broadcast[0]


def bcast_mb_list(
    mb_list: MicroBatchList | None, src_rank=0, group=None
) -> MicroBatchList:
    if dist.get_rank() == src_rank:
        assert mb_list is not None
    # bcast tensor container attributes
    data = broadcast_tensor_container(
        mb_list.data if mb_list else None, src_rank=src_rank, group=group
    )
    mbs = broadcast_tensor_container(
        mb_list.mbs if mb_list else None, src_rank=src_rank, group=group
    )
    padded_mbs = broadcast_tensor_container(
        mb_list.padded_mbs if mb_list else None, src_rank=src_rank, group=group
    )
    old_cu_seqlens_list = broadcast_tensor_container(
        mb_list.old_cu_seqlens_list if mb_list else None, src_rank=src_rank, group=group
    )
    # bcast other attributes
    to_broadcast = (
        [
            mb_list.mb_spec,
            mb_list.forward_indices,
            mb_list.backward_indices,
            mb_list.group_lens,
            mb_list.padding_lengths,
            mb_list.padded_to_lengths,
            mb_list.align_to_lengths,
        ]
        if mb_list
        else [None for _ in range(7)]
    )
    dist.broadcast_object_list(to_broadcast, src=src_rank, group=group)
    (
        mb_spec,
        forward_indices,
        backward_indices,
        group_lens,
        padding_lengths,
        padded_to_lengths,
        align_to_lengths,
    ) = to_broadcast
    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=mbs,
        forward_indices=forward_indices,
        backward_indices=backward_indices,
        group_lens=group_lens,
        padded_mbs=padded_mbs,
        padding_lengths=padding_lengths,
        padded_to_lengths=padded_to_lengths,
        old_cu_seqlens_list=old_cu_seqlens_list,
        align_to_lengths=align_to_lengths,
    )


def cycle_dataloader(dataloader: StatefulDataLoader):
    """Cycle through a dataloader indefinitely."""
    g = iter(dataloader)
    while True:
        try:
            yield next(g)
        except StopIteration:
            g = iter(dataloader)


class Normalization:
    """
    Adaptive normalization with different levels.

    Supports independent specification of normalization level for mean and std:
    - "batch": normalize across entire batch (with optional all_reduce in distributed setting)
    - "group": normalize within fixed-size groups
    - None: no centering or no std scaling
    """

    def __init__(self, config: NormConfig):
        if config.mean_level not in {"batch", "group", None}:
            raise ValueError(
                f"mean_level must be 'batch', 'group' or None, got {config.mean_level}"
            )
        if config.std_level not in {"batch", "group", None}:
            raise ValueError(
                f"std_level must be 'batch', 'group', or None, got {config.std_level}"
            )
        if (
            config.mean_level == "group" or config.std_level == "group"
        ) and config.group_size is None:
            raise ValueError("group_size must be provided if using group normalization")

        self.mean_level = config.mean_level
        self.mean_leave1out = config.mean_leave1out
        self.std_level = config.std_level
        self.std_unbiased = config.std_unbiased
        self.group_size = config.group_size
        self.eps = config.eps

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        high_precision: bool = True,
        reduce_group=None,
    ) -> torch.Tensor:
        bs = x.size(0)
        eps = self.eps

        # Early return if no elements are active (all masked out)
        if loss_mask is not None and loss_mask.sum().item() == 0:
            return x.float()

        # Step 1: Compute mean
        if self.mean_level == "batch":
            mean = self._compute_mean(
                x,
                loss_mask,
                high_precision=high_precision,
                leave_one_out=self.mean_leave1out,
                all_reduce=True,
                reduce_group=reduce_group,
            )
            mean = mean.expand_as(x)
        elif self.mean_level == "group":
            mean = torch.zeros_like(x)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                xx = x[s]
                m = loss_mask[s] if loss_mask is not None else None

                # Special case: with group_size=1 and leave_one_out=True, mean should be 0
                if self.group_size == 1 and self.mean_leave1out:
                    dtype = torch.float64 if high_precision else torch.float32
                    group_mean = torch.zeros(
                        (1, xx.shape[1]), dtype=dtype, device=xx.device
                    )
                else:
                    group_mean = self._compute_mean(
                        xx,
                        m,
                        high_precision=high_precision,
                        leave_one_out=self.mean_leave1out,
                        all_reduce=False,
                        reduce_group=None,
                    )
                mean[s] = group_mean.expand_as(xx)
        else:  # mean_level == "none"
            mean = torch.zeros_like(x)

        # Subtract mean
        x_centered = x - mean
        # mask unrelevant elements as 0
        if loss_mask is not None:
            x_centered = x_centered * loss_mask

        # Step 2: Compute std
        if self.std_level == "batch":
            std = self._compute_std(
                x,
                loss_mask,
                mean,
                unbiased=self.std_unbiased,
                high_precision=high_precision,
                all_reduce=True,
                reduce_group=reduce_group,
            )
            std = std.expand_as(x)
        elif self.std_level == "group":
            std = torch.zeros_like(x)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                xx = x[s]
                m = loss_mask[s] if loss_mask is not None else None
                group_mean_slice = mean[s]  # already computed and expanded

                # Special case: with group_size=1 and std_unbiased=True, std should be 1 for numerical stability
                if self.group_size == 1 and self.std_unbiased:
                    dtype = torch.float64 if high_precision else torch.float32
                    group_std = torch.ones(
                        (1, xx.shape[1]), dtype=dtype, device=xx.device
                    )
                else:
                    group_std = self._compute_std(
                        xx,
                        m,
                        group_mean_slice,
                        unbiased=self.std_unbiased,
                        high_precision=high_precision,
                        all_reduce=False,
                        reduce_group=reduce_group,
                    )
                std[s] = group_std.expand_as(xx)
        else:
            std = torch.ones_like(x)
            eps = 0.0

        # Normalize
        return (x_centered / (std + eps)).float()

    @staticmethod
    def _compute_mean(
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        high_precision: bool,
        leave_one_out: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute mean only, using masked_normalization internals."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        dim = tuple(range(len(x.shape)))

        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_masked = x
            x_sum = x.sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_sum = x_masked.sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)

        if leave_one_out:
            if factor.item() <= 1:
                return torch.zeros_like(x_sum)
            # For leave-one-out, we need to compute mean excluding each element individually
            # This requires broadcasting: (total_sum - each_element) / (count - 1)
            if mask is None:
                # Broadcast x_sum to original shape and subtract each element
                x_sum_broadcast = x_sum.expand_as(x)
                leave_one_out_sum = x_sum_broadcast - x
                return leave_one_out_sum / (factor - 1)
            else:
                # For masked case, only subtract where mask is 1
                x_sum_broadcast = x_sum.expand_as(x)
                leave_one_out_sum = x_sum_broadcast - x_masked
                # Only compute leave-one-out where mask is 1, elsewhere return global mean
                regular_mean = x_sum / factor
                leave_one_out_mean = leave_one_out_sum / torch.clamp(
                    factor - mask, min=1.0
                )
                return torch.where(
                    mask > 0, leave_one_out_mean, regular_mean.expand_as(x)
                )

        if factor.item() == 0:
            return torch.zeros_like(x_sum)
        return x_sum / factor

    @staticmethod
    def _compute_std(
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mean: torch.Tensor,
        unbiased: bool,
        high_precision: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute std only, given precomputed mean."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        mean = mean.to(dtype)
        dim = tuple(range(len(x.shape)))

        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_centered = x - mean
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_centered = x_masked - mean * mask  # only apply mean where mask is 1
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM, group=reduce_group)

        if unbiased:
            if factor.item() <= 1:
                return torch.ones_like(x_sum_sq)
            return (x_sum_sq / (factor - 1)).sqrt()

        if factor.item() == 0:
            return torch.ones_like(x_sum_sq)
        return (x_sum_sq / factor).sqrt()


class KLEstimator:
    """
    KL divergence estimator, supports k1, k2 and k3.
    """

    def __init__(self, kl_estimator: str = "k1", apply_clamp: bool = True):
        self.kl_estimator = kl_estimator
        if kl_estimator not in ["k1", "k2", "k3"]:
            raise ValueError(
                f"Invalid KL estimator: {kl_estimator}. Valid choices: k1, k2, k3"
            )
        self.apply_clamp = apply_clamp

    def __call__(
        self, log_probs: torch.Tensor, log_probs_base: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_approx_kl(
            log_probs, log_probs_base, self.kl_estimator, self.apply_clamp
        )

    # adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/utils.py#L7
    @staticmethod
    def _compute_approx_kl(
        log_probs: torch.Tensor,
        log_probs_base: torch.Tensor,
        kl_estimator: str = "k1",
        apply_clamp: bool = True,
    ) -> torch.Tensor:
        """
        Compute the approximate KL divergence between two distributions.
        Schulman blog: http://joschu.net/blog/kl-approx.html

        Args:
            log_probs: Log probabilities of the new distribution.
            log_probs_base: Log probabilities of the base distribution.
        """

        if kl_estimator == "k1":
            log_ratio = log_probs.float() - log_probs_base.float()

        # The k2 estimator is the non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # The k2_loss is approximately equivalent to the
        # one-step KL divergence penalty with the k1 estimator
        # used in https://arxiv.org/pdf/2310.10505.
        if kl_estimator == "k2":
            log_ratio = log_probs.float() - log_probs_base.float()
            log_ratio = log_ratio**2 / 2.0

        # The k3 estimator is the non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        if kl_estimator == "k3":
            log_ratio = log_probs.float() - log_probs_base.float()
            log_ratio = -log_ratio
            log_ratio = log_ratio.exp() - 1 - log_ratio

        if apply_clamp:
            log_ratio = log_ratio.clamp(min=-10, max=10)
        return log_ratio
