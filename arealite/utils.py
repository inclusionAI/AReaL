# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from tensorboardX import SummaryWriter

from arealite.api.cli_args import MicroBatchSpec, TrainingArgs
from realhf.base import constants, datapack


def recorder_list(xs: List, indices: List[int]) -> List:
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
    sequence_list: List[Dict[str, torch.Tensor]], pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    if not sequence_list:
        return {}
    max_length = max(len(seq) for item in sequence_list for seq in item.values())
    result = {}
    for key in sequence_list[0].keys():
        padded = [
            torch.nn.functional.pad(
                item[key], (0, max_length - len(item[key])), value=pad_value
            )
            for item in sequence_list
        ]
        result[key] = torch.stack(padded)
    attention_mask = [
        [1] * len(next(iter(item.values())))
        + [0] * (max_length - len(next(iter(item.values()))))
        for item in sequence_list
    ]
    result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
    return result


def unpad_input(
    hidden_states, attention_mask
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
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
    tensor_dicts: List[Dict[str, torch.Tensor]], pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    """Concatenate and pad tensors from multiple padded tensor dictionaries."""
    if not tensor_dicts:
        return {}

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
    return result


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


@dataclass
class MicroBatchSplitResult:
    data: Dict[str, Any]
    mb_spec: MicroBatchSpec
    mbs: List[Dict[str, Any]]
    forward_indices: List[int]
    backward_indices: List[int]


def split_dict_tensor_with_cu_seqlens(
    data: Dict[str, torch.Tensor],
    mb_spec: MicroBatchSpec,
    group: Optional[dist.ProcessGroup] = None,
) -> MicroBatchSplitResult:
    assert "cu_seqlens" in data
    cu_seqlens = data["cu_seqlens"]
    bs = cu_seqlens.shape[0] - 1
    total_lens = int(cu_seqlens[-1])
    input_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy()

    # check tensor shape, split only 1d tensors with length "total_lens"
    to_split = {}
    not_to_split = {}
    keys_to_unsqueeze = set()
    for key, value in data.items():
        if key == "cu_seqlens" or key == "max_seqlen":
            continue
        if not torch.is_tensor(value):
            not_to_split[key] = value
        else:
            assert value.numel() == total_lens, (key, value.shape)
            if value.shape[0] == 1:
                keys_to_unsqueeze.add(key)
                to_split[key] = value.squeeze()
            else:
                to_split[key] = value

    # split
    group_indices = allocate_balanced_mbs_synced(mb_spec, input_lens, group=group)
    splitted_lens = [
        [input_lens[i] for i in group_index] for group_index in group_indices
    ]
    group_lens = [sum(x) for x in splitted_lens]

    forward_indices = datapack.flat2d(group_indices)
    backward_indices = np.zeros(bs, dtype=np.int64)
    backward_indices[forward_indices] = np.arange(bs)

    to_split = dict_map(to_split, lambda x: unpack_sequence(x, cu_seqlens=cu_seqlens))
    to_split = dict_map(to_split, lambda x: recorder_list(x, forward_indices))
    to_split = dict_map(to_split, lambda x: torch.cat(x))
    to_split = dict_map(to_split, lambda x: unpack_sequence(x, lens=group_lens))
    mbs = dict_of_list2list_of_dict(to_split)

    results = []
    # organize splitted micro batches
    assert len(mbs) == len(splitted_lens), (len(mbs), len(splitted_lens))
    for i, (mb, lens) in enumerate(zip(mbs, splitted_lens)):
        mb = {
            k: v if k not in keys_to_unsqueeze else v.unsqueeze(0)
            for k, v in mb.items()
        }
        max_seqlen = max(lens)
        lens = torch.tensor(lens, device="cuda")
        batch_cu_seqlens = torch.nn.functional.pad(
            lens.cumsum(0, dtype=torch.int), (1, 0)
        )
        results.append(
            {
                **mb,
                **not_to_split,
                "max_seqlen": max_seqlen,
                "cu_seqlens": batch_cu_seqlens,
            }
        )
    return MicroBatchSplitResult(
        data=data,
        mbs=results,
        mb_spec=mb_spec,
        forward_indices=forward_indices,
        backward_indices=backward_indices,
    )


@torch.no_grad()
def compute_varlen_position_indices(
    total_seqlen: int,
    cu_seqlens: torch.Tensor,
    seqlen_offsets: Optional[torch.Tensor] = None,
) -> torch.LongTensor:
    indexing_t = torch.arange(
        total_seqlen, dtype=torch.long, device=cu_seqlens.device
    ).unsqueeze_(0)
    indexing_t = (cu_seqlens[:-1].unsqueeze(1) <= indexing_t) & (
        indexing_t < cu_seqlens[1:].unsqueeze(1)
    )
    indices = indexing_t.cumsum(1) - 1
    if seqlen_offsets is not None:
        indices += seqlen_offsets.unsqueeze(1)
    return torch.where(indexing_t, indices, 0).sum(0)


@torch.compile
@torch.no_grad()
def calc_entropy(logits, cu_seqlens):
    probs = torch.nn.functional.softmax(logits.detach().float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
    return entropy


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


def init_stats_logging(args: TrainingArgs):
    """
    Initialize wandb and/or tensorboard according to config.
    If torch.distributed is initialized

    Return:
        tensorboard SummaryWriter if args.tensorboard.path is not None
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # wandb init, connect to remote wandb host
    if args.wandb.mode != "disabled":
        wandb.login()
    wandb.init(
        mode=args.wandb.mode,
        entity=args.wandb.entity,
        project=args.wandb.project or args.experiment_name,
        name=args.wandb.name or args.trial_name,
        job_type=args.wandb.job_type,
        group=args.wandb.group or f"{args.experiment_name}_{args.trial_name}",
        notes=args.wandb.notes,
        tags=args.wandb.tags,
        config=args.wandb.config,
        dir=constants.get_log_path(args),
        force=True,
        id=f"{args.experiment_name}_{args.trial_name}_train",
        resume="allow",
        settings=wandb.Settings(start_method="fork"),
    )
    # tensorboard logging
    summary_writer = None
    if args.tensorboard.path is not None:
        summary_writer = SummaryWriter(log_dir=args.tensorboard.path)

    return summary_writer


def log_wandb_tensorboard(step, data, summary_writer=None):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    wandb.log(data, step=step)
    if summary_writer is not None:
        for key, val in data.items():
            summary_writer.add_scalar(f"{key}", val, step)


def close_wandb_tensorboard(summary_writer=None):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    wandb.finish()
    if summary_writer is not None:
        summary_writer.close()


@contextmanager
def record_timing(name, timing_stats):
    start_time = time.perf_counter()
    yield
    timing_stats[name] = time.perf_counter() - start_time


############### Logging related end ###############


############### Model load start ###############


def get_state_dict_from_repo_id_or_path(repo_id_or_path: str) -> Dict:
    """
    Obtain a state dictionary from either a Hugging Face repo ID or a local path.

    Args:
        repo_id_or_path (str): Either a Hugging Face repo ID (e.g., 'username/model-name')
                              or a local path to a directory containing model weights.

    Returns:
        Dict: The combined state dictionary from all .safetensors and .bin files.
    """
    from safetensors.torch import load_file as safetensors_load

    state_dict = {}

    # Step 1: Identify if the input is a Hugging Face repo ID or local path
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        try:
            validate_repo_id(repo_id_or_path)
            is_hf_repo = True
        except HFValidationError:
            is_hf_repo = False
    except ImportError:
        is_hf_repo = False

    if is_hf_repo:
        from huggingface_hub import snapshot_download

        # Step 2: Download the repo if it's a Hugging Face repo ID
        local_path = snapshot_download(
            repo_id=repo_id_or_path,
        )
    else:
        # Assume it's a local path
        local_path = repo_id_or_path
        if not os.path.isdir(local_path):
            raise ValueError(
                f"Local path {local_path} does not exist or is not a directory, "
                f"or {local_path} is a huggingface repo id but huggingface_hub is not installed."
            )

    # Step 3: Load all .safetensors and .bin files
    file_paths_to_load = []
    for filename in os.listdir(local_path):
        filepath = os.path.join(local_path, filename)
        if filename.endswith(".safetensors") or filename.endswith(".bin"):
            file_paths_to_load.append(filepath)

    def _load(filepath: str):
        if filepath.endswith(".safetensors"):
            state_dict = safetensors_load(filepath)
        elif filepath.endswith(".bin"):
            state_dict = torch.load(filepath, map_location="cpu")
        else:
            raise ValueError(f"{filepath} is not a torch bin or safetensor file.")
        return state_dict

    state_dict = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(
        max_workers=min(4, max(1, os.cpu_count() // 8))
    ) as executor:
        future_to_checkpoint = {
            executor.submit(_load, path): path for path in file_paths_to_load
        }

        for future in as_completed(future_to_checkpoint):
            path = future_to_checkpoint[future]
            try:
                sd = future.result()
                state_dict.update(sd)
            except Exception as e:
                raise RuntimeError(f"Error loading checkpoint from {path}: {e}")
    return state_dict


############### Model load end ###############
