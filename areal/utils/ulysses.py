# Adapted from verl

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup | None):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> dist.ProcessGroup | None:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup | None = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup | None = None) -> int:
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup | None = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(
    x: Tensor, head_dim: int, seq_dim: int, group: Optional[ProcessGroup] = None
) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    if padding_size == 0:
        return x
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    if padding_size == 0:
        return x
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def slice_input_tensor(
    x: Tensor, dim: int, padding: bool = True, group: Optional[dist.ProcessGroup] = None
) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)
    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    parts = x.size(dim) // sp_world_size
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    return x[slc].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(local_input, sp_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(sp_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> tuple[None, Tensor, None, None, None, None]:
        input_t = (
            torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
            if ctx.async_op
            else grad_output[0]
        )
        return (
            None,
            all_to_all_tensor(
                input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False
            ),
            None,
            None,
            None,
            None,
        )


def ulysses_pad(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    sp_size: int = 1,
):
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(
            input_ids_rmpad, (0, pad_size), value=0
        )
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(
                pad_size, device=position_ids_rmpad.device
            ).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(3, 1, 1)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    sp_size: int = 1,
):
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
        input_ids_rmpad, position_ids_rmpad, sp_size
    )
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(
            position_ids_rmpad, dim=1, padding=False
        )
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_prepare_inputs(
    padded_mb_input,
    ulysses_input_ids,
    ulysses_position_ids,
    sp_world_size,
):
    # init inputs with padded_mb_input and ulysses_inputs
    inputs = padded_mb_input.copy()
    inputs["input_ids"] = ulysses_input_ids
    if ulysses_position_ids is not None:
        inputs["position_ids"] = ulysses_position_ids

    # Pad and slice the loss inputs
    padded_input_ids = padded_mb_input["input_ids"]

    for key, value in list(inputs.items()):
        if key in {"input_ids", "position_ids"}:
            continue
        if not torch.is_tensor(value):
            continue

        if value.dim() >= 2 and value.shape[:2] == padded_input_ids.shape[:2]:
            # Please refer to ppo_loss_fn() in areal/engine/ppo/critic.py
            if key in {"values", "returns", "loss_mask"}:
                # For loss_mask, also keep the full version for loss function
                if key == "loss_mask":
                    inputs["full_loss_mask"] = value.squeeze(0)

                sliced_value = slice_input_tensor(value, dim=1, padding=True)
                inputs[key] = sliced_value.squeeze(0)
            else:
                inputs[key] = value.squeeze(0)

    # Roll and slice the full input_ids as the labels in Ulysses SP.
    rolled_input_ids = torch.roll(padded_input_ids, shifts=-1, dims=-1)
    rolled_input_ids, _, _ = ulysses_pad_and_slice_inputs(
        rolled_input_ids, sp_size=sp_world_size
    )
    inputs["rolled_input_ids"] = rolled_input_ids.squeeze(0)
    return inputs
