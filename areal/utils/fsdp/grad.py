from collections import defaultdict
from typing import List

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from areal.platforms import current_platform

__all__ = [
    "fsdp2_clip_grad_norm",
]

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
    )

    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        import warnings

        warnings.warn(
            f"Transformer Engine and Apex are not installed. "
            "Falling back to local implementations of multi_tensor_applier, "
            "multi_tensor_l2norm, and multi_tensor_scale"
        )

        from .multi_tensor_apply import (
            local_multi_tensor_applier,
            local_multi_tensor_l2_norm,
            local_multi_tensor_scale,
        )

        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale


def to_local_if_dtensor(tensor: Tensor | DTensor) -> Tensor:
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def device_mesh_has_dim(mesh: DeviceMesh, dim_name: str) -> bool:
    return mesh.mesh_dim_names is not None and dim_name in mesh.mesh_dim_names


def is_param_not_tensor_parallel_duplicate(param, tensor_parallel_rank: int):
    if tensor_parallel_rank == 0:
        return True

    if not isinstance(param, DTensor) or not device_mesh_has_dim(
        param.device_mesh, "tp"
    ):
        return False

    mesh = param.device_mesh
    if mesh.mesh_dim_names:
        placement = param.placements[mesh.mesh_dim_names.index("tp")]
        return not placement.is_replicate()

    return True


def get_main_grads_for_grad_norm(params, tensor_parallel_rank: int) -> List[Tensor]:
    return [
        param.grad
        for param in params
        if param.grad is not None
        and is_param_not_tensor_parallel_duplicate(param, tensor_parallel_rank)
    ]


# Adapted from Megatron-LM
def get_grad_norm_fp32(
    grads_for_norm: List[Tensor] | Tensor,
    data_parallel_group: ProcessGroup,
    model_parallel_group: ProcessGroup,
    norm_type: float = 2.0,
) -> float:
    if isinstance(grads_for_norm, Tensor):
        grads_for_norm = [grads_for_norm]

    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    norm_type = float(norm_type)
    total_norm = 0.0

    if not grads_for_norm:
        return 0.0

    device = current_platform.current_device()

    if norm_type == torch.inf:
        norms = [grad.abs().max() for grad in grads_for_norm]
        total_norm = torch.max(torch.stack(norms)) if norms else 0.0
        total_norm_cuda = torch.tensor(
            [float(total_norm)], dtype=torch.float, device=device
        )
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda,
                op=torch.distributed.ReduceOp.MAX,
                group=data_parallel_group,
            )
        torch.distributed.all_reduce(
            total_norm_cuda,
            op=torch.distributed.ReduceOp.MAX,
            group=model_parallel_group,
        )
        total_norm = float(total_norm_cuda[0].item())
    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=device)
            grad_norm, _ = multi_tensor_applier(
                l2_norm_impl,
                dummy_overflow_buf,
                [grads_for_norm],
                False,
            )
            total_norm_cuda = grad_norm**norm_type
        else:
            total_norm_cuda = torch.tensor([0.0], dtype=torch.float, device=device)
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm_cuda += grad_norm**norm_type

        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda,
                op=torch.distributed.ReduceOp.SUM,
                group=data_parallel_group,
            )
        torch.distributed.all_reduce(
            total_norm_cuda,
            op=torch.distributed.ReduceOp.SUM,
            group=model_parallel_group,
        )
        total_norm = float(total_norm_cuda.item()) ** (1.0 / norm_type)

    return total_norm


# Adapted from Megatron-LM
def clip_grad_by_total_norm_fp32(
    parameters: List[nn.Parameter],
    max_norm: int | float,
    total_norm: float,
):
    # dtype -> grad
    grads = defaultdict(list)
    for param in parameters:
        if param.grad is not None:
            # For naive FSDP, lm_head has bf16 grad while others have fp32 grad
            grad = to_local_if_dtensor(param.grad).detach()
            grads[grad.dtype].append(grad)

    assert len(grads) > 0, len(grads)
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for dtype, _grads in grads.items():
            dummy_overflow_buf = torch.tensor(
                [0], dtype=torch.int, device=current_platform.device_type
            )
            if dtype == torch.float32:
                multi_tensor_applier(
                    multi_tensor_scale_impl,
                    dummy_overflow_buf,
                    [_grads, _grads],
                    clip_coeff,
                )
            else:
                from .multi_tensor_apply import (
                    local_multi_tensor_applier,
                    local_multi_tensor_scale,
                )

                local_multi_tensor_applier(
                    local_multi_tensor_scale,
                    dummy_overflow_buf,
                    [_grads, _grads],
                    clip_coeff,
                )


def fsdp2_clip_grad_norm(
    parameters: List[nn.Parameter],
    nd_device_mesh: DeviceMesh,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    if norm_type <= 0 and norm_type != float("inf"):
        raise ValueError(
            f"Invalid norm_type {norm_type}. Must be a positive float or inf."
        )

    # These dims are guaranteed to exist by FSDPParallelDims
    fsdp_group = nd_device_mesh["dp_sp"].get_group()
    tp_group = nd_device_mesh["tp"].get_group()
    tensor_parallel_rank = dist.get_rank(tp_group)

    grads_for_norm = get_main_grads_for_grad_norm(parameters, tensor_parallel_rank)

    grad_norm = get_grad_norm_fp32(
        grads_for_norm, fsdp_group, tp_group, norm_type=norm_type
    )

    if parameters:
        clip_grad_by_total_norm_fp32(parameters, max_norm, grad_norm)

    return grad_norm
