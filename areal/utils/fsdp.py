import math
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import PreTrainedModel

from areal.platforms import current_platform
from areal.utils import logging, pkg_version

if pkg_version.is_version_greater_or_equal("torch", "2.6.0"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )
else:
    raise ModuleNotFoundError(
        "Current PyTorch version < 2.6.0 is not supported for FSDPEngine."
    )


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


__all__ = [
    "CPUOffloadPolicy",
    "MixedPrecisionPolicy",
    "fully_shard",
    "fsdp2_clip_grad_norm",
    "create_fsdp_device_mesh",
    "apply_fsdp2",
    "fsdp2_load_full_state_dict",
    "get_cosine_schedule_with_warmup",
]


logger = logging.getLogger("FSDPEngine")


def to_local_if_dtensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
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


def get_main_grads_for_grad_norm(
    params, tensor_parallel_rank: int
) -> List[torch.Tensor]:
    return [
        param.grad
        for param in params
        if param.grad is not None
        and is_param_not_tensor_parallel_duplicate(param, tensor_parallel_rank)
    ]


# Adapted from Megatron-LM
def get_grad_norm_fp32(
    grads_for_norm: List[torch.Tensor] | torch.Tensor,
    data_parallel_group: ProcessGroup,
    model_parallel_group: ProcessGroup,
    norm_type: float = 2.0,
) -> float:
    if isinstance(grads_for_norm, torch.Tensor):
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
        total_norm = total_norm_cuda[0].item()
    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=device)
            grad_norm, _ = multi_tensor_applier(
                l2_norm_impl,
                dummy_overflow_buf,
                [grads_for_norm],
                False,
            )
            total_norm = grad_norm**norm_type
        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )
        torch.distributed.all_reduce(
            total_norm,
            op=torch.distributed.ReduceOp.SUM,
            group=model_parallel_group,
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


# Adapted from Megatron-LM
def clip_grad_by_total_norm_fp32(
    parameters: List[torch.Tensor] | torch.Tensor,
    max_norm: int | float,
    total_norm: float,
):
    params = []
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == "torch.cuda.FloatTensor"
            params.append(param)
            grads.append(to_local_if_dtensor(param.grad).detach())

    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.tensor(
            [0], dtype=torch.int, device=current_platform.device_type
        )
        multi_tensor_applier(
            multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff
        )


def fsdp2_clip_grad_norm(
    parameters,
    nd_device_mesh: DeviceMesh,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    assert device_mesh_has_dim(nd_device_mesh, "fsdp") and device_mesh_has_dim(
        nd_device_mesh, "tp"
    ), "fsdp2_clip_grad_norm requires a ['fsdp', 'tp'] device mesh."

    if norm_type <= 0 and norm_type != float("inf"):
        raise ValueError(
            f"Invalid norm_type {norm_type}. Must be a positive float or inf."
        )

    fsdp_group = nd_device_mesh["fsdp"].get_group()
    tp_group = nd_device_mesh["tp"].get_group()
    tensor_parallel_rank = dist.get_rank(tp_group)

    grads_for_norm = get_main_grads_for_grad_norm(parameters, tensor_parallel_rank)

    grad_norm = get_grad_norm_fp32(
        grads_for_norm, fsdp_group, tp_group, norm_type=norm_type
    )

    if parameters:
        clip_grad_by_total_norm_fp32(parameters, max_norm, grad_norm)

    return grad_norm


def create_fsdp_device_mesh(shard_size, world_size):
    if shard_size < 0 or shard_size >= world_size:
        device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(world_size,),
            mesh_dim_names=("fsdp",),
        )
    else:
        device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(world_size // shard_size, shard_size),
            mesh_dim_names=("ddp", "fsdp"),
        )
    return device_mesh


def apply_fsdp2(model, fsdp_kwargs, wrap_policy):
    """model: AutoModelForCausalLM"""
    assert (
        CPUOffloadPolicy is not None
    ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", list())
    fsdp_transformer_layer_cls_to_wrap = (
        wrap_policy.transformer_layer_cls_to_wrap if wrap_policy is not None else list()
    )
    if not fsdp_transformer_layer_cls_to_wrap:
        fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(
        model, **fsdp_kwargs
    )  # fsdp2 will not reshard_after_forward for root module


def fsdp2_load_full_state_dict(
    model: PreTrainedModel,
    full_state: dict,
    cpu_offload=None,
    tie_word_embeddings=False,
):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    device = current_platform.current_device()
    model = model.to(device=device, non_blocking=True)
    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=cpu_offload,
        broadcast_from_rank0=True,
        strict=not tie_word_embeddings,
    )
    set_model_state_dict(model, full_state, options=options)

    if tie_word_embeddings:
        model.tie_weights()

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
