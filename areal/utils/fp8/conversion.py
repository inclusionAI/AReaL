# FP8 conversion utilities between PyTorch and Transformer Engine formats

import torch
from megatron.core.fp8_utils import is_float8tensor

from areal.platforms import current_platform


def torch_fp8_to_te_fp8(
    pytorch_fp8_tensor: torch.Tensor,
    scale_inv: torch.Tensor,
    target_te_tensor: torch.Tensor,
) -> None:
    """Convert PyTorch float8 tensor to Transformer Engine Float8BlockwiseQTensor format inplace.

    This function copies the data and scale_inv from a PyTorch float8 tensor
    to an existing TE Float8BlockwiseQTensor

    Args:
        pytorch_fp8_tensor: PyTorch float8 tensor (like torch.float8_e4m3fn)
        scale_inv: Inverse scale tensor (1/scale) with blockwise shape
        target_te_tensor: Target TE Float8BlockwiseQTensor to copy into
    """
    if not is_float8tensor(target_te_tensor):
        raise ValueError("target_te_tensor must be a Transformer Engine Float8Tensor")

    # For Float8BlockwiseQTensor, copy rowwise_data and rowwise_scale_inv
    if hasattr(target_te_tensor, "_rowwise_data") and hasattr(
        target_te_tensor, "_rowwise_scale_inv"
    ):
        assert pytorch_fp8_tensor.shape == target_te_tensor._rowwise_data.shape
        # rowwise_data is stored in uint8 format
        target_te_tensor._rowwise_data.copy_(
            pytorch_fp8_tensor.view(torch.uint8), non_blocking=False
        )
        scale_inv_shape = scale_inv.shape
        assert len(scale_inv_shape) == 2
        target_te_tensor._rowwise_scale_inv[
            : scale_inv_shape[0], : scale_inv_shape[1]
        ].copy_(scale_inv, non_blocking=False)
        current_platform.synchronize()
        with torch.cuda.device(target_te_tensor.device):
            target_te_tensor._create_columnwise()
    else:
        # Fallback for non-blockwise tensors
        target_te_tensor._data.copy_(pytorch_fp8_tensor.view(torch.uint8))
        target_te_tensor._scale_inv.copy_(scale_inv)
