"""Test FP8 conversion and matrix multiplication correctness.

This test verifies:
1. BF16 matrix multiplication baseline
2. BF16 -> TE Blockwise FP8 -> FP8 GEMM -> BF16 comparison
3. BF16 -> PyTorch FP8 -> TE FP8 (via _pytorch_fp8_to_te_fp8) -> dequant -> matmul comparison
"""

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)

from areal.models.mcore.hf_load import _pytorch_fp8_to_te_fp8
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.fp8_kernels import blockwise_cast_to_fp8_triton, weight_dequant

logger = logging.getLogger("Test FP8 Conversion")


def high_precision_to_te_blockwise_fp8(
    tensor: torch.Tensor,
    fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
    *,
    rowwise: bool = True,
    columnwise: bool = False,
    block_scaling_dim: int = 2,
    amax_epsilon: float = 0.0,
    force_pow_2_scales: bool = True,
) -> Float8BlockwiseQTensor:
    """
    Quantize high precision tensor to TE Blockwise FP8 tensor.

    Args:
        tensor: High precision tensor (float32, float16, bfloat16, etc.)
        fp8_dtype: TE FP8 format
        rowwise: Whether to use rowwise data layout
        columnwise: Whether to use columnwise data layout
        block_scaling_dim: Block scaling dimension (1 or 2)
        amax_epsilon: Epsilon for amax computation
        force_pow_2_scales: Whether to force power-of-2 scales

    Returns:
        Float8BlockwiseQTensor: TE Blockwise FP8 tensor
    """
    # Create Float8BlockQuantizer
    # Note: Always set both rowwise and columnwise to True to allow GEMM to choose the best layout
    # This matches the test pattern in TransformerEngine tests
    quantizer = Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,  # Always enable rowwise
        columnwise=True,  # Always enable columnwise for flexibility
        amax_epsilon=amax_epsilon,
        force_pow_2_scales=force_pow_2_scales,
        block_scaling_dim=block_scaling_dim,
    )

    # Check if tensor can be quantized (needs to satisfy block size requirements)
    if not quantizer.is_quantizable(tensor):
        raise ValueError(
            f"Tensor shape {tensor.shape} cannot be quantized with block size {quantizer.block_len}. "
            f"Both dimensions must be multiples of {quantizer.block_len}."
        )

    # Quantize tensor
    te_blockwise_fp8_tensor = quantizer(tensor)

    return te_blockwise_fp8_tensor


def _log_tensor_comparison(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str,
    max_threshold: float | None = None,
    mean_threshold: float | None = None,
) -> tuple[float, float]:
    """Compare two tensors and log the differences.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name: Name for logging
        max_threshold: Optional threshold for max difference
        mean_threshold: Optional threshold for mean difference

    Returns:
        Tuple of (max_diff, mean_diff)
    """
    max_diff = (tensor1 - tensor2).abs().max().item()
    mean_diff = (tensor1 - tensor2).abs().mean().item()
    logger.info(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_threshold is not None:
        assert max_diff < max_threshold, f"{name} max difference too large: {max_diff}"

    if mean_threshold is not None:
        assert mean_diff < mean_threshold, (
            f"{name} mean difference too large: {mean_diff}"
        )

    return max_diff, mean_diff


def _create_test_tensors(
    device: torch.device, M: int = 256, K: int = 512, N: int = 128
):
    """Create test tensors for matrix multiplication.

    Args:
        device: Device to create tensors on
        M: First dimension of A
        K: Shared dimension
        N: Second dimension of B

    Returns:
        Tuple of (a_bf16, b_bf16) where A is [M, K] and B is [K, N]
    """
    torch.manual_seed(42)
    a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    return a_bf16, b_bf16


def _perform_fp8_gemm(
    weight_fp8: Float8BlockwiseQTensor,
    input_fp8: Float8BlockwiseQTensor,
    output_shape: tuple[int, ...],
    device: torch.device,
    layout: str = "TN",
) -> torch.Tensor:
    """Perform FP8 GEMM using general_gemm.

    Args:
        weight_fp8: Weight tensor in FP8 format [N, K]
        input_fp8: Input tensor in FP8 format [M, K]
        output_shape: Output shape [M, N]
        device: Device to perform computation on
        layout: GEMM layout ("TN", "NN", etc.)

    Returns:
        Result tensor in BF16 format [M, N]
    """
    result = torch.empty(output_shape, device=device, dtype=torch.bfloat16)
    workspace = torch.empty(32 * 1024 * 1024 + 1024, dtype=torch.uint8, device=device)

    result, *_ = general_gemm(
        weight_fp8,
        input_fp8,
        workspace,
        out_dtype=torch.bfloat16,
        layout=layout,
        out=result,
        use_split_accumulator=False,
    )

    return result


@pytest.fixture
def device():
    return torch.device(current_platform.device_type)


@pytest.fixture
def test_tensors(device):
    """Fixture for test tensors."""
    return _create_test_tensors(device)


def test_te_fp8_gemm_vs_bf16(test_tensors, device):
    """Test BF16 -> TE Blockwise FP8 -> FP8 GEMM -> BF16 comparison."""
    a_bf16, b_bf16 = test_tensors
    M, _, N = a_bf16.shape[0], a_bf16.shape[1], b_bf16.shape[1]

    # BF16 baseline
    result_bf16 = torch.matmul(a_bf16, b_bf16)

    # Convert A to TE Blockwise FP8 with 1D scaling (input pattern)
    a_te_fp8 = high_precision_to_te_blockwise_fp8(
        a_bf16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=1,  # 1D scaling for input
    )

    # Transpose B to match Linear layer weight format [N, K]
    b_bf16_t = b_bf16.t().contiguous()
    b_te_fp8 = high_precision_to_te_blockwise_fp8(
        b_bf16_t,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,  # 2D scaling for weight
    )

    # Perform FP8 GEMM
    result_fp8 = _perform_fp8_gemm(b_te_fp8, a_te_fp8, (M, N), device, layout="TN")

    # Compare with baseline (allowing for quantization error)
    _log_tensor_comparison(
        result_bf16,
        result_fp8,
        "TE FP8 GEMM vs BF16 baseline",
        max_threshold=10.0,
        mean_threshold=1.0,
    )


def test_te_linear_autocast_vs_bf16(test_tensors):
    """Test TransformerEngine Linear with autocast FP8 vs BF16."""
    a_bf16, b_bf16 = test_tensors
    K, N = a_bf16.shape[1], b_bf16.shape[1]

    # Transpose B to match Linear layer weight format [N, K]
    b_bf16_t = b_bf16.t().contiguous()

    # Create Linear layer
    my_linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16)
    my_linear.weight.data.copy_(b_bf16_t)

    # BF16 forward
    out_bf16 = my_linear(a_bf16)

    # FP8 autocast forward
    fp8_recipe = recipe.Float8BlockScaling(fp8_format=recipe.Format.E4M3)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        auto_out_bf16 = my_linear(a_bf16)

    # Compare autocast FP8 vs BF16
    _log_tensor_comparison(
        out_bf16,
        auto_out_bf16,
        "TE Linear autocast FP8 vs BF16",
    )


def test_te_linear_autocast_vs_gemm(test_tensors):
    """Test TransformerEngine Linear autocast FP8 vs manual FP8 GEMM."""
    a_bf16, b_bf16 = test_tensors
    K, N = a_bf16.shape[1], b_bf16.shape[1]
    M = a_bf16.shape[0]

    # Transpose B to match Linear layer weight format [N, K]
    b_bf16_t = b_bf16.t().contiguous()

    # Create Linear layer
    my_linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16)
    my_linear.weight.data.copy_(b_bf16_t)

    # FP8 autocast forward
    fp8_recipe = recipe.Float8BlockScaling(fp8_format=recipe.Format.E4M3)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        auto_out_bf16 = my_linear(a_bf16)

    # Manual FP8 GEMM
    a_te_fp8 = high_precision_to_te_blockwise_fp8(
        a_bf16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=1,
    )
    b_te_fp8 = high_precision_to_te_blockwise_fp8(
        b_bf16_t,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,
    )
    result_gemm = _perform_fp8_gemm(b_te_fp8, a_te_fp8, (M, N), device, layout="TN")

    # Compare
    _log_tensor_comparison(
        auto_out_bf16,
        result_gemm,
        "TE Linear autocast vs manual GEMM",
    )


def test_pytorch_fp8_to_te_fp8_conversion(test_tensors, device):
    """Test BF16 -> PyTorch FP8 -> TE FP8 conversion."""
    a_bf16, b_bf16 = test_tensors
    block_size = [128, 128]

    # Convert BF16 to PyTorch FP8
    a_pytorch_fp8, a_scale_inv = blockwise_cast_to_fp8_triton(a_bf16, block_size)
    b_pytorch_fp8, b_scale_inv = blockwise_cast_to_fp8_triton(b_bf16, block_size)

    # Create TE Blockwise FP8 tensors (initialized with random data)
    a_rand = torch.randn(a_bf16.shape, device=device, dtype=torch.bfloat16)
    assert not torch.allclose(a_rand, a_bf16)
    a_te_fp8 = high_precision_to_te_blockwise_fp8(
        a_rand,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,
    )

    b_rand = torch.randn(b_bf16.shape, device=device, dtype=torch.bfloat16)
    assert not torch.allclose(b_rand, b_bf16)
    b_te_fp8 = high_precision_to_te_blockwise_fp8(
        b_rand,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,
    )

    # Reference: direct BF16 -> TE FP8 conversion
    b_te_fp8_ref = high_precision_to_te_blockwise_fp8(
        b_bf16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,
    )

    # Convert PyTorch FP8 to TE FP8
    _pytorch_fp8_to_te_fp8(a_pytorch_fp8, a_scale_inv, a_te_fp8)
    _pytorch_fp8_to_te_fp8(b_pytorch_fp8, b_scale_inv, b_te_fp8)

    # Compare B conversion (reference vs PyTorch->TE)
    _log_tensor_comparison(
        b_te_fp8_ref.dequantize(dtype=torch.bfloat16),
        b_te_fp8.dequantize(dtype=torch.bfloat16),
        "TE FP8 (direct) vs TE FP8 (PyTorch->TE)",
    )

    # Test dequantization
    b_bf16_dequant = weight_dequant(
        b_pytorch_fp8, b_scale_inv, dst_dtype=torch.bfloat16
    )
    _log_tensor_comparison(
        b_bf16,
        b_bf16_dequant,
        "PyTorch FP8 dequant vs original BF16",
    )

    # Test TE FP8 dequantization
    a_dequant_bf16 = a_te_fp8.dequantize(dtype=torch.bfloat16)
    b_dequant_bf16 = b_te_fp8.dequantize(dtype=torch.bfloat16)

    _log_tensor_comparison(
        a_bf16,
        a_dequant_bf16,
        "TE FP8 dequant A vs original BF16",
    )
    _log_tensor_comparison(
        b_bf16,
        b_dequant_bf16,
        "TE FP8 dequant B vs original BF16",
    )


def test_pytorch_fp8_vs_te_fp8_gemm(test_tensors, device):
    """Test GEMM using PyTorch FP8 -> TE FP8 vs direct TE FP8 conversion paths.

    This test compares two FP8 conversion paths for matrix multiplication:
    1. Direct path: BF16 -> TE Blockwise FP8 -> FP8 GEMM
    2. PyTorch path: BF16 -> PyTorch FP8 -> TE Blockwise FP8 -> FP8 GEMM

    Both paths should produce similar results since they both end up using TE FP8 tensors.
    """
    a_bf16, b_bf16 = test_tensors
    M, _, N = a_bf16.shape[0], a_bf16.shape[1], b_bf16.shape[1]
    block_size = [128, 128]

    # Path 1: Direct BF16 -> TE FP8 conversion
    # Convert input A with 1D scaling (for input pattern)
    a_te_fp8_direct = high_precision_to_te_blockwise_fp8(
        a_bf16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=1,  # 1D scaling for input
    )

    # Convert weight B (transposed) with 2D scaling (for weight pattern)
    b_bf16_t = b_bf16.t().contiguous()  # [K, N] -> [N, K]
    b_te_fp8_direct_t = high_precision_to_te_blockwise_fp8(
        b_bf16_t,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,  # 2D scaling for weight
    )

    # Perform FP8 GEMM with direct path
    result_direct = _perform_fp8_gemm(
        b_te_fp8_direct_t, a_te_fp8_direct, (M, N), device, layout="TN"
    )

    # Path 2: BF16 -> PyTorch FP8 -> TE FP8 conversion
    # Convert to PyTorch FP8 first
    b_pytorch_fp8, b_scale_inv = blockwise_cast_to_fp8_triton(b_bf16, block_size)

    # Convert PyTorch FP8 to TE FP8 for weight B (transposed)
    # Create TE FP8 tensor initialized with random data (will be overwritten)
    b_rand = torch.randn(b_bf16.shape, device=device, dtype=torch.bfloat16)
    b_te_fp8_pytorch_t = high_precision_to_te_blockwise_fp8(
        b_rand,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,  # 2D scaling for weight
    )
    # Convert transposed PyTorch FP8 to TE FP8
    b_pytorch_fp8_t = b_pytorch_fp8.t().contiguous()
    _pytorch_fp8_to_te_fp8(b_pytorch_fp8_t, b_scale_inv, b_te_fp8_pytorch_t)

    # Perform FP8 GEMM with PyTorch -> TE FP8 conversion
    result_pytorch = _perform_fp8_gemm(
        b_te_fp8_pytorch_t, a_te_fp8_direct, (M, N), device, layout="TN"
    )

    # Compare results from both paths
    _log_tensor_comparison(
        result_direct,
        result_pytorch,
        "Direct TE FP8 GEMM vs PyTorch->TE FP8 GEMM",
        max_threshold=10.0,
        mean_threshold=1.0,
    )
