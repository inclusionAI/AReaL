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
from areal.utils.fp8_kernels import blockwise_cast_to_fp8_triton, weight_dequant


def _extract_te_fp8_data(te_tensor):
    """Extract FP8 data and scale_inv from TE FP8 tensor."""
    if hasattr(te_tensor, "_rowwise_data") and hasattr(te_tensor, "_rowwise_scale_inv"):
        # Blockwise tensor
        fp8_data = te_tensor._rowwise_data.view(torch.float8_e4m3fn)
        scale_inv = te_tensor._rowwise_scale_inv
        return fp8_data, scale_inv
    else:
        # Per-tensor quantization
        fp8_data = te_tensor._data.view(torch.float8_e4m3fn)
        scale_inv = te_tensor._scale_inv
        return fp8_data, scale_inv


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp8_conversion_and_matmul():
    """Test FP8 conversion and matrix multiplication correctness."""
    device = torch.device("cuda")
    block_size = [128, 128]

    # Create two BF16 tensors for matrix multiplication
    # A: [M, K], B: [K, N]
    M, K, N = 256, 512, 128
    torch.manual_seed(42)
    a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    _ = b_bf16.transpose(0, 1).contiguous()

    # Step 1: BF16 matrix multiplication baseline
    result_bf16 = torch.matmul(a_bf16, b_bf16)

    # Step 2: Convert BF16 -> TE Blockwise FP8 -> FP8 GEMM -> dequant to BF16
    # Convert A and B to TE Blockwise FP8
    # Note: FP8 GEMM only supports 1D by 1D, 1D by 2D, or 2D by 1D block scaling
    # Not 2D by 2D. We use 1D scaling for input (A) and 2D scaling for weight (B)
    # Following Linear layer pattern: input [M, K] with 1D scaling, weight [N, K] with 2D scaling
    a_te_fp8_step2 = high_precision_to_te_blockwise_fp8(
        a_bf16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        # columnwise=True,
        block_scaling_dim=1,  # 1D scaling for input
    )

    # Transpose B from [K, N] to [N, K] to match Linear layer weight format
    # Linear layer weight is [out_features, in_features] = [N, K]
    b_bf16_t = b_bf16.t().contiguous()  # [K, N] -> [N, K]
    b_te_fp8_step2 = high_precision_to_te_blockwise_fp8(
        b_bf16_t,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        # columnwise=True,
        block_scaling_dim=2,  # 2D scaling for weight
    )

    # Perform FP8 GEMM using general_gemm (same as Linear layer)
    # general_gemm(A, B, workspace, ...) where:
    # - A is weight [N, K] (out_features, in_features)
    # - B is input [M, K] (batch, in_features)
    # - layout="TN" (default): computes B @ A^T = [M, K] @ [K, N] = [M, N]

    # Create output tensor for GEMM result [M, N]
    result_fp8_step2 = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    # Allocate workspace (required by general_gemm)
    workspace = torch.empty(32 * 1024 * 1024 + 1024, dtype=torch.uint8, device=device)

    # Perform FP8 GEMM: result = input @ weight^T where input is [M, K] and weight is [N, K]
    # layout="TN": transa=True (transpose weight), transb=False (no transpose input)
    # Result: [M, K] @ [K, N] = [M, N]
    # Note: Input uses 1D scaling, weight uses 2D scaling (1D by 2D is supported)
    result_fp8_step2, *_ = general_gemm(
        b_te_fp8_step2,  # weight [N, K] with 2D scaling
        a_te_fp8_step2,  # input [M, K] with 1D scaling
        workspace,  # workspace
        out_dtype=torch.bfloat16,  # out_dtype
        layout="TN",  # layout: transa=True, transb=False
        out=result_fp8_step2,  # output [M, N]
        use_split_accumulator=False,  # use_split_accumulator
    )

    # Result is already in BF16, no need to dequantize
    result_step2 = result_fp8_step2

    # Compare with baseline (allowing for quantization error)
    max_diff_step2 = (result_bf16 - result_step2).abs().max().item()
    mean_diff_step2 = (result_bf16 - result_step2).abs().mean().item()
    print(
        f"Step 2 comparison: max_diff={max_diff_step2:.6f}, mean_diff={mean_diff_step2:.6f}"
    )

    my_linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16)
    fp8_recipe = recipe.Float8BlockScaling(fp8_format=recipe.Format.E4M3)
    # my_linear.weight.data.copy_(b_bf16_transpose)
    my_linear.weight.data.copy_(b_bf16_t)

    with te.autocast(enabled=True, recipe=fp8_recipe):
        auto_out_bf16 = my_linear(a_bf16)

    out_bf16 = my_linear(a_bf16)
    print(auto_out_bf16)
    print(out_bf16)
    diff = (out_bf16 - auto_out_bf16).abs().max().item()
    print(f"Step 2 auto fp8 vs bf16 comparison: max_diff={diff:.6f}")
    diff = (out_bf16 - auto_out_bf16).abs().mean().item()
    print(f"Step 2 auto fp8 vs bf16 comparison: mean_diff={diff:.6f}")

    diff = (auto_out_bf16 - result_step2).abs().max().item()
    print(f"Step 2 gemm vs TE Linear comparison: max_diff={diff:.6f}")

    diff = (auto_out_bf16 - result_step2).abs().mean().item()
    print(f"Step 2 gemm vs TE Linear comparison: mean_diff={diff:.6f}")

    diff = (auto_out_bf16 - result_bf16).abs().mean().item()
    print(f"Step 2 gemm vs BF16 comparison: mean_diff={diff:.6f}")
    diff = (auto_out_bf16 - result_bf16).abs().max().item()
    print(f"Step 2 gemm vs BF16 comparison: max_diff={diff:.6f}")

    # Step 2: Allow reasonable quantization error (FP8 has limited precision)
    assert max_diff_step2 < 10.0, f"Step 2 max difference too large: {max_diff_step2}"
    assert mean_diff_step2 < 1.0, f"Step 2 mean difference too large: {mean_diff_step2}"

    # Step 3: Convert BF16 -> PyTorch FP8 -> TE FP8 (via _pytorch_fp8_to_te_fp8) -> dequant -> matmul
    # First convert BF16 to PyTorch FP8
    a_pytorch_fp8_step3, a_scale_inv_step3 = blockwise_cast_to_fp8_triton(
        a_bf16, block_size
    )

    b_pytorch_fp8_step3, b_scale_inv_step3 = blockwise_cast_to_fp8_triton(
        b_bf16, block_size
    )

    # Convert PyTorch FP8 to TE Blockwise FP8 for both A and B
    # Create TE Blockwise FP8 tensors for A
    a_rand = torch.randn(a_bf16.shape, device=device, dtype=torch.bfloat16)
    assert not torch.allclose(a_rand, a_bf16)
    a_te_fp8_step3 = high_precision_to_te_blockwise_fp8(
        a_rand,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,  # FIXME
    )
    # Convert PyTorch FP8 to TE FP8 using _pytorch_fp8_to_te_fp8
    _pytorch_fp8_to_te_fp8(a_pytorch_fp8_step3, a_scale_inv_step3, a_te_fp8_step3)

    # Create TE Blockwise FP8 tensors for B
    b_rand = torch.randn(b_bf16.shape, device=device, dtype=torch.bfloat16)
    assert not torch.allclose(b_rand, b_bf16)
    b_te_fp8_step3 = high_precision_to_te_blockwise_fp8(
        b_rand,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,
    )
    b_te_fp8_step3_ref = high_precision_to_te_blockwise_fp8(
        b_bf16,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        block_scaling_dim=2,
    )

    # Convert PyTorch FP8 to TE FP8 using _pytorch_fp8_to_te_fp8
    _pytorch_fp8_to_te_fp8(b_pytorch_fp8_step3, b_scale_inv_step3, b_te_fp8_step3)

    diff = (b_te_fp8_step3_ref - b_te_fp8_step3).abs().mean().item()
    print(f"Step 3 b te fp8 ref vs te fp8 comparison: mean_diff={diff:.6f}")
    diff = (b_te_fp8_step3_ref - b_te_fp8_step3).abs().max().item()
    print(f"Step 3 b te fp8 ref vs te fp8 comparison: max_diff={diff:.6f}")

    b_bf16_step3 = weight_dequant(
        b_pytorch_fp8_step3, b_scale_inv_step3, dst_dtype=torch.bfloat16
    )

    diff = (b_bf16 - b_bf16_step3).abs().mean().item()
    print(f"Step 3 b pytorch fp8 dequant bf16 vs bf16 comparison: mean_diff={diff:.6f}")
    diff = (b_bf16 - b_bf16_step3).abs().max().item()
    print(f"Step 3 b pytorch fp8 dequant bf16 vs bf16 comparison: max_diff={diff:.6f}")

    # Dequantize both TE FP8 tensors to BF16
    a_dequant_bf16_step3 = a_te_fp8_step3.dequantize(dtype=torch.bfloat16)
    b_dequant_bf16_step3 = b_te_fp8_step3.dequantize(dtype=torch.bfloat16)

    diff = (a_dequant_bf16_step3 - a_bf16).abs().mean().item()
    print(f"Step 3 a dequant vs bf16 comparison: mean_diff={diff:.6f}")
    diff = (a_dequant_bf16_step3 - a_bf16).abs().max().item()
    print(f"Step 3 a dequant vs bf16 comparison: max_diff={diff:.6f}")
    diff = (b_dequant_bf16_step3 - b_bf16).abs().mean().item()
    print(f"Step 3 b dequant vs bf16 comparison: mean_diff={diff:.6f}")
    diff = (b_dequant_bf16_step3 - b_bf16).abs().max().item()
    print(f"Step 3 b dequant vs bf16 comparison: max_diff={diff:.6f}")

    # b_te_fp8_step3 = high_precision_to_te_blockwise_fp8(
    #     b_bf16,
    #     fp8_dtype=tex.DType.kFloat8E4M3,
    #     rowwise=True,
    #     block_scaling_dim=2,
    # )

    # Perform matrix multiplication directly (no autocast)
    # A @ B where A is [M, K] and B is [K, N]
    # result_step3 = torch.matmul(a_dequant_bf16_step3, b_dequant_bf16_step3)
    result_step3 = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    print(b_te_fp8_step3_ref._columnwise_data[0, :10].view(torch.float8_e4m3fn))
    print(b_te_fp8_step3._columnwise_data[0, :10].view(torch.float8_e4m3fn))
    print(b_te_fp8_step3_ref._rowwise_data[:10, 0].view(torch.float8_e4m3fn))
    print(b_te_fp8_step3._rowwise_data[:10, 0].view(torch.float8_e4m3fn))

    result_step3, *_ = general_gemm(
        b_te_fp8_step3,
        # b_te_fp8_step3_ref,
        # a_te_fp8_step3,
        a_te_fp8_step2,
        workspace,
        out_dtype=torch.bfloat16,
        layout="NN",
        out=result_step3,
        use_split_accumulator=False,
    )

    # Compare step 3 with step 2 (both use FP8, but different conversion paths)
    # Step 3: BF16 -> PyTorch FP8 -> TE FP8 -> dequant -> matmul
    # Step 2: BF16 -> TE FP8 -> dequant -> matmul
    max_diff_step3_vs_step2 = (result_step2 - result_step3).abs().max().item()
    mean_diff_step3_vs_step2 = (result_step2 - result_step3).abs().mean().item()
    print(
        f"Step 3 vs Step 2 comparison: max_diff={max_diff_step3_vs_step2:.6f}, mean_diff={mean_diff_step3_vs_step2:.6f}"
    )

    # Assertions

    # Step 3 vs Step 2: Both use FP8 but different conversion paths (direct TE vs PyTorch->TE)
    # They should be reasonably close since both end up as TE FP8 tensors
    assert max_diff_step3_vs_step2 < 10.0, (
        f"Step 3 vs Step 2 max difference too large: {max_diff_step3_vs_step2}"
    )
    assert mean_diff_step3_vs_step2 < 1.0, (
        f"Step 3 vs Step 2 mean difference too large: {mean_diff_step3_vs_step2}"
    )
