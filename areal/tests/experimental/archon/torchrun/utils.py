"""Shared utilities for distributed EP tests."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.experimental.models.archon.moe import MoEArgs
from areal.experimental.models.archon.qwen3.model.args import Qwen3ModelArgs
from areal.experimental.models.archon.qwen3.model.model import Qwen3Model


def write_result(out: str, succ: bool) -> None:
    """Write test result to file for pytest verification."""
    with open(out, "w") as f:
        f.write("Passed" if succ else "Failed")


def create_moe_model_args(
    num_experts: int = 4,
    dim: int = 64,
    hidden_dim: int = 128,
    moe_inter_dim: int = 128,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 16,
    n_layers: int = 2,
    vocab_size: int = 1000,
    max_seq_len: int = 8192,
    top_k: int = 2,
) -> Qwen3ModelArgs:
    """Create model args for a small MoE model."""
    return Qwen3ModelArgs(
        dim=dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        attn_type="sdpa",
        moe_enabled=True,
        moe_inter_dim=moe_inter_dim,
        moe_args=MoEArgs(
            num_experts=num_experts,
            top_k=top_k,
            use_grouped_mm=False,
        ),
        decoder_sparse_step=1,
    )


def gather_full_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Gather all parameters including DTensors to full tensors."""
    full_state = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            full_state[name] = param.full_tensor().detach().clone()
        else:
            full_state[name] = param.detach().clone()

    for name, buffer in model.named_buffers():
        if isinstance(buffer, DTensor):
            full_state[name] = buffer.full_tensor().detach().clone()
        else:
            full_state[name] = buffer.detach().clone()

    return full_state


def create_test_input(
    num_seqs: int = 4,
    seq_len_per_seq: int = 8,
    vocab_size: int = 1000,
    device: torch.device | str = "cuda",
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Create test input tensors.

    Returns:
        tokens: (1, total_len) input token ids
        positions: (1, total_len) position ids
        cu_seqlens: (num_seqs + 1,) cumulative sequence lengths
        seq_len_per_seq: sequence length per sequence
    """
    torch.manual_seed(seed)
    total_len = num_seqs * seq_len_per_seq

    tokens = torch.randint(0, vocab_size, (1, total_len), device=device)
    positions = torch.arange(total_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [i * seq_len_per_seq for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=device,
    )

    return tokens, positions, cu_seqlens, seq_len_per_seq


def create_golden_model(
    model_args: Qwen3ModelArgs,
    device: torch.device | str,
    seed: int = 42,
) -> Qwen3Model:
    """Create non-parallelized golden model for comparison."""
    torch.manual_seed(seed)
    model = Qwen3Model(model_args)
    model.init_weights(device)
    model = model.to(device)
    return model


def verify_outputs_match(
    output1: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    max_diff_threshold: float = 1e-2,
) -> tuple[bool, float, float]:
    """Compare two outputs and return (success, max_diff, mean_diff).

    Args:
        output1: First output tensor
        output2: Second output tensor
        rtol: Relative tolerance for allclose
        atol: Absolute tolerance for allclose
        max_diff_threshold: Maximum difference threshold for pass/fail

    Returns:
        success: Whether outputs match within tolerances
        max_diff: Maximum absolute difference
        mean_diff: Mean absolute difference
    """
    max_diff = (output1 - output2).abs().max().item()
    mean_diff = (output1 - output2).abs().mean().item()

    # Use both allclose and max_diff threshold
    allclose = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    within_threshold = max_diff <= max_diff_threshold

    return (allclose or within_threshold), max_diff, mean_diff


def print_rank0(msg: str) -> None:
    """Print message only on rank 0."""
    if dist.get_rank() == 0:
        print(msg)


def create_dense_model_args(
    dim: int = 64,
    hidden_dim: int = 128,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 16,
    n_layers: int = 2,
    vocab_size: int = 1000,
    max_seq_len: int = 8192,
) -> Qwen3ModelArgs:
    """Create model args for a small dense (non-MoE) model."""
    return Qwen3ModelArgs(
        dim=dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        attn_type="sdpa",
        moe_enabled=False,
    )


def validate_gradients(model: Qwen3Model) -> tuple[bool, list[str]]:
    """Verify gradients flow through all key components.

    Args:
        model: The model to check gradients for.

    Returns:
        Tuple of (success, error_messages)
    """
    errors = []

    # Check embedding gradients
    if model.tok_embeddings.weight.grad is None:
        errors.append("tok_embeddings has no gradient")

    # Check each layer
    for layer_id, layer in model.layers.items():
        # Attention
        if layer.attention.wq.weight.grad is None:
            errors.append(f"Layer {layer_id} attention.wq has no gradient")

        # MoE or FFN
        if layer.moe is not None:
            if layer.moe.router.gate.weight.grad is None:
                errors.append(f"Layer {layer_id} MoE router has no gradient")
            if layer.moe.experts.w1.grad is None:
                errors.append(f"Layer {layer_id} MoE experts.w1 has no gradient")
        elif layer.feed_forward is not None:
            if layer.feed_forward.w1.weight.grad is None:
                errors.append(f"Layer {layer_id} FFN has no gradient")

    # Check output layer
    if model.output is not None and model.output.weight.grad is None:
        errors.append("output projection has no gradient")

    return len(errors) == 0, errors


def validate_no_nan(output: torch.Tensor) -> bool:
    """Check for NaN/Inf in output.

    Args:
        output: Output tensor to check.

    Returns:
        True if output is valid (no NaN/Inf), False otherwise.
    """
    if torch.isnan(output).any():
        return False
    if torch.isinf(output).any():
        return False
    return True
