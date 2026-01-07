"""Forward pass profiling tool for Archon models.

A simple forward profiling tool to analyze op-level performance bottlenecks
in Archon model forward passes using torch.profiler.

Usage:
    python -m areal.tools.profile_archon_forward [OPTIONS]

Examples:
    # Default configuration
    python -m areal.tools.profile_archon_forward

    # Variable-length sequences
    python -m areal.tools.profile_archon_forward --seq-lens 128,256,512,128

    # Custom output path
    python -m areal.tools.profile_archon_forward --output /tmp/my_trace.json

    # Long sequence test
    python -m areal.tools.profile_archon_forward --batch-size 1 --total-len 16384
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity

from areal.platforms import current_platform


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile Archon model forward pass to identify performance bottlenecks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Use default configuration
  %(prog)s --seq-lens 128,256,512,128   # Specify variable sequence lengths
  %(prog)s --output /tmp/trace.json     # Custom output path
  %(prog)s --batch-size 1 --total-len 16384  # Long sequence test
""",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model. Defaults to Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences in packed batch (default: 4)",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=None,
        help="Comma-separated sequence lengths, e.g. '128,256,512,128'. "
        "If not specified, generates random lengths.",
    )
    parser.add_argument(
        "--total-len",
        type=int,
        default=10240,
        help="Total token count when seq-lens is not specified (default: 10240)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=5,
        help="Number of profiling iterations (default: 5)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for Chrome trace JSON. "
        "Defaults to ./profile_archon_forward_<timestamp>.json",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory profiling",
    )
    return parser.parse_args(argv)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]


def get_output_path(output: str | None) -> Path:
    """Get output path for Chrome trace."""
    if output:
        return Path(output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"./profile_archon_forward_{timestamp}.json")


def create_packed_input(
    batch_size: int,
    seq_lens: list[int] | None,
    total_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict:
    """Create variable-length packed input with cu_seqlens.

    Args:
        batch_size: Number of sequences if seq_lens is None.
        seq_lens: Explicit sequence lengths. If None, generates random lengths.
        total_len: Total token count when seq_lens is None.
        vocab_size: Vocabulary size for random token generation.
        device: Target device.

    Returns:
        dict with input_ids, cu_seqlens, max_seqlen, and seq_lens.
    """
    if seq_lens is None:
        # Generate random variable lengths that sum to approximately total_len
        seq_lens = []
        remaining = total_len
        for i in range(batch_size - 1):
            # Random length between 64 and remaining / (batch_size - i)
            avg_remaining = remaining // (batch_size - i)
            min_len = max(64, avg_remaining // 2)
            max_len = min(remaining - 64 * (batch_size - i - 1), avg_remaining * 2)
            length = random.randint(min_len, max_len)
            seq_lens.append(length)
            remaining -= length
        seq_lens.append(remaining)

    actual_total_len = sum(seq_lens)
    input_ids = torch.randint(
        100, vocab_size - 100, (1, actual_total_len), device=device
    )

    # Build cu_seqlens from variable lengths
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(seq_lens)),
        dtype=torch.int32,
        device=device,
    )
    max_seqlen = max(seq_lens)

    return {
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "seq_lens": seq_lens,
    }


def run_profile(args: argparse.Namespace) -> None:
    """Run profiling with given configuration."""
    # Setup environment
    if "LOCAL_RANK" not in os.environ:
        current_platform.set_device(0)

    device = torch.device(current_platform.device_type)
    dtype = get_dtype(args.dtype)

    # Get model path
    from areal.tests.experimental.archon.utils import MODEL_PATHS, load_archon_model

    model_path = args.model_path or MODEL_PATHS["qwen2"]

    # Parse seq_lens if provided
    seq_lens = None
    if args.seq_lens:
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
        args.batch_size = len(seq_lens)

    print("\n" + "=" * 70)
    print("ARCHON FORWARD PROFILING")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch size: {args.batch_size}")
    if seq_lens:
        print(f"Sequence lengths: {seq_lens}")
    else:
        print(f"Total length: {args.total_len} (random variable lengths)")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Profile iterations: {args.profile_iters}")
    print(f"Memory profiling: {'disabled' if args.no_memory else 'enabled'}")
    print("=" * 70)

    # Load model
    print("\n[Profile] Loading Archon model...")
    model, _ = load_archon_model(model_path, dtype=dtype)

    # Get vocab size from model config
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = config.vocab_size

    # Create packed input
    torch.manual_seed(42)
    random.seed(42)
    inputs = create_packed_input(
        batch_size=args.batch_size,
        seq_lens=seq_lens,
        total_len=args.total_len,
        vocab_size=vocab_size,
        device=device,
    )

    print("\n[Profile] Input configuration:")
    print(f"  Sequences: {len(inputs['seq_lens'])}")
    print(f"  Sequence lengths: {inputs['seq_lens']}")
    print(f"  Total tokens: {sum(inputs['seq_lens'])}")
    print(f"  Max sequence length: {inputs['max_seqlen']}")

    # Warmup
    print(f"\n[Profile] Running warmup ({args.warmup_iters} iterations)...")
    for i in range(args.warmup_iters):
        with torch.no_grad():
            _ = model(
                inputs["input_ids"],
                positions=None,
                cu_seqlens=inputs["cu_seqlens"],
                max_seqlen=inputs["max_seqlen"],
            )
        torch.cuda.synchronize()
        print(f"  Warmup {i + 1}/{args.warmup_iters} done")

    # Profile
    print(f"\n[Profile] Running profiler ({args.profile_iters} iterations)...")
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=not args.no_memory,
    ) as prof:
        for _ in range(args.profile_iters):
            with torch.no_grad():
                _ = model(
                    inputs["input_ids"],
                    positions=None,
                    cu_seqlens=inputs["cu_seqlens"],
                    max_seqlen=inputs["max_seqlen"],
                )
            torch.cuda.synchronize()

    # Print results - sorted by CUDA time
    print("\n" + "=" * 80)
    print("PROFILING RESULTS (sorted by CUDA time)")
    print("=" * 80)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )

    # Export Chrome trace
    output_path = get_output_path(args.output)
    prof.export_chrome_trace(str(output_path))
    print(f"\n[Profile] Chrome trace exported to: {output_path}")
    print("[Profile] View with: chrome://tracing or https://ui.perfetto.dev/")

    # Print memory stats if enabled
    if not args.no_memory:
        print("\n" + "=" * 80)
        print("MEMORY STATS")
        print("=" * 80)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=10,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    run_profile(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
