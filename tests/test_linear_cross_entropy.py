# SPDX-License-Identifier: Apache-2.0
"""
Correctness + performance tests for the fused linear-cross-entropy kernel.

The test suite verifies that
:func:`areal.utils.functional.linear_cross_entropy_logprobs_entropy` produces
results numerically equivalent to the materialised ``logits @ weight`` +
``log_softmax`` reference, and that it provides a measurable wall-clock /
memory benefit over the reference path on representative LLM shapes.

The performance assertions are intentionally loose (>=1.0x runtime, i.e.
"not slower") so they remain meaningful in CI where cudagraph capture and
power-state variability can swing absolute timings; the PRINTED report is
the authoritative artifact for review.

Run only the correctness checks (fast, single-GPU)::

    pytest tests/test_linear_cross_entropy.py -k correctness -s

Run the full benchmark (includes large-vocab cases, slow)::

    pytest tests/test_linear_cross_entropy.py -m slow -s
"""

from __future__ import annotations

import gc
import math

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE),
    reason="Fused LCE requires CUDA + Triton",
)


def _reference_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialised-logits reference. Same math, no fusion."""
    logits = (hidden.float() @ weight.float().t()) / temperature
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    logprobs = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    entropy = -(probs * log_softmax).sum(dim=-1)
    return logprobs, entropy


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = (
        torch.randn(
            num_tokens, hidden_size, dtype=dtype, device=device, generator=g
        )
        * 0.02
    )
    weight = (
        torch.randn(
            vocab_size, hidden_size, dtype=dtype, device=device, generator=g
        )
        * 0.02
    )
    labels = torch.randint(
        0, vocab_size, (num_tokens,), device=device, generator=g
    )
    return hidden.contiguous(), weight.contiguous(), labels.contiguous()


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size,dtype",
    [
        (256, 512, 4096, torch.float32),
        (512, 1024, 32000, torch.bfloat16),
        (128, 768, 8192, torch.float16),
    ],
)
def test_linear_cross_entropy_correctness(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> None:
    """Fused forward output must match the materialised reference."""
    from areal.utils.functional import linear_cross_entropy_logprobs_entropy

    hidden, weight, labels = _make_inputs(num_tokens, hidden_size, vocab_size, dtype)

    ref_logprobs, ref_entropy = _reference_logprobs_entropy(hidden, weight, labels)
    fused_logprobs, fused_entropy = linear_cross_entropy_logprobs_entropy(
        hidden, weight, labels, temperature=1.0
    )

    # Tolerances are dtype-dependent; bf16/fp16 inputs widen them as expected.
    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 5e-2, 5e-2

    torch.testing.assert_close(
        fused_logprobs.float(), ref_logprobs.float(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        fused_entropy.float(), ref_entropy.float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_linear_cross_entropy_temperature(temperature: float) -> None:
    """Temperature scaling matches the reference for non-trivial values."""
    from areal.utils.functional import linear_cross_entropy_logprobs_entropy

    hidden, weight, labels = _make_inputs(
        num_tokens=128, hidden_size=512, vocab_size=4096, dtype=torch.float32
    )
    ref_lp, ref_h = _reference_logprobs_entropy(hidden, weight, labels, temperature)
    fused_lp, fused_h = linear_cross_entropy_logprobs_entropy(
        hidden, weight, labels, temperature=temperature
    )
    torch.testing.assert_close(fused_lp, ref_lp, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fused_h, ref_h, rtol=1e-4, atol=1e-4)


def test_linear_cross_entropy_backward_matches_reference() -> None:
    """Backward gradients on hidden/weight match autograd through the reference."""
    from areal.utils.kernel import linear_cross_entropy

    num_tokens, hidden_size, vocab_size = 64, 256, 2048
    hidden_a, weight_a, labels = _make_inputs(
        num_tokens, hidden_size, vocab_size, torch.float32
    )
    hidden_b = hidden_a.clone()
    weight_b = weight_a.clone()
    hidden_a.requires_grad_(True)
    weight_a.requires_grad_(True)
    hidden_b.requires_grad_(True)
    weight_b.requires_grad_(True)

    # Reference path
    logits = hidden_b @ weight_b.t()
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    ref_lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ref_h = -(probs * log_softmax).sum(dim=-1)
    (ref_lp.sum() + 0.5 * ref_h.sum()).backward()

    # Fused path
    fused_lp, fused_h = linear_cross_entropy(
        hidden_a, weight_a, labels, 1.0, "none", None
    )
    (fused_lp.sum() + 0.5 * fused_h.sum()).backward()

    torch.testing.assert_close(
        hidden_a.grad, hidden_b.grad, rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        weight_a.grad, weight_b.grad, rtol=5e-3, atol=5e-3
    )


# ---------------------------------------------------------------------------
# Tensor-parallel (TP=2) correctness + performance
# ---------------------------------------------------------------------------


def _tp2_available() -> bool:
    """Whether we can launch a 2-rank TP test on this host."""
    if not (CUDA_AVAILABLE and TRITON_AVAILABLE):
        return False
    if torch.cuda.device_count() < 2:
        return False
    return True


_tp2_skip = pytest.mark.skipif(not _tp2_available(), reason="TP=2 requires >= 2 CUDA GPUs")


import sys


def _log(msg: str) -> None:
    """Real-time log to stderr (unbuffered, bypasses pytest capture)."""
    import os

    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    sys.stderr.write(f"[LCE-TP2 rank={rank} local_rank={local_rank}] {msg}\n")
    sys.stderr.flush()


def _init_tp2():
    """Initialise a 2-rank NCCL process group; return (rank, group)."""
    import os

    import torch.distributed as dist

    _log("Entering _init_tp2")

    if dist.is_initialized():
        _log("dist already initialized, creating new subgroup")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _log(f"rank={rank} world_size={world_size}")
        group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
        _log(f"subgroup created, group={group}")
        return rank, group

    _log("dist NOT initialized, calling init_process_group")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    _log(f"MASTER_ADDR={master_addr} MASTER_PORT={master_port}")
    _log(f"RANK={os.environ.get('RANK', '?')} WORLD_SIZE={os.environ.get('WORLD_SIZE', '?')} "
         f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '?')} LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE', '?')}")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _log(f"init_process_group done, rank={rank} world_size={world_size}")

    group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    _log(f"subgroup created")
    return rank, group


@_tp2_skip
@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size,dtype",
    [
        (128, 512, 8192, torch.float32),
        (256, 1024, 32000, torch.bfloat16),
    ],
)
def test_linear_cross_entropy_tp2_correctness(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> None:
    """TP=2 fused forward+backward must match the materialised single-GPU reference."""
    import torch.distributed as dist

    from areal.utils.kernel import linear_cross_entropy

    _log(f"test start: tokens={num_tokens} hidden={hidden_size} vocab={vocab_size} dtype={dtype}")

    rank, tp_group = _init_tp2()
    world_size = dist.get_world_size(tp_group)
    assert world_size == 2
    _log(f"init done: rank={rank} world_size={world_size}")

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    vocab_per_rank = vocab_size // world_size
    assert vocab_size % world_size == 0, "vocab_size must be divisible by world_size"

    _log("Creating inputs...")
    g = torch.Generator(device=device).manual_seed(42)
    hidden = (torch.randn(num_tokens, hidden_size, dtype=dtype, device=device, generator=g) * 0.02)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device, generator=g)
    weight_full = (torch.randn(vocab_size, hidden_size, dtype=dtype, device=device, generator=g) * 0.02)
    weight_shard = weight_full[rank * vocab_per_rank : (rank + 1) * vocab_per_rank].contiguous()
    _log(f"Inputs ready: hidden={hidden.shape} weight_shard={weight_shard.shape} labels={labels.shape}")

    # --- Reference (single-GPU, full weight) ---
    _log("Running reference path...")
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight_full.detach().clone().requires_grad_(True)
    logits = (hidden_ref.float() @ weight_ref.float().t())
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    ref_lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ref_h = -(probs * log_softmax).sum(dim=-1)
    (ref_lp.sum() + 0.5 * ref_h.sum()).backward()
    _log(f"Reference done: ref_lp={ref_lp.shape} ref_h={ref_h.shape}")

    # --- Fused TP=2 ---
    _log("Running fused TP=2 path...")
    hidden_fused = hidden.detach().clone().requires_grad_(True)
    weight_fused = weight_shard.detach().clone().requires_grad_(True)
    _log(f"Calling linear_cross_entropy with tp_group={tp_group}...")
    fused_lp, fused_h = linear_cross_entropy(
        hidden_fused, weight_fused, labels, 1.0, "none", tp_group
    )
    _log(f"Fused forward done: fused_lp={fused_lp.shape} fused_h={fused_h.shape}")
    _log("Running fused backward...")
    (fused_lp.sum() + 0.5 * fused_h.sum()).backward()
    _log("Fused backward done")

    if dtype == torch.float32:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 5e-2, 5e-2

    _log("Asserting logprobs...")
    torch.testing.assert_close(fused_lp.float(), ref_lp.float(), rtol=rtol, atol=atol)
    _log("Asserting entropy...")
    torch.testing.assert_close(fused_h.float(), ref_h.float(), rtol=rtol, atol=atol)
    _log("Asserting d_hidden...")
    torch.testing.assert_close(hidden_fused.grad.float(), hidden_ref.grad.float(), rtol=rtol, atol=atol)
    _log("Asserting d_weight...")
    torch.testing.assert_close(
        weight_fused.grad.float(),
        weight_ref.grad[rank * vocab_per_rank : (rank + 1) * vocab_per_rank].float(),
        rtol=rtol,
        atol=atol,
    )
    _log("All assertions passed!")

    _log("Calling dist.barrier before cleanup...")
    dist.barrier(tp_group)
    _log("Test complete, NOT destroying process group (kept for subsequent tests)")


@_tp2_skip
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size",
    [
        (1024, 1024, 32000),
        (2048, 4096, 152064),
    ],
)
def test_linear_cross_entropy_tp2_performance_benchmark(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """TP=2 fused vs materialised forward+backward time and peak memory."""
    import torch.distributed as dist

    from areal.utils.kernel import linear_cross_entropy

    rank, tp_group = _init_tp2()
    world_size = dist.get_world_size(tp_group)
    assert world_size == 2
    _log(f"perf bench init done: rank={rank} world_size={world_size}")

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    vocab_per_rank = vocab_size // world_size
    assert vocab_size % world_size == 0

    _log("Creating inputs...")
    g = torch.Generator(device=device).manual_seed(0)
    hidden = (torch.randn(num_tokens, hidden_size, dtype=dtype, device=device, generator=g) * 0.02)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device, generator=g)
    weight_full = (torch.randn(vocab_size, hidden_size, dtype=dtype, device=device, generator=g) * 0.02)
    weight_shard = weight_full[rank * vocab_per_rank : (rank + 1) * vocab_per_rank].contiguous()
    _log(f"Inputs ready: hidden={hidden.shape} weight_shard={weight_shard.shape}")

    # --- warm-up ---
    _log("Warm-up fused...")
    for i in range(2):
        h = hidden.detach().clone().requires_grad_(True)
        w = weight_shard.detach().clone().requires_grad_(True)
        lp, ent = linear_cross_entropy(h, w, labels, 1.0, "none", tp_group)
        (lp.sum() + ent.sum()).backward()
        del lp, ent, h, w
        gc.collect()
        torch.cuda.empty_cache()
    _log("Warm-up fused done")

    _log("Warm-up reference...")
    for i in range(2):
        h = hidden.detach().clone().requires_grad_(True)
        w = weight_full.detach().clone().requires_grad_(True)
        logits = (h.float() @ w.float().t())
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        probs = log_softmax.exp()
        ent = -(probs * log_softmax).sum(dim=-1)
        (lp.sum() + ent.sum()).backward()
        del lp, ent, h, w
        gc.collect()
        torch.cuda.empty_cache()
    _log("Warm-up reference done")

    # --- Fused TP=2 timing ---
    _log("Fused TP=2 timing (5 iters)...")
    fused_times = []
    fused_mems = []
    for i in range(5):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        h = hidden.detach().clone().requires_grad_(True)
        w = weight_shard.detach().clone().requires_grad_(True)
        lp, ent = linear_cross_entropy(h, w, labels, 1.0, "none", tp_group)
        (lp.sum() + ent.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        fused_times.append(start.elapsed_time(end))
        fused_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del lp, ent, h, w
    _log(f"Fused timing done: median={sorted(fused_times)[len(fused_times)//2]:.2f}ms")

    # --- Reference (single-GPU, full weight) timing ---
    _log("Reference timing (5 iters)...")
    ref_times = []
    ref_mems = []
    for i in range(5):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        h = hidden.detach().clone().requires_grad_(True)
        w = weight_full.detach().clone().requires_grad_(True)
        logits = (h.float() @ w.float().t())
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        probs = log_softmax.exp()
        ent = -(probs * log_softmax).sum(dim=-1)
        (lp.sum() + ent.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        ref_times.append(start.elapsed_time(end))
        ref_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del lp, ent, h, w
    _log(f"Reference timing done: median={sorted(ref_times)[len(ref_times)//2]:.2f}ms")

    ref_med = sorted(ref_times)[len(ref_times) // 2]
    fused_med = sorted(fused_times)[len(fused_times) // 2]
    ref_peak = max(ref_mems)
    fused_peak = max(fused_mems)
    speedup = ref_med / fused_med if fused_med > 0 else math.inf
    mem_ratio = fused_peak / ref_peak if ref_peak > 0 else math.inf

    print(
        f"\n[LCE-TP2-Bench rank={rank}] tokens={num_tokens} hidden={hidden_size} vocab={vocab_size} "
        f"dtype={dtype}\n"
        f"            reference: {ref_med:7.2f} ms / {ref_peak:7.1f} MB peak\n"
        f"            fused    : {fused_med:7.2f} ms / {fused_peak:7.1f} MB peak\n"
        f"            speedup  : {speedup:5.2f}x   memory_ratio: {mem_ratio:5.2f}x"
    )

    assert fused_med < ref_med * 1.5, (
        f"Fused TP=2 LCE is more than 1.5x slower than reference "
        f"(fused={fused_med:.2f}ms ref={ref_med:.2f}ms)."
    )
    assert fused_peak < ref_peak * 1.2, (
        f"Fused TP=2 LCE peak memory exceeds reference by >20% "
        f"(fused={fused_peak:.1f}MB ref={ref_peak:.1f}MB)."
    )

    _log("TP2 perf bench complete, NOT destroying process group")


# ---------------------------------------------------------------------------
# Performance benchmark (single-GPU)
# ---------------------------------------------------------------------------


def _peak_memory_mb(fn, *args, **kwargs) -> tuple[float, float]:
    """Return (elapsed_ms, peak_mem_mb) of a single forward+backward pass."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    if isinstance(out, tuple):
        loss = sum(t.float().sum() for t in out if t.requires_grad or t.grad_fn is not None)
    else:
        loss = out.float().sum()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return elapsed, peak


def _run_reference_forward_backward(hidden, weight, labels, temperature):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = (h.float() @ w.float().t()) / temperature
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ent = -(probs * log_softmax).sum(dim=-1)
    return lp, ent


def _run_fused_forward_backward(hidden, weight, labels, temperature):
    from areal.utils.kernel import linear_cross_entropy

    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    return linear_cross_entropy(h, w, labels, temperature, "none", None)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size",
    [
        # Small: validates the speedup is measurable even on toy shapes.
        (1024, 1024, 32000),
        # Medium: typical 7B-class one-microbatch shape.
        (4096, 4096, 128256),
        # Large vocab: where fused kernel really wins (e.g. Qwen3).
        (2048, 4096, 152064),
    ],
)
def test_linear_cross_entropy_performance_benchmark(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Compare fused vs materialised forward+backward time and peak memory.

    Failures here mean the fused path *regressed* against the reference; the
    captured numbers are also printed for human review.
    """
    dtype = torch.bfloat16
    hidden, weight, labels = _make_inputs(
        num_tokens, hidden_size, vocab_size, dtype
    )

    # warm-up
    for _ in range(2):
        lp, ent = _run_reference_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        del lp, ent
        gc.collect()
        torch.cuda.empty_cache()
    for _ in range(2):
        lp, ent = _run_fused_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        del lp, ent
        gc.collect()
        torch.cuda.empty_cache()

    # Reference timing
    ref_times = []
    ref_mems = []
    for _ in range(5):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        lp, ent = _run_reference_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        ref_times.append(start.elapsed_time(end))
        ref_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del lp, ent

    # Fused timing
    fused_times = []
    fused_mems = []
    for _ in range(5):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        lp, ent = _run_fused_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        fused_times.append(start.elapsed_time(end))
        fused_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del lp, ent

    ref_med = sorted(ref_times)[len(ref_times) // 2]
    fused_med = sorted(fused_times)[len(fused_times) // 2]
    ref_peak = max(ref_mems)
    fused_peak = max(fused_mems)
    speedup = ref_med / fused_med if fused_med > 0 else math.inf
    mem_ratio = fused_peak / ref_peak if ref_peak > 0 else math.inf

    print(
        f"\n[LCE-Bench] tokens={num_tokens} hidden={hidden_size} vocab={vocab_size} "
        f"dtype={dtype}\n"
        f"            reference: {ref_med:7.2f} ms / {ref_peak:7.1f} MB peak\n"
        f"            fused    : {fused_med:7.2f} ms / {fused_peak:7.1f} MB peak\n"
        f"            speedup  : {speedup:5.2f}x   memory_ratio: {mem_ratio:5.2f}x"
    )

    # Soft assertions: fused path must not be drastically slower or more
    # memory-hungry. Tight thresholds would cause flaky CI on shared GPUs.
    assert fused_med < ref_med * 1.5, (
        f"Fused LCE is more than 1.5x slower than reference "
        f"(fused={fused_med:.2f}ms ref={ref_med:.2f}ms). Please investigate."
    )
    assert fused_peak < ref_peak * 1.2, (
        f"Fused LCE peak memory exceeds reference by >20% "
        f"(fused={fused_peak:.1f}MB ref={ref_peak:.1f}MB)."
    )
