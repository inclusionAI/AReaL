# SPDX-License-Identifier: Apache-2.0
"""
Standalone benchmark for the fused linear-cross-entropy kernel.

Designed to be run *outside* pytest so that NVIDIA Nsight Systems
(``nsys profile``) can capture a clean, deterministic trace covering both
the materialised reference path and the fused Triton path.

NVTX ranges are emitted around each phase so the resulting ``.nsys-rep``
file can be filtered down to just the linear-CE kernels in the Nsight UI.

Usage::

    # Plain run (sanity)
    python -m benchmark.bench_linear_cross_entropy --tokens 4096 --vocab 152064

    # Profile with Nsight Systems
    nsys profile -t nvtx,cuda,cudnn,cublas \\
        -o lce_profile --capture-range cudaProfilerApi --capture-range-end stop \\
        python -m benchmark.bench_linear_cross_entropy \\
            --tokens 4096 --vocab 152064 --use-cuda-profiler-api

See ``docs/perf/nsight_linear_cross_entropy.md`` for a full Nsight workflow.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys

import torch


def _make_inputs(num_tokens, hidden_size, vocab_size, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = (
        torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda", generator=g)
        * 0.02
    )
    weight = (
        torch.randn(vocab_size, hidden_size, dtype=dtype, device="cuda", generator=g)
        * 0.02
    )
    labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda", generator=g)
    return hidden.contiguous(), weight.contiguous(), labels.contiguous()


def _ref_step(hidden, weight, labels, temperature=1.0):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = (h.float() @ w.float().t()) / temperature
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ent = -(probs * log_softmax).sum(dim=-1)
    (lp.sum() + ent.sum()).backward()
    return h.grad, w.grad


def _fused_step(hidden, weight, labels, temperature=1.0):
    from areal.utils.kernel import linear_cross_entropy

    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    lp, ent = linear_cross_entropy(h, w, labels, temperature, "none", None)
    (lp.sum() + ent.sum()).backward()
    return h.grad, w.grad


def _measure(label, fn, hidden, weight, labels, args, warmup, iters):
    nvtx = torch.cuda.nvtx
    times = []
    mems = []

    # Warmup
    nvtx.range_push(f"{label}/warmup")
    for _ in range(warmup):
        fn(hidden, weight, labels)
        gc.collect()
        torch.cuda.empty_cache()
    nvtx.range_pop()

    nvtx.range_push(f"{label}/measure")
    for i in range(iters):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        nvtx.range_push(f"{label}/iter{i}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(hidden, weight, labels)
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()
        times.append(start.elapsed_time(end))
        mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
    nvtx.range_pop()

    return times, mems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=152064)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--use-cuda-profiler-api",
        action="store_true",
        help=(
            "Wrap the measurement region with cudaProfilerStart/Stop so that "
            "`nsys profile --capture-range cudaProfilerApi` only records the "
            "interesting region."
        ),
    )
    parser.add_argument("--mode", choices=["both", "ref", "fused"], default="both")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; aborting.", file=sys.stderr)
        sys.exit(1)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    hidden, weight, labels = _make_inputs(args.tokens, args.hidden, args.vocab, dtype)
    print(
        f"[bench] tokens={args.tokens} hidden={args.hidden} vocab={args.vocab} "
        f"dtype={args.dtype} warmup={args.warmup} iters={args.iters}"
    )

    if args.use_cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()

    results = {}
    if args.mode in ("both", "ref"):
        t, m = _measure("reference", _ref_step, hidden, weight, labels, args, args.warmup, args.iters)
        results["reference"] = (t, m)
    if args.mode in ("both", "fused"):
        t, m = _measure("fused", _fused_step, hidden, weight, labels, args, args.warmup, args.iters)
        results["fused"] = (t, m)

    if args.use_cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()

    for name, (t, m) in results.items():
        median = sorted(t)[len(t) // 2]
        peak = max(m)
        print(f"[bench] {name:9s}  median={median:7.2f}ms  peak_mem={peak:8.1f}MB")

    if "reference" in results and "fused" in results:
        ref_med = sorted(results["reference"][0])[len(results["reference"][0]) // 2]
        fused_med = sorted(results["fused"][0])[len(results["fused"][0]) // 2]
        ref_peak = max(results["reference"][1])
        fused_peak = max(results["fused"][1])
        speedup = ref_med / fused_med if fused_med > 0 else math.inf
        mem_ratio = fused_peak / ref_peak if ref_peak > 0 else math.inf
        print(
            f"[bench] speedup={speedup:.2f}x  fused_peak/ref_peak={mem_ratio:.2f}x"
        )


if __name__ == "__main__":
    main()
