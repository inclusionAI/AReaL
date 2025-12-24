"""Shared utilities for FP8/BF16 comparison tests.

This module contains common helper functions, fixtures, and constants
used across multiple FP8/BF16 comparison test files.
"""

import functools
import os
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.engine.megatron_engine import MegatronEngine
from areal.utils import logging
from areal.utils.data import (
    broadcast_tensor,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
    unpad_logits,
)
from areal.utils.functional import gather_logprobs
from areal.utils.mcore.packed_context_parallel import packed_context_parallel_forward

logger = logging.getLogger("FP8 BF16 Comparison Utils")


def extract_gemm_kernels(profiler, phase: str = "forward"):
    """Extract and summarize GEMM-related kernels from profiler output.

    Args:
        profiler: torch.profiler.profile instance
        phase: Phase name ("forward" or "backward")

    Returns:
        Dictionary with gemm kernel statistics
    """
    gemm_keywords = ["gemm", "cublas", "cutlass", "matmul", "mm", "bmm"]

    gemm_events = []

    # Get all events from profiler - iterate through all events to find CUDA kernels
    try:
        # Try to get events() which gives us raw events
        all_events = list(profiler.events())
    except Exception:
        # Fallback to key_averages() if events() is not available
        all_events = list(profiler.key_averages())

    for event in all_events:
        # Get event name - try different attributes
        event_name = None
        if hasattr(event, "key"):
            event_name = event.key
        elif hasattr(event, "name"):
            event_name = event.name
        elif hasattr(event, "__str__"):
            event_name = str(event)
        else:
            continue

        # Check if this is a CUDA kernel event
        # CUDA kernels typically have specific attributes
        is_cuda_kernel = False
        if hasattr(event, "is_cuda") and event.is_cuda:
            is_cuda_kernel = True
        elif (
            hasattr(event, "device_type") and event.device_type == 1
        ):  # CUDA device type
            is_cuda_kernel = True
        elif "cuda" in str(type(event)).lower() or "kernel" in event_name.lower():
            is_cuda_kernel = True

        # Check if this is a gemm-related kernel
        event_name_lower = event_name.lower()
        if is_cuda_kernel and any(
            keyword.lower() in event_name_lower for keyword in gemm_keywords
        ):
            # Extract kernel information
            kernel_info = {
                "name": event_name,
                "duration_us": 0.0,
                "count": 1,
            }

            # Try to get CUDA time (in microseconds)
            if hasattr(event, "cuda_time_total"):
                kernel_info["duration_us"] = event.cuda_time_total / 1000.0
            elif hasattr(event, "cuda_time"):
                kernel_info["duration_us"] = event.cuda_time / 1000.0
            elif hasattr(event, "self_cuda_time_total"):
                kernel_info["duration_us"] = event.self_cuda_time_total / 1000.0
            elif hasattr(event, "self_cuda_time"):
                kernel_info["duration_us"] = event.self_cuda_time / 1000.0

            # Try to get count
            if hasattr(event, "count"):
                kernel_info["count"] = event.count

            # Try to get input shapes if available
            if hasattr(event, "input_shapes") and event.input_shapes:
                kernel_info["input_shapes"] = event.input_shapes
            elif hasattr(event, "shapes") and event.shapes:
                kernel_info["input_shapes"] = event.shapes

            gemm_events.append(kernel_info)

    # Also check key_averages for aggregated view
    try:
        key_avgs = profiler.key_averages()
        for event in key_avgs:
            event_name = None
            if hasattr(event, "key"):
                event_name = event.key
            elif hasattr(event, "name"):
                event_name = event.name
            else:
                continue

            event_name_lower = event_name.lower()
            # Check if this is a gemm-related operation (may be at higher level)
            if any(keyword.lower() in event_name_lower for keyword in gemm_keywords):
                # Check if we already have this in gemm_events
                if not any(e["name"] == event_name for e in gemm_events):
                    kernel_info = {
                        "name": event_name,
                        "duration_us": 0.0,
                        "count": 1,
                    }

                    if hasattr(event, "cuda_time_total"):
                        kernel_info["duration_us"] = event.cuda_time_total / 1000.0
                    elif hasattr(event, "self_cuda_time_total"):
                        kernel_info["duration_us"] = event.self_cuda_time_total / 1000.0

                    if hasattr(event, "count"):
                        kernel_info["count"] = event.count

                    if hasattr(event, "input_shapes") and event.input_shapes:
                        kernel_info["input_shapes"] = event.input_shapes

                    gemm_events.append(kernel_info)
    except Exception:
        pass

    # Group by kernel name
    kernel_stats = defaultdict(
        lambda: {"count": 0, "total_time_us": 0.0, "input_shapes": []}
    )

    for event in gemm_events:
        name = event["name"]
        kernel_stats[name]["count"] += event["count"]
        kernel_stats[name]["total_time_us"] += event["duration_us"]
        if "input_shapes" in event and event["input_shapes"]:
            kernel_stats[name]["input_shapes"].extend(event["input_shapes"])

    # Calculate averages
    result = {
        "phase": phase,
        "total_gemm_kernels": len(gemm_events),
        "unique_kernel_names": len(kernel_stats),
        "kernels": {},
    }

    for name, stats in kernel_stats.items():
        result["kernels"][name] = {
            "count": stats["count"],
            "total_time_us": stats["total_time_us"],
            "avg_time_us": stats["total_time_us"] / stats["count"]
            if stats["count"] > 0
            else 0,
            "input_shapes": list(set(str(s) for s in stats["input_shapes"][:5]))
            if stats["input_shapes"]
            else [],
        }

    return result


def print_gemm_profile(profile_result: dict):
    """Print gemm profiling results in a readable format."""
    logger.info("=" * 80)
    logger.info(f"GEMM Kernel Profile - {profile_result['phase'].upper()}")
    logger.info("=" * 80)
    logger.info(f"Total GEMM kernels found: {profile_result['total_gemm_kernels']}")
    logger.info(f"Unique kernel names: {profile_result['unique_kernel_names']}")
    logger.info("")

    if not profile_result["kernels"]:
        logger.info("No GEMM kernels found in this phase.")
        return

    # Sort by total time
    sorted_kernels = sorted(
        profile_result["kernels"].items(),
        key=lambda x: x[1]["total_time_us"],
        reverse=True,
    )

    logger.info("GEMM Kernels (sorted by total time):")
    logger.info("-" * 80)
    for i, (name, stats) in enumerate(sorted_kernels, 1):
        logger.info(f"{i}. {name}")
        logger.info(f"   Count: {stats['count']}")
        logger.info(
            f"   Total time: {stats['total_time_us']:.2f} us ({stats['total_time_us'] / 1000:.2f} ms)"
        )
        logger.info(f"   Avg time: {stats['avg_time_us']:.2f} us")
        if stats["input_shapes"]:
            logger.info(f"   Sample shapes: {', '.join(stats['input_shapes'])}")
        logger.info("")

    total_time = sum(s["total_time_us"] for s in profile_result["kernels"].values())
    logger.info(f"Total GEMM time: {total_time:.2f} us ({total_time / 1000:.2f} ms)")
    logger.info("=" * 80)


def create_engine(
    model_path: str,
    fp8_enabled: bool = False,
    fp8_param: bool = False,
    port: int = 7777,
) -> MegatronEngine:
    """Create and initialize a MegatronEngine."""
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )

    megatron_config = MegatronEngineConfig()
    if fp8_enabled:
        megatron_config.fp8 = "e4m3"
        megatron_config.fp8_param = fp8_param
        megatron_config.fp8_recipe = "blockwise"
        megatron_config.ddp.fp8_param_gather = True

    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=model_path,
        optimizer=OptimizerConfig(),
        megatron=megatron_config,
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def forward_with_logits_and_logprobs(
    engine: MegatronEngine, input_: dict[str, Any], profile_gemm: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass that returns both logits and logprobs.

    Args:
        engine: MegatronEngine instance
        input_: Input dictionary
        profile_gemm: If True, profile GEMM kernels during forward pass

    Returns:
        tuple: (logits, logprobs) both with shape [batch, seq_len, ...]
    """
    engine.eval()
    if engine.is_offload:
        engine.onload()

    assert engine.model is not None, "Model is not initialized."

    # Prepare input similar to forward method
    cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
    mb_list = engine.prepare_mb_list(input_)
    mb_list = mb_list.to(engine.device)
    cu_seqlens = cu_seqlens.to(engine.device)

    output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
    max_total_len = max(m["max_seqlen"] for m in mb_list.padded_mbs)
    micro_batch_generator = [mb_list.padded_mbs] * len(engine.model)
    micro_batch_generator = [iter(b) for b in micro_batch_generator]
    forward_step_counts = [0] * len(engine.model)

    logits_list = []
    logprobs_list = []

    def forward_step(batch_iter, model):
        nonlocal forward_step_counts, logits_list, logprobs_list
        batch = next(batch_iter)
        model_vp_stage = getattr(model, "vp_stage", 0)
        forward_step_count = forward_step_counts[model_vp_stage]
        padding_length = mb_list.padding_lengths[forward_step_count]
        orig_input = mb_list.mbs[forward_step_count]
        cu_seqlens_batch = batch["cu_seqlens"]
        old_cu_seqlens = mb_list.old_cu_seqlens_list[forward_step_count]

        forward_step_counts[model_vp_stage] += 1
        output = packed_context_parallel_forward(model, batch)

        if mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=model_vp_stage):
            output_unpadded = unpad_logits(
                output,
                padding_length=padding_length,
                cu_seqlens=cu_seqlens_batch,
                old_cu_seqlens=old_cu_seqlens,
            )

            def _post_process_fn(input_, output_unpadded):
                labels = torch.roll(input_["input_ids"], shifts=-1, dims=-1)
                logprobs = gather_logprobs(
                    output_unpadded,
                    labels,
                    temperature=engine.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
                # Store logits and logprobs
                logits_list.append(output_unpadded)
                logprobs_list.append(logprobs)
                return torch.tensor(1.0, device=logprobs.device), {"output": logprobs}

            return output_unpadded, functools.partial(_post_process_fn, orig_input)

        return output, lambda x: (
            torch.tensor(1.0, device=output.device),
            {"output": None},
        )

    forward_backward_func = get_forward_backward_func()

    data_iterator = (
        micro_batch_generator if len(engine.model) > 1 else micro_batch_generator[0]
    )

    # Profile GEMM kernels if requested
    if profile_gemm:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            _ = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=engine.model if len(engine.model) > 1 else engine.model[0],
                num_microbatches=len(mb_list.padded_mbs),
                seq_length=max_total_len,
                micro_batch_size=1,
                forward_only=True,
            )
            torch.cuda.synchronize()

        # Extract and print GEMM kernels
        gemm_profile = extract_gemm_kernels(prof, phase="forward")
        print_gemm_profile(gemm_profile)
    else:
        _ = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=engine.model if len(engine.model) > 1 else engine.model[0],
            num_microbatches=len(mb_list.padded_mbs),
            seq_length=max_total_len,
            micro_batch_size=1,
            forward_only=True,
        )

    # Aggregate logits and logprobs
    if mpu.is_pipeline_last_stage():
        if logits_list:
            logits_res = torch.cat([logits for logits in logits_list], dim=0)
            logprobs_res = torch.cat([logprobs for logprobs in logprobs_list], dim=0)

            output_seqlens_filtered = [
                output_seqlens[i] for i in mb_list.forward_indices
            ]
            logits_unpacked = unpack_sequence(
                logits_res, lens=output_seqlens_filtered, dim=0
            )
            logprobs_unpacked = unpack_sequence(
                logprobs_res, lens=output_seqlens_filtered, dim=0
            )

            logits_reordered = reorder_list(logits_unpacked, mb_list.backward_indices)
            logprobs_reordered = reorder_list(
                logprobs_unpacked, mb_list.backward_indices
            )

            logits = pad_and_stack_tensors_along_first_dim(logits_reordered)
            logprobs = pad_and_stack_tensors_along_first_dim(logprobs_reordered)
        else:
            logits = None
            logprobs = None
    else:
        logits = None
        logprobs = None

    # Broadcast results
    logits = broadcast_tensor(
        logits,
        src_rank=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group(),
    )
    logprobs = broadcast_tensor(
        logprobs,
        src_rank=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group(),
    )

    return logits, logprobs


def decode_with_megatron_forward(
    engine: MegatronEngine,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """Decode using Megatron forward pass for autoregressive generation.

    Args:
        engine: MegatronEngine instance
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (None for no limit)
        top_p: Top-p (nucleus) sampling (None for no limit)

    Returns:
        Generated text (prompt + generated tokens)
    """
    engine.eval()
    if engine.is_offload:
        engine.onload()

    assert engine.model is not None, "Model is not initialized."
    assert engine.tokenizer is not None, "Tokenizer is not initialized."

    # Encode prompt
    encoded = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(engine.device)
    generated_ids = input_ids.clone()

    # Generate tokens autoregressively
    for step in range(max_new_tokens):
        # Prepare input dict
        batch_size = generated_ids.shape[0]
        seq_len = generated_ids.shape[1]
        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=engine.device
        )

        input_dict = {
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
        }

        # Forward pass to get logits
        logits, _ = forward_with_logits_and_logprobs(engine, input_dict)

        # Get logits for the last token position
        # logits shape: [batch, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = (
                next_token_logits
                < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)  # [batch, 1]

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Check for EOS token
        eos_token_id = getattr(engine.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and next_token_id[0, 0].item() == eos_token_id:
            logger.info("EOS token generated, stopping.")
            break

    # Decode full sequence
    generated_text = engine.tokenizer.decode(
        generated_ids[0], skip_special_tokens=False
    )

    return generated_text
