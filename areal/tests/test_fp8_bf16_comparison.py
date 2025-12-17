"""Test comparison between FP8 and BF16 models using Megatron Engine.

This test verifies:
1. Load FP8 model with fp8_param enabled and BF16 model using Megatron Engine
2. Compare logprobs from forward pass
3. Compare logits from forward pass
"""

import functools
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import get_fp8_context, is_float8tensor
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from torch import nn
from torch.autograd import Function
from transformers import AutoTokenizer, PretrainedConfig

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.data import (
    broadcast_tensor,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
    unpad_logits,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.mcore.packed_context_parallel import packed_context_parallel_forward
from areal.utils.megatron import all_gather_param, get_named_parameters

logger = logging.getLogger("FP8 BF16 Comparison Test")


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


# Model paths - adjust these to your actual model paths
MODEL_PATH_BF16 = "/storage/openpsi/models/Qwen__Qwen3-0.6B"
MODEL_PATH_FP8 = (
    "/storage/openpsi/models/Qwen__Qwen3-0.6B-FP8"  # Path to FP8 converted model
)
# MODEL_PATH_BF16 = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
# MODEL_PATH_FP8 = "/storage/openpsi/users/shenxujie.sxj/models/Qwen__Qwen2.5-1.5B-Instruct-FP8/"  # Path to FP8 converted model


@pytest.fixture(scope="module")
def mock_input(
    batch_size=2,
    min_seqlen=10,
    max_seqlen=128,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create mock padded input data for testing."""
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, 1000, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


@pytest.fixture(scope="module")
def fixed_input(
    questions: list[str] | None = None,
    answers: list[str] | None = None,
    model_path: str = MODEL_PATH_BF16,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create fixed input data for SFT training with question and answer.

    Args:
        questions: List of question strings. If None, uses default questions.
        answers: List of answer strings. If None, uses default answers.
        model_path: Path to the model for loading tokenizer.
        device: Device to place tensors on.

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'loss_mask' tensors.
        loss_mask: 0 for prompt tokens, 1 for answer tokens (including EOS).
    """
    if questions is None:
        questions = [
            "What is 2+2?",
            "Count from 1 to 5:",
        ]
    if answers is None:
        answers = [
            " 2+2 equals 4.",
            " 1, 2, 3, 4, 5",
        ]

    assert len(questions) == len(answers), (
        "Questions and answers must have the same length"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids_list = []
    loss_mask_list = []

    for question, answer in zip(questions, answers):
        # Encode full sequence: question + answer + eos_token
        full_text = question + answer + eos_token
        seq_token = tokenizer.encode(full_text, add_special_tokens=False)

        # Encode prompt (question) to determine where loss_mask should start
        prompt_token = tokenizer.encode(question, add_special_tokens=False)

        # Create loss_mask: 0 for prompt, 1 for answer (including EOS)
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))

        input_ids_list.append(torch.tensor(seq_token, dtype=torch.long, device=device))
        loss_mask_list.append(torch.tensor(loss_mask, dtype=torch.long, device=device))

    # Pad to same length
    max_length = max(ids.shape[0] for ids in input_ids_list)

    padded_input_ids = []
    padded_loss_masks = []
    attention_masks = []

    for input_ids, loss_mask in zip(input_ids_list, loss_mask_list):
        seq_len = input_ids.shape[0]
        padding_length = max_length - seq_len

        padded_ids = F.pad(input_ids, (0, padding_length), value=pad_token_id)
        padded_input_ids.append(padded_ids)

        padded_loss_mask = F.pad(loss_mask, (0, padding_length), value=0)
        padded_loss_masks.append(padded_loss_mask)

        attention_mask = F.pad(
            torch.ones(seq_len, dtype=torch.bool, device=device),
            (0, padding_length),
            value=0,
        )
        attention_masks.append(attention_mask)

    # Stack into batch
    input_ids = torch.stack(padded_input_ids).to(device)
    loss_mask = torch.stack(padded_loss_masks).to(device)
    attention_mask = torch.stack(attention_masks).to(device)

    logger.info("Using fixed input:")
    for i, (q, a) in enumerate(zip(questions, answers)):
        logger.info(f"  Sample {i}:")
        logger.info(f"    Question: {q}")
        logger.info(f"    Answer: {a}")
        logger.info(f"    Input IDs shape: {input_ids[i].shape}")
        logger.info(f"    Loss mask sum: {loss_mask[i].sum().item()}")

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )


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
            # "NVTE_FLASH_ATTN": "0",
            # "NVTE_FUSED_ATTN": "0",
            # "NVTE_UNFUSED_ATTN": "1",
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

    # logger.info(f"Prompt: {prompt}")
    # logger.info(f"Input IDs shape: {input_ids.shape}")
    # logger.info(f"Input IDs: {input_ids.tolist()}")

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

        # Decode current token for logging
        next_token_id_value = next_token_id[0, 0].item()
        # current_token = engine.tokenizer.decode(
        #     [next_token_id_value], skip_special_tokens=False
        # )
        # logger.info(f"Step {step + 1}: Generated token ID={next_token_id_value}, token='{current_token}'")

        # Check for EOS token
        eos_token_id = getattr(engine.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and next_token_id_value == eos_token_id:
            logger.info("EOS token generated, stopping.")
            break

    # Decode full sequence
    generated_text = engine.tokenizer.decode(
        generated_ids[0], skip_special_tokens=False
    )
    # logger.info(f"Generated text: {generated_text}")
    # logger.info(f"Generated IDs: {generated_ids[0].tolist()}")

    return generated_text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_megatron_decode_output():
    """Test decode using Megatron forward pass and print model output."""
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "The capital of France is",
        "Once upon a time",
    ]

    top_k = None
    temperature = 0.7
    max_new_tokens = 100

    # Create BF16 engine
    engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
    try:
        logger.info("=" * 80)
        logger.info("Testing decode with BF16 model")
        logger.info("=" * 80)

        for prompt in test_prompts:
            logger.info(f"{'=' * 80}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"{'=' * 80}")
            generated = decode_with_megatron_forward(
                engine_bf16,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            logger.info(f"BF16 Final output: {generated}\n")
    finally:
        engine_bf16.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Create FP8 engine with fp8_param enabled
    engine_fp8 = create_engine(
        MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778
    )
    try:
        logger.info("=" * 80)
        logger.info("Testing decode with FP8 model")
        logger.info("=" * 80)

        for prompt in test_prompts:
            logger.info(f"{'=' * 80}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"{'=' * 80}")
            generated = decode_with_megatron_forward(
                engine_fp8,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            logger.info(f"FP8 Final output: {generated}\n")
    finally:
        engine_fp8.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# def test_fp8_bf16_logprob_comparison(mock_input):
#     """Compare logprobs between FP8 and BF16 models."""
#     # Create BF16 engine
#     engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
#     try:
#         logprobs_bf16 = engine_bf16.forward(mock_input)
#         logger.info(f"BF16 logprobs shape: {logprobs_bf16.shape}")
#         logger.info(f"BF16 logprobs sample: {logprobs_bf16[0, :5]}")
#     finally:
#         engine_bf16.destroy()
#         if dist.is_initialized():
#             dist.destroy_process_group()

#     # Create FP8 engine with fp8_param enabled
#     # Note: We need to reinitialize process group after destroying the previous one
#     engine_fp8 = create_engine(MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778)
#     try:
#         logprobs_fp8 = engine_fp8.forward(mock_input)
#         logger.info(f"FP8 logprobs shape: {logprobs_fp8.shape}")
#         logger.info(f"FP8 logprobs sample: {logprobs_fp8[0, :5]}")
#     finally:
#         engine_fp8.destroy()
#         if dist.is_initialized():
#             dist.destroy_process_group()

#     # Compare logprobs
#     assert logprobs_bf16.shape == logprobs_fp8.shape, "Logprob shapes don't match"

#     # Calculate differences
#     max_diff = (logprobs_bf16 - logprobs_fp8).abs().max().item()
#     mean_diff = (logprobs_bf16 - logprobs_fp8).abs().mean().item()
#     logger.info(f"Logprob comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

#     # Allow some tolerance for FP8 quantization error
#     # FP8 has limited precision, so we expect some difference
#     assert max_diff < 1.0, f"Logprob max difference too large: {max_diff}"
#     assert mean_diff < 0.1, f"Logprob mean difference too large: {mean_diff}"


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# def test_fp8_bf16_logits_comparison(mock_input):
#     """Compare logits between FP8 and BF16 models."""
#     # Create BF16 engine
#     engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
#     try:
#         logits_bf16, logprobs_bf16 = forward_with_logits_and_logprobs(engine_bf16, mock_input)
#         logger.info(f"BF16 logits shape: {logits_bf16.shape}")
#         logger.info(f"BF16 logprobs shape: {logprobs_bf16.shape}")
#         logger.info(f"BF16 logits sample: {logits_bf16[0, 0, :5]}")
#     finally:
#         engine_bf16.destroy()
#         if dist.is_initialized():
#             dist.destroy_process_group()

#     # Create FP8 engine with fp8_param enabled
#     engine_fp8 = create_engine(MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778)
#     try:
#         logits_fp8, logprobs_fp8 = forward_with_logits_and_logprobs(engine_fp8, mock_input)
#         logger.info(f"FP8 logits shape: {logits_fp8.shape}")
#         logger.info(f"FP8 logprobs shape: {logprobs_fp8.shape}")
#         logger.info(f"FP8 logits sample: {logits_fp8[0, 0, :5]}")
#     finally:
#         engine_fp8.destroy()
#         if dist.is_initialized():
#             dist.destroy_process_group()

#     # Compare logits
#     assert logits_bf16.shape == logits_fp8.shape, "Logits shapes don't match"

#     # Calculate differences
#     max_diff = (logits_bf16 - logits_fp8).abs().max().item()
#     mean_diff = (logits_bf16 - logits_fp8).abs().mean().item()
#     logger.info(f"Logits comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

#     assert_close(logits_bf16, logits_fp8)
#     # Allow some tolerance for FP8 quantization error
#     # FP8 has limited precision, so we expect some difference
#     assert max_diff < 10.0, f"Logits max difference too large: {max_diff}"
#     assert mean_diff < 1.0, f"Logits mean difference too large: {mean_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp8_bf16_both_comparison(fixed_input):
    """Compare both logits and logprobs between FP8 and BF16 models."""
    # Create BF16 engine
    engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
    try:
        logits_bf16, logprobs_bf16 = forward_with_logits_and_logprobs(
            engine_bf16, fixed_input
        )
    finally:
        engine_bf16.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Create FP8 engine with fp8_param enabled
    engine_fp8 = create_engine(
        MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778
    )
    try:
        logits_fp8, logprobs_fp8 = forward_with_logits_and_logprobs(
            engine_fp8, fixed_input
        )
    finally:
        engine_fp8.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Get attention mask to filter out padding positions
    attention_mask = fixed_input["attention_mask"]  # [batch, seq_len]

    # Compare logprobs first
    assert logprobs_bf16.shape == logprobs_fp8.shape, "Logprob shapes don't match"
    # Only compute differences for non-padding positions
    valid_logprobs_mask = attention_mask  # [batch, seq_len]
    logprob_diff = (logprobs_bf16 - logprobs_fp8).abs()
    logprob_max_diff = (logprob_diff * valid_logprobs_mask).max().item()
    logprob_mean_diff = (
        logprob_diff * valid_logprobs_mask
    ).sum().item() / valid_logprobs_mask.sum().item()
    logger.info(
        f"Logprob comparison (non-padding only): max_diff={logprob_max_diff:.6f}, mean_diff={logprob_mean_diff:.6f}"
    )

    # Compare logits
    assert logits_bf16.shape == logits_fp8.shape, "Logits shapes don't match"
    # Only compute differences for non-padding positions
    valid_logits_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    logits_diff = (logits_bf16 - logits_fp8).abs()
    logits_max_diff = (logits_diff * valid_logits_mask).max().item()
    logits_mean_diff = (
        logits_diff * valid_logits_mask
    ).sum().item() / valid_logits_mask.sum().item()
    logger.info(
        f"Logits comparison (non-padding only): max_diff={logits_max_diff:.6f}, mean_diff={logits_mean_diff:.6f}"
    )

    # Sequence-level and token-level cosine similarity (only use valid tokens)
    batch_size, seq_len = attention_mask.shape

    # Collect data for both sequence-level and token-level cosine similarity
    cos_sim_logprobs_seq_list = []
    cos_sim_logits_seq_list = []
    logprobs_bf16_valid_list = []
    logprobs_fp8_valid_list = []
    logits_bf16_valid_list = []
    logits_fp8_valid_list = []

    for i in range(batch_size):
        valid_mask_i = attention_mask[i]  # [seq_len]
        valid_indices = valid_mask_i.nonzero(as_tuple=False).squeeze(-1)  # [num_valid]

        # Extract valid token positions: [num_valid, vocab_size]
        logprobs_bf16_valid = logprobs_bf16[i, valid_indices]  # [num_valid, vocab_size]
        logprobs_fp8_valid = logprobs_fp8[i, valid_indices]  # [num_valid, vocab_size]
        logits_bf16_valid = logits_bf16[i, valid_indices]  # [num_valid, vocab_size]
        logits_fp8_valid = logits_fp8[i, valid_indices]  # [num_valid, vocab_size]

        # For sequence-level: flatten valid tokens: [num_valid * vocab_size]
        logprobs_bf16_flat = logprobs_bf16_valid.flatten()
        logprobs_fp8_flat = logprobs_fp8_valid.flatten()
        logits_bf16_flat = logits_bf16_valid.flatten()
        logits_fp8_flat = logits_fp8_valid.flatten()

        # Compute sequence-level cosine similarity for this sample
        cos_sim_logprobs_i = F.cosine_similarity(
            logprobs_bf16_flat.unsqueeze(0), logprobs_fp8_flat.unsqueeze(0), dim=1
        ).item()
        cos_sim_logits_i = F.cosine_similarity(
            logits_bf16_flat.unsqueeze(0), logits_fp8_flat.unsqueeze(0), dim=1
        ).item()

        cos_sim_logprobs_seq_list.append(cos_sim_logprobs_i)
        cos_sim_logits_seq_list.append(cos_sim_logits_i)

        # For token-level: collect individual valid tokens
        logprobs_bf16_valid_list.append(logprobs_bf16_valid)  # [num_valid, vocab_size]
        logprobs_fp8_valid_list.append(logprobs_fp8_valid)  # [num_valid, vocab_size]
        logits_bf16_valid_list.append(logits_bf16_valid)  # [num_valid, vocab_size]
        logits_fp8_valid_list.append(logits_fp8_valid)  # [num_valid, vocab_size]

    # Sequence-level statistics
    cos_sim_logprobs_seq_mean = sum(cos_sim_logprobs_seq_list) / len(
        cos_sim_logprobs_seq_list
    )
    cos_sim_logits_seq_mean = sum(cos_sim_logits_seq_list) / len(
        cos_sim_logits_seq_list
    )

    logger.info(
        f"Seq cosine similarity of logprobs (valid tokens only): {cos_sim_logprobs_seq_mean}"
    )
    logger.info(
        f"Seq cosine similarity of logits (valid tokens only): {cos_sim_logits_seq_mean}"
    )

    # Stack token-level tensors: [num_valid_tokens, vocab_size]
    logprobs_bf16_valid = torch.cat(logprobs_bf16_valid_list, dim=0)
    logprobs_fp8_valid = torch.cat(logprobs_fp8_valid_list, dim=0)
    logits_bf16_valid = torch.cat(logits_bf16_valid_list, dim=0)
    logits_fp8_valid = torch.cat(logits_fp8_valid_list, dim=0)

    # Compute cosine similarity only for valid tokens
    cos_sim_logprobs_valid = F.cosine_similarity(
        logprobs_bf16_valid, logprobs_fp8_valid, dim=-1
    )  # [num_valid_tokens]
    cos_sim_logits_valid = F.cosine_similarity(
        logits_bf16_valid, logits_fp8_valid, dim=-1
    )  # [num_valid_tokens]

    cos_sim_logprobs_mean = cos_sim_logprobs_valid.mean().item()
    cos_sim_logprobs_min = cos_sim_logprobs_valid.min().item()
    cos_sim_logprobs_max = cos_sim_logprobs_valid.max().item()

    cos_sim_logits_mean = cos_sim_logits_valid.mean().item()
    cos_sim_logits_min = cos_sim_logits_valid.min().item()
    cos_sim_logits_max = cos_sim_logits_valid.max().item()

    logger.info(
        f"Token cosine similarity of logprobs (valid tokens only): {cos_sim_logprobs_mean}"
    )
    logger.info(
        f"Token cosine similarity of logits (valid tokens only): {cos_sim_logits_mean}"
    )
    logger.info(
        f"Token cosine similarity of logprobs - min: {cos_sim_logprobs_min:.6f}, max: {cos_sim_logprobs_max:.6f}"
    )
    logger.info(
        f"Token cosine similarity of logits - min: {cos_sim_logits_min:.6f}, max: {cos_sim_logits_max:.6f}"
    )

    if cos_sim_logprobs_mean < 0.99:
        raise AssertionError(
            f"Token cosine similarity of logprobs is less than 0.99: {cos_sim_logprobs_mean}"
        )
    if cos_sim_logits_mean < 0.99:
        raise AssertionError(
            f"Token cosine similarity of logits is less than 0.99: {cos_sim_logits_mean}"
        )
    # assert_close(logprobs_bf16, logprobs_fp8)
    # assert_close(logits_bf16, logits_fp8)
    # Assertions
    # assert logprob_max_diff < 1.0, f"Logprob max difference too large: {logprob_max_diff}"
    # assert logprob_mean_diff < 0.1, f"Logprob mean difference too large: {logprob_mean_diff}"
    # assert logits_max_diff < 10.0, f"Logits max difference too large: {logits_max_diff}"
    # assert logits_mean_diff < 1.0, f"Logits mean difference too large: {logits_mean_diff}"


def collect_gradients_after_train_batch(
    engine: MegatronEngine, input_: dict[str, Any], profile_gemm: bool = False
) -> dict[str, torch.Tensor]:
    """Execute train_batch but collect gradients before optimizer.step().

    This function replicates the train_batch logic but stops before optimizer.step()
    to collect gradients for comparison.

    Args:
        engine: MegatronEngine instance
        input_: Input dictionary
        profile_gemm: If True, profile GEMM kernels during forward and backward pass

    Returns:
        Dictionary mapping parameter names to their gradients.
    """
    if engine.is_offload:
        engine.onload()

    assert engine.model is not None, "Model is not initialized."
    assert engine.optimizer is not None, "Optimizer is not initialized."
    engine.optimizer.zero_grad()
    for model in engine.model:
        model.zero_grad_buffer()

    # print(input_)
    # print(f"input_ids: {input_["input_ids"].shape} loss_mask shape: {input_["loss_mask"].shape} attention_mask shape: {input_["attention_mask"].shape}")
    # Prepare input
    mb_list = engine.prepare_mb_list(input_)
    mb_list = mb_list.to(engine.device)

    # SFT loss function based on compute_packed_sft_loss from lm_engine.py
    def sft_loss_fn(logprobs, entropy, input_):
        """SFT loss function based on compute_packed_sft_loss.


        Args:
            logprobs: Log probabilities tensor of shape [seq_len, vocab_size] (packed format)
            entropy: Entropy (not used in SFT, ignored)
            input_: Input dictionary containing 'cu_seqlens' and 'loss_mask'

        Returns:
            Scalar loss tensor
        """
        del entropy  # SFT does not use entropy

        # Get cu_seqlens and loss_mask from input
        # These should be available after prepare_mb_list and packing
        loss_mask = input_["loss_mask"].bool()

        # Shift loss_mask to align with next-token prediction
        # In SFT, we predict the next token, so loss_mask needs to be shifted
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply loss_mask to logprobs (mask out positions where we don't compute loss)
        # logprobs shape: [seq_len, vocab_size] for packed format
        logprobs = torch.where(loss_mask, logprobs, 0)

        # Compute loss: negative log likelihood averaged over valid tokens
        device = logprobs.device
        num_valid = loss_mask.count_nonzero()
        if num_valid == 0:
            # Return zero loss if no valid tokens
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -logprobs.sum() / num_valid
        return loss

    def loss_weight_fn(mb):
        """Loss weight function based on number of valid tokens."""
        return mb["loss_mask"].count_nonzero()

    total_loss_weight = (
        torch.stack([loss_weight_fn(mb) for mb in mb_list.padded_mbs])
        .sum()
        .detach()
        .clone()
        .to(dtype=torch.float32)
    )
    assert total_loss_weight != 0
    dist.all_reduce(total_loss_weight, group=mpu.get_data_parallel_group())
    max_total_len = max(m["cu_seqlens"][-1].item() for m in mb_list.padded_mbs)
    micro_batch_generator = [mb_list.padded_mbs] * len(engine.model)
    micro_batch_generator = [iter(b) for b in micro_batch_generator]
    forward_step_counts = [0] * len(engine.model)

    def forward_step(batch_iter, model):
        nonlocal forward_step_counts
        batch = next(batch_iter)
        model_vp_stage = getattr(model, "vp_stage", 0)
        forward_step_count = forward_step_counts[model_vp_stage]
        padding_length = mb_list.padding_lengths[forward_step_count]
        orig_input = mb_list.mbs[forward_step_count]
        cu_seqlens = batch["cu_seqlens"]
        old_cu_seqlens = mb_list.old_cu_seqlens_list[forward_step_count]

        forward_step_counts[model_vp_stage] += 1
        output = packed_context_parallel_forward(model, batch)
        # print(f"batch: {batch}")
        # print(f"forward output: {output.shape}")

        if mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=model_vp_stage):
            output = unpad_logits(
                output,
                padding_length=padding_length,
                cu_seqlens=cu_seqlens,
                old_cu_seqlens=old_cu_seqlens,
            )

        def _scaled_loss_fn(input_, output):
            # Prepare input dict with cu_seqlens for loss function
            loss_input = input_.copy()

            labels = torch.roll(input_["input_ids"], shifts=-1, dims=-1)
            # print(loss_input["input_ids"].shape)
            # print(labels.shape)
            # print(f"output shape: {output.shape}")
            logprobs, entropy = gather_logprobs_entropy(
                output,
                labels,
                temperature=engine.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            loss = sft_loss_fn(logprobs, entropy, loss_input)
            loss_scale = loss_weight_fn(input_) / total_loss_weight
            loss_scale *= mpu.get_data_parallel_world_size()
            loss_scale *= engine.optimizer.get_loss_scale().item()
            loss *= loss_scale
            return loss, {}

        return output, functools.partial(_scaled_loss_fn, orig_input)

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
            forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=engine.model if len(engine.model) > 1 else engine.model[0],
                num_microbatches=len(mb_list.padded_mbs),
                seq_length=max_total_len,
                micro_batch_size=1,
                forward_only=False,
            )
            torch.cuda.synchronize()

        # Extract and print GEMM kernels
        gemm_profile = extract_gemm_kernels(prof, phase="backward")
        print_gemm_profile(gemm_profile)
    else:
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=engine.model if len(engine.model) > 1 else engine.model[0],
            num_microbatches=len(mb_list.padded_mbs),
            seq_length=max_total_len,
            micro_batch_size=1,
            forward_only=False,
        )

    # Collect gradients before optimizer.step()
    # Note: In Megatron, gradients might be in param.grad or param.main_grad
    # Also need to handle DDP wrapping - unwrap if needed
    gradients = {}
    for name, param in get_named_parameters(engine.model, num_experts=None):
        if param.requires_grad:
            # Try to get gradient from param.grad or param.main_grad
            grad = None
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grad = param.main_grad.clone()
            elif hasattr(param, "grad") and param.grad is not None:
                grad = param.grad.clone()
            else:
                raise ValueError(f"No gradient found for {name}")

            if grad is not None:
                # All-gather gradient if it's tensor parallel
                # For single GPU tests (d1p1t1), tensor parallel is not used, so we can skip this
                # For multi-GPU tensor parallel, we would need to all-gather gradients
                if (
                    hasattr(param, "tensor_model_parallel")
                    and param.tensor_model_parallel
                ):
                    try:
                        # Create a temporary parameter with gradient as data for all_gather_param
                        temp_param = torch.nn.Parameter(grad)
                        # Copy tensor_model_parallel and other attributes from original param
                        temp_param.tensor_model_parallel = param.tensor_model_parallel
                        if hasattr(param, "partition_dim"):
                            temp_param.partition_dim = param.partition_dim
                        if hasattr(param, "partition_stride"):
                            temp_param.partition_stride = param.partition_stride
                        if hasattr(param, "parallel_mode"):
                            temp_param.parallel_mode = param.parallel_mode
                        grad = all_gather_param(name, temp_param)
                    except Exception as e:
                        logger.warning(f"Failed to all_gather gradient for {name}: {e}")
                        # If all_gather fails, use the local gradient
                gradients[name] = grad

    return gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp8_bf16_gradient_comparison(fixed_input):
    """Compare gradients between FP8 and BF16 models after train_batch.

    This test verifies that gradients computed from FP8 and BF16 models
    are consistent across all layers.
    """
    # Create BF16 engine
    engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
    try:
        engine_bf16.train()
        gradients_bf16 = collect_gradients_after_train_batch(engine_bf16, fixed_input)
        logger.info(f"BF16 model: collected {len(gradients_bf16)} parameter gradients")
    finally:
        engine_bf16.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Create FP8 engine with fp8_param enabled
    engine_fp8 = create_engine(
        MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778
    )
    try:
        engine_fp8.train()
        gradients_fp8 = collect_gradients_after_train_batch(engine_fp8, fixed_input)
        logger.info(f"FP8 model: collected {len(gradients_fp8)} parameter gradients")
    finally:
        engine_fp8.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Compare gradients
    assert len(gradients_bf16) == len(gradients_fp8), (
        f"Number of parameters with gradients don't match: "
        f"BF16={len(gradients_bf16)}, FP8={len(gradients_fp8)}"
    )

    # Get common parameter names
    common_names = set(gradients_bf16.keys()) & set(gradients_fp8.keys())
    logger.info(f"Comparing {len(common_names)} common parameters")

    # Statistics for all layers
    all_max_diffs = []
    all_mean_diffs = []
    all_cos_sims = []
    layer_stats = []

    for name in sorted(common_names):
        grad_bf16 = gradients_bf16[name]
        grad_fp8 = gradients_fp8[name]

        # Check shapes match
        assert grad_bf16.shape == grad_fp8.shape, (
            f"Gradient shapes don't match for {name}: "
            f"BF16={grad_bf16.shape}, FP8={grad_fp8.shape}"
        )

        # Compute differences
        grad_diff = (grad_bf16 - grad_fp8).abs()
        max_diff = grad_diff.max().item()
        mean_diff = grad_diff.mean().item()

        # Compute cosine similarity
        grad_bf16_flat = grad_bf16.flatten()
        grad_fp8_flat = grad_fp8.flatten()
        cos_sim = F.cosine_similarity(
            grad_bf16_flat.unsqueeze(0), grad_fp8_flat.unsqueeze(0), dim=1
        ).item()

        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)
        all_cos_sims.append(cos_sim)

        # Extract layer index from parameter name for grouping
        layer_match = re.search(r"layers\.(\d+)", name)
        layer_idx = int(layer_match.group(1)) if layer_match else -1

        layer_stats.append(
            {
                "name": name,
                "layer_idx": layer_idx,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "cos_sim": cos_sim,
                "shape": grad_bf16.shape,
            }
        )

    # Log statistics by layer
    layer_indices = sorted(
        set(s["layer_idx"] for s in layer_stats if s["layer_idx"] >= 0)
    )
    for layer_idx in layer_indices:
        layer_grads = [s for s in layer_stats if s["layer_idx"] == layer_idx]
        layer_max_diffs = [s["max_diff"] for s in layer_grads]
        layer_mean_diffs = [s["mean_diff"] for s in layer_grads]
        layer_cos_sims = [s["cos_sim"] for s in layer_grads]

        logger.info(
            f"Layer {layer_idx}: "
            f"max_diff={max(layer_max_diffs):.6f}, "
            f"mean_diff={sum(layer_mean_diffs) / len(layer_mean_diffs):.6f}, "
            f"cos_sim={sum(layer_cos_sims) / len(layer_cos_sims):.6f}, "
            f"n_params={len(layer_grads)}, "
            f"names={','.join([s['name'] for s in layer_grads])}"
        )
    # log lay_idx < 0
    layer_stats_less_than_0 = [s for s in layer_stats if s["layer_idx"] < 0]
    logger.info(f"Do not have layer indices: {len(layer_stats_less_than_0)} params")
    for stat in layer_stats_less_than_0:
        name_str = f"Layer {stat['name']}"
        logger.info(
            f"{name_str:<50} "
            f"max_diff={stat['max_diff']:>12.6f}, "
            f"mean_diff={stat['mean_diff']:>12.6f}, "
            f"cos_sim={stat['cos_sim']:>10.6f}"
        )

    # Overall statistics
    overall_max_diff = max(all_max_diffs)
    overall_mean_diff = sum(all_mean_diffs) / len(all_mean_diffs)
    overall_cos_sim = sum(all_cos_sims) / len(all_cos_sims)
    overall_min_cos_sim = min(all_cos_sims)

    logger.info("=" * 80)
    logger.info("Overall gradient comparison statistics:")
    logger.info(f"  Max difference: {overall_max_diff:.6f}")
    logger.info(f"  Mean difference: {overall_mean_diff:.6f}")
    logger.info(f"  Mean cosine similarity: {overall_cos_sim:.6f}")
    logger.info(f"  Min cosine similarity: {overall_min_cos_sim:.6f}")
    logger.info("=" * 80)

    # Log parameters with largest differences
    layer_stats_sorted = sorted(layer_stats, key=lambda x: x["max_diff"], reverse=True)
    logger.info("Top 10 parameters with largest gradient differences:")
    for i, stat in enumerate(layer_stats_sorted[:10]):
        logger.info(
            f"  {i + 1}. {stat['name']}: "
            f"max_diff={stat['max_diff']:.6f}, "
            f"mean_diff={stat['mean_diff']:.6f}, "
            f"cos_sim={stat['cos_sim']:.6f}"
        )

    # Assertions - allow some tolerance for FP8 quantization
    # FP8 has limited precision, so we expect some difference
    assert overall_cos_sim > 0.95, (
        f"Overall cosine similarity too low: {overall_cos_sim:.6f}. "
        f"This suggests gradients are not consistent between BF16 and FP8 models."
    )
    assert overall_min_cos_sim > 0.90, (
        f"Minimum cosine similarity too low: {overall_min_cos_sim:.6f}. "
        f"Some parameters have very different gradients."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_gemm_kernels(fixed_input):
    """Profile and print GEMM kernels used in forward and backward pass.

    This test profiles the GEMM kernels (matrix multiplication operations) used
    during forward and backward passes to understand which operators are being used.
    """
    # Create BF16 engine
    engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
    try:
        logger.info("=" * 80)
        logger.info("Profiling GEMM kernels - BF16 Model")
        logger.info("=" * 80)

        # Profile forward pass
        logger.info("\n>>> Profiling FORWARD pass...")
        logits_bf16, logprobs_bf16 = forward_with_logits_and_logprobs(
            engine_bf16, fixed_input, profile_gemm=True
        )

        # Profile backward pass
        logger.info("\n>>> Profiling BACKWARD pass...")
        engine_bf16.train()
        gradients_bf16 = collect_gradients_after_train_batch(
            engine_bf16, fixed_input, profile_gemm=True
        )
        logger.info(f"Collected {len(gradients_bf16)} parameter gradients")

    finally:
        engine_bf16.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Create FP8 engine with fp8_param enabled
    engine_fp8 = create_engine(
        MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778
    )
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Profiling GEMM kernels - FP8 Model")
        logger.info("=" * 80)

        # Profile forward pass
        logger.info("\n>>> Profiling FORWARD pass...")
        logits_fp8, logprobs_fp8 = forward_with_logits_and_logprobs(
            engine_fp8, fixed_input, profile_gemm=True
        )

        # Profile backward pass
        logger.info("\n>>> Profiling BACKWARD pass...")
        engine_fp8.train()
        gradients_fp8 = collect_gradients_after_train_batch(
            engine_fp8, fixed_input, profile_gemm=True
        )
        logger.info(f"Collected {len(gradients_fp8)} parameter gradients")

    finally:
        engine_fp8.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def extract_single_layer(engine: MegatronEngine, layer_idx: int):
    """Extract a single transformer layer from the model.

    Args:
        engine: MegatronEngine instance
        layer_idx: Index of the layer to extract (0-based)

    Returns:
        The transformer layer module
    """
    assert engine.model is not None, "Model is not initialized."

    # Get the actual model module (unwrap DDP if needed)
    model = engine.model[0]
    if hasattr(model, "module"):
        model = model.module

    # Handle Float16Module wrapper (if present)
    if hasattr(model, "module"):
        model = model.module

    # Access decoder.layers[layer_idx]
    # Structure: model.decoder.layers[layer_idx] or model.module.decoder.layers[layer_idx]
    decoder = None
    if hasattr(model, "decoder"):
        decoder = model.decoder
    elif hasattr(model, "module") and hasattr(model.module, "decoder"):
        decoder = model.module.decoder

    if decoder is not None and hasattr(decoder, "layers"):
        layers = decoder.layers
        if layer_idx < len(layers):
            return layers[layer_idx]
        else:
            raise ValueError(
                f"Layer index {layer_idx} out of range. Model has {len(layers)} layers."
            )
    else:
        raise ValueError(
            f"Model does not have decoder.layers structure. Available attributes: {dir(model)}"
        )


def get_model_from_engine(engine: MegatronEngine):
    """Get the actual model module from engine, unwrapping DDP and Float16Module."""
    assert engine.model is not None, "Model is not initialized."
    model = engine.model[0]
    if hasattr(model, "module"):
        model = model.module
    # Handle Float16Module wrapper
    if hasattr(model, "module"):
        model = model.module
    return model


def reduce_model_to_layers(engine: MegatronEngine, layer_indices: list[int] | int):
    """Reduce the model to specified transformer layers while keeping full structure.

    This function modifies the model in-place by replacing decoder.layers (ModuleList)
    with a new ModuleList containing only the specified layers. This allows the model
    to maintain its full structure (embedding, rotary_pos_emb, final_layernorm, output_layer)
    so that forward pass and loss computation work correctly.

    Args:
        engine: MegatronEngine instance
        layer_indices: Index or list of indices of layers to keep (0-based).
                      If int, keeps only that layer. If list, keeps multiple layers.

    Returns:
        The original number of layers (for potential restoration)
    """
    model = get_model_from_engine(engine)

    # Get decoder
    decoder = None
    if hasattr(model, "decoder"):
        decoder = model.decoder
    elif hasattr(model, "module") and hasattr(model.module, "decoder"):
        decoder = model.module.decoder

    if decoder is None or not hasattr(decoder, "layers"):
        raise ValueError("Cannot find decoder.layers")

    original_layers = decoder.layers
    original_num_layers = len(original_layers)

    # Convert single int to list
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    # Validate layer indices
    for layer_idx in layer_indices:
        if layer_idx >= original_num_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range. Model has {original_num_layers} layers."
            )

    # Remove duplicates and sort to maintain order
    layer_indices = sorted(list(set(layer_indices)))

    # Create new ModuleList with only the specified layers
    selected_layers = [original_layers[idx] for idx in layer_indices]
    new_layers = torch.nn.ModuleList(selected_layers)

    # Replace the layers ModuleList
    decoder.layers = new_layers

    if len(layer_indices) == 1:
        logger.info(
            f"Reduced model from {original_num_layers} layers to 1 layer (keeping layer {layer_indices[0]})"
        )
    else:
        logger.info(
            f"Reduced model from {original_num_layers} layers to {len(layer_indices)} layers (keeping layers {layer_indices})"
        )

    return original_num_layers


def forward_backward_model_with_hooks(
    engine: MegatronEngine,
    input_: dict[str, Any],
    layer_indices: list[int] | int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Perform forward and backward pass on model with specified layers and activation hooks.

    This function reduces the model to specified layers, then performs forward and backward
    using the full model structure (embedding -> layers -> final_layernorm -> output_layer),
    allowing for real loss computation.

    Args:
        engine: MegatronEngine instance
        input_: Input dictionary with 'input_ids', 'attention_mask', 'loss_mask'
        layer_indices: Index or list of indices of layers to keep (0-based).
                      If int, keeps only that layer. If list, keeps multiple layers.

    Returns:
        tuple: (logits, activations_dict, gradients_dict)
        - logits: Output logits from the model
        - activations_dict: Dictionary mapping op names to their output activations
        - gradients_dict: Dictionary mapping parameter names to their gradients
    """
    # Convert single int to list for consistency
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    # Reduce model to specified layers
    _ = reduce_model_to_layers(engine, layer_indices)

    activations = {}
    gradients = {}
    output_gradients = {}  # Gradients flowing back to module outputs
    hooks = []

    def make_activation_hook(name):
        def hook(module, input, output):
            try:
                if isinstance(output, tuple):
                    activations[name] = (
                        output[0].clone().detach() if len(output) > 0 else None
                    )
                else:
                    activations[name] = output.clone().detach()
                logger.info(
                    f"Captured activation for {name}: {activations[name].dtype}"
                )
            except Exception as e:
                logger.warning(f"Failed to capture activation for {name}: {e}")

        return hook

    # Get model and register hooks
    model = get_model_from_engine(engine)

    # Register hooks for components
    hook_names = []

    # Embedding
    if hasattr(model, "embedding"):
        hook_names.append(("embedding", model.embedding))
        if hasattr(model.embedding, "word_embeddings"):
            hook_names.append(
                ("embedding.word_embeddings", model.embedding.word_embeddings)
            )

    # Rotary position embedding
    if hasattr(model, "rotary_pos_emb"):
        hook_names.append(("rotary_pos_emb", model.rotary_pos_emb))

    # Decoder and layers
    if hasattr(model, "decoder"):
        decoder = model.decoder
        hook_names.append(("decoder", decoder))

        # Selected layers (after reduction)
        if hasattr(decoder, "layers") and len(decoder.layers) > 0:
            # Register hooks for each layer
            for layer_idx_in_reduced, layer in enumerate(decoder.layers):
                # Use original layer index in naming if we know it, otherwise use position in reduced list
                # For now, use position in reduced list
                layer_prefix = f"layer_{layer_idx_in_reduced}"

                hook_names.append((f"{layer_prefix}", layer))

                # Input layernorm
                if hasattr(layer, "input_layernorm"):
                    hook_names.append(
                        (f"{layer_prefix}.input_layernorm", layer.input_layernorm)
                    )

                # Self attention
                if hasattr(layer, "self_attention"):
                    hook_names.append(
                        (f"{layer_prefix}.self_attention", layer.self_attention)
                    )
                    if hasattr(layer.self_attention, "linear_qkv"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.linear_qkv",
                                layer.self_attention.linear_qkv,
                            )
                        )
                    if hasattr(layer.self_attention, "linear_proj"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.linear_proj",
                                layer.self_attention.linear_proj,
                            )
                        )
                    if hasattr(layer.self_attention, "core_attention"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.core_attention",
                                layer.self_attention.core_attention,
                            )
                        )
                    if hasattr(layer.self_attention, "q_layernorm"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.q_layernorm",
                                layer.self_attention.q_layernorm,
                            )
                        )

                        # Add pre-hook to capture input to q_layernorm
                        def make_q_layernorm_input_hook(prefix):
                            def q_layernorm_input_hook(module, input):
                                try:
                                    if isinstance(input, tuple):
                                        activations[
                                            f"{prefix}.self_attention.q_layernorm.input"
                                        ] = (
                                            input[0].clone().detach()
                                            if len(input) > 0
                                            else None
                                        )
                                    else:
                                        activations[
                                            f"{prefix}.self_attention.q_layernorm.input"
                                        ] = input.clone().detach()
                                    logger.info(
                                        f"Captured q_layernorm input for {prefix}: {activations[f'{prefix}.self_attention.q_layernorm.input'].shape}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to capture q_layernorm input for {prefix}: {e}"
                                    )

                            return q_layernorm_input_hook

                        pre_hook = (
                            layer.self_attention.q_layernorm.register_forward_pre_hook(
                                make_q_layernorm_input_hook(layer_prefix)
                            )
                        )
                        hooks.append(pre_hook)

                        # Add backward hook to capture gradient flowing back to q_layernorm output
                        def make_q_layernorm_backward_hook(prefix):
                            def q_layernorm_backward_hook(
                                module, grad_input, grad_output
                            ):
                                try:
                                    if grad_output is not None and len(grad_output) > 0:
                                        if grad_output[0] is not None:
                                            output_gradients[
                                                f"{prefix}.self_attention.q_layernorm.output_grad"
                                            ] = grad_output[0].clone().detach()
                                            logger.info(
                                                f"Captured q_layernorm output grad for {prefix}: {output_gradients[f'{prefix}.self_attention.q_layernorm.output_grad'].shape}"
                                            )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to capture q_layernorm output grad for {prefix}: {e}"
                                    )

                            return q_layernorm_backward_hook

                        backward_hook = layer.self_attention.q_layernorm.register_full_backward_hook(
                            make_q_layernorm_backward_hook(layer_prefix)
                        )
                        hooks.append(backward_hook)
                    if hasattr(layer.self_attention, "k_layernorm"):
                        hook_names.append(
                            (
                                f"{layer_prefix}.self_attention.k_layernorm",
                                layer.self_attention.k_layernorm,
                            )
                        )

                        # Add pre-hook to capture input to k_layernorm
                        def make_k_layernorm_input_hook(prefix):
                            def k_layernorm_input_hook(module, input):
                                try:
                                    if isinstance(input, tuple):
                                        activations[
                                            f"{prefix}.self_attention.k_layernorm.input"
                                        ] = (
                                            input[0].clone().detach()
                                            if len(input) > 0
                                            else None
                                        )
                                    else:
                                        activations[
                                            f"{prefix}.self_attention.k_layernorm.input"
                                        ] = input.clone().detach()
                                    logger.info(
                                        f"Captured k_layernorm input for {prefix}: {activations[f'{prefix}.self_attention.k_layernorm.input'].shape}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to capture k_layernorm input for {prefix}: {e}"
                                    )

                            return k_layernorm_input_hook

                        pre_hook = (
                            layer.self_attention.k_layernorm.register_forward_pre_hook(
                                make_k_layernorm_input_hook(layer_prefix)
                            )
                        )
                        hooks.append(pre_hook)

                        # Add backward hook to capture gradient flowing back to k_layernorm output
                        def make_k_layernorm_backward_hook(prefix):
                            def k_layernorm_backward_hook(
                                module, grad_input, grad_output
                            ):
                                try:
                                    if grad_output is not None and len(grad_output) > 0:
                                        if grad_output[0] is not None:
                                            output_gradients[
                                                f"{prefix}.self_attention.k_layernorm.output_grad"
                                            ] = grad_output[0].clone().detach()
                                            logger.info(
                                                f"Captured k_layernorm output grad for {prefix}: {output_gradients[f'{prefix}.self_attention.k_layernorm.output_grad'].shape}"
                                            )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to capture k_layernorm output grad for {prefix}: {e}"
                                    )

                            return k_layernorm_backward_hook

                        backward_hook = layer.self_attention.k_layernorm.register_full_backward_hook(
                            make_k_layernorm_backward_hook(layer_prefix)
                        )
                        hooks.append(backward_hook)

                # Post attention layernorm
                if hasattr(layer, "post_attention_layernorm"):
                    hook_names.append(
                        (
                            f"{layer_prefix}.post_attention_layernorm",
                            layer.post_attention_layernorm,
                        )
                    )
                elif hasattr(layer, "pre_mlp_layernorm"):
                    hook_names.append(
                        (f"{layer_prefix}.pre_mlp_layernorm", layer.pre_mlp_layernorm)
                    )

                # MLP
                if hasattr(layer, "mlp"):
                    hook_names.append((f"{layer_prefix}.mlp", layer.mlp))
                    if hasattr(layer.mlp, "linear_fc1"):
                        hook_names.append(
                            (f"{layer_prefix}.mlp.linear_fc1", layer.mlp.linear_fc1)
                        )
                    if hasattr(layer.mlp, "linear_fc2"):
                        hook_names.append(
                            (f"{layer_prefix}.mlp.linear_fc2", layer.mlp.linear_fc2)
                        )

                    # Add pre-hook to capture activation output
                    if hasattr(layer.mlp, "linear_fc2"):

                        def make_mlp_activation_hook(prefix):
                            def mlp_activation_output_hook(module, input):
                                try:
                                    if isinstance(input, tuple):
                                        activations[
                                            f"{prefix}.mlp.activation_output"
                                        ] = (
                                            input[0].clone().detach()
                                            if len(input) > 0
                                            else None
                                        )
                                    else:
                                        activations[
                                            f"{prefix}.mlp.activation_output"
                                        ] = input.clone().detach()
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to capture MLP activation output for {prefix}: {e}"
                                    )

                            return mlp_activation_output_hook

                        activation_hook = (
                            layer.mlp.linear_fc2.register_forward_pre_hook(
                                make_mlp_activation_hook(layer_prefix)
                            )
                        )
                        hooks.append(activation_hook)

        # Final layernorm
        if hasattr(decoder, "final_layernorm"):
            hook_names.append(("decoder.final_layernorm", decoder.final_layernorm))

    # Output layer
    if hasattr(model, "output_layer"):
        hook_names.append(("output_layer", model.output_layer))

    # Register forward hooks and backward hooks for all modules
    for name, module in hook_names:
        try:
            # Register forward hook
            hook = module.register_forward_hook(make_activation_hook(name))
            hooks.append(hook)

            # Register backward hook to capture output gradients
            def make_backward_hook(hook_name):
                def backward_hook(module, grad_input, grad_output):
                    try:
                        if grad_output is not None and len(grad_output) > 0:
                            if grad_output[0] is not None:
                                output_gradients[f"{hook_name}.output_grad"] = (
                                    grad_output[0].clone().detach()
                                )
                                logger.debug(
                                    f"Captured output grad for {hook_name}: {output_gradients[f'{hook_name}.output_grad'].shape}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to capture output grad for {hook_name}: {e}"
                        )

                return backward_hook

            backward_hook = module.register_full_backward_hook(make_backward_hook(name))
            hooks.append(backward_hook)
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    # Forward and backward using engine's train_batch method
    engine.train()

    # Prepare loss function
    def sft_loss_fn(logprobs, entropy, input_):
        del entropy
        loss_mask = input_["loss_mask"].bool()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        logprobs = torch.where(loss_mask, logprobs, 0)
        device = logprobs.device
        num_valid = loss_mask.count_nonzero()
        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = -logprobs.sum() / num_valid
        return loss

    def loss_weight_fn(mb):
        return mb["loss_mask"].count_nonzero()

    # Use engine's train_batch but collect gradients before optimizer step
    engine.optimizer.zero_grad()
    for model_chunk in engine.model:
        model_chunk.zero_grad_buffer()

    # Forward and backward
    engine.train_batch(input_, sft_loss_fn, loss_weight_fn)

    # Collect gradients from all components (focusing on the selected layers)
    model = get_model_from_engine(engine)

    # Collect gradients from all selected layers
    if (
        hasattr(model, "decoder")
        and hasattr(model.decoder, "layers")
        and len(model.decoder.layers) > 0
    ):
        for layer_idx_in_reduced, layer in enumerate(model.decoder.layers):
            layer_prefix = f"layer_{layer_idx_in_reduced}"
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    grad = None
                    if hasattr(param, "main_grad") and param.main_grad is not None:
                        grad = param.main_grad.clone().detach()
                    elif hasattr(param, "grad") and param.grad is not None:
                        grad = param.grad.clone().detach()
                    else:
                        raise ValueError(f"No gradient found for {layer_prefix}.{name}")

                    if grad is not None:
                        # Use layer_X. prefix to match activation naming
                        gradients[f"{layer_prefix}.{name}"] = grad
                    else:
                        logger.warning(f"No gradient found for {layer_prefix}.{name}")

    # Get logits by doing a forward pass
    engine.eval()
    logits = engine.forward(input_)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return logits, activations, gradients, output_gradients


def forward_backward_single_layer_with_hooks(
    layer: torch.nn.Module,
    input_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    rotary_pos_emb: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Perform forward and backward pass on a single layer with activation hooks.

    Args:
        layer: The transformer layer module
        input_hidden_states: Input hidden states [batch, seq_len, hidden_size]
        attention_mask: Optional attention mask [batch, seq_len]
        rotary_pos_emb: Optional rotary position embedding module (from model level)

    Returns:
        tuple: (output_hidden_states, activations_dict, gradients_dict)
        - output_hidden_states: Output from the layer
        - activations_dict: Dictionary mapping op names to their output activations
        - gradients_dict: Dictionary mapping parameter names to their gradients
    """
    activations = {}
    gradients = {}

    # Register forward hooks to capture activations
    hooks = []

    def make_activation_hook(name):
        def hook(module, input, output):
            # Store the output activation
            try:
                if isinstance(output, tuple):
                    activations[name] = (
                        output[0].clone().detach() if len(output) > 0 else None
                    )
                else:
                    activations[name] = output.clone().detach()
            except Exception as e:
                logger.warning(f"Failed to capture activation for {name}: {e}")

        return hook

    # Register hooks for different components
    # Based on actual Megatron structure:
    # - input_layernorm
    # - self_attention (with linear_qkv, linear_proj, core_attention)
    # - mlp (with linear_fc1, linear_fc2)
    hook_names = []

    # Input layernorm
    if hasattr(layer, "input_layernorm"):
        hook_names.append(("input_layernorm", layer.input_layernorm))

    # Self attention module
    if hasattr(layer, "self_attention"):
        hook_names.append(("self_attention", layer.self_attention))
        # Hook attention submodules
        if hasattr(layer.self_attention, "linear_qkv"):
            hook_names.append(
                ("self_attention.linear_qkv", layer.self_attention.linear_qkv)
            )
        if hasattr(layer.self_attention, "linear_proj"):
            hook_names.append(
                ("self_attention.linear_proj", layer.self_attention.linear_proj)
            )
        if hasattr(layer.self_attention, "core_attention"):
            hook_names.append(
                ("self_attention.core_attention", layer.self_attention.core_attention)
            )
        # Hook Q/K layernorms (Qwen3 style)
        if hasattr(layer.self_attention, "q_layernorm"):
            hook_names.append(
                ("self_attention.q_layernorm", layer.self_attention.q_layernorm)
            )
        if hasattr(layer.self_attention, "k_layernorm"):
            hook_names.append(
                ("self_attention.k_layernorm", layer.self_attention.k_layernorm)
            )
        # Also try legacy names for compatibility
        if hasattr(layer.self_attention, "q_proj"):
            hook_names.append(("self_attention.q_proj", layer.self_attention.q_proj))
        if hasattr(layer.self_attention, "o_proj"):
            hook_names.append(("self_attention.o_proj", layer.self_attention.o_proj))

    # Hook rotary_pos_emb if provided (it's at model level, not layer level)
    if rotary_pos_emb is not None:
        hook_names.append(("rotary_pos_emb", rotary_pos_emb))
    # Also try legacy name 'self_attn' for compatibility
    elif hasattr(layer, "self_attn"):
        hook_names.append(("self_attn", layer.self_attn))
        if hasattr(layer.self_attn, "q_proj"):
            hook_names.append(("self_attn.q_proj", layer.self_attn.q_proj))
        if hasattr(layer.self_attn, "k_proj"):
            hook_names.append(("self_attn.k_proj", layer.self_attn.k_proj))
        if hasattr(layer.self_attn, "v_proj"):
            hook_names.append(("self_attn.v_proj", layer.self_attn.v_proj))
        if hasattr(layer.self_attn, "o_proj"):
            hook_names.append(("self_attn.o_proj", layer.self_attn.o_proj))

    # Post attention layernorm (may be named differently)
    if hasattr(layer, "post_attention_layernorm"):
        hook_names.append(("post_attention_layernorm", layer.post_attention_layernorm))
    elif hasattr(layer, "pre_mlp_layernorm"):
        hook_names.append(("pre_mlp_layernorm", layer.pre_mlp_layernorm))

    # MLP module
    if hasattr(layer, "mlp"):
        hook_names.append(("mlp", layer.mlp))
        # Hook MLP submodules (Megatron uses linear_fc1 and linear_fc2)
        if hasattr(layer.mlp, "linear_fc1"):
            hook_names.append(("mlp.linear_fc1", layer.mlp.linear_fc1))
        if hasattr(layer.mlp, "linear_fc2"):
            hook_names.append(("mlp.linear_fc2", layer.mlp.linear_fc2))
        # Also try legacy names for compatibility
        if hasattr(layer.mlp, "gate_proj"):
            hook_names.append(("mlp.gate_proj", layer.mlp.gate_proj))
        if hasattr(layer.mlp, "up_proj"):
            hook_names.append(("mlp.up_proj", layer.mlp.up_proj))
        if hasattr(layer.mlp, "down_proj"):
            hook_names.append(("mlp.down_proj", layer.mlp.down_proj))

        # Hook activation function if it exists as a module or attribute
        # For TransformerEngine MLP, activation might be applied in forward
        # We'll add a special hook to capture activation output
        if hasattr(layer.mlp, "activation_fn"):
            hook_names.append(("mlp.activation_fn", layer.mlp.activation_fn))

    # Register all hooks
    for name, module in hook_names:
        try:
            hook = module.register_forward_hook(make_activation_hook(name))
            hooks.append(hook)
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    # Add pre-hook to linear_fc2 to capture activation function output
    # (linear_fc2's input is the output of activation function)
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "linear_fc2"):

        def mlp_activation_output_hook(module, input):
            """Capture the output of activation function (input to linear_fc2)."""
            try:
                if isinstance(input, tuple):
                    # input[0] is the activation output
                    activations["mlp.activation_output"] = (
                        input[0].clone().detach() if len(input) > 0 else None
                    )
                else:
                    activations["mlp.activation_output"] = input.clone().detach()
            except Exception as e:
                logger.warning(f"Failed to capture MLP activation output: {e}")

        try:
            activation_hook = layer.mlp.linear_fc2.register_forward_pre_hook(
                mlp_activation_output_hook
            )
            hooks.append(activation_hook)
        except Exception as e:
            logger.warning(f"Failed to register MLP activation output hook: {e}")

    # Also try for legacy names
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):

        def mlp_activation_output_hook_legacy(module, input):
            """Capture the output of activation function (input to down_proj)."""
            try:
                if isinstance(input, tuple):
                    activations["mlp.activation_output"] = (
                        input[0].clone().detach() if len(input) > 0 else None
                    )
                else:
                    activations["mlp.activation_output"] = input.clone().detach()
            except Exception as e:
                logger.warning(f"Failed to capture MLP activation output (legacy): {e}")

        try:
            activation_hook = layer.mlp.down_proj.register_forward_pre_hook(
                mlp_activation_output_hook_legacy
            )
            hooks.append(activation_hook)
        except Exception as e:
            logger.warning(
                f"Failed to register MLP activation output hook (legacy): {e}"
            )

    # Also register a hook for the final layer output
    def final_output_hook(module, input, output):
        try:
            if isinstance(output, tuple):
                activations["layer_output"] = (
                    output[0].clone().detach() if len(output) > 0 else None
                )
            else:
                activations["layer_output"] = output.clone().detach()
        except Exception as e:
            logger.warning(f"Failed to capture layer output: {e}")

    final_hook = layer.register_forward_hook(final_output_hook)
    hooks.append(final_hook)

    # Forward pass
    layer.train()
    layer.zero_grad()

    # Ensure input is on the same device as layer
    device = next(layer.parameters()).device
    input_hidden_states = input_hidden_states.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Prepare input - Megatron layers typically expect (hidden_states, attention_mask, ...)
    # We need to check the actual signature, but for now assume standard format
    try:
        # Try standard forward signature with attention_mask as kwarg
        if attention_mask is not None:
            output = layer(input_hidden_states, attention_mask=attention_mask)
        else:
            output = layer(input_hidden_states)
    except Exception as e:
        logger.warning(f"Standard forward failed: {e}, trying alternative signature")
        # Try alternative signatures
        try:
            # Try positional attention_mask
            if attention_mask is not None:
                output = layer(input_hidden_states, attention_mask)
            else:
                output = layer(input_hidden_states)
        except Exception as e2:
            logger.warning(
                f"Positional attention_mask failed: {e2}, trying hidden_states only"
            )
            # Last resort: just pass hidden states
            output = layer(input_hidden_states)

    if isinstance(output, tuple):
        output_hidden_states = output[0]
    else:
        output_hidden_states = output

    # Create a dummy loss for backward
    # Use mean of output as loss to get gradients
    loss = output_hidden_states.mean()

    # Backward pass
    loss.backward()

    # Collect gradients
    for name, param in layer.named_parameters():
        if param.requires_grad:
            # Try to get gradient from param.grad or param.main_grad
            grad = None
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grad = param.main_grad.clone().detach()
            elif hasattr(param, "grad") and param.grad is not None:
                grad = param.grad.clone().detach()
            else:
                raise ValueError(f"No gradient found for {name}")

            if grad is not None:
                gradients[name] = grad

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return output_hidden_states, activations, gradients


def categorize_op_name(name: str) -> str:
    """Categorize operation name into op type.

    Args:
        name: Parameter or activation name

    Returns:
        Op type category: 'attention', 'mlp', 'layernorm', 'embedding', 'other'
    """
    name_lower = name.lower()
    if "attn" in name_lower or "attention" in name_lower:
        if (
            "qkv" in name_lower
            or "q_proj" in name_lower
            or "k_proj" in name_lower
            or "v_proj" in name_lower
        ):
            return "attention_proj"
        elif (
            "linear_proj" in name_lower
            or "o_proj" in name_lower
            or "out_proj" in name_lower
        ):
            return "attention_out"
        elif "core_attention" in name_lower:
            return "attention_core"
        else:
            return "attention"
    elif "mlp" in name_lower or "feedforward" in name_lower or "ffn" in name_lower:
        if "activation" in name_lower:
            return "mlp_activation"
        elif "fc1" in name_lower or "gate" in name_lower or "up" in name_lower:
            return "mlp_gate_up"
        elif "fc2" in name_lower or "down" in name_lower:
            return "mlp_down"
        else:
            return "mlp"
    elif "rotary" in name_lower or "rope" in name_lower:
        return "rope"
    elif "layernorm" in name_lower or "norm" in name_lower:
        # Distinguish Q/K layernorms from regular layernorms
        if "q_layernorm" in name_lower or "k_layernorm" in name_lower:
            return "qk_layernorm"
        return "layernorm"
    elif "embedding" in name_lower or "embed" in name_lower:
        return "embedding"
    else:
        return "other"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp8_bf16_single_layer_comparison(fixed_input, save_data: bool = False):
    """Compare FP8 and BF16 on a model reduced to specified layers.

    This test reduces the model to specified transformer layers while keeping the full
    structure (embedding, rotary_pos_emb, final_layernorm, output_layer), performs
    forward and backward with real loss computation, and compares activations and
    gradients between FP8 and BF16 models to identify which operations have precision issues.
    """
    # Test specific layers - can be a single layer index or a list of indices
    layer_indices = list(
        range(2)
    )  # Test the first layer, or use [0, 1] to test first two layers

    # Create BF16 engine
    engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
    try:
        logger.info("=" * 80)
        logger.info(f"Testing model with layers {layer_indices} - BF16 Model")
        logger.info("=" * 80)

        # Forward and backward on model with specified layers
        logits_bf16, activations_bf16, gradients_bf16, output_gradients_bf16 = (
            forward_backward_model_with_hooks(
                engine_bf16,
                fixed_input,
                layer_indices=layer_indices,
            )
        )

        logger.info(f"BF16 - Logits shape: {logits_bf16.shape}")
        logger.info(f"BF16 - Collected {len(activations_bf16)} activations")
        logger.info(f"BF16 - Collected {len(gradients_bf16)} gradients")
        logger.info(f"BF16 - Collected {len(output_gradients_bf16)} output gradients")

    finally:
        engine_bf16.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Create FP8 engine
    engine_fp8 = create_engine(
        MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778
    )
    try:
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing model with layers {layer_indices} - FP8 Model")
        logger.info("=" * 80)

        # Forward and backward on model with specified layers
        logits_fp8, activations_fp8, gradients_fp8, output_gradients_fp8 = (
            forward_backward_model_with_hooks(
                engine_fp8,
                fixed_input,
                layer_indices=layer_indices,
            )
        )

        logger.info(f"FP8 - Logits shape: {logits_fp8.shape}")
        logger.info(f"FP8 - Collected {len(activations_fp8)} activations")
        logger.info(f"FP8 - Collected {len(gradients_fp8)} gradients")
        logger.info(f"FP8 - Collected {len(output_gradients_fp8)} output gradients")

    finally:
        engine_fp8.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    # Compare logits
    logger.info("\n" + "=" * 80)
    logger.info("Logits Comparison")
    logger.info("=" * 80)
    if logits_bf16.shape == logits_fp8.shape:
        logits_diff = (logits_bf16 - logits_fp8).abs()
        logits_max_diff = logits_diff.max().item()
        logits_mean_diff = logits_diff.mean().item()
        logits_cos_sim = F.cosine_similarity(
            logits_bf16.flatten().unsqueeze(0), logits_fp8.flatten().unsqueeze(0), dim=1
        ).item()
        logger.info(f"Logits max diff: {logits_max_diff:.6f}")
        logger.info(f"Logits mean diff: {logits_mean_diff:.6f}")
        logger.info(f"Logits cosine similarity: {logits_cos_sim:.6f}")
    else:
        logger.warning(
            f"Logits shapes don't match: BF16={logits_bf16.shape}, FP8={logits_fp8.shape}"
        )

    # Compare activations by op type
    logger.info("\n" + "=" * 80)
    logger.info("Activation Comparison by Operation Type")
    logger.info("=" * 80)

    activation_stats_by_type = defaultdict(
        lambda: {"max_diffs": [], "mean_diffs": [], "cos_sims": [], "names": []}
    )

    common_activation_names = set(activations_bf16.keys()) & set(activations_fp8.keys())
    for name in sorted(common_activation_names):
        act_bf16 = activations_bf16[name]
        act_fp8 = activations_fp8[name]

        if act_bf16 is None or act_fp8 is None:
            continue

        if act_bf16.shape != act_fp8.shape:
            logger.warning(
                f"Activation {name} shapes don't match: BF16={act_bf16.shape}, FP8={act_fp8.shape}"
            )
            continue

        act_diff = (act_bf16 - act_fp8).abs()
        max_diff = act_diff.max().item()
        mean_diff = act_diff.mean().item()

        act_bf16_flat = act_bf16.flatten()
        act_fp8_flat = act_fp8.flatten()
        if name == "embedding":
            print(f"Embedding BF16: {act_bf16.shape}, FP8: {act_fp8.shape}")
        cos_sim = F.cosine_similarity(
            act_bf16_flat.unsqueeze(0), act_fp8_flat.unsqueeze(0), dim=1
        ).item()

        # if cos_sim > 0.9:
        #     print(f"scale ratio: {torch.norm(act_bf16_flat, 2) / torch.norm(act_fp8_flat, 2)}")

        op_type = categorize_op_name(name)
        activation_stats_by_type[op_type]["max_diffs"].append(max_diff)
        activation_stats_by_type[op_type]["mean_diffs"].append(mean_diff)
        activation_stats_by_type[op_type]["cos_sims"].append(cos_sim)
        activation_stats_by_type[op_type]["names"].append(name)

        # Format with fixed width for alignment
        name_str = f"{name} ({op_type})"
        logger.info(
            f"{name_str:<50} "
            f"max_diff={max_diff:>12.6f}, "
            f"mean_diff={mean_diff:>12.6f}, "
            f"cos_sim={cos_sim:>10.6f}"
        )

    # Summary by op type
    logger.info("\n" + "-" * 80)
    logger.info("Activation Summary by Operation Type")
    logger.info("-" * 80)
    for op_type in sorted(activation_stats_by_type.keys()):
        stats = activation_stats_by_type[op_type]
        if stats["max_diffs"]:
            max_diff_val = max(stats["max_diffs"])
            mean_diff_val = sum(stats["mean_diffs"]) / len(stats["mean_diffs"])
            cos_sim_val = sum(stats["cos_sims"]) / len(stats["cos_sims"])
            logger.info(
                f"{op_type:<50} "
                f"max_diff={max_diff_val:>12.6f}, "
                f"mean_diff={mean_diff_val:>12.6f}, "
                f"cos_sim={cos_sim_val:>10.6f}, "
                f"n_ops={len(stats['names']):>4}"
            )

    # Compare gradients by op type
    logger.info("\n" + "=" * 80)
    logger.info("Gradient Comparison by Operation Type")
    logger.info("=" * 80)

    gradient_stats_by_type = defaultdict(
        lambda: {"max_diffs": [], "mean_diffs": [], "cos_sims": [], "names": []}
    )

    common_gradient_names = set(gradients_bf16.keys()) & set(gradients_fp8.keys())
    for name in sorted(common_gradient_names):
        grad_bf16 = gradients_bf16[name]
        grad_fp8 = gradients_fp8[name]

        if grad_bf16.shape != grad_fp8.shape:
            logger.warning(
                f"Gradient {name} shapes don't match: BF16={grad_bf16.shape}, FP8={grad_fp8.shape}"
            )
            continue

        # Check for NaN or Inf
        bf16_has_nan = torch.isnan(grad_bf16).any().item()
        bf16_has_inf = torch.isinf(grad_bf16).any().item()
        fp8_has_nan = torch.isnan(grad_fp8).any().item()
        fp8_has_inf = torch.isinf(grad_fp8).any().item()

        if bf16_has_nan or bf16_has_inf or fp8_has_nan or fp8_has_inf:
            logger.warning(
                f"Gradient {name} has NaN/Inf: "
                f"BF16 NaN={bf16_has_nan}, Inf={bf16_has_inf}, "
                f"FP8 NaN={fp8_has_nan}, Inf={fp8_has_inf}"
            )

        # Check if gradients are zero
        bf16_norm = grad_bf16.norm().item()
        fp8_norm = grad_fp8.norm().item()

        if bf16_norm == 0.0 or fp8_norm == 0.0:
            logger.warning(
                f"Gradient {name} has zero norm: BF16 norm={bf16_norm:.6e}, FP8 norm={fp8_norm:.6e}"
            )
            # If one is zero, cosine similarity will be undefined (0/0), set to 0
            cos_sim = 0.0
        else:
            grad_bf16_flat = grad_bf16.flatten()
            grad_fp8_flat = grad_fp8.flatten()
            cos_sim = F.cosine_similarity(
                grad_bf16_flat.unsqueeze(0), grad_fp8_flat.unsqueeze(0), dim=1
            ).item()

            # Check if cosine similarity is NaN (can happen if both vectors are zero or very small)
            if torch.isnan(torch.tensor(cos_sim)):
                logger.warning(
                    f"Gradient {name} cosine similarity is NaN, setting to 0.0"
                )
                cos_sim = 0.0

        grad_diff = (grad_bf16 - grad_fp8).abs()
        max_diff = grad_diff.max().item()
        mean_diff = grad_diff.mean().item()

        op_type = categorize_op_name(name)
        gradient_stats_by_type[op_type]["max_diffs"].append(max_diff)
        gradient_stats_by_type[op_type]["mean_diffs"].append(mean_diff)
        gradient_stats_by_type[op_type]["cos_sims"].append(cos_sim)
        gradient_stats_by_type[op_type]["names"].append(name)

        # Log detailed info for problematic gradients
        if cos_sim < 0.1 or bf16_norm == 0.0 or fp8_norm == 0.0:
            name_str = f"{name} ({op_type})"
            logger.warning(
                f"{name_str:<50} "
                f"max_diff={max_diff:>12.6f}, "
                f"mean_diff={mean_diff:>12.6f}, "
                f"cos_sim={cos_sim:>10.6f}, "
                f"BF16_norm={bf16_norm:>12.6e}, FP8_norm={fp8_norm:>12.6e}, "
                f"BF16_shape={str(grad_bf16.shape):<20}, FP8_shape={str(grad_fp8.shape):<20}, "
                f"BF16_min={grad_bf16.min().item():>12.6e}, BF16_max={grad_bf16.max().item():>12.6e}, "
                f"FP8_min={grad_fp8.min().item():>12.6e}, FP8_max={grad_fp8.max().item():>12.6e}"
            )
        else:
            # Format with fixed width for alignment
            name_str = f"{name} ({op_type})"
            logger.info(
                f"{name_str:<80} "
                f"max_diff={max_diff:>12.6f}, "
                f"mean_diff={mean_diff:>12.6f}, "
                f"cos_sim={cos_sim:>10.6f}"
            )

    # Summary by op type
    logger.info("\n" + "-" * 80)
    logger.info("Gradient Summary by Operation Type")
    logger.info("-" * 80)
    for op_type in sorted(gradient_stats_by_type.keys()):
        stats = gradient_stats_by_type[op_type]
        if stats["max_diffs"]:
            max_diff_val = max(stats["max_diffs"])
            mean_diff_val = sum(stats["mean_diffs"]) / len(stats["mean_diffs"])
            cos_sim_val = sum(stats["cos_sims"]) / len(stats["cos_sims"])
            logger.info(
                f"{op_type:<50} "
                f"max_diff={max_diff_val:>12.6f}, "
                f"mean_diff={mean_diff_val:>12.6f}, "
                f"cos_sim={cos_sim_val:>10.6f}, "
                f"n_params={len(stats['names']):>4}, "
                f"names={','.join(stats['names'])}"
            )

    # Collect all output gradients for statistics
    logger.info("\n" + "=" * 80)
    logger.info("Output Gradient Statistics")
    logger.info("=" * 80)

    # Compare output gradients by operation
    common_output_grad_names = set(output_gradients_bf16.keys()) & set(
        output_gradients_fp8.keys()
    )

    output_grad_stats_by_type = defaultdict(
        lambda: {"max_diffs": [], "mean_diffs": [], "cos_sims": [], "names": []}
    )

    for name in sorted(common_output_grad_names):
        grad_bf16 = output_gradients_bf16[name]
        grad_fp8 = output_gradients_fp8[name]

        if grad_bf16.shape != grad_fp8.shape:
            logger.warning(
                f"Output grad {name} shapes don't match: BF16={grad_bf16.shape}, FP8={grad_fp8.shape}"
            )
            continue

        # Calculate differences
        grad_diff = (grad_bf16 - grad_fp8).abs()
        max_diff = grad_diff.max().item()
        mean_diff = grad_diff.mean().item()

        # Cosine similarity
        grad_bf16_flat = grad_bf16.flatten()
        grad_fp8_flat = grad_fp8.flatten()
        cos_sim = F.cosine_similarity(
            grad_bf16_flat.unsqueeze(0), grad_fp8_flat.unsqueeze(0), dim=1
        ).item()

        # Norms
        grad_bf16_norm = grad_bf16.norm().item()
        grad_fp8_norm = grad_fp8.norm().item()

        op_type = categorize_op_name(name.replace(".output_grad", ""))
        output_grad_stats_by_type[op_type]["max_diffs"].append(max_diff)
        output_grad_stats_by_type[op_type]["mean_diffs"].append(mean_diff)
        output_grad_stats_by_type[op_type]["cos_sims"].append(cos_sim)
        output_grad_stats_by_type[op_type]["names"].append(name)

        # Format with fixed width for alignment
        logger.info(
            f"{name:<80} "
            f"max_diff={max_diff:>12.6f}, "
            f"mean_diff={mean_diff:>12.6f}, "
            f"cos_sim={cos_sim:>10.6f}, "
            f"BF16_norm={grad_bf16_norm:>12.6f}, FP8_norm={grad_fp8_norm:>12.6f}"
        )

    # Summary by op type
    logger.info("\n" + "-" * 80)
    logger.info("Output Gradient Summary by Operation Type")
    logger.info("-" * 80)
    for op_type in sorted(output_grad_stats_by_type.keys()):
        stats = output_grad_stats_by_type[op_type]
        if stats["max_diffs"]:
            max_diff_val = max(stats["max_diffs"])
            mean_diff_val = sum(stats["mean_diffs"]) / len(stats["mean_diffs"])
            cos_sim_val = sum(stats["cos_sims"]) / len(stats["cos_sims"])
            logger.info(
                f"{op_type:<50} "
                f"max_diff={max_diff_val:>12.6f}, "
                f"mean_diff={mean_diff_val:>12.6f}, "
                f"cos_sim={cos_sim_val:>10.6f}, "
                f"n_ops={len(stats['names']):>4}"
            )

    if save_data:
        # Save q_layernorm and k_layernorm inputs and output gradients for separate testing
        layernorm_inputs_bf16 = {}
        layernorm_inputs_fp8 = {}
        layernorm_output_grads_bf16 = {}
        layernorm_output_grads_fp8 = {}
        for name in activations_bf16.keys():
            if name.endswith(".q_layernorm.input") or name.endswith(
                ".k_layernorm.input"
            ):
                layernorm_inputs_bf16[name] = activations_bf16[name]
        for name in activations_fp8.keys():
            if name.endswith(".q_layernorm.input") or name.endswith(
                ".k_layernorm.input"
            ):
                layernorm_inputs_fp8[name] = activations_fp8[name]
        for name in output_gradients_bf16.keys():
            if name.endswith(".q_layernorm.output_grad") or name.endswith(
                ".k_layernorm.output_grad"
            ):
                layernorm_output_grads_bf16[name] = output_gradients_bf16[name]
        for name in output_gradients_fp8.keys():
            if name.endswith(".q_layernorm.output_grad") or name.endswith(
                ".k_layernorm.output_grad"
            ):
                layernorm_output_grads_fp8[name] = output_gradients_fp8[name]

        if layernorm_inputs_bf16 or layernorm_inputs_fp8:
            logger.info("\n" + "=" * 80)
            logger.info("Found layernorm inputs for separate testing")
            logger.info(f"BF16 layernorm inputs: {list(layernorm_inputs_bf16.keys())}")
            logger.info(f"FP8 layernorm inputs: {list(layernorm_inputs_fp8.keys())}")
            logger.info(
                f"BF16 layernorm output grads: {list(layernorm_output_grads_bf16.keys())}"
            )
            logger.info(
                f"FP8 layernorm output grads: {list(layernorm_output_grads_fp8.keys())}"
            )
            logger.info("=" * 80)

            # Save activation inputs to files
            save_dir = Path("activation_inputs")
            save_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save BF16 activation inputs
            if layernorm_inputs_bf16:
                bf16_save_path = save_dir / f"bf16_layernorm_inputs_{timestamp}.pt"
                torch.save(layernorm_inputs_bf16, bf16_save_path)
                logger.info(f"Saved BF16 layernorm inputs to: {bf16_save_path}")
                logger.info(
                    f"  Total size: {bf16_save_path.stat().st_size / 1024 / 1024:.2f} MB"
                )
                for name, tensor in layernorm_inputs_bf16.items():
                    logger.info(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

            # Save FP8 activation inputs
            if layernorm_inputs_fp8:
                fp8_save_path = save_dir / f"fp8_layernorm_inputs_{timestamp}.pt"
                torch.save(layernorm_inputs_fp8, fp8_save_path)
                logger.info(f"Saved FP8 layernorm inputs to: {fp8_save_path}")
                logger.info(
                    f"  Total size: {fp8_save_path.stat().st_size / 1024 / 1024:.2f} MB"
                )
                for name, tensor in layernorm_inputs_fp8.items():
                    logger.info(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

            # Also save a combined file with metadata
            # Save all output gradients, not just layernorm ones
            combined_data = {
                "bf16_inputs": layernorm_inputs_bf16,
                "fp8_inputs": layernorm_inputs_fp8,
                "bf16_output_grads": layernorm_output_grads_bf16,
                "fp8_output_grads": layernorm_output_grads_fp8,
                # 'bf16_all_output_grads': output_gradients_bf16,  # All output gradients
                # 'fp8_all_output_grads': output_gradients_fp8,  # All output gradients
                "timestamp": timestamp,
                "layer_indices": layer_indices,
            }
            combined_save_path = save_dir / f"layernorm_inputs_combined_{timestamp}.pt"
            torch.save(combined_data, combined_save_path)
            logger.info(f"Saved combined layernorm inputs to: {combined_save_path}")
            logger.info(
                f"  Total size: {combined_save_path.stat().st_size / 1024 / 1024:.2f} MB"
            )

    # Identify problematic operations
    logger.info("\n" + "=" * 80)
    logger.info("Problematic Operations (low cosine similarity)")
    logger.info("=" * 80)

    threshold = 0.95
    problematic_activations = []
    problematic_gradients = []

    for op_type, stats in activation_stats_by_type.items():
        for i, (name, cos_sim) in enumerate(zip(stats["names"], stats["cos_sims"])):
            if cos_sim < threshold:
                problematic_activations.append(
                    (op_type, name, cos_sim, stats["max_diffs"][i])
                )

    for op_type, stats in gradient_stats_by_type.items():
        for i, (name, cos_sim) in enumerate(zip(stats["names"], stats["cos_sims"])):
            if cos_sim < threshold:
                problematic_gradients.append(
                    (op_type, name, cos_sim, stats["max_diffs"][i])
                )

    if problematic_activations:
        logger.info("Problematic Activations:")
        for op_type, name, cos_sim, max_diff in sorted(
            problematic_activations, key=lambda x: x[2]
        ):
            logger.info(
                f"  {name} ({op_type}): cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}"
            )
    else:
        logger.info("No problematic activations found (all cos_sim >= 0.95)")

    if problematic_gradients:
        logger.info("Problematic Gradients:")
        for op_type, name, cos_sim, max_diff in sorted(
            problematic_gradients, key=lambda x: x[2]
        ):
            logger.info(
                f"  {name} ({op_type}): cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}"
            )
    else:
        logger.info("No problematic gradients found (all cos_sim >= 0.95)")

    logger.info("=" * 80)


def dequantize_fp8_param(tensor: torch.Tensor) -> torch.Tensor:
    if is_float8tensor(tensor):
        return tensor.dequantize(dtype=torch.bfloat16)
    else:
        logger.info("Not a quantized tensor, converting to bfloat16")
        return tensor.to(torch.bfloat16)


def forward_backward_rmsnorm_module(
    layernorm_module: torch.nn.Module,
    input_activation: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    name: str = "rmsnorm",
    collect_gradients: bool = True,
    output_grad: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Forward and backward a single RMSNorm module with given input activation.

    This function tests a RMSNorm module in isolation by:
    1. Setting the module to train mode (for gradients)
    2. Converting input to the specified dtype
    3. Running forward pass
    4. Running backward pass with a dummy loss
    5. Collecting output statistics and gradients

    Args:
        layernorm_module: The RMSNorm module to test
        input_activation: Input activation tensor
        dtype: Data type to use (torch.bfloat16 or torch.float16)
        name: Name identifier for logging
        collect_gradients: Whether to collect gradients (requires backward pass)
        output_grad: Optional gradient from downstream layers for backward pass

    Returns:
        Dictionary with output tensor, statistics, and gradients
    """

    layernorm_module.train()  # Set to train mode for gradients

    # Convert input to specified dtype and ensure it requires grad
    input_activation = input_activation.to(dtype=dtype)
    if collect_gradients:
        input_activation = input_activation.clone().detach().requires_grad_(True)

    # Forward pass
    output = layernorm_module(input_activation)

    # Calculate statistics
    output_norm = output.norm().item()
    output_max = output.abs().max().item()
    output_mean = output.mean().item()
    output_std = output.std().item()

    gradients = {}
    if collect_gradients:
        # Zero gradients first
        layernorm_module.zero_grad()
        if input_activation.grad is not None:
            input_activation.grad.zero_()

        # Use provided output gradient if available, otherwise use dummy loss
        if output_grad is not None:
            # Use the real gradient from downstream layers
            output_grad = output_grad.to(dtype=dtype, device=output.device)
            output.backward(output_grad)
        else:
            # Create a dummy loss (sum of output)
            loss = output.sum()
            # Backward pass
            loss.backward()

        # Collect gradients from module parameters
        for param_name, param in layernorm_module.named_parameters():
            if param.requires_grad:
                grad = None
                # Check different gradient storage locations
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad = param.main_grad.clone().detach()
                elif hasattr(param, "grad") and param.grad is not None:
                    grad = param.grad.clone().detach()
                else:
                    raise ValueError(f"No gradient found for {param_name}")
                if grad is not None:
                    gradients[param_name + "_grad"] = grad
                    logger.debug(
                        f"{name} gradient {param_name}: "
                        f"shape={grad.shape}, norm={grad.norm().item():.6f}, "
                        f"min={grad.min().item():.6f}, max={grad.max().item():.6f}"
                    )

        # # Also collect input gradient
        # if input_activation.grad is not None:
        #     gradients['input'] = input_activation.grad.clone().detach()
        gradients["input"] = input_activation.clone().detach()
        gradients["output"] = output.clone().detach()

        if output_grad is not None:
            gradients["output_grad"] = output_grad.clone().detach()

    logger.info(
        f"{name} ({dtype}): "
        f"input_shape={input_activation.shape}, output_shape={output.shape}, "
        f"output_norm={output_norm:.6f}, output_max={output_max:.6f}, "
        f"output_mean={output_mean:.6f}, output_std={output_std:.6f}, "
        f"n_gradients={len(gradients)}"
    )

    return {
        "output": output,
        "output_norm": output_norm,
        "output_max": output_max,
        "output_mean": output_mean,
        "output_std": output_std,
        "input_shape": input_activation.shape,
        "output_shape": output.shape,
        "gradients": gradients,
    }


def load_layernorm_inputs_from_file(file_path: str | Path) -> dict[str, Any]:
    """Load layernorm activation inputs from saved file.

    Args:
        file_path: Path to the saved .pt file (can be combined file or individual file)

    Returns:
        Dictionary with 'bf16_inputs', 'fp8_inputs', 'timestamp', 'layer_indices'
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = torch.load(file_path, map_location="cpu")

    # Check if it's a combined file or individual file
    if isinstance(data, dict) and "bf16_inputs" in data and "fp8_inputs" in data:
        # Combined file
        return data
    elif isinstance(data, dict):
        # Individual file - determine if BF16 or FP8 based on keys or filename
        if "bf16" in file_path.name.lower():
            return {
                "bf16_inputs": data,
                "fp8_inputs": {},
                "timestamp": file_path.stem.split("_")[-1]
                if "_" in file_path.stem
                else "",
                "layer_indices": [],
            }
        elif "fp8" in file_path.name.lower():
            return {
                "bf16_inputs": {},
                "fp8_inputs": data,
                "timestamp": file_path.stem.split("_")[-1]
                if "_" in file_path.stem
                else "",
                "layer_indices": [],
            }
        else:
            # Assume it's BF16 if can't determine
            return {
                "bf16_inputs": data,
                "fp8_inputs": {},
                "timestamp": file_path.stem.split("_")[-1]
                if "_" in file_path.stem
                else "",
                "layer_indices": [],
            }
    else:
        raise ValueError(f"Unexpected file format in {file_path}")


def get_custom_rmsnorm(
    layernorm_module: torch.nn.Module,
    hf_config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    weight: torch.Tensor | None = None,
) -> torch.nn.Module:
    # Extract weight parameter
    if hasattr(layernorm_module, "weight"):
        weight_param = layernorm_module.weight
    else:
        # Try to find weight in named_parameters
        weight_param = None
        for name, param in layernorm_module.named_parameters():
            if "weight" in name.lower():
                weight_param = param
                break

    if weight_param is None:
        raise ValueError(f"Cannot find weight parameter in {layernorm_module}")

    # Dequantize if FP8, or convert to bfloat16
    dequantized_weight_data = dequantize_fp8_param(weight_param.data)

    # Get hidden_size from weight shape
    hidden_size = hf_config.head_dim
    eps = hf_config.rms_norm_eps

    # Create custom RMSNorm module
    custom_rmsnorm = Qwen3RMSNorm(hidden_size, eps=eps)
    if weight is not None:
        custom_rmsnorm.weight.data = (
            weight.clone().detach().to(device=device, dtype=dtype)
        )
    else:
        custom_rmsnorm.weight.data = dequantized_weight_data.clone().detach()
    custom_rmsnorm = custom_rmsnorm.to(device=device, dtype=dtype)

    logger.info(
        f"Using custom Qwen3RMSNorm for to replace {layernorm_module} with dtype {dtype}"
    )

    return custom_rmsnorm


def compare_rmsnorm_bf16_fp8(
    engine_bf16: MegatronEngine,
    engine_fp8: MegatronEngine,
    q_layernorm_input_bf16: torch.Tensor,
    q_layernorm_input_fp8: torch.Tensor,
    layer_path: str,
    output_grad_bf16: torch.Tensor | None = None,
    output_grad_fp8: torch.Tensor | None = None,
    use_custom_rmsnorm: bool = False,
    save_data: bool = False,
) -> dict[str, Any]:
    """Compare RMSNorm module outputs between BF16 and FP8 engines.

    This function extracts the q_layernorm module from both engines and compares
    their outputs when given the respective input activations.

    Args:
        engine_bf16: BF16 MegatronEngine
        engine_fp8: FP8 MegatronEngine
        q_layernorm_input_bf16: Input activation from BF16 model
        q_layernorm_input_fp8: Input activation from FP8 model
        layer_path: Path to identify the layer (e.g., "layer_0.self_attention.q_layernorm")

    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 80)
    logger.info(f"Testing RMSNorm module: {layer_path}")
    logger.info("=" * 80)

    # Extract q_layernorm module from both engines
    model_bf16 = get_model_from_engine(engine_bf16)
    model_fp8 = get_model_from_engine(engine_fp8)

    # Parse layer path (e.g., "layer_0.self_attention.q_layernorm" or "layer_0.self_attention.k_layernorm")
    matches = re.match(
        r"layer_(\d+)\.self_attention\.(q_layernorm|k_layernorm)", layer_path
    )
    if not matches:
        raise ValueError(
            f"Invalid layer path: {layer_path}. Expected format: layer_X.self_attention.(q_layernorm|k_layernorm)"
        )
    layer_idx = int(matches.group(1))
    layernorm_type = matches.group(2)

    fp8_context = get_fp8_context(get_model_config(model_fp8), layer_no=layer_idx)

    # Get decoder and layer
    decoder_bf16 = model_bf16.decoder if hasattr(model_bf16, "decoder") else None
    decoder_fp8 = model_fp8.decoder if hasattr(model_fp8, "decoder") else None

    if decoder_bf16 is None or decoder_fp8 is None:
        raise ValueError("Cannot find decoder in model")

    if layer_idx >= len(decoder_bf16.layers) or layer_idx >= len(decoder_fp8.layers):
        raise ValueError(f"Layer index {layer_idx} out of range")

    layer_bf16 = decoder_bf16.layers[layer_idx]
    layer_fp8 = decoder_fp8.layers[layer_idx]

    if not hasattr(layer_bf16.self_attention, layernorm_type) or not hasattr(
        layer_fp8.self_attention, layernorm_type
    ):
        raise ValueError(f"Layer {layer_idx} does not have {layernorm_type}")

    layernorm_bf16 = getattr(layer_bf16.self_attention, layernorm_type)
    layernorm_fp8 = getattr(layer_fp8.self_attention, layernorm_type)

    # Test BF16
    logger.info("Testing BF16 RMSNorm...")
    if use_custom_rmsnorm:
        layernorm_bf16 = get_custom_rmsnorm(
            layernorm_bf16, engine_bf16.hf_config, engine_bf16.device, torch.bfloat16
        )
    result_bf16 = forward_backward_rmsnorm_module(
        layernorm_bf16,
        q_layernorm_input_bf16,
        output_grad=output_grad_bf16,
        dtype=torch.bfloat16,
        name=f"{layer_path} (BF16)",
        collect_gradients=True,
    )

    # Test FP8
    logger.info("Testing FP8 RMSNorm...")
    if use_custom_rmsnorm:
        # For custom RMSNorm, we dequantize params first, so no need for FP8 context
        layernorm_fp8 = get_custom_rmsnorm(
            layernorm_fp8, engine_fp8.hf_config, engine_fp8.device, torch.bfloat16
        )
        result_fp8 = forward_backward_rmsnorm_module(
            layernorm_fp8,
            q_layernorm_input_fp8,
            output_grad=output_grad_fp8,
            dtype=torch.bfloat16,  # Will use dequantized params
            name=f"{layer_path} (FP8, dequantized)",
            collect_gradients=True,
        )
    else:
        # Use original FP8 module with FP8 context
        with fp8_context:
            result_fp8 = forward_backward_rmsnorm_module(
                layernorm_fp8,
                q_layernorm_input_fp8,
                output_grad=output_grad_fp8,
                dtype=torch.bfloat16,  # Input will be converted, but module may use FP8 internally
                name=f"{layer_path} (FP8)",
                collect_gradients=True,
            )

    if save_data:
        # save input, weight, output_grad for both BF16 and FP8
        save_dir = Path("layernorm_inputs")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"layernorm_inputs_{layer_path}_{timestamp}.pt"
        torch.save(
            {
                "bf16": {
                    "input": q_layernorm_input_bf16,
                    "weight": layernorm_bf16.weight.data.clone().detach(),
                    "output_grad": output_grad_bf16.clone().detach(),
                },
                "fp8": {
                    "input": q_layernorm_input_fp8,
                    "weight": layernorm_fp8.weight.data.clone().detach(),
                    "output_grad": output_grad_fp8.clone().detach(),
                },
            },
            save_path,
        )
        logger.info(f"Saved layernorm inputs to: {save_path}")
        logger.info(f"  Total size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(
            f"  BF16 - Input shape: {q_layernorm_input_bf16.shape}, dtype: {q_layernorm_input_bf16.dtype}"
        )
        logger.info(
            f"  BF16 - Weight shape: {layernorm_bf16.weight.data.shape}, dtype: {layernorm_bf16.weight.data.dtype}"
        )
        logger.info(
            f"  BF16 - Output grad shape: {output_grad_bf16.shape}, dtype: {output_grad_bf16.dtype}"
        )
        logger.info(
            f"  FP8 - Input shape: {q_layernorm_input_fp8.shape}, dtype: {q_layernorm_input_fp8.dtype}"
        )
        logger.info(
            f"  FP8 - Weight shape: {layernorm_fp8.weight.data.shape}, dtype: {layernorm_fp8.weight.data.dtype}"
        )
        logger.info(
            f"  FP8 - Output grad shape: {output_grad_fp8.shape}, dtype: {output_grad_fp8.dtype}"
        )

    # Compare outputs
    output_bf16 = result_bf16["output"]
    output_fp8 = result_fp8["output"]

    if output_bf16.shape != output_fp8.shape:
        logger.warning(
            f"Output shapes don't match: BF16={output_bf16.shape}, FP8={output_fp8.shape}"
        )
        return {
            "layer_path": layer_path,
            "shape_mismatch": True,
            "bf16_shape": output_bf16.shape,
            "fp8_shape": output_fp8.shape,
        }

    # Calculate differences
    output_diff = (output_bf16 - output_fp8).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()

    # Cosine similarity
    output_bf16_flat = output_bf16.flatten()
    output_fp8_flat = output_fp8.flatten()
    cos_sim = F.cosine_similarity(
        output_bf16_flat.unsqueeze(0), output_fp8_flat.unsqueeze(0), dim=1
    ).item()

    logger.info("=" * 80)
    logger.info(f"RMSNorm Comparison Results for {layer_path}")
    logger.info("=" * 80)
    logger.info(
        f"Output - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}"
    )
    logger.info(
        f"BF16 output_norm={result_bf16['output_norm']:.6f}, FP8 output_norm={result_fp8['output_norm']:.6f}"
    )
    logger.info(
        f"BF16 output_max={result_bf16['output_max']:.6f}, FP8 output_max={result_fp8['output_max']:.6f}"
    )

    # Compare gradients
    gradients_bf16 = result_bf16.get("gradients", {})
    gradients_fp8 = result_fp8.get("gradients", {})

    gradient_comparison = {}
    common_gradient_names = set(gradients_bf16.keys()) & set(gradients_fp8.keys())

    if common_gradient_names:
        logger.info("\n" + "-" * 80)
        logger.info("Gradient Comparison")
        logger.info("-" * 80)

        for grad_name in sorted(common_gradient_names):
            grad_bf16 = gradients_bf16[grad_name]
            grad_fp8 = gradients_fp8[grad_name]

            if grad_bf16.shape != grad_fp8.shape:
                logger.warning(
                    f"Gradient {grad_name} shapes don't match: "
                    f"BF16={grad_bf16.shape}, FP8={grad_fp8.shape}"
                )
                continue

            # Calculate differences
            grad_diff = (grad_bf16 - grad_fp8).abs()
            grad_max_diff = grad_diff.max().item()
            grad_mean_diff = grad_diff.mean().item()

            # Cosine similarity
            grad_bf16_flat = grad_bf16.flatten()
            grad_fp8_flat = grad_fp8.flatten()
            grad_cos_sim = F.cosine_similarity(
                grad_bf16_flat.unsqueeze(0), grad_fp8_flat.unsqueeze(0), dim=1
            ).item()

            # Norms
            grad_bf16_norm = grad_bf16.norm().item()
            grad_fp8_norm = grad_fp8.norm().item()

            gradient_comparison[grad_name] = {
                "max_diff": grad_max_diff,
                "mean_diff": grad_mean_diff,
                "cos_sim": grad_cos_sim,
                "bf16_norm": grad_bf16_norm,
                "fp8_norm": grad_fp8_norm,
            }

            # Format with fixed width for alignment
            logger.info(
                f"{layer_path + '.' + grad_name:<80} "
                f"max_diff={grad_max_diff:>12.6f}, "
                f"mean_diff={grad_mean_diff:>12.6f}, "
                f"cos_sim={grad_cos_sim:>10.6f}, "
                f"BF16_norm={grad_bf16_norm:>12.6f}, FP8_norm={grad_fp8_norm:>12.6f}"
            )

        # Summary
        if gradient_comparison:
            avg_cos_sim = sum(g["cos_sim"] for g in gradient_comparison.values()) / len(
                gradient_comparison
            )
            max_grad_diff = max(g["max_diff"] for g in gradient_comparison.values())
            logger.info("-" * 80)
            logger.info(
                f"Gradient Summary: "
                f"avg_cos_sim={avg_cos_sim:.6f}, "
                f"max_diff={max_grad_diff:.6f}, "
                f"n_gradients={len(gradient_comparison)}"
            )
    else:
        logger.warning("No common gradients found for comparison")
        logger.info(f"BF16 gradients: {list(gradients_bf16.keys())}")
        logger.info(f"FP8 gradients: {list(gradients_fp8.keys())}")

    logger.info("=" * 80)

    return {
        "layer_path": layer_path,
        "output_max_diff": max_diff,
        "output_mean_diff": mean_diff,
        "output_cos_sim": cos_sim,
        "bf16_output_norm": result_bf16["output_norm"],
        "fp8_output_norm": result_fp8["output_norm"],
        "bf16_output_max": result_bf16["output_max"],
        "fp8_output_max": result_fp8["output_max"],
        "output_bf16": output_bf16,
        "output_fp8": output_fp8,
        "gradient_comparison": gradient_comparison,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_custom_rmsnorm", [True, False])
def test_rmsnorm_from_file(
    use_custom_rmsnorm: bool,
    activation_inputs_file: str | Path | None = None,
    layer_path: str | None = None,
    save_data: bool = False,
):
    """Test RMSNorm modules using activation inputs loaded from file.

    This test loads previously saved activation inputs from file and tests
    RMSNorm modules (q_layernorm and k_layernorm) in isolation.

    Args:
        activation_inputs_file: Path to the saved activation inputs file.
                               If None, will look for the most recent file in activation_inputs/
        layer_path: Specific layer path to test (e.g., "layer_0.self_attention.q_layernorm").
                   If None, will test all available layers.
        use_custom_rmsnorm: If True, use custom Qwen3RMSNorm with dequantized FP8 params.
                           For FP8, params will be dequantized to bfloat16 before RMSNorm.
    """
    activation_inputs_file = (
        "activation_inputs/layernorm_inputs_combined_20251216_170822.pt"
    )
    # Find activation inputs file
    if activation_inputs_file is None:
        save_dir = Path("activation_inputs")
        if not save_dir.exists():
            raise FileNotFoundError(
                "activation_inputs directory not found. "
                "Please run test_fp8_bf16_single_layer_comparison first to generate activation inputs."
            )

        # Find the most recent combined file
        combined_files = list(save_dir.glob("layernorm_inputs_combined_*.pt"))
        if not combined_files:
            raise FileNotFoundError(
                f"No combined activation inputs file found in {save_dir}. "
                f"Please run test_fp8_bf16_single_layer_comparison first."
            )

        activation_inputs_file = max(combined_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using most recent file: {activation_inputs_file}")

    # Load activation inputs
    logger.info("=" * 80)
    logger.info(f"Loading activation inputs from: {activation_inputs_file}")
    logger.info("=" * 80)

    data = load_layernorm_inputs_from_file(activation_inputs_file)
    bf16_inputs = data.get("bf16_inputs", {})
    fp8_inputs = data.get("fp8_inputs", {})
    bf16_output_grads = data.get("bf16_output_grads", {})
    fp8_output_grads = data.get("fp8_output_grads", {})
    layer_indices = data.get("layer_indices", [])

    logger.info(f"Loaded BF16 inputs: {list(bf16_inputs.keys())}")
    logger.info(f"Loaded FP8 inputs: {list(fp8_inputs.keys())}")
    logger.info(f"Loaded BF16 output grads: {list(bf16_output_grads.keys())}")
    logger.info(f"Loaded FP8 output grads: {list(fp8_output_grads.keys())}")
    if layer_indices:
        logger.info(f"Layer indices: {layer_indices}")

    # Create engines
    engine_bf16 = create_engine(MODEL_PATH_BF16, fp8_enabled=False, port=7777)
    engine_fp8 = create_engine(
        MODEL_PATH_FP8, fp8_enabled=True, fp8_param=True, port=7778
    )

    try:
        # Find matching layer paths
        common_keys = set(bf16_inputs.keys()) & set(fp8_inputs.keys())
        if not common_keys:
            logger.warning("No common layer paths found between BF16 and FP8 inputs")
            return

        # Filter by layer_path if specified
        if layer_path:
            # Convert layer_path to input key format
            if layer_path.endswith(".q_layernorm"):
                input_key = layer_path.replace(".q_layernorm", ".q_layernorm.input")
            elif layer_path.endswith(".k_layernorm"):
                input_key = layer_path.replace(".k_layernorm", ".k_layernorm.input")
            else:
                input_key = f"{layer_path}.input"

            if input_key not in common_keys:
                logger.warning(f"Layer path {layer_path} not found in loaded inputs")
                logger.info(f"Available keys: {sorted(common_keys)}")
                return

            common_keys = {input_key}

        # only test q_layernorm
        common_keys = {k for k in common_keys if k.endswith(".q_layernorm.input")}

        # Test each matching layer
        results = []
        for input_key in sorted(common_keys):
            # Extract layer path from input key
            if input_key.endswith(".q_layernorm.input"):
                test_layer_path = input_key.replace(".input", "")
                layernorm_type = "q_layernorm"
            elif input_key.endswith(".k_layernorm.input"):
                test_layer_path = input_key.replace(".input", "")
                layernorm_type = "k_layernorm"
            else:
                logger.warning(f"Unexpected input key format: {input_key}")
                continue

            logger.info("\n" + "=" * 80)
            logger.info(f"Testing {layernorm_type} for {test_layer_path}")
            logger.info("=" * 80)

            # Get input activations
            q_layernorm_input_bf16 = bf16_inputs[input_key]
            q_layernorm_input_fp8 = fp8_inputs[input_key]

            # Get output gradients (from downstream layers)
            output_grad_key = input_key.replace(".input", ".output_grad")
            output_grad_bf16 = bf16_output_grads.get(output_grad_key, None)
            output_grad_fp8 = fp8_output_grads.get(output_grad_key, None)

            logger.info(f"BF16 input shape: {q_layernorm_input_bf16.shape}")
            logger.info(f"FP8 input shape: {q_layernorm_input_fp8.shape}")
            if output_grad_bf16 is not None:
                logger.info(f"BF16 output grad shape: {output_grad_bf16.shape}")
                logger.info(f"BF16 output grad dtype: {output_grad_bf16.dtype}")
            if output_grad_fp8 is not None:
                logger.info(f"FP8 output grad shape: {output_grad_fp8.shape}")
                logger.info(f"FP8 output grad dtype: {output_grad_fp8.dtype}")
            if output_grad_bf16 is None or output_grad_fp8 is None:
                logger.warning(
                    f"Output gradient not found for {test_layer_path}, will use dummy loss"
                )

            q_layernorm_input_bf16 = q_layernorm_input_bf16.to(engine_bf16.device)
            q_layernorm_input_fp8 = q_layernorm_input_fp8.to(engine_fp8.device)
            if output_grad_bf16 is not None:
                output_grad_bf16 = output_grad_bf16.to(engine_bf16.device)
            if output_grad_fp8 is not None:
                output_grad_fp8 = output_grad_fp8.to(engine_fp8.device)

            # Compare RMSNorm
            result = compare_rmsnorm_bf16_fp8(
                engine_bf16,
                engine_fp8,
                q_layernorm_input_bf16,
                q_layernorm_input_fp8,
                test_layer_path,
                output_grad_bf16=output_grad_bf16,
                output_grad_fp8=output_grad_fp8,
                use_custom_rmsnorm=use_custom_rmsnorm,
                save_data=save_data,
            )
            results.append(result)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("RMSNorm Test Summary")
        logger.info("=" * 80)
        for result in results:
            if "shape_mismatch" in result and result["shape_mismatch"]:
                logger.warning(
                    f"{result['layer_path']}: Shape mismatch - "
                    f"BF16={result['bf16_shape']}, FP8={result['fp8_shape']}"
                )
            else:
                logger.info(
                    f"{result['layer_path']}: "
                    f"output_max_diff={result['output_max_diff']:.6f}, "
                    f"output_mean_diff={result['output_mean_diff']:.6f}, "
                    f"output_cos_sim={result['output_cos_sim']:.6f}"
                )

                # Gradient summary
                if "gradient_comparison" in result and result["gradient_comparison"]:
                    grad_comp = result["gradient_comparison"]
                    avg_grad_cos_sim = sum(
                        g["cos_sim"] for g in grad_comp.values()
                    ) / len(grad_comp)
                    max_grad_diff = max(g["max_diff"] for g in grad_comp.values())
                    logger.info(
                        f"  Gradients: "
                        f"avg_cos_sim={avg_grad_cos_sim:.6f}, "
                        f"max_diff={max_grad_diff:.6f}, "
                        f"n_gradients={len(grad_comp)}"
                    )
        logger.info("=" * 80)

    finally:
        engine_bf16.destroy()
        engine_fp8.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def print_tensor_stats(tensor, name):
    """Print mean, max, min statistics of a tensor."""
    if tensor is None:
        print(f"{name}: None")
        return
    tensor_flat = tensor.flatten()
    print(
        f"{name}: mean={tensor_flat.mean().item():.6f}, max={tensor_flat.max().item():.6f}, min={tensor_flat.min().item():.6f}, shape={tensor.shape}, dtype={tensor.dtype}"
    )


class Qwen3RMSNormFunction(Function):
    """Custom autograd Function for Qwen3RMSNorm backward."""

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):
        """
        Forward pass for RMSNorm.

        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            weight: Weight parameter of shape [hidden_size]
            variance_epsilon: Epsilon value for numerical stability

        Returns:
            Normalized and weighted output tensor
        """
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)

        # Compute variance: mean(x^2) along last dimension
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)

        # Compute normalized: x / sqrt(variance + eps)
        inv_std = torch.rsqrt(variance + variance_epsilon)
        normalized = hidden_states_fp32 * inv_std

        # Apply weight and convert back to input dtype
        output = (weight * normalized).to(input_dtype)

        # Save tensors for backward
        ctx.save_for_backward(hidden_states_fp32, weight, inv_std, normalized)
        ctx.variance_epsilon = variance_epsilon
        ctx.input_dtype = input_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for RMSNorm.

        Args:
            grad_output: Gradient w.r.t. output, shape [..., hidden_size]

        Returns:
            grad_input: Gradient w.r.t. input
            grad_weight: Gradient w.r.t. weight
            grad_eps: None (variance_epsilon is not a tensor)
        """
        hidden_states, weight, inv_std, normalized = ctx.saved_tensors
        # variance_epsilon = ctx.variance_epsilon
        input_dtype = ctx.input_dtype

        # print_tensor_stats(grad_output, "[backward] grad_output (input)")
        # print_tensor_stats(hidden_states, "[backward] hidden_states")
        # print_tensor_stats(weight, "[backward] weight")
        # print_tensor_stats(inv_std, "[backward] inv_std")
        # print_tensor_stats(normalized, "[backward] normalized")

        # Convert grad_output to float32 for computation
        grad_output_fp32 = grad_output.to(torch.float32)
        # print_tensor_stats(grad_output_fp32, "[backward] grad_output_fp32 (after to float32)")

        # Gradient w.r.t. weight: sum over all dimensions except last
        grad_weight = (grad_output_fp32 * normalized).sum(
            dim=tuple(range(grad_output_fp32.dim() - 1))
        )
        # print_tensor_stats(grad_weight, "[backward] grad_weight (after sum)")

        # Gradient w.r.t. normalized: weight * grad_output
        grad_normalized = grad_output_fp32 * weight.unsqueeze(0)
        # print_tensor_stats(grad_normalized, "[backward] grad_normalized (after weight * grad_output)")

        # Gradient w.r.t. variance
        # d(normalized)/d(variance) = -0.5 * x * (variance + eps)^(-3/2)
        # = -0.5 * x * inv_std^3
        # We need to sum over the last dimension for grad_variance
        inv_std_pow3 = inv_std.pow(3)
        # print_tensor_stats(inv_std_pow3, "[backward] inv_std_pow3")
        grad_variance = (grad_normalized * hidden_states * -0.5 * inv_std_pow3).sum(
            -1, keepdim=True
        )
        # print_tensor_stats(grad_variance, "[backward] grad_variance (after sum)")

        # Gradient w.r.t. hidden_states
        # d(variance)/d(hidden_states) = 2 * hidden_states / hidden_size
        hidden_size = hidden_states.shape[-1]
        grad_input_from_variance = grad_variance * 2.0 * hidden_states / hidden_size
        # print_tensor_stats(grad_input_from_variance, "[backward] grad_input_from_variance")

        # d(normalized)/d(hidden_states) = inv_std (direct contribution)
        grad_input_from_normalized = grad_normalized * inv_std
        # print_tensor_stats(grad_input_from_normalized, "[backward] grad_input_from_normalized")

        # Total gradient w.r.t. input
        grad_input = grad_input_from_normalized + grad_input_from_variance
        # print_tensor_stats(grad_input, "[backward] grad_input (before dtype conversion)")

        # Convert back to input dtype
        grad_input = grad_input.to(input_dtype)
        grad_weight = grad_weight.to(input_dtype)
        # print_tensor_stats(grad_input, "[backward] grad_input (final, after dtype conversion)")
        # print_tensor_stats(grad_weight, "[backward] grad_weight (final, after dtype conversion)")

        return grad_input, grad_weight, None


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Qwen3RMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
