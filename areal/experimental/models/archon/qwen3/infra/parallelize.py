from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from areal.utils import logging

if TYPE_CHECKING:
    from areal.experimental.models.archon.activation_checkpoint import (
        ActivationCheckpointConfig,
    )

logger = logging.getLogger(__name__)


def parallelize_qwen3(
    model: nn.Module,
    tp_mesh: DeviceMesh | None = None,
    dp_mesh: DeviceMesh | None = None,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    loss_parallel: bool = True,
    cpu_offload: bool = False,
    reshard_after_forward: bool = True,
    ac_config: ActivationCheckpointConfig | None = None,
    enable_compile: bool = True,
) -> nn.Module:
    """Apply parallelization to Qwen3 model.

    This is the main entry point for parallelizing a Qwen3 model.
    It applies TP (if tp_mesh provided), AC (if ac_config provided),
    torch.compile (if enable_compile), and FSDP (if dp_mesh provided).

    Order of operations:
    1. Apply TP (Tensor Parallelism)
    2. Apply AC (Activation Checkpointing) - must be after TP
    3. Apply torch.compile - must be after AC, before FSDP
    4. Apply FSDP (Fully Sharded Data Parallelism)

    Args:
        model: The Qwen3 model to parallelize.
        tp_mesh: Device mesh for tensor parallelism. If None, TP is not applied.
        dp_mesh: Device mesh for data parallelism (FSDP). If None, FSDP is not applied.
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        loss_parallel: Whether to keep output sharded for loss parallelism.
        cpu_offload: Whether to enable CPU offloading for FSDP.
        reshard_after_forward: Whether to reshard after forward pass.
        ac_config: Activation checkpointing configuration. If None, AC is not applied.
        enable_compile: Whether to apply torch.compile to TransformerBlocks.

    Returns:
        The parallelized model.

    Note:
        Only packed sequences are supported (cu_seqlens must be provided).
        SDPA is used for all attention computations.
    """
    # Step 1: Apply tensor parallelism
    if tp_mesh is not None:
        apply_tp(model, tp_mesh, loss_parallel=loss_parallel)

    # Step 2: Apply activation checkpointing (must be after TP)
    if ac_config is not None and ac_config.mode != "none":
        from areal.experimental.models.archon.activation_checkpoint import (
            apply_activation_checkpointing,
        )

        apply_activation_checkpointing(model, ac_config)

    # Step 3: Apply torch.compile (must be after AC, before FSDP)
    if enable_compile:
        from areal.experimental.models.archon.compile import apply_compile

        apply_compile(model)

    # Step 4: Apply FSDP
    if dp_mesh is not None:
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cpu_offload=cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

    # Step 5: Enable weight tying after applying parallelisms
    if getattr(model.model_args, "enable_weight_tying", False):
        if model.output is not None and model.tok_embeddings is not None:
            model.output.weight = model.tok_embeddings.weight

    return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = True,
) -> None:
    """Apply tensor parallelism to Qwen3 model.

    This applies TP with Sequence Parallelism to the model:
    - Embedding: RowwiseParallel (output sharded on sequence dim)
    - Attention: ColwiseParallel for q/k/v, RowwiseParallel for output
    - FFN: ColwiseParallel for w1/w3, RowwiseParallel for w2
    - Q/K norm: SequenceParallel
    - Final norm: SequenceParallel
    - Output (lm_head): ColwiseParallel
    - Score (critic): Replicated (small layer, no need to shard)

    Args:
        model: The model to apply TP to.
        tp_mesh: Device mesh for tensor parallelism.
        loss_parallel: Whether to keep output sharded for loss parallelism.
    """
    # Build root-level TP plan
    root_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
    }

    # Add output layer based on model type (actor vs critic)
    if model.output is not None:
        # Actor: lm_head with loss parallel
        # use_local_output=True: return local tensor (not DTensor) for vocab_parallel loss
        root_plan["output"] = ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=True,
        )

    if model.score is not None:
        # Critic: score layer - input is SP (Shard(1)), output should be Replicate
        # Using PrepareModuleInput to handle the input redistribution
        root_plan["score"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )

    # Parallelize root module
    parallelize_module(model, tp_mesh, root_plan)

    # Apply TP to each transformer block
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            # Prepare attention input:
            # - x: Shard(1) -> Replicate() (unshard sequence dim)
            # - rope_cache: Replicate() -> Replicate() (must be DTensor for xq*cos)
            # - positions: Replicate() -> Replicate() (must be DTensor for rope_cache indexing)
            # - cu_seqlens: None (pass-through, can be None or tensor)
            # - max_seqlen: None (pass-through, int not tensor)
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), Replicate(), None, None),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    Replicate(),
                    None,
                    None,
                ),
            ),
            # Q/K/V projections: column-parallel
            # wq/wk stay DTensor (needed for q_norm/k_norm which are SequenceParallel)
            # wv outputs local tensor for attention compatibility
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=True),
            # Q/K norms: sequence parallel on head dim
            # Output is DTensor, converted to local in model.py via maybe_to_local()
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            # Output projection: row-parallel, output sharded on sequence
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            # Prepare FFN input: unshard sequence dim
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            # FFN projections
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to the model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    cpu_offload: bool = False,
    reshard_after_forward: bool = True,
) -> None:
    """Apply FSDP2 to Qwen3 model.

    This wraps each component with FSDP for memory-efficient training:
    - Token embeddings (separately wrapped)
    - Each TransformerBlock (separately wrapped)
    - Final norm + output/score (wrapped together)
    - Root model (for any remaining params)

    Args:
        model: The model to apply FSDP to.
        dp_mesh: Device mesh for data parallelism.
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        cpu_offload: Whether to enable CPU offloading.
        reshard_after_forward: Whether to reshard parameters after forward.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Wrap token embeddings
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Wrap each transformer block
    for transformer_block in model.layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Wrap final layers together (optimization: don't reshard for last layers)
    # Handle both actor (output) and critic (score)
    final_layers = [model.norm] if model.norm is not None else []
    if model.output is not None:
        final_layers.append(model.output)
    if hasattr(model, "score") and model.score is not None:
        final_layers.append(model.score)

    if final_layers:
        fully_shard(
            final_layers,
            **fsdp_config,
            reshard_after_forward=False,  # Don't reshard - would be prefetched anyway
        )

    # Wrap root model
    fully_shard(model, **fsdp_config)

    logger.info("Applied FSDP to the model")
    if cpu_offload:
        logger.info("Applied CPU Offloading to the model")


__all__ = [
    "parallelize_qwen3",
    "apply_tp",
    "apply_fsdp",
]
