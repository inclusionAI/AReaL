"""Model manipulation utilities for FP8/BF16 comparison tests.

This module contains functions for extracting layers, reducing models,
and collecting activations/gradients using hooks.
"""

import functools
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from areal.engine.megatron_engine import MegatronEngine
from areal.tests.fp8.engine_utils import (
    extract_gemm_kernels,
    print_gemm_profile,
)
from areal.utils import logging
from areal.utils.data import unpad_logits
from areal.utils.functional import gather_logprobs_entropy
from areal.utils.mcore.packed_context_parallel import packed_context_parallel_forward
from areal.utils.megatron import all_gather_param, get_named_parameters

logger = logging.getLogger("FP8 BF16 Model Utils")


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

    # Prepare input
    mb_list = engine.prepare_mb_list(input_)
    mb_list = mb_list.to(engine.device)

    # SFT loss function based on compute_packed_sft_loss from lm_engine.py
    def sft_loss_fn(logprobs, entropy, input_):
        """SFT loss function based on compute_packed_sft_loss."""
        del entropy  # SFT does not use entropy

        # Get cu_seqlens and loss_mask from input
        loss_mask = input_["loss_mask"].bool()

        # Shift loss_mask to align with next-token prediction
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply loss_mask to logprobs
        logprobs = torch.where(loss_mask, logprobs, 0)

        # Compute loss: negative log likelihood averaged over valid tokens
        device = logprobs.device
        num_valid = loss_mask.count_nonzero()
        if num_valid == 0:
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


def forward_backward_model_with_hooks(
    engine: MegatronEngine,
    input_: dict[str, Any],
    layer_indices: list[int] | int = 0,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
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
        tuple: (logits, activations_dict, gradients_dict, output_gradients_dict)
        - logits: Output logits from the model
        - activations_dict: Dictionary mapping op names to their output activations
        - gradients_dict: Dictionary mapping parameter names to their gradients
        - output_gradients_dict: Dictionary mapping op names to their output gradients
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
