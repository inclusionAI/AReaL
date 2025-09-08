# Adapted from verl

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from areal.utils import logging
from areal.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)

logger = logging.getLogger("Ulysses Monkey Patch")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep).
    The hidden states go from (batch, seqlen, num_key_value_heads, head_dim)
    to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    **kwargs,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (1, total_seqlen / sp_size, num_heads, head_dim)
        # -> (1, total_seqlen, num_heads / sp_size, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

    # (1, total_seqlen, num_heads / sp_size, head_dim)
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        *args,
        **kwargs,
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen, num_heads / sp_size, head_dim)
        # -> (1, total_seqlen / sp_size, num_heads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


# NOTE: For vision models, inputs_embeds will be sliced instead of input_ids.
def patch_vlm_for_ulysses_input_slicing(model_class: type):
    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(
                    inputs_embeds, dim=1, padding=False
                )
                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    logger.info(f"Patched {model_class.__name__}.forward")


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
):
    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert (
        num_attention_heads % ulysses_sp_size == 0
    ), f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    assert (
        num_key_value_heads % ulysses_sp_size == 0
        or ulysses_sp_size % num_key_value_heads == 0
    ), (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "
        f"{ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
        f"kv heads are repeated to ensure correctness."
    )

    vl_model_mappings = {
        "qwen2_5_vl": {
            "module": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
            "attn_class": "Qwen2_5_VLAttention",
            "model_class": "Qwen2_5_VLTextModel",
            "patch_module": "areal.models.transformers.qwen2_vl",
            "patch_attn_func": "ulysses_flash_attn_forward",
        },
        "qwen2_vl": {
            "module": "transformers.models.qwen2_vl.modeling_qwen2_vl",
            "attn_class": "Qwen2VLAttention",
            "model_class": "Qwen2VLTextModel",
            "patch_module": "areal.models.transformers.qwen2_vl",
            "patch_attn_func": "ulysses_flash_attn_forward",
        },
    }

    if ulysses_sp_size <= 1:
        return

    if model.config.model_type in vl_model_mappings:
        # NOTE: The following code segment will patch only TextModel's attention and forward methods
        mapping = vl_model_mappings[model.config.model_type]
        module_name = mapping["module"]
        attn_class_name = mapping["attn_class"]
        model_class_name = mapping["model_class"]
        patch_module_name = mapping["patch_module"]
        patch_attn_func_name = mapping["patch_attn_func"]

        module = __import__(
            module_name,
            fromlist=[attn_class_name, model_class_name],
        )
        getattr(module, attn_class_name)
        attn_class = getattr(module, attn_class_name)
        model_class = getattr(module, model_class_name)

        patch_module = __import__(
            patch_module_name,
            fromlist=[patch_attn_func_name],
        )
        patch_attn_func = getattr(patch_module, patch_attn_func_name)

        attn_class.forward = patch_attn_func
        logger.info(
            f"Patched {module_name}.{attn_class_name}.forward using {patch_module_name}.{patch_attn_func_name} and {model_class_name}.forward"
        )

        patch_vlm_for_ulysses_input_slicing(model_class)
        logger.info(f"Patched {model_class_name}.forward")
    else:
        from transformers.integrations import flash_attention

        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        logger.info(
            "Patched transformers.integrations.flash_attention._flash_attention_forward"
        )
