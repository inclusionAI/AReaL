# Adapted from verl

from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from areal.utils import logging
from areal.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)

logger = logging.getLogger("Ulysses Monkey Patch")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
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
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    get_ulysses_sequence_parallel_rank()

    if ulysses_sp_size > 1:
        assert (
            position_ids is not None
        ), "position_ids is required for Ulysses sequence parallelism"

        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (1, total_seqlen/n, n_head, head_dim) -> (1, total_seqlen, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # (1, total_seqlen/n) -> (1, total_seqlen)
        position_ids_list = [
            torch.empty_like(position_ids) for _ in range(ulysses_sp_size)
        ]
        torch.distributed.all_gather(
            position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group()
        )
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (1, total_seqlen, n_head/n, head_dim)
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        *args,
        position_ids=position_ids,
        **kwargs,
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen, n_head/n, head_dim) -> (1, total_seqlen/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


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

    if ulysses_sp_size > 1:
        from transformers import modeling_flash_attention_utils
        from transformers.integrations import flash_attention

        if hasattr(modeling_flash_attention_utils, "_flash_attention_forward"):
            modeling_flash_attention_utils._flash_attention_forward = (
                _ulysses_flash_attention_forward
            )
            logger.info(
                "Patched modeling_flash_attention_utils._flash_attention_forward"
            )

        if hasattr(flash_attention, "_flash_attention_forward"):
            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            logger.info("Patched flash_attention._flash_attention_forward")
