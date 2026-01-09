import os
from contextlib import contextmanager

import torch
from mbridge.core import LLMBridge
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from areal.utils import logging

logger = logging.getLogger(__name__)

_TORCH_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}
_FLEX_DYNAMIC = not (os.environ.get("AREAL_DISABLE_FLEX_ATTENTION_DYNAMIC", "0") == "1")
logger.info(
    "Compiled torch flex attention. Options: %s, dynamic: %s",
    str(_TORCH_COMPILE_OPTIONS),
    str(_FLEX_DYNAMIC),
)
_flex_attention = torch.compile(
    flex_attention,
    dynamic=_FLEX_DYNAMIC,
    options=_TORCH_COMPILE_OPTIONS,
)
USE_BLOCK_MASK = not (
    os.environ.get("AREAL_DISABLE_FLEX_ATTENTION_BLOCK_MASK", "0") == "1"
)
BLOCK_SIZE = int(os.environ.get("AREAL_FLEX_ATTENTION_BLOCK_SIZE", "128"))
logger.info(
    "Using block mask in flex attention: %s, block size: %d",
    str(USE_BLOCK_MASK),
    BLOCK_SIZE,
)


# Global cache for block mask (size 1)
# Stores: {"key": (mask_data_ptr, q_len, device), "block_mask": block_mask}
_block_mask_cache: dict = {"key": None, "block_mask": None}

def make_block_mask_or_score_mod(
    attention_mask: torch.Tensor,
    q_len: int,
    device: torch.device,
):
    if USE_BLOCK_MASK:
        # Check cache for existing block mask
        cache_key = (attention_mask.data_ptr(), q_len, device)
        if _block_mask_cache["key"] == cache_key:
            block_mask = _block_mask_cache["block_mask"]
        else:
            def arbitrary_mask(
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                k_idx: torch.Tensor,
            ):
                return attention_mask[q_idx, k_idx]

            block_mask = create_block_mask(
                arbitrary_mask,
                B=1,  # Broadcast across batch
                H=1,  # Broadcast across heads
                Q_LEN=q_len,
                KV_LEN=q_len,
                BLOCK_SIZE=BLOCK_SIZE,
                device=device,
                _compile=False,
            )
            # Update cache
            _block_mask_cache["key"] = cache_key
            _block_mask_cache["block_mask"] = block_mask

        score_mod = None
    else:
        def arbitrary_score_mod(score, b, h, q_idx, k_idx):
            mask_value = attention_mask[q_idx, k_idx]
            score = score.masked_fill(~mask_value, float("-inf"))
            return score

        block_mask = None
        score_mod = arbitrary_score_mod
    return block_mask, score_mod

class PytorchFlexAttention(torch.nn.Module):
    """Pytorch flex attention implementation that supports arbitrary attention mask type."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale
        logger.info("Using PytorchFlexAttention for tree training attention")

        # PytorchFlexAttention does not support context parallel
        if config.context_parallel_size != 1:
            raise ValueError(
                "PytorchFlexAttention does not support context parallelism."
            )

        if attention_type != "self":
            raise ValueError("PytorchFlexAttention only supports self-attention.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # query: [S, B, H, D] in which B should be 1 in current tree training implementation
        # key: [S, B, H, D]
        # value: [S, B, H, D]
        # attention_mask: [1, 1, S, S]
        # attention_mask_type: arbitrary

        if attention_bias is not None:
            raise NotImplementedError(
                "PytorchFlexAttention does not support attention_bias yet."
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "PytorchFlexAttention does not support packed sequences yet."
            )

        # query, key, value shape: [S, B, H, D] -> [B, H, S, D]
        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        enable_gqa = query.shape[1] != key.shape[1]

        q_len = attention_mask.shape[0]

        block_mask, score_mod = make_block_mask_or_score_mod(
            attention_mask,
            q_len,
            query.device,
        )

        output = _flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            score_mod=score_mod,
            scale=self.softmax_scale,
            enable_gqa=enable_gqa,
        )

        # output shape: [B, H, S, D] -> [S, B, H, D] -> [S, B, H*D]
        output = (
            output.permute(2, 0, 1, 3)
            .contiguous()
            .view(output.shape[2], output.shape[0], -1)
        )
        return output


def _tree_attn_fwd_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    *args,
    **kwargs
):
    assert "full_attention_mask" in kwargs, "full_attention_mask is required for tree attention"
    attention_mask = kwargs["full_attention_mask"]
    # [B, S, H, D] -> [B, H, S, D]
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    enable_gqa = query.shape[1] != key.shape[1]

    q_len = attention_mask.shape[0]
    block_mask, score_mod = make_block_mask_or_score_mod(
        attention_mask,
        q_len,
        query.device,
    )
    
    output = _flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        score_mod=score_mod,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
    )
    # [B, H, S, D] -> [B, S, H, D]
    output = output.permute(0, 2, 1, 3).contiguous()
    return output


@contextmanager
def patch_bridge_for_tree_training(enable: bool = True):
    """Context manager to patch LLMBridge for tree training with arbitrary attention mask.

    Parameters
    ----------
    enable : bool, default=True
        If True, apply the patch. If False, the context manager is a no-op.

    Yields
    ------
    None

    Examples
    --------
    >>> with patch_bridge_for_tree_training(enable=True):
    ...     # LLMBridge is patched here
    ...     model = create_model()
    ... # Patch is reverted after exiting the context
    """
    if not enable:
        yield
        return

    # Store original method
    original_layer_spec_getter = LLMBridge._get_transformer_layer_spec

    def _patched_getter(self, vp_stage: int | None = None):
        spec: TransformerBlockSubmodules = original_layer_spec_getter(self, vp_stage)
        for layer_spec in spec.layer_specs:
            if layer_spec.module is not TransformerLayer:
                logger.info(f"Skipping patch module: {layer_spec.module}")
                continue
            submodules: TransformerLayerSubmodules = layer_spec.submodules
            self_attn_spec = submodules.self_attention
            if self_attn_spec.module is not SelfAttention:
                logger.info(f"Skipping patch module: {self_attn_spec.module}")
                continue
            self_attn_spec.params["attn_mask_type"] = AttnMaskType.arbitrary
            self_attn_spec.submodules.core_attention = PytorchFlexAttention
        return spec

    # Apply patch
    LLMBridge._get_transformer_layer_spec = _patched_getter
    try:
        yield
    finally:
        # Revert patch
        LLMBridge._get_transformer_layer_spec = original_layer_spec_getter


