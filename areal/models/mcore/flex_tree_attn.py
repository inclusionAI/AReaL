import os

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class MegatronFlexTreeAttention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str = "p2p",
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config)
        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = "sbhd"
        self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        self.layer_number = layer_number

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        packed_seq_kwargs = (
            {
                key: getattr(packed_seq_params, key)
                for key in self.kept_packed_seq_params
            }
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)
        if qkv_format != "sbhd":
            raise NotImplementedError(
                f"TEDotProductAttention only supports 'sbhd' qkv_format, got {qkv_format}"
            )

        # [b, h, s, d]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        def mask_mod(batch_idx, head_idx, q_idx, k_idx):
            is_causal = k_idx <= q_idx
            return is_causal & attention_mask[k_idx, q_idx]

        # FIXME: avoid creating block mask here
        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q.shape[2],
            KV_LEN=k.shape[2],
        )

        out = flex_attention(
            q,
            k,
            v,
            scale=self.softmax_scale,
            block_mask=block_mask,
            enable_gqa=q.shape[1] != k.shape[1],
        )

        return out.permute(2, 0, 1, 3).view(query.shape[0], query.shape[1], -1)
