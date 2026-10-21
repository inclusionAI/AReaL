# SPDX-License-Identifier: Apache-2.0

"""NoPE MLA with GLM-5.3's frozen, key-pooled sparse indexer.

The projection/absorption order follows radixark/miles ee36bb2's GLM-5.3
implementation. Tensor-parallel MLA projections are supplied by Megatron.
"""

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.multi_latent_attention import MLASelfAttention


class Glm5NextDSAAttention(MLASelfAttention):
    def __init__(self, config, submodules, layer_number, glm_config, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        if config.context_parallel_size != 1 or config.qk_pos_emb_head_dim != 0:
            raise ValueError("GLM-5.3 DSA requires NoPE and CP=1")
        t = glm_config.text
        self.index_heads = t["index_n_heads"]
        self.index_dim = t["index_head_dim"]
        self.index_topk = t["index_topk"]
        self.index_kpool = t["index_kpool"]
        self.softmax_scale = config.qk_head_dim**-0.5
        dtype = config.params_dtype
        self.wq_b = torch.nn.Linear(
            config.q_lora_rank,
            self.index_heads * self.index_dim,
            bias=False,
            dtype=dtype,
        )
        self.wk = torch.nn.Linear(
            config.hidden_size, self.index_dim, bias=False, dtype=dtype
        )
        self.k_norm = torch.nn.LayerNorm(self.index_dim, eps=1e-6, dtype=torch.float32)
        self.weights_proj = torch.nn.Linear(
            config.hidden_size, self.index_heads, bias=False, dtype=dtype
        )
        self.index_kpool_compress_gate = torch.nn.Parameter(
            torch.zeros(self.index_dim, config.hidden_size, dtype=dtype)
        )
        self.index_kpool_compress_ape = mark_keep_in_fp32(
            torch.nn.Parameter(
                torch.zeros(self.index_kpool, self.index_dim, dtype=torch.float32)
            )
        )
        # This is the frozen indexer path used by the reference RL/SFT recipe.
        for module in (self.wq_b, self.wk, self.k_norm, self.weights_proj):
            module.requires_grad_(False)
        for param in self.k_norm.parameters():
            mark_keep_in_fp32(param)
        self.index_kpool_compress_gate.requires_grad_(False)
        self.index_kpool_compress_ape.requires_grad_(False)

    def _gather(self, tensor, *, output_grad=True):
        if not self.config.sequence_parallel:
            return tensor
        return tensor_parallel.gather_from_sequence_parallel_region(
            tensor,
            tensor_parallel_output_grad=output_grad,
            group=self.pg_collection.tp,
        )

    def forward(
        self, hidden_states, attention_mask=None, packed_seq_params=None, **kwargs
    ):
        from areal.experimental.ops.glm5_next.indexer import kpool_select_topk
        from areal.experimental.ops.glm5_next.sparse_attention import sparse_attention

        if packed_seq_params is None or hidden_states.shape[1] != 1:
            raise ValueError("GLM-5.3 DSA requires a packed THD batch")
        if (
            kwargs.get("inference_context") is not None
            or kwargs.get("inference_params") is not None
        ):
            raise ValueError(
                "GLM-5.3 actor attention does not support cached inference"
            )
        if kwargs.get("attention_bias") is not None:
            raise ValueError("GLM-5.3 DSA does not support an attention bias")

        compressed_q, _ = self.linear_q_down_proj(hidden_states)
        compressed_q = compressed_q.squeeze(1)
        query, _ = self.linear_q_up_proj(compressed_q)
        query = query.unflatten(
            -1, (self.num_attention_heads_per_partition, self.config.qk_head_dim)
        )
        key_weight, value_weight = self.linear_kv_up_proj.weight.unflatten(
            0, (-1, self.config.qk_head_dim + self.config.v_head_dim)
        ).split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)
        query = torch.einsum("thd,hdm->thm", query, key_weight)
        compressed_kv, _ = self.linear_kv_down_proj(hidden_states)
        key = compressed_kv.squeeze(1)
        key = F.rms_norm(
            key.float(),
            (key.shape[-1],),
            self.linear_kv_up_proj.layer_norm_weight.float(),
            self.config.layernorm_epsilon,
        ).to(key.dtype)
        if self.config.sequence_parallel:
            # Reduce-scatter the head-local gradients before the replicated norm.
            # Megatron then sums the norm's sequence shards through its SP tag.
            key = self._gather(key)
        else:
            key = tensor_parallel.copy_to_tensor_model_parallel_region(
                key,
                group=self.pg_collection.tp,
            )

        with torch.no_grad():
            x = hidden_states.squeeze(1).detach()
            q_lora = F.rms_norm(
                compressed_q.float(),
                (compressed_q.shape[-1],),
                self.linear_q_up_proj.layer_norm_weight.float(),
                self.config.layernorm_epsilon,
            ).to(compressed_q.dtype)
            index_q = self._gather(
                self.wq_b(q_lora).unflatten(-1, (self.index_heads, self.index_dim)),
                output_grad=False,
            )
            index_k = self._gather(
                self.k_norm(self.wk(x).float()).to(x.dtype), output_grad=False
            )
            gates = self._gather(
                F.linear(x, self.index_kpool_compress_gate), output_grad=False
            )
            head_weights = (
                F.linear(x.float(), self.weights_proj.weight.float())
                * (self.index_heads * self.index_dim) ** -0.5
            )
            head_weights = self._gather(head_weights, output_grad=False)
            indices = kpool_select_topk(
                index_q,
                index_k,
                head_weights,
                gates,
                self.index_kpool_compress_ape,
                packed_seq_params.cu_seqlens_kv,
                self.index_topk,
                self.index_kpool,
            )

        # The sparse MLA kernels consume a 64-wide RoPE tail. GLM's tail is zero.
        query = F.pad(query, (0, 64)).contiguous()
        key = F.pad(key, (0, 64)).contiguous()
        output = sparse_attention(
            query.unsqueeze(0),
            key.unsqueeze(0).unsqueeze(2),
            indices.unsqueeze(0),
            self.softmax_scale,
            d_v=self.config.kv_lora_rank,
        ).squeeze(0)
        output = torch.einsum("thm,hdm->thd", output, value_weight)
        return self.linear_proj(output.reshape(output.shape[0], 1, -1))
