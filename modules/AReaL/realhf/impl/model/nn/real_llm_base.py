# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import contextlib
import dataclasses
import functools
import itertools
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers

import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.parallelism.tensor_parallel.mappings as tensor_parallel
from realhf.api.core import model_api
from realhf.impl.model.modules import (
    CausalSelfAttentionLayer,
    GemmaRMSNorm,
    LayerNormMLP,
    LayerNormMoELayer,
    LlamaLayerNormMLP,
    LlamaRMSNorm,
    OffsetParallelPositionalEmbedding,
    OffsetPositionalEmbedding,
)
from realhf.impl.model.parallelism.tensor_parallel.modules import (
    ColumnParallelLinear,
    ParallelEmbedding,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    parallel_lm_logits,
    scatter_to_tensor_model_parallel_region,
)
from realhf.impl.model.utils.functional import compute_varlen_position_indices

logger = logging.getLogger("ReaLModelBase")


@dataclasses.dataclass
class PipeTransferData:
    """Data structure for transferring data between stages.

    Each pipeline stage has exactly one PipeTransferData as the input and the output,
    no matter how many layers are in this stage.

    Attributes:
        pp_input: The input to the current stage. Usually hidden states
            with shape [bs, seq_len, hidden_dim].
        pp_output: The output of the current stage, also the input to the next stage.
            Usually hidden states with shape [bs, seq_len, hidden_dim].
        cu_seqlens: The cumulative sequence lengths of packed input_ids.
            Used by flash_attn_varlen_func. Will not be used during generation.
            It's configuration-like data that must be transfered from the first stage
            to the last. Shape [bs + 1].
        max_seqlen: The maximum sequence length of packed input_ids.
            Used by flash_attn_varlen_func. Will not be used during generation.
            It's configuration-like data that must be transfered from the first stage
            to the last.
        store_kv_cache: Whether to store the key and value cache for generation.
        attention_mask: The attention mask of the input, the same as huggingface transformers.
            Used by torch_attn_func to examine the outputs of PyTorch attention and flash
            attention are the same. Only for debugging. Shape [bs, seq_len].
    """

    pp_input: torch.Tensor = None
    pp_output: torch.Tensor = None

    # The followings are "configuration"-like data that should be passed across all stages.
    cu_seqlens: torch.Tensor = None
    max_seqlen: int = None
    store_kv_cache: bool = False


@dataclasses.dataclass
class PipeCacheData:
    """Data structure for caching data locally that will not be trasferred.

    Each layer has exactly one PipeCacheData as the input.
    If a pipeline stage has multiple layers, a list of PipeCacheData should be passed
    as the input. The cached tensors will be changed in-place.

    Attributes:
        input_ids: The input token ids. Used only at the first stage.
            Can be packed with shape [total_seq_len] or unpacked with shape [bs, seq].
        prompt_mask: Prompt mask used
        position_ids: Input position IDs. Can be resolved automatically in most cases.
            Used only at the first stage. The same shape as input_ids.
            If None, will be resolved automatically.
        k_cache: Key cache used for generation, shape [bs, max_seq, n_kv_heads, head_dim].
            Note that this is the cache for a specific layer, not for all layers.
        v_cache: Value cache used for generation, shape [bs, max_seq, n_kv_heads, head_dim].
            Note that this is the cache for a specific layer, not for all layers.
        cache_seqlens: The sequence lengths of the cached tokens. Used for generation. Shape [bs].
    """

    # Only cached in the first stage.
    packed_input_ids: torch.Tensor = None
    packed_position_ids: torch.Tensor = None
    # Cached in each transformer layer.
    k_cache: torch.Tensor = None
    v_cache: torch.Tensor = None
    cache_seqlens: torch.Tensor = None


class ReaLModelBlock(nn.Module):

    def __init__(
        self,
        config: model_api.ReaLModelConfig,
        layer_index: int,
        output_layernorm: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.config = config
        self.layer_index = layer_index
        self.attn = CausalSelfAttentionLayer(
            hidden_dim=config.hidden_dim,
            n_kv_heads=config.n_kv_heads,
            n_q_heads=config.n_q_heads,
            head_dim=config.head_dim,
            resid_pdrop=config.resid_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_index=layer_index,
            layer_norm_epsilon=config.layer_norm_epsilon,
            scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            scale_attn_weights=config.scale_attn_weights,
            layer_norm_type=config.layer_norm_type,
            use_attention_bias=config.use_attention_bias,
            use_attn_proj_bias=config.use_attn_proj_bias,
            do_layernorm_before=config.do_layernorm_before,
            qk_layernorm=config.qk_layernorm,
            apply_rotary=config.apply_rotary,
            rotary_base=config.rotary_base,
            rotary_interleaved=config.rotary_interleaved,
            rotary_scaling=config.rotary_scaling,
            rotary_scaling_type=config.rotary_scaling_type,
            rotary_special_impl=config.rotary_special_impl,
            dtype=dtype,
            device=device,
        )
        if config.mlp_type is None:
            self.mlp = LayerNormMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                resid_pdrop=config.resid_pdrop,
                use_bias=config.use_mlp_bias,
                do_layernorm_before=config.do_layernorm_before,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                device=device,
            )
        elif config.mlp_type == "llama":
            self.mlp = LlamaLayerNormMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                layer_norm_type=config.layer_norm_type,
                use_bias=config.use_mlp_bias,
                dtype=dtype,
                device=device,
            )
        elif config.mlp_type == "moe":
            self.mlp = LayerNormMoELayer(
                config=config,
                layer_idx=layer_index,
                dtype=dtype,
                device=device,
            )
        else:
            raise NotImplementedError(f"Unknown MLP type: {config.mlp_type}")

        self.output_layernorm = output_layernorm
        if output_layernorm:
            if config.layer_norm_type is None:
                layer_norm_fn = nn.LayerNorm
            elif config.layer_norm_type == "rms":
                layer_norm_fn = LlamaRMSNorm
            elif config.layer_norm_type == "gemma":
                layer_norm_fn = GemmaRMSNorm
            self.ln_f = layer_norm_fn(
                config.hidden_dim,
                eps=config.layer_norm_epsilon,
                dtype=dtype,
                device=device,
            )

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        pp_input = x.pp_input
        cu_seqlens = x.cu_seqlens
        k_cache = y.k_cache
        v_cache = y.v_cache
        cache_seqlens = y.cache_seqlens
        max_seqlen = x.max_seqlen
        if constants.gradient_checkpointing():
            pp_output, k, v = torch.utils.checkpoint.checkpoint(
                self._forward,
                pp_input,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                max_seqlen,
                use_reentrant=True,
            )
        else:
            pp_output, k, v = self._forward(
                pp_input,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                max_seqlen,
            )

        x.pp_output = pp_output
        if x.store_kv_cache:
            if y.k_cache is None:
                y.k_cache = k.detach()
            if y.v_cache is None:
                y.v_cache = v.detach()
            if y.cache_seqlens is None and x.cu_seqlens is not None:
                y.cache_seqlens = x.cu_seqlens[1:] - x.cu_seqlens[:-1]
        return x

    def _forward(
        self,
        pp_input: torch.Tensor,
        cu_seqlens: torch.Tensor,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        max_seqlen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = pp_input
        attn_out, k, v = self.attn(
            hidden_states=h,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
        )
        h = h + attn_out

        # For opt-350m
        if not self.config.do_layernorm_before:
            h = self.attn.c_attn.ln(h)

        h = self.mlp(h) + h

        # For opt-350m
        if not self.config.do_layernorm_before:
            h = self.mlp.ln(h)

        if self.output_layernorm:
            h = self.ln_f(h)
        return h, k, v


class VocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        config: model_api.ReaLModelConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.n_positions = config.n_positions
        self.hidden_dim = config.hidden_dim

        tensor_parallel = constants.tensor_parallel_world_size() > 1
        if tensor_parallel:
            embed_cls = ParallelEmbedding
        else:
            embed_cls = nn.Embedding

        self.wte = embed_cls(
            config.vocab_size, config.hidden_dim, dtype=dtype, device=device
        )

        self.apply_abs_pos_embed = not config.apply_rotary
        if self.apply_abs_pos_embed:
            p_embed_cls = (
                OffsetParallelPositionalEmbedding
                if tensor_parallel
                else OffsetPositionalEmbedding
            )
            self.wpe = p_embed_cls(
                config.n_positions,
                config.hidden_dim,
                offset=config.abs_position_embedding_offset,
                dtype=dtype,
                device=device,
            )

        self.normalize_embed = config.normalize_embed
        self.embed_drop = nn.Dropout(config.embd_pdrop)

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        # Set position ids.
        # if y.packed_position_ids is not None:
        #     raise ValueError("In our use cases, position_ids must be None.")
        y.packed_position_ids = compute_varlen_position_indices(
            total_seqlen=y.packed_input_ids.shape[0],
            cu_seqlens=x.cu_seqlens,
            seqlen_offsets=y.cache_seqlens,
        )
        if x.max_seqlen > self.n_positions:
            raise ValueError(
                f"max_seqlen ({x.max_seqlen}) must be <= n_positions ({self.n_positions})."
            )
        assert y.packed_position_ids.shape == y.packed_input_ids.shape, (
            y.packed_position_ids.shape,
            y.packed_input_ids.shape,
            x.cu_seqlens,
        )

        x.pp_output = self._forward(y.packed_input_ids, y.packed_position_ids)
        return x

    def _forward(
        self, input_ids: torch.LongTensor, position_ids: torch.LongTensor
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        if self.apply_abs_pos_embed:
            inputs_embeds = inputs_embeds + self.wpe(position_ids)
        if self.normalize_embed:
            normalizer = torch.tensor(self.hidden_dim**0.5, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds * normalizer
        if constants.sequence_parallel():
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(
                inputs_embeds
            )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            inputs_embeds = inputs_embeds.clone()
            # with tensor_parallel.get_cuda_rng_tracker().fork():
            x = self.embed_drop(inputs_embeds)
        else:
            x = self.embed_drop(inputs_embeds)
        return x


class OutputHead(nn.Linear):
    def __init__(
        self, *args, norm_head: bool = False, norm_softmax: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._norm_head = norm_head
        self._norm_softmax = norm_softmax

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        x.pp_output = self._forward(x.pp_input)
        return x

    def _forward(self, x: torch.Tensor):
        if self._norm_head and self.out_features != 1:
            unnormed_head = nn.functional.linear(
                torch.eye(
                    self.in_features, dtype=self.weight.dtype, device=self.weight.device
                ),
                self.weight,
            ).transpose(1, 0)
            head_norm = unnormed_head.norm(dim=0, keepdim=True, p=2)
            normed_head = unnormed_head / (head_norm + 1e-7)
            logits = nn.functional.linear(x, normed_head, None)
        else:
            logits = super().forward(x)

        if self._norm_softmax and self.out_features != 1:
            logits = logits / (torch.std(logits, dim=-1, keepdim=True) + 1e-6)
        return logits


class SequenceParallelCriticHead(nn.Linear):

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        all_gather_buffer = tensor_parallel.gather_from_sequence_parallel_region(
            x.pp_input
        )
        x.pp_output = nn.functional.linear(all_gather_buffer, self.weight, self.bias)
        return x

    def _forward(self, x: torch.Tensor):
        x = tensor_parallel.gather_from_sequence_parallel_region(x)
        return super().forward(x)


class ParallelActorHead(ColumnParallelLinear):

    def __init__(
        self, *args, norm_head: bool = False, norm_softmax: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._norm_head = norm_head
        self._norm_softmax = norm_softmax

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        x.pp_output = self._forward(x.pp_input)
        return x

    def _forward(self, x: torch.Tensor):
        weight = self.weight
        if self._norm_head:
            from realhf.impl.model.parallelism.tensor_parallel.mappings import (
                gather_from_sequence_parallel_region,
            )

            # HACK: This is a terrible implementation for Bailing's head norming,
            # because we basically eliminates TP for the LM head.
            whole_weight = gather_from_sequence_parallel_region(self.weight)
            unnormed_head = nn.functional.linear(
                torch.eye(
                    self.input_size, dtype=self.weight.dtype, device=self.weight.device
                ),
                whole_weight,
            ).transpose(1, 0)
            head_norm = unnormed_head.norm(dim=0, keepdim=True, p=2)
            normed_head = unnormed_head / (head_norm + 1e-7)
            weight = tensor_parallel.scatter_to_sequence_parallel_region(normed_head)

        output = parallel_lm_logits(
            x,
            weight,
            parallel_output=True,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            bias=self.bias,
        )

        if self._norm_softmax:
            whole_output = gather_from_tensor_model_parallel_region(output)
            whole_output = whole_output / (
                torch.std(whole_output, dim=-1, keepdim=True) + 1e-6
            )
            output = scatter_to_tensor_model_parallel_region(whole_output)

        return output


class ReaLModelParamKeys:
    """The keys of parameters in ReaLModel, used for parameter reallocation.

    **IMPORTANT**: The returned keys are **ordered**. They should have
    the same order as we iterate layer indices and call
    layer.state_dict().
    """

    @staticmethod
    def embed(config: model_api.ReaLModelConfig) -> int:
        keys = ["0.wte.weight"]
        if not config.apply_rotary:
            keys += ["0.wpe.weight"]
        return keys

    @staticmethod
    def tblock(config: model_api.ReaLModelConfig, idx: int) -> List[str]:
        # NOTE: `idx`` is the index of transformer blocks,
        # i.e, 0 for the first block or the second layer of the transformer.
        # NOTE: The order matters, we should not change the order of keys.
        keys = [f"{idx + 1}.attn.c_attn.ln.weight"]
        if config.layer_norm_type is None:
            keys += [f"{idx + 1}.attn.c_attn.ln.bias"]
        keys += [f"{idx + 1}.attn.c_attn.q_attn.weight"]
        if config.use_attention_bias:
            keys += [f"{idx + 1}.attn.c_attn.q_attn.bias"]
        keys += [f"{idx + 1}.attn.c_attn.k_attn.weight"]
        if config.use_attention_bias:
            keys += [f"{idx + 1}.attn.c_attn.k_attn.bias"]
        keys += [f"{idx + 1}.attn.c_attn.v_attn.weight"]
        if config.use_attention_bias:
            keys += [f"{idx + 1}.attn.c_attn.v_attn.bias"]
        keys += [f"{idx + 1}.attn.c_proj.weight"]
        if config.use_attn_proj_bias:
            keys += [f"{idx + 1}.attn.c_proj.bias"]
        if config.qk_layernorm:
            keys += [f"{idx + 1}.attn.q_ln.weight"]
            keys += [f"{idx + 1}.attn.k_ln.weight"]
            if config.layer_norm_type is None:
                keys += [f"{idx + 1}.attn.q_ln.bias"]
                keys += [f"{idx + 1}.attn.k_ln.bias"]
        keys += [f"{idx + 1}.mlp.ln.weight"]
        if config.layer_norm_type is None:
            keys += [f"{idx + 1}.mlp.ln.bias"]

        if config.mlp_type is None:
            if config.use_mlp_bias:
                keys += [
                    f"{idx + 1}.mlp.c_fc.weight",
                    f"{idx + 1}.mlp.c_fc.bias",
                    f"{idx + 1}.mlp.c_proj.weight",
                    f"{idx + 1}.mlp.c_proj.bias",
                ]
            else:
                keys += [
                    f"{idx + 1}.mlp.c_fc.weight",
                    f"{idx + 1}.mlp.c_proj.weight",
                ]
        elif config.mlp_type == "llama":
            weights_key = [
                f"{idx + 1}.mlp.gate_proj.weight",
                f"{idx + 1}.mlp.up_proj.weight",
                f"{idx + 1}.mlp.down_proj.weight",
            ]
            if config.use_mlp_bias:
                keys += list(
                    itertools.chain.from_iterable(
                        zip(
                            weights_key,
                            [
                                f"{idx + 1}.mlp.gate_proj.bias",
                                f"{idx + 1}.mlp.up_proj.bias",
                                f"{idx + 1}.mlp.down_proj.bias",
                            ],
                        )
                    )
                )
            else:
                keys += weights_key
        elif config.mlp_type == "moe":
            num_experts = config.moe.num_experts
            keys += [
                f"{idx + 1}.mlp.router.weight",
            ]
            for j in range(num_experts):
                weights_key = [
                    f"{idx + 1}.mlp.experts.local_experts.{j}.gate_proj.weight",
                    f"{idx + 1}.mlp.experts.local_experts.{j}.up_proj.weight",
                    f"{idx + 1}.mlp.experts.local_experts.{j}.down_proj.weight",
                ]
                if config.use_mlp_bias:
                    keys += list(
                        itertools.chain.from_iterable(
                            zip(
                                weights_key,
                                [
                                    f"{idx + 1}.mlp.experts.local_experts.{j}.gate_proj.bias",
                                    f"{idx + 1}.mlp.experts.local_experts.{j}.up_proj.bias",
                                    f"{idx + 1}.mlp.experts.local_experts.{j}.down_proj.bias",
                                ],
                            )
                        )
                    )
                else:
                    keys += weights_key
        else:
            raise NotImplementedError()
        if idx == config.n_layers - 1:
            keys += [f"{idx + 1}.ln_f.weight"]
            if config.layer_norm_type is None:
                keys += [f"{idx + 1}.ln_f.bias"]
        return keys

    @staticmethod
    def head(config: model_api.ReaLModelConfig) -> List[str]:
        return [f"{config.n_layers + 1}.weight"]


def keys_from_layer_indices(
    config: model_api.ReaLModelConfig, layer_indices: List[int]
) -> List[str]:
    # assert _is_integer_list_contiguous(layer_indices)
    sd_keys = []
    for layer_idx in sorted(layer_indices):
        if layer_idx == 0:
            sd_keys += ReaLModelParamKeys.embed(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += ReaLModelParamKeys.head(config)
        else:
            sd_keys += ReaLModelParamKeys.tblock(config, layer_idx - 1)
    return sd_keys
