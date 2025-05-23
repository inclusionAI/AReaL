# Modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

import math
from typing import Literal, Optional, Tuple, Union

import torch
from einops import rearrange, repeat

try:
    from flash_attn.ops.triton.rotary import apply_rotary
except ModuleNotFoundError:
    pass


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(
                cos, sin, cu_seqlens
            )  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x,
        cos,
        sin,
        interleaved,
        inplace,
        seqlen_offsets,
        cu_seqlens,
        max_seqlen,
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class RotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings from RoFormer_ (Su et. al). A crucial
    insight from the method is that the query and keys are transformed by
    rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_factor: float = None,
        scale_type: Optional[Literal["linear", "dynamic"]] = None,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
        special_impl: Optional[str] = None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_factor = scale_factor
        self.scale_type = scale_type

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

        self.special_impl = special_impl

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            if self.scale_factor is not None:
                if self.scale_type == "linear":
                    t = t / self.scale_factor
                elif self.scale_type == "dynamic":
                    base = self.base * (
                        (self.scale_factor * seqlen / self._seq_len_cached)
                        - (self.scale_factor - 1)
                    ) ** (self.dim / (self.dim - 2))
                    inv_freq = 1.0 / (
                        base
                        ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
                    )
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                else:
                    raise NotImplementedError("Unsupported scale type", self.scale_type)
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.special_impl == "bailing":
                freqs = torch.cat((freqs, freqs), dim=-1)
                self._cos_cached = torch.cos(freqs)[:seqlen].to(dtype)
                self._sin_cached = torch.sin(freqs)[:seqlen].to(dtype)
            elif self.special_impl is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                raise NotImplementedError()
            # power = (
            #     torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
            #     - seqlen // 2
            # ) / self.scale_base
            # scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # # We want the multiplication by scale to happen in fp32
            # self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            # self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            # self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            # self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qk: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, nq + nkv, headdim),
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        assert max_seqlen is None or isinstance(max_seqlen, int)
        if cu_seqlens is not None:
            assert max_seqlen is not None
        seqlen = qk.shape[1] if max_seqlen is None else max_seqlen
        self._update_cos_sin_cache(seqlen, device=qk.device, dtype=qk.dtype)
        return apply_rotary_emb_func(
            qk,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            inplace=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
