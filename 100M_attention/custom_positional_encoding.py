"""Custom positional encoding modules for context-free aware attention.

This module provides the custom positional encoding implementations from
inference_partial.py, packaged for easy integration with LLaMA-Factory.

Reference: "Towards 100M context windows via context-free aware attention"
"""

import math
from typing import Tuple

import torch
from torch import nn


class CustomRotaryPositionalEncoding(nn.Module):
    """Custom rotary positional embedding with periodic encoding and dimension masking.
    
    This implementation provides:
    1. Periodic positional encoding via modulo operation
    2. Dimension masking to zero out higher dimensions (context-free aware attention)
    3. Learned scaling via alpha/beta parameters
    
    Args:
        inv_freq: Inverse frequency tensor for rotary encoding
        max_length: Maximum sequence length
        attention_scaling: Scaling factor for attention
        dropout: Dropout rate applied to embeddings
        learned_scaling: Whether to use learnable alpha/beta parameters
        period: Periodicity for positional encoding (default: 128000 for ~100M context)
    """

    def __init__(
        self,
        inv_freq: torch.Tensor,
        max_length: int,
        attention_scaling: float = 1.0,
        dropout: float = 0.0,
        learned_scaling: bool = True,
        period: int = 128000,
    ) -> None:
        super().__init__()
        if inv_freq.ndim != 1:
            raise ValueError("`inv_freq` is expected to be a 1-D tensor.")

        head_dim = inv_freq.numel() * 2
        if head_dim % 2 != 0:
            raise ValueError("Rotary head dimension must be even.")

        self.head_dim = head_dim
        self.max_seq_len_cached = max_length
        self.original_max_seq_len = max_length
        self.attention_scaling = attention_scaling
        self.dropout_p = dropout
        self.learned_scaling = learned_scaling
        self.period = period  # Periodicity for positional encoding

        inv_freq = inv_freq.detach().to(dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if learned_scaling:
            self.alpha = nn.Parameter(torch.ones_like(inv_freq))
            self.beta = nn.Parameter(torch.zeros_like(inv_freq))
        else:
            self.register_buffer("alpha", torch.ones_like(inv_freq), persistent=False)
            self.register_buffer("beta", torch.zeros_like(inv_freq), persistent=False)
        
        # Create dimension masking buffer once during initialization
        # Apply masking: zero out dimensions beyond m, where m ≈ 0.8 * n
        n = self.head_dim
        # Ensure n is even (it should be by construction)
        if n % 2 != 0:
            n = n - 1
        
        # Calculate m as approximately 0.8 * n, ensuring it's even
        m = int(0.8 * n)
        if m % 2 != 0:
            m = m - 1
        
        # Create a mask that zeros out dimensions beyond m
        mask = torch.zeros(n, dtype=torch.float32)
        mask[:m] = 1.0
        self.register_buffer("dim_mask", mask, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute cos and sin embeddings.
        
        Args:
            x: Input tensor (used for device/dtype inference)
            position_ids: Position indices [batch_size, seq_len]
            
        Returns:
            Tuple of (cos_emb, sin_emb) with masking applied
        """
        if position_ids.dtype != torch.long:
            position_ids = position_ids.long()

        batch, seq_len = position_ids.shape
        inv_freq = self.inv_freq.to(device=x.device)
        
        # Apply modulo operation to create periodic positional encoding
        # For any index i and i+period, they will have the same encoding
        position = position_ids.to(device=x.device, dtype=torch.float32)
        position = position % self.period  # Apply periodicity

        freqs = torch.einsum("bs,d->bsd", position, inv_freq)
        alpha = self.alpha.to(device=x.device)
        beta = self.beta.to(device=x.device)
        freqs = freqs * alpha + beta

        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply dropout only if needed and in training mode
        if self.dropout_p > 0 and self.training:
            emb = torch.nn.functional.dropout(emb, p=self.dropout_p, training=True)

        # Apply the pre-created dimension mask
        # emb shape: [batch, seq_len, head_dim]
        dim_mask = self.dim_mask.to(device=emb.device, dtype=emb.dtype)
        emb = emb * dim_mask.unsqueeze(0).unsqueeze(0)  # broadcast to [1, 1, head_dim]

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CustomPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learned per-dimension scaling.
    
    For models that use traditional learned positional embeddings (e.g., GPT-2).
    Not used for rotary models like Qwen/LLaMA.
    
    The module produces deterministic sinusoidal embeddings and then applies a
    lightweight, trainable affine transformation (per-dimension scale and
    offset). This keeps the positional information explicit while granting the
    model flexibility to adapt the encoding distribution during fine-tuning.
    """

    def __init__(
        self,
        hidden_size: int,
        max_length: int,
        dropout: float = 0.0,
        learned_scaling: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(
                "CustomPositionalEncoding expects an even hidden size; received"
                f" {hidden_size}."
            )

        self.hidden_size = hidden_size
        self.embedding_dim = hidden_size
        self.num_embeddings = max_length
        self.dropout_p = dropout
        self.learned_scaling = learned_scaling

        pe = self._build_sinusoidal_table(max_length, hidden_size)
        self.register_buffer("pe", pe, persistent=False)

        if learned_scaling:
            self.alpha = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer("alpha", torch.ones(hidden_size), persistent=False)
            self.register_buffer("beta", torch.zeros(hidden_size), persistent=False)

    @staticmethod
    def _build_sinusoidal_table(max_len: int, hidden_size: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(max_len, hidden_size, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @property
    def weight(self) -> torch.Tensor:
        """Expose the internal table to mimic `nn.Embedding`'s API."""
        return self.pe

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        if position_ids.dtype != torch.long:
            position_ids = position_ids.long()

        max_pos = int(position_ids.max().item()) + 1 if position_ids.numel() else 0
        if max_pos > self.num_embeddings:
            raise ValueError(
                f"Position ids require length {max_pos}, but buffer is built"
                f" for {self.num_embeddings}. Increase `max_length`."
            )

        device = position_ids.device
        pe = self.pe.to(device)
        embeddings = pe[position_ids]

        alpha = self.alpha.to(device)
        beta = self.beta.to(device)
        embeddings = embeddings * alpha + beta
        
        # Apply dropout only if needed and in training mode
        if self.dropout_p > 0 and self.training:
            embeddings = torch.nn.functional.dropout(embeddings, p=self.dropout_p, training=True)
        
        return embeddings
