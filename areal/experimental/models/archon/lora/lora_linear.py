"""LoRALinear module implementation following torchtune patterns.

Reference: torchtune/torchtune/modules/peft/lora.py
This implementation provides FSDP2-compatible LoRA layers that work naturally
with Archon's meta device initialization flow.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA).

    LoRA decomposes weight updates into low-rank matrices A and B:
        W' = W + (alpha/rank) * B @ A

    During forward pass:
        output = x @ W^T + (alpha/rank) * x @ A^T @ B^T

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        rank: LoRA rank (r parameter)
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path (default: 0.0)
        use_bias: Whether to include bias term (default: False)

    Attributes:
        weight: Base weight parameter (frozen during LoRA training)
        bias: Optional bias parameter (frozen during LoRA training)
        lora_a: Low-rank matrix A (trainable)
        lora_b: Low-rank matrix B (trainable)
        dropout: Dropout layer for LoRA path
        scaling: Computed scaling factor (alpha/rank)
        disabled: Flag to disable LoRA during forward pass (for reference models)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.disabled = False

        # Base weight (frozen during LoRA training)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        # LoRA adapters (trainable)
        # Note: naming lora_a, lora_b (lowercase) matches PEFT convention
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following torchtune/PEFT conventions.

        Base weight: Kaiming uniform (will be overwritten by checkpoint loading)
        Bias: Zeros
        lora_a: Kaiming uniform (random initialization)
        lora_b: Zeros (so initial LoRA contribution is 0)
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # LoRA init: lora_a=kaiming, lora_b=zeros
        # This ensures initial output matches base model output
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA.

        Args:
            x: Input tensor of shape [..., in_dim]

        Returns:
            Output tensor of shape [..., out_dim]
        """
        # Base forward pass
        base_out = F.linear(x, self.weight, self.bias)

        # If LoRA is disabled (e.g., for reference model), return base output
        if self.disabled:
            return base_out

        # LoRA forward pass: dropout -> A -> B -> scale
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))
        return base_out + self.scaling * lora_out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Convert an existing nn.Linear to LoRALinear.

        Args:
            linear: Existing linear layer
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability (default: 0.0)

        Returns:
            LoRALinear with base weights copied from input linear layer
        """
        lora_linear = cls(
            in_dim=linear.in_features,
            out_dim=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_bias=linear.bias is not None,
        )

        # Copy base weights from original linear layer
        lora_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_linear.bias.data.copy_(linear.bias.data)

        return lora_linear

    def adapter_params(self) -> list[str]:
        """Return list of adapter parameter names.

        Implements AdapterModule protocol for parameter extraction.

        Returns:
            List of parameter names relative to this module
        """
        return ["lora_a.weight", "lora_b.weight"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout.p}, bias={self.bias is not None})"
        )
