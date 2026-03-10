"""LoRALinear module implementation following torchtune patterns.

Reference: torchtune/torchtune/modules/peft/lora.py

LoRA weights are stored as **plain tensors** (not ``nn.Parameter``) so that
FSDP2 does not register ``post_accumulate_grad_hook`` on them.  This avoids
FSDP DP reduce-scatter operations interleaving with DTensor TP operations
during backward, which would otherwise create a diamond deadlock across the
TP and DP communicators.

After backward, ``sync_lora_grads`` must be called to all-reduce LoRA
gradients across both TP and DP groups before the optimizer step.
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

    LoRA weights (_lora_a_weight, _lora_b_weight) are plain tensors stored
    via ``object.__setattr__`` to keep them invisible to ``nn.Module``
    parameter/buffer tracking and therefore to FSDP2.
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
        self._dropout_p = dropout

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        _a = torch.empty(rank, in_dim)
        _b = torch.empty(out_dim, rank)
        _a.requires_grad_(True)
        _b.requires_grad_(True)
        object.__setattr__(self, "_lora_a_weight", _a)
        object.__setattr__(self, "_lora_b_weight", _b)

        self._tp_enabled = False

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self._lora_a_weight, a=math.sqrt(5))
        nn.init.zeros_(self._lora_b_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)

        if self.disabled:
            return base_out

        if self._tp_enabled:
            result = self._tp_lora_forward(x, base_out)
            if result.requires_grad and hasattr(self, "_debug_name"):
                _name = self._debug_name

                result.register_hook(lambda grad: grad)
            return result

        h = F.dropout(x, p=self._dropout_p, training=self.training)
        h = F.linear(h, self._lora_a_weight)
        lora_out = F.linear(h, self._lora_b_weight)
        return base_out + self.scaling * lora_out

    def _tp_lora_forward(
        self, x: torch.Tensor, base_out: torch.Tensor
    ) -> torch.Tensor:
        """LoRA forward compatible with TP + FSDP2.

        1. Input is DETACHED for the LoRA path so the input gradient comes
           entirely from the base ``F.linear`` path (which handles TP).
        2. The LoRA output is wrapped as a DTensor with the SAME placements
           as ``base_out`` and added in DTensor space, keeping the autograd
           connection to ``base_out`` intact.
        3. LoRA weights are plain tensors; their gradients are synced after
           backward via ``sync_lora_grads``.
        """
        from torch.distributed.tensor import DTensor

        local_x = x._local_tensor.detach() if isinstance(x, DTensor) else x.detach()
        h = F.dropout(local_x, p=self._dropout_p, training=self.training)

        if self._tp_style == "rowwise":
            s = self._tp_rank * self._tp_local_in
            lora_a_w = self._lora_a_weight[:, s : s + self._tp_local_in]
            h = F.linear(h, lora_a_w)
        else:
            h = F.linear(h, self._lora_a_weight)

        lora_out = F.linear(h, self._lora_b_weight)
        lora_out = self.scaling * lora_out

        if self._tp_style == "colwise":
            s = self._tp_rank * self._tp_local_out
            lora_out = lora_out[..., s : s + self._tp_local_out]

        if isinstance(base_out, DTensor):
            lora_dtensor = DTensor.from_local(
                lora_out,
                base_out.device_mesh,
                list(base_out.placements),
                run_check=False,
            )
            return base_out + lora_dtensor

        return base_out + lora_out

    # ------------------------------------------------------------------
    # LoRA weight access helpers
    # ------------------------------------------------------------------

    def lora_parameters(self) -> list[torch.Tensor]:
        """Return the raw LoRA weight tensors (for the optimizer)."""
        return [self._lora_a_weight, self._lora_b_weight]

    def materialize_lora(self, device: torch.device) -> None:
        """Move LoRA weights from meta device to *device* and re-init."""
        if self._lora_a_weight.device.type == "meta":
            a = torch.empty(
                self._lora_a_weight.shape,
                dtype=self._lora_a_weight.dtype,
                device=device,
            ).requires_grad_(True)
            object.__setattr__(self, "_lora_a_weight", a)
        if self._lora_b_weight.device.type == "meta":
            b = torch.empty(
                self._lora_b_weight.shape,
                dtype=self._lora_b_weight.dtype,
                device=device,
            ).requires_grad_(True)
            object.__setattr__(self, "_lora_b_weight", b)

    # ------------------------------------------------------------------
    # State dict helpers (plain tensors are invisible to nn.Module)
    # ------------------------------------------------------------------

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._lora_a_weight.device.type == "meta":
            return
        a = self._lora_a_weight if keep_vars else self._lora_a_weight.detach()
        b = self._lora_b_weight if keep_vars else self._lora_b_weight.detach()
        destination[prefix + "_lora_a_weight"] = a
        destination[prefix + "_lora_b_weight"] = b

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        a_key = prefix + "_lora_a_weight"
        b_key = prefix + "_lora_b_weight"
        if a_key in state_dict:
            self._lora_a_weight.data.copy_(state_dict.pop(a_key))
        elif strict:
            missing_keys.append(a_key)
        if b_key in state_dict:
            self._lora_b_weight.data.copy_(state_dict.pop(b_key))
        elif strict:
            missing_keys.append(b_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------
    # Factory & protocol
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Convert an existing nn.Linear to LoRALinear.

        After TP, ``linear.weight`` is a DTensor.  LoRA weights are kept as
        plain tensors (NOT nn.Parameter) so FSDP2 ignores them entirely,
        preventing DP reduce-scatter from interleaving with TP operations
        during backward.
        """
        lora_linear = cls.__new__(cls)
        nn.Module.__init__(lora_linear)

        lora_linear.in_dim = linear.in_features
        lora_linear.out_dim = linear.out_features
        lora_linear.rank = rank
        lora_linear.alpha = alpha
        lora_linear.scaling = alpha / rank
        lora_linear.disabled = False
        lora_linear._dropout_p = dropout

        lora_linear.weight = linear.weight
        if linear.bias is not None:
            lora_linear.bias = linear.bias
        else:
            lora_linear.register_parameter("bias", None)

        local_w = getattr(linear.weight, "_local_tensor", linear.weight)
        _a = torch.empty(rank, linear.in_features, device=local_w.device, dtype=local_w.dtype)
        _b = torch.empty(linear.out_features, rank, device=local_w.device, dtype=local_w.dtype)
        _a.requires_grad_(True)
        _b.requires_grad_(True)
        object.__setattr__(lora_linear, "_lora_a_weight", _a)
        object.__setattr__(lora_linear, "_lora_b_weight", _b)

        from torch.distributed.tensor import DTensor

        lora_linear._tp_enabled = False
        if isinstance(linear.weight, DTensor):
            from torch.distributed.tensor import Shard

            tp_mesh = linear.weight.device_mesh
            placement = linear.weight.placements[0]
            local_shape = linear.weight._local_tensor.shape

            lora_linear._tp_enabled = True
            lora_linear._tp_rank = tp_mesh.get_local_rank(0)
            lora_linear._tp_size = tp_mesh.size(0)

            if isinstance(placement, Shard) and placement.dim == 0:
                lora_linear._tp_style = "colwise"
                lora_linear._tp_local_out = local_shape[0]
            elif isinstance(placement, Shard) and placement.dim == 1:
                lora_linear._tp_style = "rowwise"
                lora_linear._tp_local_in = local_shape[1]
            else:
                lora_linear._tp_style = "replicate"

        nn.init.kaiming_uniform_(lora_linear._lora_a_weight, a=math.sqrt(5))
        nn.init.zeros_(lora_linear._lora_b_weight)

        # Preserve TP forward hooks registered by parallelize_module.
        lora_linear._forward_pre_hooks = linear._forward_pre_hooks.copy()
        lora_linear._forward_hooks = linear._forward_hooks.copy()
        lora_linear._forward_hooks_with_kwargs = (
            linear._forward_hooks_with_kwargs.copy()
        )
        lora_linear._forward_hooks_always_called = (
            linear._forward_hooks_always_called.copy()
        )
        lora_linear._forward_pre_hooks_with_kwargs = (
            linear._forward_pre_hooks_with_kwargs.copy()
        )

        return lora_linear

    def adapter_params(self) -> list[str]:
        return ["_lora_a_weight", "_lora_b_weight"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self._dropout_p}, bias={self.bias is not None})"
        )


def sync_lora_grads(
    model: nn.Module,
    tp_group,
    dp_group=None,
) -> None:
    """All-reduce LoRA weight gradients across TP and DP groups.

    Because LoRA weights are plain tensors (not nn.Parameter), FSDP2 does
    not handle their gradient synchronisation.  This function must be
    called between backward and optimizer_step.

    Args:
        model: The model containing LoRALinear modules.
        tp_group: Process group for tensor parallelism (required).
        dp_group: Process group for data parallelism (optional but
            recommended; without it gradients are only TP-synced).
    """
    import torch.distributed as dist

    if tp_group is None and dp_group is None:
        return

    for module in model.modules():
        if isinstance(module, LoRALinear) and module._tp_enabled:
            for _pname, tensor in [
                ("a", module._lora_a_weight),
                ("b", module._lora_b_weight),
            ]:
                if tensor.grad is not None:
                    grad = tensor.grad
                    if tp_group is not None:
                        dist.all_reduce(grad, group=tp_group)
                    if dp_group is not None:
                        dist.all_reduce(grad, group=dp_group)
