from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelArgsProtocol(Protocol):
    """Protocol for model args that have head configuration."""

    n_heads: int
    n_kv_heads: int | None


def validate_tp_constraints(
    model_args: ModelArgsProtocol,
    tp_size: int,
) -> None:
    """Validate tensor parallelism constraints for attention heads.

    This validates that the model's attention head configuration is compatible
    with the requested tensor parallelism configuration.

    Args:
        model_args: Model arguments containing n_heads and n_kv_heads.
        tp_size: Tensor parallelism size.

    Raises:
        ValueError: If head counts don't satisfy TP constraints.
    """
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else n_heads

    if n_heads % tp_size != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by tp_size ({tp_size})"
        )

    if n_kv_heads % tp_size != 0:
        raise ValueError(
            f"n_kv_heads ({n_kv_heads}) must be divisible by tp_size ({tp_size})"
        )


def validate_cp_constraints(
    model_args: ModelArgsProtocol,
    cp_size: int,
    tp_size: int = 1,
) -> None:
    """Validate context parallelism constraints for attention heads.

    This validates that the model's attention head configuration is compatible
    with the requested context parallelism (Ulysses SP) configuration.

    Args:
        model_args: Model arguments containing n_heads and n_kv_heads.
        cp_size: Context parallelism size (world size of CP group).
        tp_size: Tensor parallelism size, used to compute local head counts.

    Raises:
        ValueError: If head counts don't satisfy CP constraints.
    """
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else n_heads
    n_rep = n_heads // n_kv_heads

    q_heads = n_heads // tp_size
    kv_heads = n_kv_heads // tp_size

    # Constraint 1: q_heads must be divisible by cp_size (no repeat allowed for q)
    if q_heads % cp_size != 0:
        raise ValueError(
            f"q_heads after TP ({q_heads}) must be divisible by cp_size ({cp_size})"
        )

    # Constraint 2: kv_heads must be divisible by or divide cp_size,
    # and if repeat is needed, repeat count must be a factor of n_rep
    if kv_heads >= cp_size:
        if kv_heads % cp_size != 0:
            raise ValueError(
                f"kv_heads after TP ({kv_heads}) must be divisible by cp_size ({cp_size})"
            )
    else:
        if cp_size % kv_heads != 0:
            raise ValueError(
                f"cp_size ({cp_size}) must be divisible by kv_heads after TP ({kv_heads})"
            )
        repeats = cp_size // kv_heads
        if n_rep % repeats != 0:
            raise ValueError(
                f"repeat count ({repeats}) must be a factor of n_rep ({n_rep})"
            )


__all__ = ["ModelArgsProtocol", "validate_cp_constraints", "validate_tp_constraints"]
