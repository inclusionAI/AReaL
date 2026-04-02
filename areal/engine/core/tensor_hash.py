from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from areal.infra.platforms import current_platform

_TENSOR_HASH_PERSON = b"AReaLTensorV1"
_STRING_HASH_PERSON = b"AReaLNamesV1"
_HASH_DTYPE = torch.int64
_UINT64_SIGN_WRAP = 1 << 63
_UINT64_MODULUS = 1 << 64


def _tensor_hash_device() -> torch.device:
    return cast(torch.device, current_platform.current_device())


def _to_local_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach()


def _blake2b_uint64(*, person: bytes, payloads: Sequence[bytes]) -> int:
    hasher = hashlib.blake2b(digest_size=8, person=person)
    for payload in payloads:
        hasher.update(payload)
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)


def _to_hash_tensor(hash_value: int, *, device: torch.device) -> torch.Tensor:
    hash_value = _to_signed_hash_value(hash_value)
    return torch.tensor(hash_value, dtype=_HASH_DTYPE, device=device)


def _to_signed_hash_value(hash_value: int) -> int:
    if hash_value >= _UINT64_SIGN_WRAP:
        hash_value -= _UINT64_MODULUS
    return hash_value


def fallback_hash_tensor(
    tensor: torch.Tensor | DTensor,
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    local = _to_local_tensor(tensor).contiguous()
    byte_view = local.reshape(-1).view(torch.uint8)
    if byte_view.device.type != "cpu":
        byte_view = byte_view.cpu()
    byte_payload = bytes(byte_view.tolist())

    hash_value = _blake2b_uint64(
        person=_TENSOR_HASH_PERSON,
        payloads=(
            str(local.dtype).encode("utf-8"),
            repr(tuple(local.shape)).encode("utf-8"),
            byte_payload,
        ),
    )
    return _to_hash_tensor(hash_value, device=output_device or _tensor_hash_device())


def hash_tensor(
    tensor: torch.Tensor | DTensor,
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    device = output_device or _tensor_hash_device()
    local = _to_local_tensor(tensor)

    if local.numel() == 0:
        return torch.tensor(0, dtype=_HASH_DTYPE, device=device)

    # TODO(agent): Remove the fallback path once the full SGLang/vLLM stack
    # upgrades to PyTorch 2.10+ and torch.hash_tensor is always available.
    if hasattr(torch, "hash_tensor"):
        hash_input = local if local.device.type != "cpu" else local.to(device)
        return torch.hash_tensor(hash_input).to(dtype=_HASH_DTYPE, device=device)

    return fallback_hash_tensor(local, output_device=device)


def hash_parameter_shards(
    param_list: list[tuple[str, nn.Parameter]],
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    device = output_device or _tensor_hash_device()
    hashes = torch.empty(len(param_list), dtype=_HASH_DTYPE, device=device)
    for i, (_name, param) in enumerate(param_list):
        hashes[i] = hash_tensor(param.data, output_device=device)
    return hashes


def hash_string_sequence(
    values: Sequence[str],
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    device = output_device or _tensor_hash_device()
    encoded_values = tuple(value.encode("utf-8") for value in values)
    payloads: list[bytes] = []
    for encoded in encoded_values:
        payloads.append(len(encoded).to_bytes(8, byteorder="little", signed=False))
        payloads.append(encoded)

    digest = _blake2b_uint64(person=_STRING_HASH_PERSON, payloads=payloads)
    return torch.tensor(
        [len(values), _to_signed_hash_value(digest)],
        dtype=_HASH_DTYPE,
        device=device,
    )
