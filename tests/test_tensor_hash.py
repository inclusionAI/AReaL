import pytest
import torch
from torch import nn

import areal.engine.core.tensor_hash as tensor_hash_module
from areal.engine.core.tensor_hash import (
    fallback_hash_tensor,
    hash_parameter_shards,
    hash_string_sequence,
    hash_tensor,
)


def _mock_max_uint64_digest(*, person: bytes, payloads: list[bytes]) -> int:
    del person, payloads
    return 2**64 - 1


def _mock_sign_wrap_digest(*, person: bytes, payloads: list[bytes]) -> int:
    del person, payloads
    return 2**63


def test_fallback_hash_tensor_changes_when_tensor_bytes_change():
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    changed_tensor = tensor.clone()
    changed_tensor[2, 1] += 0.5

    original_hash = fallback_hash_tensor(tensor, output_device=torch.device("cpu"))
    changed_hash = fallback_hash_tensor(
        changed_tensor, output_device=torch.device("cpu")
    )

    assert original_hash.item() != changed_hash.item()


def test_fallback_hash_tensor_changes_when_shape_changes():
    flat = torch.arange(8, dtype=torch.int32)
    matrix = flat.view(2, 4)

    flat_hash = fallback_hash_tensor(flat, output_device=torch.device("cpu"))
    matrix_hash = fallback_hash_tensor(matrix, output_device=torch.device("cpu"))

    assert flat_hash.item() != matrix_hash.item()


def test_fallback_hash_tensor_is_deterministic_across_repeated_calls():
    tensor = torch.randn(5, 7, dtype=torch.float32)

    first_hash = fallback_hash_tensor(tensor, output_device=torch.device("cpu"))
    second_hash = fallback_hash_tensor(tensor, output_device=torch.device("cpu"))

    torch.testing.assert_close(first_hash, second_hash)


def test_fallback_hash_tensor_is_dtype_sensitive():
    int_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    float_tensor = int_tensor.to(torch.float32)

    int_hash = fallback_hash_tensor(int_tensor, output_device=torch.device("cpu"))
    float_hash = fallback_hash_tensor(float_tensor, output_device=torch.device("cpu"))

    assert int_hash.item() != float_hash.item()


def test_fallback_hash_tensor_normalizes_non_contiguous_inputs():
    tensor = torch.arange(24, dtype=torch.float32).view(4, 6)[:, ::2]

    non_contiguous_hash = fallback_hash_tensor(
        tensor, output_device=torch.device("cpu")
    )
    contiguous_hash = fallback_hash_tensor(
        tensor.contiguous(), output_device=torch.device("cpu")
    )

    torch.testing.assert_close(non_contiguous_hash, contiguous_hash)


def test_hash_string_sequence_is_order_sensitive():
    first = hash_string_sequence(
        ["a.weight", "b.bias"], output_device=torch.device("cpu")
    )
    second = hash_string_sequence(
        ["b.bias", "a.weight"], output_device=torch.device("cpu")
    )
    repeated = hash_string_sequence(
        ["a.weight", "b.bias"], output_device=torch.device("cpu")
    )

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, second)


def test_hash_string_sequence_changes_when_length_changes():
    short = hash_string_sequence(["a.weight"], output_device=torch.device("cpu"))
    long = hash_string_sequence(
        ["a.weight", "b.bias"], output_device=torch.device("cpu")
    )

    assert not torch.equal(short, long)


def test_hash_parameter_shards_fallback_returns_one_hash_per_param(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delattr(torch, "hash_tensor", raising=False)

    params = [
        ("first", nn.Parameter(torch.arange(6, dtype=torch.float32))),
        ("second", nn.Parameter(torch.arange(6, dtype=torch.float32) + 10)),
    ]
    initial_hashes = hash_parameter_shards(params, output_device=torch.device("cpu"))

    params[1][1].data[3] += 1.0
    updated_hashes = hash_parameter_shards(params, output_device=torch.device("cpu"))

    assert initial_hashes.shape == (2,)
    assert updated_hashes.shape == (2,)
    assert initial_hashes.dtype == torch.int64
    assert updated_hashes.dtype == torch.int64
    assert initial_hashes[0].item() == updated_hashes[0].item()
    assert initial_hashes[1].item() != updated_hashes[1].item()


def test_hash_tensor_returns_zero_for_empty_tensor():
    empty = torch.empty(0, dtype=torch.float32)

    hashed = hash_tensor(empty, output_device=torch.device("cpu"))

    assert hashed.dtype == torch.int64
    assert hashed.device.type == "cpu"
    assert hashed.item() == 0


def test_hash_string_sequence_returns_int64_tensor():
    hashed = hash_string_sequence(
        ["a.weight", "b.bias"], output_device=torch.device("cpu")
    )

    assert hashed.dtype == torch.int64
    assert hashed.shape == (2,)


def test_fallback_hash_tensor_wraps_upper_uint64_range_to_int64(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(tensor_hash_module, "_blake2b_uint64", _mock_max_uint64_digest)

    hashed = fallback_hash_tensor(
        torch.arange(4, dtype=torch.float32), output_device=torch.device("cpu")
    )

    assert hashed.dtype == torch.int64
    assert hashed.item() == -1


def test_hash_string_sequence_wraps_upper_uint64_range_to_int64(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(tensor_hash_module, "_blake2b_uint64", _mock_sign_wrap_digest)

    hashed = hash_string_sequence(["a.weight"], output_device=torch.device("cpu"))

    assert hashed.dtype == torch.int64
    assert int(hashed[0].item()) == 1
    assert int(hashed[1].item()) == -(2**63)
