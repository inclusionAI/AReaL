# SPDX-License-Identifier: Apache-2.0

import hashlib
import pickle
from types import SimpleNamespace

import pytest
import torch

from areal.engine import sglang_mm_tensor_storage as patch


@pytest.fixture
def compact_items(monkeypatch):
    source = (
        "import copy\nimport torch\n"
        + patch.ANCHOR
        + "    expanded_mm_items = original_mm_items\n"
        + patch.OLD
    ).encode()
    expected = source.replace(
        patch.ANCHOR.encode(), (patch.HELPER + patch.ANCHOR).encode()
    ).replace(patch.OLD.encode(), patch.NEW.encode())
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(patch, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    patched = patch.patched_source(source)
    assert patch.patched_source(patched) == patched
    namespace = {}
    exec(compile(patched, "mm_utils.py", "exec"), namespace)
    return namespace["get_new_expanded_mm_items"]


@pytest.mark.parametrize("count", [1, 16])
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_views_pickle_without_backing_storage_amplification(
    compact_items, count, strided
):
    source = torch.arange(32 * 1024, dtype=torch.float32).reshape(32, 1024)
    views = [source[i : i + 1, :: 2 if strided else 1] for i in range(count)]
    items = [
        SimpleNamespace(
            feature=view,
            precomputed_embeddings=None,
            model_specific_data={"nested": [view, (view,)]},
            offsets=[(i, i + 1)],
            hash=i,
        )
        for i, view in enumerate(views)
    ]
    compact = compact_items(items)
    encoded = pickle.dumps(compact)
    restored = pickle.loads(encoded)
    logical_bytes = sum(view.numel() * view.element_size() for view in views)
    assert len(encoded) < logical_bytes + 1024 * count
    assert len(pickle.dumps(items)) > len(encoded) * 8
    for original, result, decoded in zip(items, compact, restored, strict=True):
        assert result is not original
        assert original.feature.untyped_storage().nbytes() == source.numel() * 4
        assert result.model_specific_data["nested"][0] is result.feature
        assert result.model_specific_data["nested"][1][0] is result.feature
        assert result.offsets == original.offsets
        assert result.hash == original.hash
        torch.testing.assert_close(decoded.feature, original.feature, rtol=0, atol=0)
        assert decoded.feature.untyped_storage().nbytes() == (
            decoded.feature.numel() * decoded.feature.element_size()
        )


def test_embeddings_empty_views_and_compact_tensors(compact_items):
    full = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    marker = object()
    item = SimpleNamespace(
        feature=[full, full[:0]],
        precomputed_embeddings=full[1:3],
        model_specific_data={"proxy": marker, "none": None},
    )
    result = compact_items([item])[0]
    assert result.feature[0] is full
    assert result.feature[1].untyped_storage().nbytes() == 0
    assert result.model_specific_data["proxy"] is marker
    torch.testing.assert_close(result.precomputed_embeddings, full[1:3], rtol=0, atol=0)
    assert result.precomputed_embeddings.dtype == torch.bfloat16


def test_unknown_multimodal_source_is_rejected():
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.patched_source(b"# unvalidated runtime")
