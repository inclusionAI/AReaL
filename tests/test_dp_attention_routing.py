"""Tests for DP attention routing through ModelRequest and SGLang payload."""

from areal.api.io_struct import ModelRequest
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai.cache import InteractionCache


class TestModelRequestDpRank:
    def test_default_none(self):
        req = ModelRequest()
        assert req.data_parallel_rank is None

    def test_set_dp_rank(self):
        req = ModelRequest(data_parallel_rank=2)
        assert req.data_parallel_rank == 2

    def test_copy_preserves_dp_rank(self):
        req = ModelRequest(data_parallel_rank=3, rid="test-rid")
        copied = req.copy()
        assert copied.data_parallel_rank == 3
        assert copied.rid == "test-rid"

    def test_copy_none_dp_rank(self):
        req = ModelRequest()
        copied = req.copy()
        assert copied.data_parallel_rank is None


class TestSGLangPayload:
    def test_rid_in_payload(self):
        req = ModelRequest(rid="test-123", input_ids=[1, 2, 3])
        backend = SGLangBackend()
        http_req = backend.build_generation_request(req, with_lora=False, version=0)
        assert http_req.payload["rid"] == "test-123"

    def test_dp_rank_in_payload(self):
        req = ModelRequest(rid="test-123", input_ids=[1, 2, 3], data_parallel_rank=2)
        backend = SGLangBackend()
        http_req = backend.build_generation_request(req, with_lora=False, version=0)
        assert http_req.payload["data_parallel_rank"] == 2

    def test_dp_rank_none_not_in_payload(self):
        req = ModelRequest(rid="test-123", input_ids=[1, 2, 3])
        backend = SGLangBackend()
        http_req = backend.build_generation_request(req, with_lora=False, version=0)
        assert "data_parallel_rank" not in http_req.payload


class TestInteractionCacheRidBase:
    def test_default_values(self):
        cache = InteractionCache()
        assert cache.rid_base is None
        assert cache.sample_idx == 0

    def test_set_values(self):
        cache = InteractionCache()
        cache.rid_base = "django-123-0"
        cache.sample_idx = 1
        assert cache.rid_base == "django-123-0"
        assert cache.sample_idx == 1
