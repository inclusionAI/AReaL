"""Tests for areal.utils.rollout_id.RolloutIdBuilder."""

import pytest

from areal.utils.rollout_id import RolloutIdBuilder


@pytest.fixture(autouse=True)
def reset_counter():
    """Reset the global counter before each test."""
    RolloutIdBuilder.reset()
    yield
    RolloutIdBuilder.reset()


class TestFillRidBase:
    def test_basic_query_id(self):
        data = {"query_id": "gsm8k-42", "prompt": "What is 2+2?"}
        RolloutIdBuilder.fill_rid_base(data)
        assert data[RolloutIdBuilder.EXT_QID_FIELD] == ["gsm8k-42-0"]

    def test_counter_increments(self):
        data1 = {"query_id": "q1"}
        data2 = {"query_id": "q1"}
        data3 = {"query_id": "q1"}
        RolloutIdBuilder.fill_rid_base(data1)
        RolloutIdBuilder.fill_rid_base(data2)
        RolloutIdBuilder.fill_rid_base(data3)
        assert data1[RolloutIdBuilder.EXT_QID_FIELD] == ["q1-0"]
        assert data2[RolloutIdBuilder.EXT_QID_FIELD] == ["q1-1"]
        assert data3[RolloutIdBuilder.EXT_QID_FIELD] == ["q1-2"]

    def test_different_qids_independent_counters(self):
        data_a = {"query_id": "a"}
        data_b = {"query_id": "b"}
        data_a2 = {"query_id": "a"}
        RolloutIdBuilder.fill_rid_base(data_a)
        RolloutIdBuilder.fill_rid_base(data_b)
        RolloutIdBuilder.fill_rid_base(data_a2)
        assert data_a[RolloutIdBuilder.EXT_QID_FIELD] == ["a-0"]
        assert data_b[RolloutIdBuilder.EXT_QID_FIELD] == ["b-0"]
        assert data_a2[RolloutIdBuilder.EXT_QID_FIELD] == ["a-1"]

    def test_fallback_keys(self):
        """Test qid extraction from different key names."""
        for key in ("qid", "id", "instance_id"):
            RolloutIdBuilder.reset()
            data = {key: "test-val"}
            RolloutIdBuilder.fill_rid_base(data)
            assert data[RolloutIdBuilder.EXT_QID_FIELD] == ["test-val-0"]

    def test_fallback_hash(self):
        """When no ID key exists, hash the prompt."""
        data = {"prompt": "What is 2+2?"}
        RolloutIdBuilder.fill_rid_base(data)
        rid_base = data[RolloutIdBuilder.EXT_QID_FIELD][0]
        # Should be "{16-char-hash}-0"
        parts = rid_base.rsplit("-", 1)
        assert parts[1] == "0"
        assert len(parts[0]) == 16

    def test_priority_order(self):
        """query_id takes priority over qid, id, instance_id."""
        data = {"query_id": "winner", "qid": "loser", "id": "also-loser"}
        RolloutIdBuilder.fill_rid_base(data)
        assert data[RolloutIdBuilder.EXT_QID_FIELD] == ["winner-0"]


class TestGetRidBase:
    def test_returns_rid_base(self):
        data = {"query_id": "q1"}
        RolloutIdBuilder.fill_rid_base(data)
        assert RolloutIdBuilder.get_rid_base(data) == "q1-0"

    def test_returns_uuid_when_missing(self):
        data = {}
        rid_base = RolloutIdBuilder.get_rid_base(data)
        # Should be a UUID string
        assert len(rid_base) == 36  # UUID format


class TestBuildRid:
    def test_without_round(self):
        rid = RolloutIdBuilder.build_rid("django-123-0", sample_idx=0)
        assert rid == "django-123-0_0"

    def test_with_round(self):
        rid = RolloutIdBuilder.build_rid("django-123-0", sample_idx=0, round_idx=2)
        assert rid == "django-123-0-r-2_0"

    def test_different_samples(self):
        rid0 = RolloutIdBuilder.build_rid("q-0", sample_idx=0, round_idx=1)
        rid1 = RolloutIdBuilder.build_rid("q-0", sample_idx=1, round_idx=1)
        assert rid0 == "q-0-r-1_0"
        assert rid1 == "q-0-r-1_1"
        assert rid0 != rid1


class TestInferRoundIdx:
    def test_no_assistant(self):
        messages = [{"role": "user", "content": "hi"}]
        assert RolloutIdBuilder.infer_round_idx(messages) == 0

    def test_one_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert RolloutIdBuilder.infer_round_idx(messages) == 1

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how?"},
            {"role": "assistant", "content": "fine"},
            {"role": "user", "content": "bye"},
        ]
        assert RolloutIdBuilder.infer_round_idx(messages) == 2

    def test_empty(self):
        assert RolloutIdBuilder.infer_round_idx([]) == 0


class TestParseRoutingKey:
    def test_with_round(self):
        assert (
            RolloutIdBuilder.parse_routing_key("django-123-0-r-2_0") == "django-123-0_0"
        )

    def test_without_round(self):
        assert RolloutIdBuilder.parse_routing_key("django-123-0_0") == "django-123-0_0"

    def test_different_rounds_same_routing_key(self):
        """Different rounds should produce the same routing key."""
        key0 = RolloutIdBuilder.parse_routing_key("q-0-r-0_0")
        key1 = RolloutIdBuilder.parse_routing_key("q-0-r-1_0")
        key2 = RolloutIdBuilder.parse_routing_key("q-0-r-5_0")
        assert key0 == key1 == key2 == "q-0_0"

    def test_different_samples_different_routing_key(self):
        key0 = RolloutIdBuilder.parse_routing_key("q-0-r-1_0")
        key1 = RolloutIdBuilder.parse_routing_key("q-0-r-1_1")
        assert key0 != key1
        assert key0 == "q-0_0"
        assert key1 == "q-0_1"

    def test_plain_uuid(self):
        """UUID-style rid should pass through unchanged."""
        uuid_rid = "550e8400-e29b-41d4-a716-446655440000"
        assert RolloutIdBuilder.parse_routing_key(uuid_rid) == uuid_rid


class TestComputeDpRank:
    def test_dp_size_1(self):
        assert RolloutIdBuilder.compute_dp_rank("any-rid", dp_size=1) == 0

    def test_deterministic(self):
        """Same rid should always produce the same rank."""
        rid = "django-123-0-r-2_0"
        rank1 = RolloutIdBuilder.compute_dp_rank(rid, dp_size=4)
        rank2 = RolloutIdBuilder.compute_dp_rank(rid, dp_size=4)
        assert rank1 == rank2

    def test_same_routing_key_same_rank(self):
        """Different rounds of the same episode → same DP rank."""
        rank_r0 = RolloutIdBuilder.compute_dp_rank("q-0-r-0_0", dp_size=4)
        rank_r1 = RolloutIdBuilder.compute_dp_rank("q-0-r-1_0", dp_size=4)
        rank_r2 = RolloutIdBuilder.compute_dp_rank("q-0-r-2_0", dp_size=4)
        assert rank_r0 == rank_r1 == rank_r2

    def test_rank_in_range(self):
        """Rank should always be in [0, dp_size)."""
        for dp_size in (2, 4, 8):
            for i in range(100):
                rank = RolloutIdBuilder.compute_dp_rank(f"q-{i}-0_0", dp_size=dp_size)
                assert 0 <= rank < dp_size

    def test_distribution(self):
        """Ranks should be roughly evenly distributed across many keys."""
        dp_size = 4
        counts = [0] * dp_size
        for i in range(1000):
            rank = RolloutIdBuilder.compute_dp_rank(f"q-{i}-0_0", dp_size=dp_size)
            counts[rank] += 1
        # Each bucket should get roughly 250 ± 100
        for c in counts:
            assert c > 100, f"Uneven distribution: {counts}"


class TestEndToEnd:
    """Test the full flow: fill_rid_base → build_rid → routing."""

    def test_multi_turn_affinity(self):
        """Simulate a multi-turn agent episode. All rounds should route to same DP rank."""
        data = {"instance_id": "swe-bench-123"}
        RolloutIdBuilder.fill_rid_base(data)
        rid_base = RolloutIdBuilder.get_rid_base(data)

        dp_size = 4
        ranks = []
        for round_idx in range(5):
            rid = RolloutIdBuilder.build_rid(
                rid_base, sample_idx=0, round_idx=round_idx
            )
            rank = RolloutIdBuilder.compute_dp_rank(rid, dp_size)
            ranks.append(rank)
            # Also verify routing key is the same
            routing_key = RolloutIdBuilder.parse_routing_key(rid)
            assert routing_key == f"{rid_base}_0"

        # All rounds should route to the same DP rank
        assert len(set(ranks)) == 1

    def test_different_episodes_can_differ(self):
        """Different episodes of the same qid get different rid_base."""
        data1 = {"query_id": "math-1"}
        data2 = {"query_id": "math-1"}
        RolloutIdBuilder.fill_rid_base(data1)
        RolloutIdBuilder.fill_rid_base(data2)
        rid_base1 = RolloutIdBuilder.get_rid_base(data1)
        rid_base2 = RolloutIdBuilder.get_rid_base(data2)
        assert rid_base1 == "math-1-0"
        assert rid_base2 == "math-1-1"
        assert rid_base1 != rid_base2
