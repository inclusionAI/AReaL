# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from areal.v2.weight_update.gateway.config import PairInfo
from areal.v2.weight_update.gateway.pair_registry import PairRegistry


@pytest.fixture
def registry() -> PairRegistry:
    return PairRegistry()


@pytest.fixture
def sample_pair_info() -> PairInfo:
    return PairInfo(
        pair_name="test_pair",
        train_worker_urls=["http://train:8000"],
        inference_worker_urls=["http://infer:9000"],
        train_world_size=4,
        inference_world_size=2,
    )


def test_get_by_name_returns_pair_info(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    registry.register(sample_pair_info)
    retrieved = registry.get_by_name("test_pair")
    assert retrieved is sample_pair_info


def test_get_by_name_returns_none_for_unregistered_name(registry: PairRegistry) -> None:
    result = registry.get_by_name("nonexistent_pair")
    assert result is None


def test_register_raises_for_duplicate_pair_name(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    registry.register(sample_pair_info)

    duplicate_pair = PairInfo(
        pair_name="test_pair",
        train_worker_urls=["http://train:8000"],
        inference_worker_urls=["http://infer:9000"],
    )

    with pytest.raises(ValueError, match="Pair 'test_pair' already registered"):
        registry.register(duplicate_pair)


def test_unregister_removes_pair(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    registry.register(sample_pair_info)
    removed = registry.unregister("test_pair")

    assert removed is sample_pair_info
    assert registry.get_by_name("test_pair") is None


def test_unregister_returns_none_for_nonexistent_pair(registry: PairRegistry) -> None:
    result = registry.unregister("nonexistent_pair")
    assert result is None


def test_list_pairs_returns_all_names(registry: PairRegistry) -> None:
    pair1 = PairInfo(
        pair_name="pair_1",
        train_worker_urls=["http://train1:8000"],
        inference_worker_urls=["http://infer1:9000"],
    )
    pair2 = PairInfo(
        pair_name="pair_2",
        train_worker_urls=["http://train2:8000"],
        inference_worker_urls=["http://infer2:9000"],
    )

    registry.register(pair1)
    registry.register(pair2)

    pairs = registry.list_pairs()
    assert set(pairs) == {"pair_1", "pair_2"}


def test_list_pairs_returns_empty_for_no_pairs(registry: PairRegistry) -> None:
    pairs = registry.list_pairs()
    assert pairs == []


def test_list_pairs_after_unregister(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    registry.register(sample_pair_info)
    assert registry.list_pairs() == ["test_pair"]

    registry.unregister("test_pair")
    assert registry.list_pairs() == []


def test_connection_reservation_must_exist_before_commit():
    registry = PairRegistry()
    pair = PairInfo("reserved", ["http://train"], ["http://infer"])
    with pytest.raises(RuntimeError, match="reservation"):
        registry.finish_connect(pair)
    registry.register(
        PairInfo(
            "reserved",
            pair.train_worker_urls,
            pair.inference_worker_urls,
            state="connecting",
        )
    )
    registry.finish_connect(pair)
    assert registry.get_by_name("reserved") is pair
    with pytest.raises(RuntimeError, match="reservation"):
        registry.finish_connect(pair)


def test_worker_pair_identity_survives_adapter_replacement():
    from areal.v2.weight_update.awex.lifecycle import WeightUpdateLifecycle

    lifecycle = WeightUpdateLifecycle()
    assert lifecycle.begin("old", fenced=True)
    lifecycle.connected()
    assert not lifecycle.begin("old", fenced=True)
    lifecycle.require("old")
    with pytest.raises(RuntimeError, match="inactive"):
        lifecycle.require(None)
    with pytest.raises(RuntimeError, match="inactive"):
        lifecycle.require("new")
    assert lifecycle.begin_teardown("old")
    with pytest.raises(RuntimeError, match="inactive"):
        lifecycle.require("old")
    lifecycle.closed()
    assert lifecycle.begin("new", fenced=True)
    lifecycle.connected()
    assert not lifecycle.begin_teardown("old")
    with pytest.raises(RuntimeError, match="retired"):
        lifecycle.begin("old", fenced=True)
    lifecycle.require("new")


def test_uncertain_worker_initialization_requires_teardown_before_retry():
    from areal.v2.weight_update.awex.lifecycle import WeightUpdateLifecycle

    lifecycle = WeightUpdateLifecycle()
    assert lifecycle.begin("partial", fenced=True)
    with pytest.raises(RuntimeError, match="cleanup"):
        lifecycle.begin("partial", fenced=True)
    with pytest.raises(RuntimeError, match="inactive"):
        lifecycle.require("partial")
    assert lifecycle.begin_teardown("partial")
    lifecycle.closed()
    assert lifecycle.begin("replacement", fenced=True)


def test_teardown_before_delayed_init_retires_that_identity():
    from areal.v2.weight_update.awex.lifecycle import WeightUpdateLifecycle

    lifecycle = WeightUpdateLifecycle()
    assert not lifecycle.begin_teardown("delayed")
    with pytest.raises(RuntimeError, match="retired"):
        lifecycle.begin("delayed", fenced=True)
