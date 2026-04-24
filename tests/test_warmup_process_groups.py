"""Unit tests for ``warmup_process_groups``.

The collective warmup itself needs a distributed environment, but the
no-op short-circuits can be exercised with plain unit tests so we catch
regressions in the guard logic without requiring a GPU.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch.distributed as dist

from areal.engine.core.distributed import warmup_process_groups


@pytest.fixture
def mock_all_reduce():
    with patch.object(dist, "all_reduce") as m:
        yield m


def test_noop_when_dist_uninitialized(mock_all_reduce):
    with patch.object(dist, "is_initialized", return_value=False):
        warmup_process_groups(object(), object())
    mock_all_reduce.assert_not_called()


def test_noop_on_cpu_platform(mock_all_reduce):
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch("areal.infra.platforms.current_platform") as platform,
    ):
        platform.device_type = "cpu"
        warmup_process_groups(object(), object())
    mock_all_reduce.assert_not_called()


def test_noop_when_all_groups_are_none(mock_all_reduce):
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch("areal.infra.platforms.current_platform") as platform,
    ):
        platform.device_type = "cuda"
        warmup_process_groups(None, None)
    mock_all_reduce.assert_not_called()


def test_noop_with_no_arguments(mock_all_reduce):
    warmup_process_groups()
    mock_all_reduce.assert_not_called()


def test_dedupes_repeated_groups(mock_all_reduce, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    group = object()
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch("areal.infra.platforms.current_platform") as platform,
        patch("torch.zeros") as zeros,
    ):
        platform.device_type = "cuda"
        platform.set_device = lambda _: None
        zeros.return_value = object()
        warmup_process_groups(group, group, None, group)

    assert mock_all_reduce.call_count == 1
    kwargs = mock_all_reduce.call_args.kwargs
    assert kwargs["group"] is group
