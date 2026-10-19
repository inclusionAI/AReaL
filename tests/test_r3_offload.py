# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine import megatron_engine as module


@pytest.fixture
def offload_engine(monkeypatch):
    engine = object.__new__(module.MegatronEngine)
    engine._weight_residency = None
    engine.is_offload = False
    engine._cpu_group = object()
    engine.process_group_initialized = True
    engine.mcore_config = SimpleNamespace(disable_grad_buffers_cpu_backup=True)
    engine.get_device_stats = Mock(return_value=Mock())
    engine._log_weight_residency_stats = Mock()
    ddp = Mock()
    monkeypatch.setattr(module, "DDP", type(ddp))
    engine.model = [ddp]
    platform, saver, barrier = Mock(), Mock(), Mock()
    monkeypatch.setattr(module, "current_platform", platform)
    monkeypatch.setattr(module, "torch_memory_saver", saver)
    monkeypatch.setattr(module, "is_tms_enabled", lambda: True)
    monkeypatch.setattr(module.dist, "barrier", barrier)
    return engine, saver, barrier, ddp


def test_repeated_tms_offload_releases_memory_once(offload_engine):
    engine, saver, barrier, ddp = offload_engine
    engine.offload()
    engine.offload()
    assert engine.is_offload
    saver.pause.assert_called_once()
    ddp.offload_grad_buffers.assert_called_once()
    barrier.assert_called_once_with(group=engine.cpu_group)


def test_tms_onload_allows_next_offload(offload_engine):
    engine, saver, _, ddp = offload_engine
    engine.offload()
    engine.onload()
    assert not engine.is_offload
    engine.offload()
    assert saver.pause.call_count == 2
    saver.resume.assert_called_once()
    ddp.restore_grad_buffers.assert_called_once()


def test_explicit_residency_still_releases_tags_when_offloaded(offload_engine):
    engine, saver, barrier, _ = offload_engine
    engine._weight_residency = Mock()
    engine.is_offload = True
    engine.offload()
    engine._weight_residency.release_memory.assert_called_once_with(
        tags=["optimizer", "weights"]
    )
    barrier.assert_called_once_with(group=engine.cpu_group)
    saver.pause.assert_not_called()


def test_first_offload_requires_tms_enabled(offload_engine, monkeypatch):
    engine, saver, _, _ = offload_engine
    monkeypatch.setattr(module, "is_tms_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="enable_offload=True"):
        engine.offload()
    saver.pause.assert_not_called()
