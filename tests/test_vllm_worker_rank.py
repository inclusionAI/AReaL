# SPDX-License-Identifier: Apache-2.0
"""XCCL ranks must include both internal DP and the external server offset."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.vllm


@pytest.mark.parametrize(
    ("dp_index", "worker_rank", "rank_offset", "expected_rank"),
    [(0, 2, 9, 11), (2, 3, 1, 12)],
    ids=["external-dp", "internal-dp"],
)
def test_weight_group_uses_physical_rank(
    monkeypatch, dp_index, worker_rank, rank_offset, expected_rank
):
    """Compose unique ranks from the DP index, replica rank and server offset."""
    pytest.importorskip("vllm")
    import areal.engine.vllm_ext.vllm_worker_extension as module

    initialize = Mock(return_value=object())
    monkeypatch.setattr(module, "init_custom_process_group", initialize)
    worker = module.VLLMWorkerExtension()
    worker.rank = worker_rank
    worker.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=4, data_parallel_index=dp_index)
        ),
        lora_manager=None,
    )

    success, message = worker.areal_init_update_weight_group(
        "127.0.0.1", "12345", rank_offset, 17, "gloo", "test-group"
    )

    assert success, message
    assert initialize.call_args.kwargs["rank"] == expected_rank
