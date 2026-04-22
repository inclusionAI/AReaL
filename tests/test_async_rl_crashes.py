from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from areal.infra.controller.train_controller import TrainController
from areal.utils.seqpack import balanced_greedy_partition


def test_balanced_greedy_partition_handles_remainders():
    groups = balanced_greedy_partition([10, 8, 6, 4, 2], K=3)

    assert len(groups) == 3
    assert sorted(len(group) for group in groups) == [1, 2, 2]
    assert sorted(idx for group in groups for idx in group) == [0, 1, 2, 3, 4]


def test_custom_function_call_trims_padded_eval_items():
    controller = TrainController.__new__(TrainController)
    controller.train_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=4))

    sample = {"attention_mask": torch.ones(1, dtype=torch.long)}
    captured: dict[str, int] = {}

    def fake_prepare_dispatch(*args, **kwargs):
        captured["padded_len"] = len(args[0])
        return [], {}, None

    controller._prepare_dispatch = fake_prepare_dispatch
    controller._call_workers = lambda *args, **kwargs: []
    controller._collect_results = (
        lambda results, group_indices: list(range(captured["padded_len"]))
    )

    with patch(
        "areal.infra.controller.train_controller.make_dummy_eval_item",
        lambda template: {"attention_mask": torch.zeros_like(template["attention_mask"])},
    ), patch(
        "areal.infra.controller.train_controller.run_async_task",
        lambda fn, *args, **kwargs: [],
    ):
        result = controller._custom_function_call("noop", [sample] * 5)

    assert captured["padded_len"] == 8
    assert result == [0, 1, 2, 3, 4]
