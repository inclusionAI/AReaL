# SPDX-License-Identifier: Apache-2.0

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from areal.autopilot.elastic.runtime.state.checkpoint import (
    capture_runtime_state,
    restore_runtime_state,
)


def make_engine(device):
    parameter = torch.nn.Parameter(torch.ones(2, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=6
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    engine = SimpleNamespace(
        rank=0,
        world_size=4,
        device=torch.device(device),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        version=1,
    )
    engine.get_version = lambda: engine.version
    engine.set_version = lambda version: setattr(engine, "version", version)
    return engine


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_runtime_roundtrip_continues_all_random_streams(device):
    """A fresh optimizer/scheduler and RNG streams must resume the saved point."""
    engine = make_engine(device)
    state = json.loads(json.dumps(capture_runtime_state(engine)))
    optimizer = copy.deepcopy(engine.optimizer.state_dict())

    def draw():
        result = [random.random(), float(np.random.random()), torch.rand(3)]
        if device == "cuda":
            result.append(torch.rand(3, device="cuda").cpu())
        return result

    expected = draw()
    draw()
    restored = make_engine(device)
    restored.optimizer.load_state_dict(optimizer)
    restore_runtime_state(restored, state)
    actual = draw()

    assert actual[:2] == expected[:2]
    for a, b in zip(actual[2:], expected[2:]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert restored.lr_scheduler.state_dict() == engine.lr_scheduler.state_dict()
    assert restored.version == 1


def test_runtime_wrong_rank_rejected_before_restore():
    """Never load another rank's random stream by accident."""
    engine = make_engine("cpu")
    state = capture_runtime_state(engine)
    state["rank"] = 1

    with pytest.raises(ValueError, match="rank/world"):
        restore_runtime_state(engine, state)


def test_runtime_lr_mismatch_rejected():
    """Restoring the scheduler alone cannot hide a missing optimizer restore."""
    engine = make_engine("cpu")
    state = capture_runtime_state(engine)
    engine.optimizer.param_groups[0]["lr"] *= 2

    with pytest.raises(ValueError, match="learning rates"):
        restore_runtime_state(engine, state)
