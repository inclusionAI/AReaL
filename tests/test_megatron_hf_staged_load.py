# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api import SaveLoadMeta
from areal.engine.megatron_engine import MegatronEngine
from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for staged AdamW"
)


def _make_engine(param: torch.nn.Parameter, optimizer: GPUStagedAdamW):
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.optimizer = SimpleNamespace(optimizer=optimizer)
    engine.is_offload = False
    engine._weight_residency = None
    model = torch.nn.Module()
    model.register_parameter("weight", param)
    engine.model = [model]
    engine.lr_scheduler = object()
    engine.checkpointer = MegatronCheckpointManager.__new__(MegatronCheckpointManager)
    engine.checkpointer.optimizer = engine.optimizer
    engine.checkpointer.load_failed = False
    return engine


@pytest.mark.parametrize("slots_released", [False, True])
def test_hf_reload_rebuilds_staged_state_and_next_update(slots_released: bool) -> None:
    """A model-only HF reload continues from loaded weights, not old masters."""
    param = torch.nn.Parameter(torch.ones(23, device="cuda", dtype=torch.bfloat16))
    kwargs = dict(lr=0.02, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.07)
    optimizer = GPUStagedAdamW(
        [param],
        staged_config=GPUStagedAdamWConfig(
            buffer_count=2, bucket_size_mb=7 * 4 / (1024 * 1024)
        ),
        **kwargs,
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    param.decoupled_grad = torch.ones_like(param)
    optimizer.step()
    optimizer.drain()
    assert optimizer.param_groups[0]["step"] == 1
    if slots_released:
        optimizer.offload_to_cpu()
    engine = _make_engine(param, optimizer)
    scheduler = engine.lr_scheduler
    loaded = torch.linspace(-2, 2, 23, device="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def load_weights(path):
        param.copy_(loaded)

    engine._load_model_from_hf = load_weights
    engine.load(SaveLoadMeta(path="model", weight_format="hf", with_optim=False))

    assert next(engine.model[0].parameters()) is param
    assert engine.lr_scheduler is scheduler
    assert optimizer.param_groups[0]["step"] == 0
    assert optimizer.param_groups[0]["lr"] == kwargs["lr"]
    assert bool(optimizer._slots) is not slots_released
    torch.testing.assert_close(
        optimizer.cpu_slabs.master, loaded.cpu().float(), rtol=0.0, atol=0.0
    )
    for moment in (optimizer.cpu_slabs.exp_avg, optimizer.cpu_slabs.exp_avg_sq):
        torch.testing.assert_close(moment, torch.zeros_like(moment), rtol=0.0, atol=0.0)

    optimizer.restore_from_cpu()
    baseline_param = torch.nn.Parameter(loaded.float().clone())
    baseline = torch.optim.AdamW([baseline_param], **kwargs)
    grad = torch.linspace(-1, 1, 23, device="cuda", dtype=torch.bfloat16)
    param.decoupled_grad = grad
    baseline_param.grad = grad.float()
    optimizer.step()
    optimizer.drain()
    baseline.step()
    torch.testing.assert_close(param, baseline_param.bfloat16(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        optimizer.cpu_slabs.master, baseline_param.detach().cpu(), rtol=2e-6, atol=2e-6
    )


def test_partial_hf_reload_failure_stops_staged_updates() -> None:
    """A failed in-place model load cannot continue with stale optimizer state."""
    param = torch.nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    optimizer = GPUStagedAdamW([param])
    optimizer.bind_owned_params(optimizer.param_groups)
    engine = _make_engine(param, optimizer)

    @torch.no_grad()
    def failed_load(path):
        param[:4].zero_()
        raise RuntimeError("HF load failed")

    engine._load_model_from_hf = failed_load
    with pytest.raises(RuntimeError, match="HF load failed"):
        engine.load(SaveLoadMeta(path="model", weight_format="hf", with_optim=False))
    assert engine.checkpointer.load_failed
    param.decoupled_grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="checkpoint load|restart"):
        optimizer.step()
