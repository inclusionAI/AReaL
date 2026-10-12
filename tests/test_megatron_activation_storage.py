# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from areal.engine import megatron_engine as engine_module
from areal.engine.megatron_utils.activation_storage import ActivationStorageTracker


@pytest.mark.parametrize("awex_residency", [False, True])
def test_training_tracks_activation_storage_only_for_awex(monkeypatch, awex_residency):
    """Ordinary training bypasses tracking; AWEX releases after one backward."""
    engine = engine_module.MegatronEngine.__new__(engine_module.MegatronEngine)
    engine._weight_residency = (
        SimpleNamespace(ensure_grad_buffers=lambda: None) if awex_residency else None
    )
    engine.model = [torch.nn.Linear(2, 2)]
    engine.model[0].config = SimpleNamespace(calculate_per_token_loss=True)
    engine.hf_config = None
    engine._ensure_ready = lambda: None
    engine._normalize_batch_input = lambda batch: (batch, None)
    engine.optimizer_zero_grad = lambda: None
    engine._prepare_mb_list = lambda *args, **kwargs: SimpleNamespace(mbs=[{}])
    engine._collect_mtp_loss = lambda count: None
    calls = []
    engine.forward_backward_batch = lambda *args, **kwargs: calls.append("backward")

    def optimizer_step():
        calls.append("optimizer")
        return {}

    engine.optimizer_step = optimizer_step
    monkeypatch.setattr(
        engine_module, "record_training_batch", lambda *args: nullcontext()
    )
    monkeypatch.setattr(
        engine_module, "tensor_container_to", lambda value, device: value
    )
    monkeypatch.setattr(
        engine_module,
        "install_activation_storage_tracking",
        lambda: calls.append("install"),
    )

    def release():
        calls.append("release")
        return 0, 0

    @contextmanager
    def tracking(protected):
        assert {id(t) for t in protected} == {
            id(t) for t in engine.model[0].parameters()
        }
        yield SimpleNamespace(release=release)

    monkeypatch.setattr(engine_module, "track_activation_storage", tracking)
    engine.train_batch({}, lambda *args: None, lambda *args: None)
    assert calls == (
        ["install", "backward", "release", "optimizer"]
        if awex_residency
        else ["backward", "optimizer"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tracker_releases_activation_and_preserves_model_alias():
    """A tracked view of model storage must stay resident after release."""
    model_param = torch.nn.Parameter(torch.ones(8, device="cuda"))
    activation = torch.ones(8, device="cuda")
    tracker = ActivationStorageTracker([model_param])
    tracker.append(model_param.view(2, 4))
    tracker.append(activation)

    released, size = tracker.release()

    assert released == 1
    assert size == 8 * activation.element_size()
    assert activation.untyped_storage().nbytes() == 0
    torch.testing.assert_close(
        model_param, torch.ones_like(model_param), rtol=0, atol=0
    )
