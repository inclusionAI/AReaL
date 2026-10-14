# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for the engine's Megatron loss callback contracts."""

import functools
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.test_megatron_bridge_config import _load_function

ENGINE_SOURCE = "areal/engine/megatron_engine.py"


class _MicroBatches(list):
    @property
    def mbs(self):
        return self

    def to(self, device):
        return self


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp_size", [1, 8])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("n_mbs", [1, 3])
@pytest.mark.parametrize("loss_scale", [1.0, 128.0])
def test_train_gradient_matches_global_mean(
    per_token, cp_size, dp_size, n_mbs, loss_scale
):
    """Unequal and empty microbatches retain one global normalization."""
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    batches = [
        _MicroBatches(
            {"weight": torch.tensor(float(w)), "target": float(rank + i)}
            for i, w in enumerate([7] if n_mbs == 1 else [3, 0, 4 + rank])
        )
        for rank in range(dp_size)
    ]

    def weight_fn(inputs):
        return inputs["weight"]

    total_weight = sum(weight_fn(mb) for batch in batches for mb in batch)
    reference = (
        sum(
            (parameter - mb["target"]).square() * weight_fn(mb)
            for batch in batches
            for mb in batch
        )
        / total_weight
    )
    reference_grad = torch.autograd.grad(reference, parameter)[0]
    scheduled_losses, counts = [], []

    for batch in batches:
        for cp_rank in range(cp_size):
            configs = [
                SimpleNamespace(
                    calculate_per_token_loss=per_token, grad_scale_func=None
                )
                for _ in range(2)
            ]
            namespace = {
                "torch": torch,
                "get_model_config": lambda model: model.config,
                "mpu": SimpleNamespace(
                    get_data_parallel_group=lambda **kw: None,
                    get_data_parallel_world_size=lambda: dp_size,
                    get_context_parallel_world_size=lambda: cp_size,
                    get_context_parallel_rank=lambda: cp_rank,
                ),
                "compute_total_loss_weight": lambda *args: total_weight * cp_size,
            }
            train = _load_function(ENGINE_SOURCE, "train_batch", namespace)
            compute = _load_function(
                ENGINE_SOURCE, "_compute_logprobs_and_loss", namespace
            )
            engine = SimpleNamespace(
                _ensure_ready=lambda: None,
                optimizer_zero_grad=lambda: None,
                optimizer_step=lambda: {},
                _normalize_batch_input=lambda inputs: (inputs, None),
                _prepare_mb_list=lambda inputs: batch,
                model=[SimpleNamespace(config=config) for config in configs],
                # Bridge can mutate the model config without changing tf_config.
                tf_config=SimpleNamespace(calculate_per_token_loss=not per_token),
                config=SimpleNamespace(is_critic=True),
                enable_tree_training=False,
                device="cpu",
                optimizer=SimpleNamespace(
                    get_loss_scale=lambda: torch.tensor(loss_scale),
                    scale_loss=lambda loss: loss * loss_scale,
                ),
            )
            engine._compute_logprobs_and_loss = lambda *a, **kw: compute(
                engine, *a, **kw
            )

            def forward_backward(mbs, process_output, forward_only):
                assert not forward_only
                for mb in mbs:
                    result = process_output(parameter.reshape(1), mb)
                    if per_token:
                        loss, count = result
                        assert count.dtype == torch.int64
                        counts.append(count)
                        assert all(c.grad_scale_func is not None for c in configs)
                        scheduled_losses.append(configs[0].grad_scale_func(loss))
                    else:
                        assert all(c.grad_scale_func is None for c in configs)
                        # MCore two-value callback and DDP averaging.
                        scheduled_losses.append(
                            result * cp_size / n_mbs / (dp_size * cp_size)
                        )

            engine.forward_backward_batch = forward_backward
            train(
                engine,
                {},
                lambda output, mb: (output - mb["target"]).square().mean(),
                weight_fn,
            )

    loss = sum(scheduled_losses)
    if per_token:
        count = sum(counts)
        torch.testing.assert_close(count.float(), total_weight, rtol=0, atol=0)
        loss = loss / count
    gradient = torch.autograd.grad(loss / loss_scale, parameter)[0]
    torch.testing.assert_close(gradient, reference_grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("with_count", [False, True])
def test_forward_callback_preserves_loss_contract(with_count):
    """Per-token training emits three values; eval and forward keep two."""
    output = torch.tensor([2.0])
    mb = SimpleNamespace(
        orig_mb={}, padded_mb={}, padding_length=0, old_cu_seqlens=None
    )
    batches = _MicroBatches([mb])
    batches.max_seqlen = 1
    captured = []

    def schedule(**kwargs):
        result, callback = kwargs["forward_step_func"](
            kwargs["data_iterator"], kwargs["model"]
        )
        captured.append(callback(result))

    forward = _load_function(
        ENGINE_SOURCE,
        "forward_backward_batch",
        {
            "torch": torch,
            "functools": functools,
            "mpu": SimpleNamespace(
                get_context_parallel_world_size=lambda: 1,
                is_pipeline_last_stage=lambda **kw: True,
            ),
            "packed_context_parallel_forward": lambda *a, **kw: output,
            "unpad_logits": lambda output, **kw: output,
            "get_forward_backward_func": lambda: schedule,
            "trace_scope": lambda *args: nullcontext(),
        },
    )
    engine = SimpleNamespace(
        _ensure_ready=lambda: None,
        enable_tree_training=False,
        is_vision_model=False,
        use_padded_seq=False,
        use_model_packed_seq=False,
        model=[SimpleNamespace()],
    )
    count = torch.tensor(3)
    forward(
        engine,
        batches,
        lambda output, inputs: (output.sum(), count) if with_count else output.sum(),
        forward_only=not with_count,
    )
    result = captured[0]
    assert len(result) == (3 if with_count else 2)
    assert result[-1] == {}
    torch.testing.assert_close(result[0], output.sum(), rtol=0, atol=0)
    if with_count:
        assert result[1] is count
