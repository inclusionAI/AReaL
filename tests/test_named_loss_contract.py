# SPDX-License-Identifier: Apache-2.0

"""Engine contracts, including bridges from main's scalar PPO reductions."""

from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.loss import get_named_loss_names, named_loss_normalizer
from areal.engine.core import (
    compute_microbatch_loss_weight,
    compute_total_loss_weight,
    normalize_loss,
)
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import TRANSPORT_DUMMY_KEY, pack_tensor_dict


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip("CPU Gloo is unavailable")
    rendezvous = tmp_path_factory.mktemp("named-loss") / "rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    yield dist.group.WORLD
    dist.destroy_process_group()


def test_named_schema_reads_partial_and_bound_method_without_calling():
    class Owner:
        @named_loss_normalizer(names=("policy", "aux"))
        def normalizer(self, batch):
            pytest.fail("Schema lookup must not execute callbacks")

    assert get_named_loss_names(partial(Owner().normalizer)) == ("aux", "policy")
    assert get_named_loss_names(lambda batch: None) is None


@pytest.mark.parametrize("names", [(), ("",), ("a", "a"), "policy", (1,)])
def test_named_schema_rejects_invalid_names(names):
    with pytest.raises(ValueError):
        named_loss_normalizer(names=names)


def test_transport_dummy_uses_declared_schema_without_callback():
    @named_loss_normalizer(names=("b", "a"))
    def normalizer(batch):
        pytest.fail("Transport data must not reach the normalizer")

    batch = {TRANSPORT_DUMMY_KEY: True, "input_ids": torch.ones(1)}
    weights = compute_microbatch_loss_weight(batch, normalizer)
    assert tuple(weights) == ("a", "b")
    for mass in weights.values():
        torch.testing.assert_close(mass, torch.tensor(0.0), rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor(-1.0),
        torch.tensor(float("nan")),
        torch.ones(2),
        torch.tensor(1.0, requires_grad=True),
    ],
)
def test_named_mass_rejects_invalid_values(bad):
    @named_loss_normalizer(names=("policy",))
    def normalizer(batch):
        return {"policy": bad}

    with pytest.raises((ValueError, RuntimeError)):
        compute_microbatch_loss_weight({}, normalizer)


def test_named_normalizer_rejects_missing_schema_or_mismatched_keys():
    with pytest.raises(ValueError, match="named_loss_normalizer"):
        compute_microbatch_loss_weight({}, lambda batch: {"a": torch.tensor(1.0)})

    @named_loss_normalizer(names=("a",))
    def wrong_keys(batch):
        return {"b": torch.tensor(1.0)}

    with pytest.raises(ValueError, match="declared names"):
        compute_microbatch_loss_weight({}, wrong_keys)


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("n_steps", [1, 2])
def test_scalar_to_named_bridge_preserves_main_loss_and_gradient(
    mode, packed, n_steps, process_group
):
    # Group A spans rows 0,1,2; optimizer steps split it after row 1.
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    data = {
        "input_ids": torch.ones_like(mask),
        "attention_mask": torch.ones_like(mask),
        "loss_mask": mask,
        "features": torch.arange(1, 13).float().reshape(4, 3),
    }
    prepared = prepare_policy_gradient_batch(
        data, mode=mode, divisor=4.0 if mode == "constant" else None, group_sizes=[3, 1]
    )
    schedule = (
        [data]
        if n_steps == 1
        else [{k: v[i : i + 2] for k, v in data.items()} for i in (0, 2)]
    )
    steps = prepared.for_steps(schedule, dp_group=process_group)
    for batch, step in zip(schedule, steps, strict=True):
        parts = [
            {k: v[i : i + 1] for k, v in batch.items()}
            for i in reversed(range(len(batch["loss_mask"])))
        ]
        if packed:
            parts = [pack_tensor_dict(part) for part in parts]
        parts.append({TRANSPORT_DUMMY_KEY: True, "input_ids": torch.ones(1)})

        @named_loss_normalizer(names=("policy",))
        def normalizer(part):
            return {"policy": step.loss_weight(part)}

        minibatches = SimpleNamespace(mbs=parts)
        scalar_total = compute_total_loss_weight(
            minibatches, step.loss_weight, process_group
        )
        named_total = compute_total_loss_weight(minibatches, normalizer, process_group)
        parameter = torch.tensor(0.3, requires_grad=True)
        scalar, named = parameter * 0, parameter * 0
        for part in parts:
            if part.get(TRANSPORT_DUMMY_KEY):
                continue
            reduction = step.bind(part)
            mask = part["loss_mask"].bool()
            # Narrow numerator support without changing original denominators.
            retained = mask & (part["features"].long() % 2 == 0)
            local = reduction.aggregate(
                (parameter * part["features"]).square(), retained
            )
            mass = reduction.normalizer()
            scalar = scalar + normalize_loss(local, mass, scalar_total)
            named = named + normalize_loss(
                {"policy": local * mass}, {"policy": mass}, named_total
            )
        torch.testing.assert_close(named, scalar, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            torch.autograd.grad(named, parameter, retain_graph=True)[0],
            torch.autograd.grad(scalar, parameter)[0],
            rtol=1e-6,
            atol=1e-7,
        )


def test_real_empty_prompt_step_keeps_positive_mass(process_group):
    data = {
        "loss_mask": torch.tensor([[1], [0]]),
        "input_ids": torch.ones(2, 1, dtype=torch.long),
    }
    prepared = prepare_policy_gradient_batch(data, mode="prompt_mean", group_sizes=[2])
    batches = [{key: value[i : i + 1] for key, value in data.items()} for i in range(2)]
    step = prepared.for_steps(batches, dp_group=process_group)[1]
    mass = step.loss_weight(batches[1])
    assert mass > 0
    value = torch.tensor([[1.0]], requires_grad=True)
    loss = step.bind(batches[1]).aggregate(value, batches[1]["loss_mask"])
    named = normalize_loss({"policy": loss * mass}, {"policy": mass}, {"policy": mass})
    named.backward()
    torch.testing.assert_close(value.grad, torch.zeros_like(value), rtol=0, atol=0)


def _distributed_named_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for dummy_rank in (False, True):
            if dummy_rank and rank == 1:
                batches = [{TRANSPORT_DUMMY_KEY: True, "input_ids": torch.ones(1)}]
            else:
                batches = [
                    {
                        "x": torch.tensor(float(1 + rank)),
                        "policy": torch.tensor(0.25 if rank == 0 else 0.0),
                        "aux": torch.tensor(0.0 if rank == 0 else 2.0),
                    }
                ]

            @named_loss_normalizer(names=("policy", "aux", "empty"))
            def normalizer(batch):
                assert not batch.get(TRANSPORT_DUMMY_KEY)
                return {
                    "policy": batch["policy"],
                    "aux": batch["aux"],
                    "empty": torch.tensor(0.0),
                }

            totals = compute_total_loss_weight(
                SimpleNamespace(mbs=batches), normalizer, dist.group.WORLD
            )
            theta = torch.tensor(0.7, requires_grad=True)
            loss = theta * 0
            for batch in batches:
                if batch.get(TRANSPORT_DUMMY_KEY):
                    continue
                pred = theta * batch["x"]
                numerators = {
                    "policy": pred.square() * batch["policy"],
                    "aux": 0.3 * (pred - 1).square() * batch["aux"],
                    "empty": pred * 0,
                }
                # Model engine DP multiplier followed by averaged distributed gradients.
                loss = loss + normalize_loss(
                    numerators, normalizer(batch), totals, multiplier=2
                )
            gradient = torch.autograd.grad(loss, theta)[0]
            dist.all_reduce(gradient)
            gradient /= 2
            actual = loss.detach().clone()
            dist.all_reduce(actual)
            actual /= 2
            ref = torch.tensor(0.7, requires_grad=True)
            expected = ref.square()
            if not dummy_rank:
                expected = expected + 0.3 * (2 * ref - 1).square()
            torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-7)
            expected_gradient = torch.autograd.grad(expected, ref)[0]
            torch.testing.assert_close(
                gradient, expected_gradient, rtol=1e-6, atol=1e-7
            )
            # SGD once, after the same global gradient, must match the oracle.
            torch.testing.assert_close(
                theta.detach() - 0.1 * gradient,
                ref.detach() - 0.1 * expected_gradient,
                rtol=1e-6,
                atol=1e-7,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo unavailable")
def test_named_losses_distributed_fractional_empty_and_transport_rank(tmp_path):
    mp.spawn(
        _distributed_named_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("bad", [torch.tensor(float("inf")), torch.ones(2)])
def test_named_numerator_rejects_nonfinite_or_nonscalar(bad):
    with pytest.raises((ValueError, RuntimeError)):
        normalize_loss({"a": bad}, {"a": torch.tensor(1.0)}, {"a": torch.tensor(1.0)})


def test_normalization_rejects_mixed_contracts_and_keys():
    one = torch.tensor(1.0)
    for loss, local, total in [
        ({"a": one}, one, one),
        (one, {"a": one}, {"a": one}),
        ({"a": one}, {"b": one}, {"a": one}),
    ]:
        with pytest.raises((TypeError, ValueError)):
            normalize_loss(loss, local, total)
