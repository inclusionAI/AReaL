# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import torch

from areal.api import TrainEngine
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import logging, stats_tracker
from areal.utils.data import batched_call
from areal.utils.functional import dpo_pair_logratios, dpo_preference_loss
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("DPOEngine")


def _dpo_valid_pairs(x: dict[str, Any]) -> torch.Tensor:
    seqlens = x["cu_seqlens"][1:] - x["cu_seqlens"][:-1]
    return seqlens.view(-1, 2).ne(0).all(dim=1)


def _dpo_loss_weight(x: dict[str, Any]) -> torch.Tensor:
    return _dpo_valid_pairs(x).count_nonzero().float()


def _log_empty_dpo_stats(device: torch.device) -> None:
    n_pairs = torch.zeros(1, dtype=torch.bool, device=device)
    stats_tracker.denominator(n_pairs=n_pairs)
    stats_tracker.stat(
        loss=torch.zeros(1, dtype=torch.float32, device=device),
        chosen_reward=torch.zeros(1, dtype=torch.float32, device=device),
        rejected_reward=torch.zeros(1, dtype=torch.float32, device=device),
        reward_accuracy=torch.zeros(1, dtype=torch.float32, device=device),
        reward_margin=torch.zeros(1, dtype=torch.float32, device=device),
        denominator="n_pairs",
    )


class DPOEngine:
    def __init__(
        self,
        engine: TrainEngine,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        self.engine = engine
        self.beta = beta
        self.loss_type = loss_type

    @trace_perf("dpo_engine.train_dpo", category="compute")
    @stats_tracker.scope_func_wrapper("dpo")
    def train_dpo(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._train_dpo, data, unpack=False)

    def _train_dpo(self, data: dict[str, Any]) -> None:
        self.engine.train()
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=functools.partial(
                compute_dpo_loss, beta=self.beta, loss_type=self.loss_type
            ),
            loss_weight_fn=_dpo_loss_weight,
        )
        stats_tracker.scalar(**stats)

    @trace_perf("dpo_engine.evaluate_dpo", category="compute")
    @stats_tracker.scope_func_wrapper("dpo-eval")
    def evaluate_dpo(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._evaluate_dpo, data, unpack=False)

    def _evaluate_dpo(self, data: dict[str, Any]) -> None:
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=functools.partial(
                compute_dpo_loss, beta=self.beta, loss_type=self.loss_type
            ),
            loss_weight_fn=_dpo_loss_weight,
        )

    @trace_perf("dpo_engine.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        """Compute per-token log-probabilities. Used by the ref engine to
        provide reference log-probs for DPO training."""
        return batched_call(self._compute_logp, data)

    def _compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )


class DPOController(TrainController):
    def train_dpo(self, *args, **kwargs):
        # dpo_collate produces 2 sequences (chosen + rejected) per example;
        # group_size=2 keeps each pair on the same DP rank.
        self._custom_function_call(
            "train_dpo", *args, group_size=2, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_dpo(self, *args, **kwargs):
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=2)
        self._custom_function_call(
            "evaluate_dpo",
            *args,
            group_size=2,
            rpc_meta={"broadcast": True},
            **kwargs,
        )

    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_logp", *args, group_size=2, rpc_meta={"broadcast": True}, **kwargs
        )


class DPOControllerV2(GatewayTrainController):
    def train_dpo(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/dpo/train", payload)

    def evaluate_dpo(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("group_size", 2)
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/dpo/evaluate", payload)

    def compute_logp(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("group_size", 2)
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/dpo/compute_logp", payload)


def compute_dpo_loss(
    logprobs: torch.Tensor,
    entropy: torch.Tensor | None,
    input_: dict[str, Any],
    *,
    beta: float,
    loss_type: str = "sigmoid",
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """DPO loss on packed, interleaved [chosen_0, rejected_0, chosen_1, ...] pairs.

    See ``areal.utils.functional.dpo_preference_loss`` for supported
    ``loss_type`` variants. ``input_`` must contain ``cu_seqlens`` and
    ``loss_mask``; ``ref_logprobs`` is optional (defaults to zero-ref).
    ``entropy`` / ``vocab_{min,max}_logits`` are engine ``loss_fn`` contract
    parameters and unused by DPO.
    """
    device = logprobs.device
    cu_seqlens = input_["cu_seqlens"].to(device=device, dtype=torch.long)
    loss_mask = input_["loss_mask"].bool().to(device=device)
    ref_logprobs = input_.get("ref_logprobs")
    ref_logprobs = (
        ref_logprobs.to(device)
        if ref_logprobs is not None
        else torch.zeros_like(logprobs)
    )

    valid_pairs = _dpo_valid_pairs(input_).to(device=device)
    if not valid_pairs.any():
        _log_empty_dpo_stats(device)
        return torch.zeros((), dtype=torch.float32, device=device)

    policy_logps, ref_logps = dpo_pair_logratios(
        logprobs, ref_logprobs, cu_seqlens, loss_mask, valid_pairs
    )
    logits = (policy_logps[:, 0] - policy_logps[:, 1]) - (
        ref_logps[:, 0] - ref_logps[:, 1]
    )
    per_pair_loss = dpo_preference_loss(logits, beta=beta, loss_type=loss_type)

    with torch.no_grad():
        chosen_rewards = beta * (policy_logps[:, 0] - ref_logps[:, 0]).float()
        rejected_rewards = beta * (policy_logps[:, 1] - ref_logps[:, 1]).float()
        stats_tracker.denominator(
            n_pairs=torch.ones(
                chosen_rewards.shape[0], dtype=torch.bool, device=device
            ),
        )
        stats_tracker.stat(
            loss=per_pair_loss.detach().float(),
            chosen_reward=chosen_rewards,
            rejected_reward=rejected_rewards,
            reward_accuracy=(chosen_rewards > rejected_rewards).float(),
            reward_margin=(chosen_rewards - rejected_rewards).float(),
            denominator="n_pairs",
        )

    return per_pair_loss.mean()
