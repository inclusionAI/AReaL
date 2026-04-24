# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.distributed as dist

from areal.api import TrainEngine
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import stats_tracker
from areal.utils.data import batched_call
from areal.utils.perf_tracer import trace_perf


class LMEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    @trace_perf("lm_engine.train_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft")
    def train_lm(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._train_lm, data, unpack=False)

    def _train_lm(self, data: dict[str, Any]) -> None:
        self.engine.train()
        data["loss_mask"] = torch.roll(data["loss_mask"].bool(), shifts=-1, dims=-1)
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        stats_tracker.scalar(**stats)

    @trace_perf("lm_engine.evaluate_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft-eval")
    def evaluate_lm(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._evaluate_lm, data, unpack=False)

    def _evaluate_lm(self, data: dict[str, Any]) -> None:
        self.engine.eval()
        data["loss_mask"] = torch.roll(data["loss_mask"].bool(), shifts=-1, dims=-1)
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class LMController(TrainController):
    def train_lm(self, *args, **kwargs):
        self._custom_function_call(
            "train_lm", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_lm(self, *args, **kwargs):
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=1)
        self._custom_function_call(
            "evaluate_lm", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class LMControllerV2(GatewayTrainController):
    def train_lm(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/sft/train", payload)

    def evaluate_lm(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/sft/evaluate", payload)


def compute_packed_sft_loss(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_: dict[str, Any],
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SFT loss from logprobs."""
    del entropy  # SFT does not use entropy
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    loss_mask = input_["loss_mask"].bool()

    logprobs = torch.where(loss_mask, logprobs, 0)

    device = logprobs.device
    # NOTE: The returned `loss` scalar is what `forward_backward_func` uses for
    # backward. Under CP-local loss, each CP rank computes its own local mean
    # loss and backward; Megatron's CP ring-attention handles the cross-rank
    # gradient coupling. We therefore must NOT replace this with a globally
    # averaged loss here -- doing so would change the gradient on CP shards.
    # The stats reported below are a separate concern and are made CP-invariant
    # via per-key reduce_group overrides (see #1242 follow-up).
    loss = -logprobs.sum() / (1e-5 + loss_mask.count_nonzero())

    # CP-related process groups (populated by MegatronEngine when CP > 1).
    #   `_cp_reduce_group`    : context-parallel group, used to aggregate
    #                           per-sequence partial sums into a full-sequence
    #                           view so ppl / seqlogp match the pre-CP-local
    #                           reporting semantics.
    #   `_cp_dp_reduce_group` : DP + CP group, passed as per-key `reduce_group`
    #                           to `stats_tracker.stat` so that ratio metrics
    #                           (loss / entropy / vocab_*) whose numerator and
    #                           denominator live on CP-local tensors reduce
    #                           across DP + CP at export time, yielding a
    #                           globally averaged value instead of a CP-slice
    #                           average. Scalar reductions only; does NOT
    #                           reintroduce the expensive logits all-gather.
    cp_reduce_group = input_.get("_cp_reduce_group")
    cp_dp_reduce_group = input_.get("_cp_dp_reduce_group")

    with torch.no_grad():
        batch_size = cu_seqlens.shape[0] - 1
        # Collect per-sequence partial sums first, then CP-reduce once at the
        # end to avoid one collective per sequence. Contributions from CP ranks
        # other than this one are zero locally and get added in the all-reduce.
        seq_logp_sum = torch.zeros(batch_size, dtype=torch.float64, device=device)
        seq_valid_count = torch.zeros(batch_size, dtype=torch.int64, device=device)
        for i in range(batch_size):
            m = loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]]
            logp = logprobs[cu_seqlens[i] : cu_seqlens[i + 1]]
            seq_logp_sum[i] = torch.where(m, logp.detach(), 0.0).sum().double()
            seq_valid_count[i] = m.sum()

        if cp_reduce_group is not None:
            # Sum partial contributions across CP ranks so each element of
            # `seq_logp_sum` / `seq_valid_count` reflects the full sequence
            # rather than this rank's CP slice. Tensors are tiny (batch_size).
            dist.all_reduce(seq_logp_sum, group=cp_reduce_group)
            dist.all_reduce(seq_valid_count, group=cp_reduce_group)

        n_seqs = seq_valid_count > 0
        safe_valid = seq_valid_count.clamp(min=1).double()
        seqlogp = torch.where(
            n_seqs, seq_logp_sum / safe_valid, torch.zeros_like(seq_logp_sum)
        )

    ## Logging stats
    # Use the pre-CP-split loss_mask (when available) for token-count
    # denominators so these metrics are invariant to CP topology. The local
    # loss_mask is also recorded as `n_valid_tokens_local` because
    # `stats_tracker.stat` requires denominators whose shape matches the
    # recorded tensor, and `logprobs` / `vocab_*` are CP-local. See #1242.
    global_loss_mask = input_.get("_global_loss_mask", loss_mask)
    stats_tracker.denominator(
        n_seqs=n_seqs,
        n_tokens=torch.ones(
            global_loss_mask.shape[0], dtype=torch.bool, device=global_loss_mask.device
        ),
        n_valid_tokens=global_loss_mask,
        prompt_tokens=global_loss_mask.logical_not(),
        n_tokens_local=torch.ones(logprobs.shape[0], dtype=torch.bool, device=device),
        n_valid_tokens_local=loss_mask,
    )
    # `ppl` is already CP-invariant (seqlogp was CP-reduced above), and
    # `n_seqs` is DP-distinct but CP-replicated, so the default DP-only
    # reduce at export time gives the correct global per-sequence ppl.
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    # loss / vocab_* live on CP-local tensors. Overriding the reduce_group to
    # DP + CP means the scalar numerator (sum of `-logprobs * mask`) and scalar
    # denominator (sum of `mask`) are all-reduced across DP + CP, yielding the
    # global weighted average. No tensor gather is involved -- the collectives
    # see only scalars.
    stats_tracker.stat(
        loss=-logprobs.detach(),
        denominator="n_valid_tokens_local",
        reduce_group=cp_dp_reduce_group,
    )

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens_local",
            reduce_group=cp_dp_reduce_group,
        )

    return loss
