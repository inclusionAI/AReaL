from typing import Any

import torch

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.scheduler_api import Scheduler
from areal.controller.train_controller import TrainController
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.utils import stats_tracker
from areal.utils.perf_tracer import trace_perf


class LMEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    @trace_perf("lm_engine.train_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft")
    def train_lm(self, data: dict[str, Any]):
        self.engine.train()
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        stats_tracker.scalar(**stats)

    @trace_perf("lm_engine.evaluate_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft-eval")
    def evaluate_lm(self, data):
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class LMController(TrainController):
    def train_lm(self, *args, **kwargs):
        self._custom_function_call("train_lm", *args, **kwargs)

    def evaluate_lm(self, *args, **kwargs):
        self._custom_function_call("evaluate_lm", *args, **kwargs)


class FSDPLMEngine(FSDPEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(
        cls, config: TrainEngineConfig, scheduler: Scheduler
    ) -> LMController:
        return LMController(train_engine=cls, config=config, scheduler=scheduler)


class MegatronLMEngine(MegatronEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(
        cls, config: TrainEngineConfig, scheduler: Scheduler
    ) -> LMController:
        return LMController(train_engine=cls, config=config, scheduler=scheduler)


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

    loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    logprobs = torch.where(loss_mask, logprobs, 0)

    device = logprobs.device
    loss = -logprobs.sum() / loss_mask.count_nonzero()
    with torch.no_grad():
        batch_size = cu_seqlens.shape[0] - 1
        seqlogp = torch.zeros(batch_size, dtype=torch.float64, device=device)
        n_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            m = loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]]
            logp = logprobs[cu_seqlens[i] : cu_seqlens[i + 1]]
            valid_tokens = int(m.count_nonzero().item())
            if valid_tokens == 0:
                # This is a padded dummy sequence created in `padded_mb_input`.
                # When Ulysses SP is enabled, padded inputs are passed into the loss function.
                # So we skip it.
                continue

            n_seqs[i] = True
            seqlogp[i] = torch.where(m, logp.detach(), 0.0).sum() / valid_tokens

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=n_seqs,
        n_tokens=torch.ones(logprobs.shape[0], dtype=torch.bool, device=device),
        n_valid_tokens=loss_mask,
        prompt_tokens=loss_mask.logical_not(),
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    return loss
