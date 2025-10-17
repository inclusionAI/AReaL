from typing import Any, Dict

import torch

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.utils import stats_tracker
from areal.utils.functional import gather_logprobs


class LMEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_lm(self, data: Dict[str, Any]):
        self.engine.train()
        return self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )

    def evaluate_lm(self, data):
        self.engine.eval()
        return self.engine.eval_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class FSDPLMEngine(FSDPEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)


class MegatronLMEngine(MegatronEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)


def compute_packed_sft_loss(
    logits: torch.Tensor, input_: Dict[str, Any]
) -> torch.Tensor:
    # Use rolled input_ids. Ulysses SP will roll input_ids in ulysses_prepare_inputs().
    labels: torch.Tensor = input_.get(
        "rolled_input_ids",
        torch.roll(input_["input_ids"], shifts=-1, dims=-1),
    )
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    # Use full loss_mask. Ulysses SP will slice loss_mask in ulysses_prepare_inputs().
    loss_mask = input_.get("full_loss_mask", input_["loss_mask"]).bool()

    logprobs = gather_logprobs(logits, labels)
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    logprobs = torch.where(loss_mask, logprobs, 0)

    device = logits.device
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
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=device),
        n_valid_tokens=loss_mask,
        prompt_tokens=loss_mask.logical_not(),
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    return loss
