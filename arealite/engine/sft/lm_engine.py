from typing import Dict

import torch
import torch.utils.data
from tensordict import TensorDict

from arealite.api.cli_args import TrainEngineConfig
from arealite.api.engine_api import TrainEngine
from arealite.engine.fsdp_engine import FSDPEngine
from arealite.utils.functional import gather_logprobs
from realhf.base import stats_tracker


class LMEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_lm(self, data: TensorDict):
        self.engine.train()
        return self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["prompt_mask"].logical_not().count_nonzero(),
        )

    def evaluate_lm(self, data):
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["prompt_mask"].logical_not().count_nonzero(),
        )


class FSDPLMEngine(FSDPEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)


def compute_packed_sft_loss(
    logits: torch.Tensor, input_: Dict[str, torch.Tensor]
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["input_ids"]
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    prompt_mask = input_["prompt_mask"].bool()

    logprobs = gather_logprobs(logits, torch.roll(packed_input_ids, shifts=-1, dims=-1))
    prompt_mask = torch.roll(prompt_mask, shifts=-1, dims=-1)
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss = -logprobs.sum() / prompt_mask.logical_not().count_nonzero()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp.detach()).sum() / (
                m.numel() - m.count_nonzero()
            )

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=prompt_mask.logical_not(),
        prompt_tokens=prompt_mask,
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
