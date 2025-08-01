import torch
import torch.utils.data
from tensordict import TensorDict

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.functional import gather_logprobs
from realhf.base import stats_tracker


class LMEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_lm(self, data: TensorDict):
        self.engine.train()
        return self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )

    def evaluate_lm(self, data):
        self.engine.eval()
        self.engine.eval_batch(
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


def compute_packed_sft_loss(logits: torch.Tensor, input_: TensorDict) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["input_ids"]
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    loss_mask = input_["loss_mask"].bool()

    logprobs = gather_logprobs(logits, torch.roll(packed_input_ids, shifts=-1, dims=-1))
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    logprobs = torch.where(loss_mask, logprobs, 0)

    loss = -logprobs.sum() / loss_mask.count_nonzero()
    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]]
            logp = logprobs[cu_seqlens[i] : cu_seqlens[i + 1]]
            seqlogp[i] = torch.where(m, logp.detach(), 0.0).sum() / (m.count_nonzero())

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
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
