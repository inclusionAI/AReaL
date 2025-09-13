import torch
from tensordict import TensorDict

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils import stats_tracker


class RWEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_rw(self, data: TensorDict):
        self.engine.train()
        return self.engine.rw_train_batch(
            input_=data,
            loss_fn=compute_rw_loss,
        )

    def evaluate_rw(self, data):
        self.engine.eval()
        self.engine.rw_eval_batch(
            input_=data,
            loss_fn=compute_rw_loss,
        )


class FSDPRWEngine(FSDPEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.rw_engine = RWEngine(self)

    def train_rw(self, data):
        return self.rw_engine.train_rw(data)

    def evaluate_rw(self, data):
        return self.rw_engine.evaluate_rw(data)


def compute_rw_loss(rewards: torch.Tensor, input_: TensorDict) -> torch.Tensor:
    input_ids, attention_masks = input_["input_ids"], input_["attention_mask"]
    bs = input_ids.shape[0] // 2

    chosen_ids = input_ids[:bs]
    rejected_ids = input_ids[bs:]
    chosen_rewards = rewards[:bs]
    rejected_rewards = rewards[bs:]
    c_attention_masks = attention_masks[:bs]
    r_attention_masks = attention_masks[bs:]

    loss_sum = 0.0
    for i in range(bs):
        chosen_id = chosen_ids[i]
        rejected_id = rejected_ids[i]
        chosen_reward = chosen_rewards[i]
        rejected_reward = rejected_rewards[i]
        c_attention_mask = c_attention_masks[i]
        r_attention_mask = r_attention_masks[i]

        # append the length of input to inds (corner case : no padding on the right)
        c_inds = torch.cat(
            [
                (c_attention_mask == 0).nonzero(),
                torch.tensor([[len(chosen_id)]]).to(c_attention_mask.device),
            ]
        )
        r_inds = torch.cat(
            [
                (r_attention_mask == 0).nonzero(),
                torch.tensor([[len(chosen_id)]]).to(r_attention_mask.device),
            ]
        )
        c_ind, r_ind = c_inds[0], r_inds[0]
        end_ind = max(c_ind, r_ind)

        # the index of first different input_id (prompt + chosen/rejected response)
        divergence_ind = (chosen_id != rejected_id).nonzero()[0]
        assert divergence_ind > 0

        c_truncated_reward = chosen_reward[divergence_ind:end_ind]
        r_truncated_reward = rejected_reward[divergence_ind:end_ind]
        # the score of last token
        chosen_score = chosen_reward[c_ind - 1]
        rejected_score = rejected_reward[r_ind - 1]
        acc = 1.0 if chosen_score > rejected_score else 0.0

        loss = -torch.nn.functional.logsigmoid(
            c_truncated_reward - r_truncated_reward
        ).mean()
        loss_sum += loss

        # logging stats
        stats_tracker.denominator(
            n_seqs=torch.ones(1, dtype=torch.bool, device=loss.device),
        )

        stats_tracker.stat(
            loss=loss.detach().unsqueeze(0).to(torch.float32),
            chosen_score=chosen_score.detach().to(torch.float32),
            rejected_score=rejected_score.detach().to(torch.float32),
            acc=torch.tensor([acc], dtype=torch.float32).to(loss.device),
            denominator="n_seqs",
        )

    return loss_sum / bs
