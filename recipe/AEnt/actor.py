import functools
from typing import Dict, List

import torch
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.ppo.actor import PPOActor
from areal.utils import stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.functional import (
    ppo_actor_loss_fn,
    gather_logprobs_entropy,
)

from recipe.AEnt.aent_args import AEntPPOActorConfig
from recipe.AEnt.functional import gather_logprobs_clamped_entropy


class AEntPPOActor(PPOActor):

    def __init__(self, config: AEntPPOActorConfig, engine: TrainEngine):
        super().__init__(config, engine)
        self.entropy_coeff = config.aent.entropy_coeff
        self.entropy_clamp = config.aent.entropy_clamp
        self.adaptive_coeff = config.aent.adaptive_coeff
        if self.adaptive_coeff:
            self.entropy_high = config.aent.entropy_high
            self.entropy_low = config.aent.entropy_low
            self.coeff_lr = config.aent.coeff_lr
            self.coeff_box_high = config.aent.coeff_box_high
            self.coeff_box_low = config.aent.coeff_box_low
            self.warmup_steps = config.aent.warmup_steps

    def aent_ppo_update(self, data: TensorDict, global_step: int) -> List[Dict[str, float]]:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        all_stats = []
        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = []
        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behav_imp_weight_cap is not None:
            scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
        stats_tracker.scalar(**scalars)

        global_stats = stats_tracker.export(reduce_group=self.engine.parallelism_group)
        for k in global_denominators:
            keys = list(global_stats.keys())
            for k2 in keys:
                if k2.endswith(k):
                    global_stats.pop(k2)
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        ent_trace = []
        for mb in mb_inputs.mbs:
            train_stat = self.engine.train_batch(
                mb,
                loss_fn=functools.partial(
                    aent_grpo_loss_fn,
                    temperature=self.temperature,
                    eps_clip=self.config.eps_clip,
                    entropy_coeff=self.entropy_coeff,
                    entropy_clamp=self.entropy_clamp,
                    c_clip=self.config.c_clip,
                    behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )
            stats_tracker.scalar(**train_stat)
            all_stats.append(
                stats_tracker.export(reduce_group=self.engine.parallelism_group)
            )
            ent_trace.append(float(all_stats[-1]['grpo_actor/entropy/avg']))
        if self.adaptive_coeff and global_step > self.warmup_steps:
            entropy = sum(ent_trace)/len(ent_trace)
            self.entropy_coeff -= self.coeff_lr*(min(0,entropy-self.entropy_low)+max(0,entropy-self.entropy_high))
            self.entropy_coeff = min(max(self.entropy_coeff, self.coeff_box_low), self.coeff_box_high)

        all_stats[0].update(global_stats)
        return all_stats


class FSDPAEntPPOActor(FSDPEngine):

    def __init__(self, config: AEntPPOActorConfig):
        super().__init__(config)
        self.actor = AEntPPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def aent_ppo_update(self, *args, **kwargs) -> List[Dict[str, float]]:
        return self.actor.aent_ppo_update(*args, **kwargs)

# AEnt regularized grpo loss
def aent_grpo_loss_fn(
    logits: torch.Tensor,
    input_data: Dict,
    temperature: float,
    eps_clip: float,
    entropy_coeff: float,
    entropy_clamp: float,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
):
    """Loss function for actor step, all inputs should be split into
    pipeline micro batches, returns loss and logging stats."""
    input_ids = input_data["input_ids"]
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp = input_data["prox_logp"]

    if entropy_clamp>0:
        logprobs, clamped_entropy = gather_logprobs_clamped_entropy(
            logits, torch.roll(input_ids, shifts=-1, dims=-1), entropy_clamp, temperature
        )
    else:
        logprobs, clamped_entropy = gather_logprobs_entropy(
            logits, torch.roll(input_ids, shifts=-1, dims=-1), temperature
        )
    ppo_loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=loss_mask,
        c_clip=c_clip,
        proximal_logprobs=prox_logp,
        behav_imp_weight_cap=behav_imp_weight_cap,
    )
    # add AEnt's clamped entropy regularizer
    clamped_entropy_loss = clamped_entropy_loss_fn(clamped_entropy, loss_mask)
    loss = ppo_loss - entropy_coeff * clamped_entropy_loss

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=clamped_entropy.detach().float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
    clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
    stats_tracker.stat(
        clipped_new_logp=clipped_new_logp,
        clipped_old_logp=clipped_old_logp,
        denominator="clipped_tokens",
    )
    return loss

def clamped_entropy_loss_fn(clamped_entropy: torch.Tensor, loss_mask: torch.Tensor):
    loss_mask_count = loss_mask.count_nonzero() or 1
    clamped_ent_loss = torch.where(loss_mask.bool(), clamped_entropy, 0.0).sum() / loss_mask_count
    return clamped_ent_loss
