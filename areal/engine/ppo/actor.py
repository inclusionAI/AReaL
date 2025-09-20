import functools
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils import stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.fsdp import offload_fsdp2_model_to_cpu, offload_fsdp_optimizer
from areal.utils.functional import (
    dynamic_sampling,
    gather_logprobs,
    gather_logprobs_entropy,
    masked_normalization,
    ppo_actor_loss_fn,
    reward_overlong_penalty,
)


class PPOActor:

    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.group_reward_norm = config.group_reward_norm
        self.group_size = config.group_size

        self.kl_ctl = config.kl_ctl

        self.adv_norm = AdvNorm(config.adv_norm) if config.adv_norm else None

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature
        self.dynamic_sampling = config.dynamic_sampling

    @torch.no_grad()
    def compute_logp(
        self,
        data: TensorDict,
        temperature: Optional[float] = None,
    ) -> torch.Tensor | None:

        def calc_logprobs(logits, input_data):
            labels = torch.roll(input_data["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(logits, labels, temperature or 1.0)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    def compute_advantages(self, data: TensorDict) -> None:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # TODO:rewrite the reward into "reward" class __call__ method should be good. Like VeRL does.
        # Reward Penalty on length
        if self.config.overlong_reward_penalty:

            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.group_reward_norm:
            for i in range(bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                r = reward_score[s]
                reward_score[s] = (r - r.mean()) / (r.std() + 1e-9)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            old_logp = data["logprobs"] = data["prox_logp"]
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp", torch.zeros_like(old_logp))
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * (old_logp - ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        for t in reversed(range(max_seqlen - 1)):
            nextvalues = values[:, t + 1]
            if t == max_seqlen - 2:
                nextvalues *= seq_no_eos_mask
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            lastgaelam = delta + self.discount * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

    def ppo_update(self, data: TensorDict) -> List[Dict[str, float]]:

        if self.dynamic_sampling and len(data["rewards"]) % self.group_size == 0:
            data, sampling_stat = dynamic_sampling(data, self.group_size)

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
        if self.config.log_agent_stats:
            assert (
                "begin_of_trajectory" in data
            ), "'begin_of_trajectory' is expected to log agent statistics"
            assert (
                len(self.config.log_agent_stats_keys) > 0
            ), "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
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

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )

        global_stats = stats_tracker.export(
            reduce_group=self.engine.data_parallel_group
        )
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

        if self.engine._is_offload_param:
            load_fsdp2_model_to_cpu(self.actor_module_fsdp)

        if self.engine._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.optimizer)

        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        for mb in mb_inputs.mbs:
            train_stat = self.engine.train_batch(
                mb,
                loss_fn=functools.partial(
                    grpo_loss_fn,
                    temperature=self.temperature,
                    eps_clip=self.config.eps_clip,
                    eps_clip_higher=self.config.eps_clip_higher,
                    c_clip=self.config.c_clip,
                    behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )
            stats_tracker.scalar(**train_stat)
            all_stats.append(
                stats_tracker.export(reduce_group=self.engine.data_parallel_group)
            )
        all_stats[0].update(global_stats)

        if self.engine._is_offload_param:
            offload_fsdp2_model_to_cpu(self.actor_module_fsdp)

        if self.engine._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.optimizer)
        return all_stats


class FSDPPPOActor(FSDPEngine):

    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> List[Dict[str, float]]:
        return self.actor.ppo_update(*args, **kwargs)


def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: Dict,
    temperature: float,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    input_ids = input_data["input_ids"]
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp = input_data["prox_logp"]

    logprobs, entropy = gather_logprobs_entropy(
        logits, torch.roll(input_ids, shifts=-1, dims=-1), temperature
    )
    entropy = entropy.detach()
    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        proximal_logprobs=prox_logp,
        behav_imp_weight_cap=behav_imp_weight_cap,
    )

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
        entropy=entropy.float(),
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


class AdvNorm:
    """
    Adaptive Advantage Normalization.

    Supports independent specification of normalization level for mean and std:
    - "batch": normalize across entire batch (with optional all_reduce in distributed setting)
    - "group": normalize within fixed-size groups
    - std_level can be None: only center the data (no scaling)

    If mean_level == std_level, uses original masked_normalization for efficiency.
    Otherwise, computes mean and std separately and combines.

    Args:
        mean_level (str): "batch" or "group"
        std_level (str or None): "batch", "group", or None (no std scaling)
        group_size (int, optional): required if any level is "group"
    """

    def __init__(
        self,
        advNorm_cfg,
    ):
        if advNorm_cfg is None:
            return None

        if advNorm_cfg.mean_level not in {"batch", "group", "none"}:
            raise ValueError(
                f"mean_level must be 'batch', 'group' or 'none', got {advNorm_cfg.mean_level}"
            )
        if advNorm_cfg.std_level not in {"batch", "group", "none"}:
            raise ValueError(
                f"std_level must be 'batch', 'group', or 'none', got {advNorm_cfg.std_level}"
            )
        if (
            advNorm_cfg.mean_level == "group" or advNorm_cfg.std_level == "group"
        ) and advNorm_cfg.group_size is None:
            raise ValueError("group_size must be provided if using group normalization")

        self.mean_level = advNorm_cfg.mean_level
        self.std_level = advNorm_cfg.std_level
        self.group_size = advNorm_cfg.group_size

    @torch.no_grad()
    def __call__(
        self,
        advantages: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        unbiased: bool = False,
        high_precision: bool = True,
        reduce_group=None,
    ) -> torch.Tensor:
        """
        Normalize advantages tensor according to mean_level and std_level.

        Args:
            advantages (torch.Tensor): [...]
            loss_mask (torch.Tensor, optional): same shape as advantages
            eps (float): small constant for numerical stability
            unbiased (bool): whether to use unbiased variance
            high_precision (bool): use float64 for computation
            reduce_group: distributed group for all_reduce

        Returns:
            normalized advantages (same shape, dtype=float32)
        """

        bs = advantages.size(0)

        # Case: same level → use original masked_normalization to maxize the robust to original code
        if self.mean_level == self.std_level:
            if self.mean_level == "batch":
                return masked_normalization(
                    advantages,
                    mask=loss_mask,
                    unbiased=unbiased,
                    eps=eps,
                    high_precision=high_precision,
                    all_reduce=True,  # follow original code
                    reduce_group=reduce_group,
                )
            else:  # group or none
                if self.mean_level == "none":
                    return advantages.float()
                adv_list = []
                for i in range(0, bs // self.group_size):
                    s = slice(i * self.group_size, (i + 1) * self.group_size)
                    adv = advantages[s]
                    m = loss_mask[s] if loss_mask is not None else None
                    adv_list.append(
                        masked_normalization(
                            adv,
                            mask=m,
                            unbiased=unbiased,
                            eps=eps,
                            high_precision=high_precision,
                            all_reduce=False,  # follow original code
                            reduce_group=reduce_group,
                        )
                    )
                return torch.cat(adv_list, 0)

        # Cases: mean and std levels differ, or std_level is None

        # Early return for no normalization case
        if self.mean_level == "none" and self.std_level == "none":
            return advantages.float()

        # Step 1: Compute mean
        if self.mean_level == "batch":
            mean = self._compute_mean(
                advantages, loss_mask, high_precision, True, reduce_group
            )
            # Expand batch mean to match input shape for mixed normalization
            mean = mean.expand_as(advantages)
        elif self.mean_level == "group":
            mean = torch.zeros_like(advantages)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                adv = advantages[s]
                m = loss_mask[s] if loss_mask is not None else None
                group_mean = self._compute_mean(
                    adv, m, high_precision, False, reduce_group
                )
                mean[s] = group_mean.expand_as(adv)
        else:  # mean_level == "none"
            mean = torch.zeros_like(advantages)

        # Subtract mean
        x_centered = advantages - mean

        # Step 2: Compute std
        if self.std_level == "none":
            return x_centered.float()

        if self.std_level == "batch":
            std = self._compute_std(
                advantages,
                loss_mask,
                mean,
                unbiased,
                high_precision,
                True,
                reduce_group,
            )
            # Expand batch std to match input shape
            std = std.expand_as(advantages)
        else:  # group
            std = torch.zeros_like(advantages)
            for i in range(0, bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                adv = advantages[s]
                m = loss_mask[s] if loss_mask is not None else None
                group_mean_slice = mean[s]  # already computed and expanded
                group_std = self._compute_std(
                    adv,
                    m,
                    group_mean_slice,
                    unbiased,
                    high_precision,
                    False,
                    reduce_group,
                )
                std[s] = group_std.expand_as(adv)

        # Normalize
        return (x_centered / (std + eps)).float()

    @staticmethod
    def _compute_mean(
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        high_precision: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute mean only, using masked_normalization internals."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        dim = tuple(range(len(x.shape)))
        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_sum = x.sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_sum = x_masked.sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)

        mean = x_sum / factor
        return mean

    @staticmethod
    def _compute_std(
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mean: torch.Tensor,
        unbiased: bool,
        high_precision: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute std only, given precomputed mean."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        dim = tuple(range(len(x.shape)))
        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_centered = x_masked - mean * mask  # only apply mean where mask is 1
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)
        if mask is None:
            x_centered = x - mean
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM, group=reduce_group)

        var = x_sum_sq / factor
        if unbiased:
            var *= factor / (factor - 1)
        std = var.sqrt()
        return std
