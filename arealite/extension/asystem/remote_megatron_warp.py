import dataclasses
import json
import os
from typing import Dict, List, Literal, Optional

import torch
import torch.distributed as dist

from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine

import realhf.api.core.model_api as model_api
import realhf.impl.model.utils.ppo_functional as ppo_functional
from realhf.api.core.data_api import (
    RL_TASKS,
    MicroBatchSpec,
    SequenceSample,
    SequenceSplitSpec,
)
from realhf.base import constants, logging, stats_tracker
from realhf.base.datapack import flat2d
from realhf.impl.model.utils.functional import (
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("RemoteMegatronWarp")


def _ppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
    kl_adapter: ppo_functional.KLController,  # const
    eps_clip: float,  # const
    c_clip: float | None,
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
    temperature: Optional[float] = 1,
) -> torch.Tensor:
    """Loss function for ppo actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    packed_input_ids = input_.data["packed_input_ids"]
    cu_seqlens = (
        torch.nn.functional.pad(
            torch.tensor(flat2d(input_.seqlens["packed_input_ids"])).cumsum(0),
            (1, 0),
        )
        .int()
        .to(logits.device)
    )
    ppo_loss_mask = input_.data["ppo_loss_mask"]
    advantages = input_.data["advantages"]
    old_logp = input_.data["old_logp"]
    kl_rewards = input_.data["kl_rewards"]

    if temperature is not None:
        logits /= temperature
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    old_logp = logprobs.detach()
    loss, ppo_stat = ppo_functional.actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=ppo_loss_mask,
        c_clip=c_clip,
        proximal_logprobs=input_.data.get("prox_logp", None),
    )

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=ppo_loss_mask.bool(),
        clipped_tokens=ppo_stat["clip_mask"],
        dual_clipped_tokens=ppo_stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=ppo_stat["importance_weight"],
        approx_kl=ppo_stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        actor_loss=ppo_stat["loss"],
        clip_ratio=ppo_stat["clip_mask"].float(),
        dual_clip_ratio=ppo_stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in ppo_stat:
        stats_tracker.denominator(unclipped_behave_tokens=ppo_stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=ppo_stat["behave_imp_weight"],
            behave_approx_kl=ppo_stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    dist.all_reduce(
        vocab_min_logits, group=constants.tensor_parallel_group(), op=dist.ReduceOp.MIN
    )
    dist.all_reduce(
        vocab_max_logits, group=constants.tensor_parallel_group(), op=dist.ReduceOp.MAX
    )
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    clip_mask = ppo_stat["clip_mask"]
    dual_clip_mask = ppo_stat["dual_clip_mask"]
    clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
    dual_clipped_new_logp = torch.where(dual_clip_mask, logprobs.detach(), 0.0)
    clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
    dual_clipped_old_logp = torch.where(dual_clip_mask, old_logp, 0.0)
    stats_tracker.stat(
        clipped_new_logp=clipped_new_logp,
        clipped_old_logp=clipped_old_logp,
        denominator="clipped_tokens",
    )
    stats_tracker.stat(
        dual_clipped_new_logp=dual_clipped_new_logp,
        dual_clipped_old_logp=dual_clipped_old_logp,
        denominator="dual_clipped_tokens",
    )

    # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
    mean_ref_kl = (kl_rewards.detach().float() * ppo_loss_mask).sum()
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    _imp = (ppo_stat["importance_weight"].float() * ppo_loss_mask).sum()
    dist.all_reduce(_imp, group=constants.data_parallel_group())
    _kl = (ppo_stat["approx_kl"].float() * ppo_loss_mask).sum()
    dist.all_reduce(_kl, group=constants.data_parallel_group())
    _n_valid_tokens = ppo_loss_mask.count_nonzero().clone()
    dist.all_reduce(_n_valid_tokens, group=constants.data_parallel_group())
    mean_ref_kl /= _n_valid_tokens
    _imp /= _n_valid_tokens
    _kl /= _n_valid_tokens
    # Early stopping.
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)
    if early_stop_imp_ratio is not None and _imp > early_stop_imp_ratio:
        logger.warning(
            f"Current importance ratio {_imp.item():.4f} is larger "
            f"than early stop threshold {early_stop_imp_ratio}. Abandon this minibatch."
        )
        loss = loss * 0.0
    if early_stop_kl is not None and _kl > early_stop_kl:
        logger.warning(
            f"Current approximate KL divergence {_kl.item():.4f} is larger "
            f"than early stop threshold {early_stop_kl}. Abort actor update."
        )
        loss = loss * 0.0

    return loss


@dataclasses.dataclass
class RemoteMegatronWarp:
    n_minibatches: int = 4
    kl_ctl: float = 0.1
    adv_norm: bool = True
    discount: float = 1.0
    gae_lambda: float = 1.0

    eps_clip: float = 0.2
    c_clip: Optional[float] = None
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    disable_value: bool = False

    early_stop_kl: Optional[float] = None  # e.g. 0.1
    early_stop_imp_ratio: Optional[float] = None  # e.g., 10.0

    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    enable_save: bool = True

    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    group_size: int = 1
    generation_size: Optional[int] = None
    mask_no_eos_with_zero: bool = False
    group_adv_norm: bool = False
    mask_too_long: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True
    token_normalize_scope: Literal["global", "dp"] = "global"

    sample_reuse: int = 1

    engine: RemoteMegatronEngine

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.value_norm_beta, epsilon=self.value_norm_eps
                )
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

    def train_step(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> Dict | List[Dict]:

        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.tensor(
            flat2d(input_.seqlens["packed_input_ids"]), device=model.device
        )
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        prompt_lens = []
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            prompt_lens.append(prompt_mask[s:e].sum())
        prompt_lens = torch.tensor(prompt_lens, device=model.device)
        reward_score = input_.data["rewards"].float()
        task_ids = input_.data["task_ids"]
        task_ids = task_ids.repeat(self.group_size, 1).transpose(0, 1).reshape(-1)

        if "dense_rewards" in input_.data:
            dense_reward_score = input_.data["dense_rewards"].float()
        if not self.disable_value:
            values = input_.data["values"].float()
        else:
            values = torch.zeros_like(
                input_.data["packed_input_ids"], dtype=torch.float32
            )
        seq_no_eos_mask = input_.data["seq_no_eos_mask"]
        if self.kl_adapter.value == 0:
            ref_logp: torch.FloatTensor = reward_score.new_zeros(
                int(input_lens.sum()) - len(input_lens)
            )
        else:
            ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()

        if not self.disable_value:
            if self.value_norm:
                denormalized_values = self.rms.denormalize(values)
            else:
                denormalized_values = values
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()

        if self.mask_too_long:
            for i in range(seq_no_eos_mask.shape[0]):
                if seq_no_eos_mask[i]:
                    loss_mask[cu_seqlens[i]: cu_seqlens[i + 1]] = False

        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        new_reward_score = reward_score
        if not self.adv_norm and self.group_adv_norm:
            n_seqs = len(input_lens)
            reward_score_grpo = reward_score.clone().detach()
            new_reward_score = reward_score_grpo
            for i in range(n_seqs // self.group_size):
                group_rewards = reward_score[i * self.group_size: (i + 1) * self.group_size]
                grouped_std = group_rewards.std(dim=-1)
                normed_rewards = (group_rewards - group_rewards.mean(-1, keepdim=True)) / (grouped_std + 1e-9)
                reward_score_grpo[i * self.group_size: (i + 1) * self.group_size] = normed_rewards

        # Compute rewards and GAEs.
        if self.use_dense_reward:
            kl_rewards, rewards = ppo_functional.get_packed_reward_dense(
                kl_ctl=self.kl_adapter.value,
                clip_reward_value=self.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                dense_reward_score=dense_reward_score,
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                reward_delta=self.reward_delta,
            )
        else:
            kl_rewards, rewards = ppo_functional.get_packed_rewards(
                kl_ctl=self.kl_adapter.value,
                clip_reward_value=self.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                reward_score=(new_reward_score),
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                mask_no_eos_with_zero=self.mask_no_eos_with_zero,
            )
        advantages, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=(
                denormalized_values
                if not self.disable_value
                else denormalized_values.new_zeros(denormalized_values.shape)
            ),
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
        if self.adv_norm:
            if self.group_adv_norm == False:
                advantages = masked_normalization(advantages, loss_mask)
            else:
                logger.info(f"adv_shape: {advantages.shape}")
                logger.info(f"prompt_mask_shape: {prompt_mask.shape}")
                n_samples = len(cu_seqlens) - 1
                assert n_samples % self.group_size == 0
                adv_list = []
                for i in range(0, n_samples, self.group_size):
                    for j in range(1, self.group_size):
                        assert (
                            prompt_mask[cu_seqlens[i]: cu_seqlens[i + 1]].sum()
                            == prompt_mask[
                               cu_seqlens[i + j]: cu_seqlens[i + j + 1]
                               ].sum()
                        )
                    adv_list.append(
                        masked_normalization(
                            advantages[
                            short1cu_seqlens[i]: short1cu_seqlens[
                                i + self.group_size
                                ]
                            ],
                            loss_mask[
                            short1cu_seqlens[i]: short1cu_seqlens[
                                i + self.group_size
                                ]
                            ],
                            all_reduce=False,
                        )
                    )

                advantages = torch.cat(adv_list, 0)

        # Prepare data to be splitted into mini-batches.
        flat_data = dict(
            advantages=advantages,
            old_logp=old_logp,
            ppo_loss_mask=loss_mask,
            packed_input_ids=input_.data["packed_input_ids"],
            kl_rewards=kl_rewards,
        )
        use_prox_logp = "proximal_logprobs" in input_.data
        if use_prox_logp:
            flat_data["prox_logp"] = input_.data["proximal_logprobs"].float()

        flat_input = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            data=flat_data,
            seqlens=[int(x) for x in input_lens.cpu().numpy().tolist()],
        )

        if self.use_dense_reward:
            dense_reward_score = dense_reward_score[shift_one_indices]

        ### Logging code starts. ###
        all_stats = []
        with stats_tracker.scope("ppo_actor"):
            assert (
                task_ids.shape == reward_score.shape
            ), f"task_ids ({task_ids.shape}) and reward_score ({reward_score.shape}) must have the same shape"

            task_denominators = {
                f"{task}_n_seqs": (task_ids == idx).bool()
                for idx, task in enumerate(RL_TASKS)
            }

            global_denominators = dict(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                n_tokens=torch.ones_like(prompt_mask, dtype=torch.bool),
                n_valid_tokens=loss_mask.bool(),
                **task_denominators,
            )
            stats_tracker.denominator(**global_denominators)

            for task in RL_TASKS:
                stats_tracker.stat(
                    **{f"{task}_reward": reward_score}, denominator=f"{task}_n_seqs"
                )

            stats = dict(
                advantages=advantages,
                kl_rewards=kl_rewards,
                final_reward=rewards,
            )
            if self.use_dense_reward:
                stats["dense_reward"] = dense_reward_score
            stats_tracker.stat(**stats, denominator="n_valid_tokens")

            seq_stats = dict(
                no_eos_ratios=seq_no_eos_mask.float(),
                task_reward=reward_score,
                prompt_len=prompt_lens.float(),
                seq_len=input_lens.float(),
            )
            if "version_start" in input_.data:
                seq_stats["head_offpolicyness"] = (
                    model.version.global_step - input_.data["version_start"]
                ).float()
            if "version_end" in input_.data:
                seq_stats["tail_offpolicyness"] = (
                    model.version.global_step - input_.data["version_end"]
                ).float()
            stats_tracker.stat(
                **seq_stats,
                denominator="n_seqs",
            )
            scalars = dict(
                disable_value=self.disable_value,
                mask_no_eos_with_zero=self.mask_no_eos_with_zero,
                eps_clip=self.eps_clip,
                use_prox_logp=use_prox_logp,
            )
            if self.c_clip is not None:
                scalars["c_clip"] = self.c_clip
                scalars["use_dual_clip"] = 1
            else:
                scalars["use_dual_clip"] = 0
            stats_tracker.scalar(**scalars)

            global_stats = stats_tracker.export()
            for k in global_denominators:
                global_stats.pop(f"ppo_actor/{k}")

            # Run mini-batched PPO training!
            def _loss_fn(logits, input_):
                return _ppo_actor_loss_from_model_outputs(
                    logits,
                    input_,
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                    c_clip=self.c_clip,
                    temperature=self.gconfig.temperature,
                )

            for reuse in range(self.sample_reuse):
                # NOTE: We split PPO minibatches in terms of #seqs instead of #tokens.
                flat_input = SequenceSample.shuffled(flat_input)
                bs = flat_input.bs
                sizes = [0 for _ in range(self.n_minibatches)]
                for idx in range(bs):
                    sizes[idx % self.n_minibatches] += 1
                spec = SequenceSplitSpec(sizes=sizes)
                datas = flat_input.split_with_spec(spec)
                logger.info(
                    f"PPO minibatch split (size {self.n_minibatches}): "
                    f"#seqs: {[s.bs for s in datas]}, "
                    f"#tokens: {[sum([sum(lens) for lens in s.seqlens[s._get_split_key()]]) for s in datas]}"
                )

                for mb_i, data in enumerate(datas):
                    train_stat = module.train_batch(
                        input_=data,
                        mb_spec=mb_spec,
                        version_steps=model.version.global_step,
                        loss_fn=_loss_fn,
                        loss_weight_fn=lambda x: x.data[
                            "ppo_loss_mask"
                        ].count_nonzero(),
                        token_normalize_scope=self.token_normalize_scope,
                    )
                    stats_tracker.scalar(**train_stat)
                    all_stats.append(stats_tracker.export())

        model.inc_version()
        all_stats[0].update(global_stats)

        return all_stats
