# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
from typing import Dict, List, Literal, Optional

import torch
import torch.distributed as dist

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
from realhf.impl.dataset.math_parser import parse_lines_in_parallel
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import concat_prompt_to_generation_output
from realhf.impl.model.utils.functional import (
    build_leave_one_indices,
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("PackedPPOInterface")


def get_score(prompt_ids, generated, query_ids, tokenizer):
    prompt_strs = tokenizer.batch_decode(
        prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    seq_strs = tokenizer.batch_decode(
        generated, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    query_id_strs = [query_id.split("@")[0] for query_id in query_ids]
    return parse_lines_in_parallel(seq_strs, query_id_strs)


def topk(scores, gen_lengths, k) -> list:
    indexed = list(enumerate(zip(scores, gen_lengths)))

    sorted_indices = sorted(indexed, key=lambda x: (x[1][0], x[1][1]), reverse=True)[:k]

    return [idx for idx, _ in sorted_indices]


@torch.compile
@torch.no_grad()
def calc_entropy(logits, cu_seqlens):
    leave_one_indices = build_leave_one_indices(logits, cu_seqlens)
    probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)[leave_one_indices]
    return entropy


def _ppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
    kl_adapter: ppo_functional.KLController,  # const
    eps_clip: float,  # const
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
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
    loss, ppo_stat = ppo_functional.actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=ppo_loss_mask,
        c_clip=c_clip,
        proximal_logprobs=input_.data.get("prox_logp", None),
        behav_imp_weight_cap=behav_imp_weight_cap,
    )

    entropy = calc_entropy(logits=logits, cu_seqlens=cu_seqlens)

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
        entropy=entropy.float(),
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


def splited_sum_bool_tensor(t: torch.BoolTensor, chunk_size=256 * 1024 * 1024) -> int:
    """Sum a boolean tensor by splitting them into chunks and sum the chunks
    separately.

    to avoid memory overhead introduced by torch default sum method
    (which will apply for a block of memory of size `8 * t.numel()`
    bytes.)
    """
    flatten = t.flatten()
    splitted = flatten.split(chunk_size // 8, dim=0)
    r = 0
    for chunk in splitted:
        r += chunk.sum()
    return r


@dataclasses.dataclass
class PPOActor(model_api.ModelInterface):
    engine: xxx
    n_minibatches: int = 4

    # Use dict here to allow argument passing through commandline.
    generation_config: Dict = dataclasses.field(default_factory=dict)

    kl_ctl: float = 0.1

    adv_norm: bool = True
    discount: float = 1.0
    gae_lambda: float = 1.0

    eps_clip: float = 0.2
    c_clip: Optional[float] = None
    behav_imp_weight_cap: Optional[float] = None
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

        self.gconfig = model_api.GenerationHyperparameters(**self.generation_config)
        if self.generation_size is not None:
            assert self.generation_size >= self.group_size
        else:
            self.generation_size = self.group_size
        self.gconfig.n = self.generation_size

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def compute_logps(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
        module = model.module

        module.eval()
        self.engine.forward()

        # Remap the key `packed_prompts` to `packed_input_ids`,
        # because the pipe runner only recognizes `packed_input_ids`.
        # x = SequenceSample.from_default(
        #     ids=input_.ids,
        #     seqlens=input_.seqlens["packed_prompts"],
        #     data=dict(packed_input_ids=input_.data["packed_prompts"]),
        # )

        packed_input_ids = input_.data["packed_prompts"]
        new_input_ids = []
        offset = 0
        for x in input_.seqlens["packed_prompts"]:
            new_input_ids += [
                packed_input_ids[offset : offset + x[0]]
            ] * self.generation_size
            offset += x[0]
        assert offset == sum(x[0] for x in input_.seqlens["packed_prompts"])

        if model.backend_name not in ["vllm", "sglang"]:
            # Replicate prompts
            grouped_input = SequenceSample.from_default(
                ids=list(range(input_.bs * self.generation_size)),
                seqlens=[
                    x[0]
                    for x in input_.seqlens["packed_prompts"]
                    for _ in range(self.generation_size)
                ],
                data=dict(packed_input_ids=torch.cat(new_input_ids)),
            )
        else:
            grouped_input = SequenceSample(
                ids=input_.ids,
                seqlens=dict(packed_input_ids=input_.seqlens["packed_prompts"]),
                keys=["packed_input_ids"],
                dtypes=dict(packed_input_ids=torch.long),
                trailing_shapes=dict(packed_input_ids=()),
                data=dict(packed_input_ids=input_.data["packed_prompts"]),
            )

        res = module.generate(
            input_=grouped_input,
            tokenizer=model.tokenizer,
            gconfig=self.gconfig,
            mb_spec=mb_spec,
        )
        if res is None or res[0] is None:
            return None

        gen_tokens, logprobs, _ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(
            gen_tokens[:, -1] != pad_token_id
        )
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(
            gen_tokens != eos_token_id
        ).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])
        input_seq_lens = [
            x for x in input_.seqlens["packed_prompts"] for _ in range(self.group_size)
        ]
        input_token_ids = torch.cat(new_input_ids)

        if self.generation_size is not None and self.generation_size > self.group_size:

            # best of k
            query_ids = [
                query_id for query_id in input_.ids for _ in range(self.generation_size)
            ]
            scores = get_score(new_input_ids, gen_tokens, query_ids, model.tokenizer)
            input_ids_topk, gen_tokens_topk, logprobs_topk, gen_lengths_topk = (
                [],
                [],
                [],
                [],
            )
            for data_idx in range(0, len(gen_tokens), self.generation_size):
                topk_idx = topk(
                    scores[data_idx : data_idx + self.generation_size],
                    gen_lengths[data_idx : data_idx + self.generation_size],
                    self.group_size,
                )
                topk_idx = [data_idx + x for x in topk_idx]
                gen_tokens_topk += gen_tokens[topk_idx]
                logprobs_topk += logprobs[topk_idx]
                gen_lengths_topk += gen_lengths[topk_idx]
                input_ids_topk += [new_input_ids[x] for x in topk_idx]

            input_token_ids = torch.cat(input_ids_topk)

            gen_tokens = torch.stack(gen_tokens_topk)
            logprobs = torch.stack(logprobs_topk)
            gen_lengths = torch.stack(gen_lengths_topk)
            seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(
                gen_tokens[:, -1] != pad_token_id
            )

        (
            packed_input_ids,
            packed_logprobs,
            _,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=input_token_ids,
            prompt_lengths=torch.tensor(flat2d(input_seq_lens)).to(model.device),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=None,
            gen_lengths=gen_lengths,
        )

        # Partition generated data into groups.
        seqlens = [
            seq_lengths[i * self.group_size : (i + 1) * self.group_size]
            .cpu()
            .numpy()
            .tolist()
            for i in range(input_.bs)
        ]

        data = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_input_ids=packed_input_ids,
            packed_logprobs=packed_logprobs,
            prompt_mask=prompt_mask,
        )

        res = SequenceSample(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "seq_no_eos_mask",
            ],
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                packed_logprobs=(),
                seq_no_eos_mask=(),
            ),
            dtypes=dict(
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                packed_logprobs=torch.float,
                seq_no_eos_mask=torch.bool,
            ),
            seqlens=dict(
                packed_input_ids=seqlens,
                packed_logprobs=[[x - 1 for x in slens] for slens in seqlens],
                prompt_mask=seqlens,
                seq_no_eos_mask=[[1] * self.group_size for _ in seqlens],
            ),
            data=data,
            ids=input_.ids,
            prompt_mask=prompt_mask,
        )

        return res

    @torch.no_grad()
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
        module = model.module
        module.eval()

        # This post_hook will gather log probabilities in mini-batches,
        # reducing peak memory usage.
        def calc_logprobs(logits, input_):
            logits /= self.gconfig.temperature

            input_lens = torch.tensor(input_.seqlens["packed_input_ids"]).view(-1)
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()

            logprobs = gather_packed_shifted_log_probs(
                logits, cu_seqlens, input_.data["packed_input_ids"]
            )
            return logprobs

        input_flattend = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            seqlens=flat2d(input_.seqlens["packed_input_ids"]),
            data=dict(packed_input_ids=input_.data["packed_input_ids"]),
        )
        # add posthook to avoid storing full logits
        logprobs = module.forward(
            input_=input_flattend,
            post_hook=calc_logprobs,
            output_seqlens=[
                [x - 1 for x in slens]
                for slens in input_flattend.seqlens["packed_input_ids"]
            ],
            mb_spec=mb_spec,
        )

        res = SequenceSample(
            keys=["logprobs"],
            ids=input_.ids,
            dtypes=dict(logprobs=model.module.dtype),
            trailing_shapes=dict(logprobs=()),
            data=dict(logprobs=logprobs),
            seqlens=dict(
                logprobs=[
                    [x - 1 for x in slen] for slen in input_.seqlens["packed_input_ids"]
                ]
            ),
        )

        return res

    def train_step(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> Dict | List[Dict]:
        module = model.module
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

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
                    loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]] = False

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
                reward_score=(reward_score),
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
                            prompt_mask[cu_seqlens[i] : cu_seqlens[i + 1]].sum()
                            == prompt_mask[
                                cu_seqlens[i + j] : cu_seqlens[i + j + 1]
                            ].sum()
                        )
                    adv_list.append(
                        masked_normalization(
                            advantages[
                                short1cu_seqlens[i] : short1cu_seqlens[
                                    i + self.group_size
                                ]
                            ],
                            loss_mask[
                                short1cu_seqlens[i] : short1cu_seqlens[
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

            result_denominators = {
                "correct_n_seqs": (reward_score > 0).bool(),
                "incorrect_n_seqs": (reward_score <= 0).bool(),
            }

            global_denominators = dict(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                n_tokens=torch.ones_like(prompt_mask, dtype=torch.bool),
                n_valid_tokens=loss_mask.bool(),
                **task_denominators,
                **result_denominators,
            )
            stats_tracker.denominator(**global_denominators)

            for task in RL_TASKS:
                stats_tracker.stat(
                    **{f"{task}_reward": reward_score}, denominator=f"{task}_n_seqs"
                )

            stats_tracker.stat(
                correct_seq_len=input_lens.float(), denominator="correct_n_seqs"
            )
            stats_tracker.stat(
                incorrect_seq_len=input_lens.float(), denominator="incorrect_n_seqs"
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
            if self.behav_imp_weight_cap is not None:
                scalars["behav_imp_weight_cap"] = self.behav_imp_weight_cap
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
                    behav_imp_weight_cap=self.behav_imp_weight_cap,
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

    # Mock methods for profiling only.
    def _mock_inference(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> SequenceSample:
        prompt_lens = flat2d(dataset_input.seqlens["packed_prompts"])
        seqlens = [x + self.gconfig.max_new_tokens for x in prompt_lens]
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(packed_input_ids=packed_input_ids),
        )

    # Mock methods for profiling only.
    def _mock_train_step(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> Dict:
        prompt_lens = flat2d(dataset_input.seqlens["packed_prompts"])
        bs = len(prompt_lens)
        seqlens = [x + self.gconfig.max_new_tokens for x in prompt_lens]
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        mdtype = module.dtype
        short1_seqlens = [x - 1 for x in seqlens]

        packed_logprobs = torch.randn(
            (sum(short1_seqlens),), dtype=mdtype, device=model.device
        )
        packed_ref_logprobs = torch.randn_like(packed_logprobs)
        prompt_mask = torch.zeros(
            (sum(seqlens),), dtype=torch.bool, device=model.device
        )
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )
        rewards = torch.randn(bs, dtype=mdtype, device=model.device)
        seq_no_eos_mask = torch.randint(
            0, 2, (bs,), dtype=torch.bool, device=model.device
        )
        values = torch.randn(
            (sum(seqlens),),
            dtype=mdtype,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(
                packed_logprobs=packed_logprobs,
                packed_ref_logprobs=packed_ref_logprobs,
                prompt_mask=prompt_mask,
                packed_input_ids=packed_input_ids,
                rewards=rewards,
                seq_no_eos_mask=seq_no_eos_mask,
                values=values,
            ),
        )


def _ppo_critic_loss_from_model_outputs(
    new_values: torch.FloatTensor,
    input_: SequenceSample,
    value_eps_clip: float,
    kl_adapter: ppo_functional.KLController,
    rms=None,
) -> torch.Tensor:

    cu_seqlens = (
        torch.nn.functional.pad(
            torch.tensor(flat2d(input_.seqlens["packed_input_ids"])).cumsum(0),
            (1, 0),
        )
        .int()
        .to(new_values.device)
    )
    ppo_loss_mask = input_.data["ppo_loss_mask"]
    returns = input_.data["returns"]
    values = input_.data["values"]
    kl_rewards = input_.data["kl_rewards"]

    leave_one_indices = torch.cat(
        [
            torch.arange(
                cu_seqlens[i],
                cu_seqlens[i + 1] - 1,
                dtype=torch.long,
                device=cu_seqlens.device,
            )
            for i in range(cu_seqlens.shape[0] - 1)
        ]
    )
    new_values = new_values[leave_one_indices].view(-1).float()
    values = values[leave_one_indices].view(-1).float()

    loss, loss_stat = ppo_functional.critic_loss_fn(
        value=new_values,
        old_value=values,
        target_value=returns,
        value_eps_clip=value_eps_clip,
        loss_mask=ppo_loss_mask,
    )

    if rms is not None:
        denormalized_values = rms.denormalize(new_values)
    else:
        denormalized_values = new_values

    # Logging.
    stats_tracker.denominator(n_valid_tokens=ppo_loss_mask.bool())
    stats_tracker.stat(
        value_loss=loss_stat["loss"],
        clip_ratio=loss_stat["clip_mask"].float(),
        denormalized_values=denormalized_values.detach().float(),
        denominator="n_valid_tokens",
    )

    # Update KL coefficient to be consistent with actor.
    mean_ref_kl = (kl_rewards.detach().float() * ppo_loss_mask).sum()
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    _n_valid_tokens = ppo_loss_mask.count_nonzero().clone()
    dist.all_reduce(_n_valid_tokens, group=constants.data_parallel_group())
    mean_ref_kl /= _n_valid_tokens
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)

    return loss


@dataclasses.dataclass
class PPOCriticInterface(model_api.ModelInterface):
    n_minibatches: int = 4
    enable_save: bool = True
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    disable_value: bool = False

    group_size: int = 1
    mask_no_eos_with_zero: bool = False
    mask_too_long: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True
    token_normalize_scope: Literal["global", "dp"] = "global"

    sample_reuse: int = 1

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

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
        assert model.module.module.config.is_critic
        module = model.module
        module.eval()

        input_flattend = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            seqlens=flat2d(input_.seqlens["packed_input_ids"]),
            data=dict(packed_input_ids=input_.data["packed_input_ids"]),
        )
        if self.disable_value:
            scores = input_.data["packed_input_ids"].new_zeros(dtype=module.dtype)
        else:
            scores = module.forward(input_=input_flattend, mb_spec=mb_spec)

        if scores is None:
            return None
        scores = scores.view(-1)
        # res = SequenceSample.from_default(
        #     ids=input_.ids,
        #     data=dict(values=scores),
        #     seqlens=input_.seqlens["packed_input_ids"],
        # )
        res = SequenceSample(
            keys=["values"],
            ids=input_.ids,
            dtypes=dict(values=module.dtype),
            trailing_shapes=dict(values=()),
            data=dict(values=scores),
            seqlens=dict(values=input_.seqlens["packed_input_ids"]),
        )

        return res

    def train_step(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> Dict | List[Dict]:
        assert model.module.module.config.is_critic

        if self.disable_value:
            return dict()

        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.tensor(
            flat2d(input_.seqlens["packed_input_ids"]), device=model.device
        )
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        reward_score = input_.data["rewards"].float()
        if "dense_rewards" in input_.data:
            dense_reward_score = input_.data["dense_rewards"].float()
        values = input_.data["values"].float()
        seq_no_eos_mask = input_.data["seq_no_eos_mask"]
        if self.kl_adapter.value == 0:
            ref_logp: torch.FloatTensor = reward_score.new_zeros(
                int(input_lens.sum()) - len(input_lens)
            )
        else:
            ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()

        if self.mask_too_long:
            for i in range(seq_no_eos_mask.shape[0]):
                if seq_no_eos_mask[i]:
                    loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]] = False

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
                reward_score=(reward_score),
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                mask_no_eos_with_zero=self.mask_no_eos_with_zero,
            )
        _, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
            normalized_returns = self.rms.normalize(returns)
        else:
            normalized_returns = returns

        # Prepare data to be splitted into mini-batches.
        flat_input = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            data=dict(
                returns=normalized_returns,
                values=values,
                ppo_loss_mask=loss_mask,
                packed_input_ids=input_.data["packed_input_ids"],
                kl_rewards=kl_rewards,
            ),
            seqlens=[int(x) for x in input_lens.cpu().numpy().tolist()],
        )

        # Logging.
        with stats_tracker.scope("ppo_critic"):
            stats_tracker.denominator(n_valid_tokens=loss_mask)
            stats_tracker.stat(returns=returns, denominator="n_valid_tokens")

            def _loss_fn(out, inp):
                return _ppo_critic_loss_from_model_outputs(
                    out,
                    inp,
                    value_eps_clip=self.value_eps_clip,
                    kl_adapter=self.kl_adapter,
                    rms=None if not self.value_norm else self.rms,
                )

            # Run mini-batched PPO training!
            for reuse in range(self.sample_reuse):
                with stats_tracker.scope(f"reuse{reuse}"):
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
                        with stats_tracker.scope(f"mb{mb_i}"):
                            stats = module.train_batch(
                                input_=data,
                                mb_spec=mb_spec,
                                version_steps=model.version.global_step,
                                loss_fn=_loss_fn,
                                loss_weight_fn=lambda x: x.data[
                                    "ppo_loss_mask"
                                ].count_nonzero(),
                                token_normalize_scope=self.token_normalize_scope,
                            )
                            stats_tracker.scalar(**stats)

        model.inc_version()

        return stats_tracker.export()

    # Mock methods for profiling only.
    def _mock_inference(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> SequenceSample:
        seqlens = flat2d(dataset_input.seqlens["packed_prompts"])
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(packed_input_ids=packed_input_ids),
        )

    # Mock methods for profiling only.
    def _mock_train_step(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> Dict:
        seqlens = flat2d(dataset_input.seqlens["packed_prompts"])
        bs = len(seqlens)
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        mdtype = module.dtype
        short1_seqlens = [x - 1 for x in seqlens]

        packed_logprobs = torch.randn(
            (sum(short1_seqlens),), dtype=mdtype, device=model.device
        )
        packed_ref_logprobs = torch.randn_like(packed_logprobs)
        prompt_mask = torch.zeros(
            (sum(seqlens),), dtype=torch.bool, device=model.device
        )
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )
        rewards = torch.randn(bs, dtype=mdtype, device=model.device)
        seq_no_eos_mask = torch.randint(
            0, 2, (bs,), dtype=torch.bool, device=model.device
        )
        values = torch.randn(
            (sum(seqlens),),
            dtype=mdtype,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(
                packed_logprobs=packed_logprobs,
                packed_ref_logprobs=packed_ref_logprobs,
                prompt_mask=prompt_mask,
                packed_input_ids=packed_input_ids,
                rewards=rewards,
                seq_no_eos_mask=seq_no_eos_mask,
                values=values,
            ),
        )


model_api.register_interface("ppo_actor", PPOActorInterface)
model_api.register_interface("ppo_critic", PPOCriticInterface)
