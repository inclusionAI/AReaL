# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from datasets import Dataset

from arealite import ppo_functional
from arealite.api.cli_args import MicroBatchSpec, TrainerConfig, TrainingArgs
from arealite.api.engine_api import EngineFactory
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.trainer_api import Trainer
from arealite.impl.rollout_controller import RolloutController
from arealite.utils import (
    calc_entropy,
    compute_varlen_position_indices,
    concat_padded_tensors,
    dict_of_list2list_of_dict,
    dict_split_mbs,
    gather_logprobs,
    list_of_dict2dict_of_list,
    masked_normalization,
    to_device,
    unpack_sequence,
    unpad_input,
)
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, stats_tracker

logger = logging.getLogger("SPMD RLVR PPO Trainer")


@dataclass
class UnpaddedRolloutOutput:
    loaded_data: Dict[str, Any]
    model_inputs: Dict[str, Any]
    prompt_mask: torch.BoolTensor
    seq_no_eos_mask: Optional[torch.BoolTensor] = None
    logprobs: Optional[torch.Tensor] = None
    rewards: Optional[torch.FloatTensor] = None


class SpmdPPOTrainer(Trainer):

    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(
            args,
            trainer_config,
            train_dataset,
            valid_dataset,
            rollout_controller,
        )
        if self.rollout_controller is None:
            raise ValueError("PPO Trainer requires a rollout controller.")

        self.config = config = trainer_config.ppo

        # Create actor model
        engine_factory = EngineFactory(args)
        self.actor = engine_factory.make_engine(config.actor)

        self.actor_tokenizer = load_hf_tokenizer(config.actor.path)
        self.gconfig = args.rollout.gconfig

        # Create reference model is specified
        self.ref = None
        if config.ref is not None:
            self.ref = engine_factory.make_engine(config.ref)

        # Create a client to generate responses and update weights
        client_factory = LLMClientFactory(args)
        self.llm_client = client_factory.make_client(config.inf_service)

        # Algorithm related attributes
        self.kl_ctl = self.config.kl_ctl
        self.discount = self.config.discount
        self.gae_lambda = self.config.gae_lambda
        self.adv_norm = self.config.adv_norm
        self.max_reward_clip = self.config.max_reward_clip
        self.group_adv_norm = self.config.group_adv_norm
        self.group_size = self.args.rollout.gconfig.n_samples

    def _setup_models(self):
        # TODO: disable dropout
        self.actor.init_distributed()
        if self.ref is not None:
            self.ref.init_distributed()

    def _get_rollout_batch(self):
        if self.config.async_training:
            # Wait until enough trajectories has been collected.
            trajs = self.rollout_controller.prepare_batch(
                batch_size=self.args.train_dataset.batch_size // dist.get_world_size()
            )
            data = concat_padded_tensors([traj.data for traj in trajs])
            prompt = list_of_dict2dict_of_list([traj.prompt for traj in trajs])
            return prompt, data

        # Run batched rollout by submitting requests to LLM servers
        prompt = next(self.data_generator)
        env_options = dict_of_list2list_of_dict(prompt)
        trajs = self.rollout_controller.generate_batch(
            batch_size=len(env_options),
            env_options=env_options,
        )
        data = concat_padded_tensors([traj.data for traj in trajs])
        prompt = list_of_dict2dict_of_list([traj.prompt for traj in trajs])
        return prompt, data

    def _rollout_step(self) -> UnpaddedRolloutOutput:
        # Run generation or rollout to collect data
        loaded_data, rollout = self._get_rollout_batch()
        [int(m.sum()) for m in rollout["attention_mask"]]
        rollout = to_device(rollout)

        # Marks which sequence does not has an EOS token, i.e.,
        # generation is truncated by the configured maximum generation length
        batch_tokens = rollout["input_ids"]
        seq_no_eos_mask = (
            batch_tokens[:, -1] != self.actor_tokenizer.eos_token_id
        ).logical_and(batch_tokens[:, -1] != self.actor_tokenizer.pad_token_id)

        # Remove padding to use flash-attn
        attn_mask = rollout["attention_mask"]
        input_ids, cu_seqlens, max_seqlen, _ = unpad_input(
            rollout["input_ids"], attn_mask
        )
        position_ids = compute_varlen_position_indices(input_ids.shape[0], cu_seqlens)

        # Transformer forward input data
        input_data = dict(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_cache=False,
        )
        old_logp, *_ = unpad_input(rollout["logprobs"], attn_mask)
        prompt_mask, *_ = unpad_input(rollout["prompt_mask"], attn_mask)
        return UnpaddedRolloutOutput(
            loaded_data=loaded_data,
            model_inputs=input_data,
            logprobs=old_logp,
            prompt_mask=prompt_mask,
            seq_no_eos_mask=seq_no_eos_mask,
            rewards=rollout.get("rewards", None),
        )

    def _train_step(self, rollout_output: UnpaddedRolloutOutput):
        input_ids = rollout_output.model_inputs["input_ids"].squeeze(0)
        n_seqs = rollout_output.seq_no_eos_mask.shape[0]
        assert n_seqs == self.local_train_batch_size * self.group_size, (
            n_seqs,
            self.group_size,
            self.local_train_batch_size,
        )

        # Run reference model forward
        def calc_logprobs(logits, input_data):
            logits = logits.squeeze(0).float()
            labels = torch.roll(input_data["input_ids"].squeeze(0), shifts=-1)
            logits /= self.gconfig.temperature
            logprobs = gather_logprobs(logits, labels)
            return logprobs

        if self.ref is not None and self.config.kl_ctl != 0.0:
            ref_logp = self.ref.forward(
                rollout_output.model_inputs,
                mb_spec=self.config.mb_spec,
                post_hook=calc_logprobs,
            )
        else:
            ref_logp = torch.zeros_like(input_ids, dtype=torch.float32)

        # Recompute logprobs using the current actor model.
        prox_logp = None
        old_logp = rollout_output.logprobs
        if self.config.recompute_logprob:
            _logp = self.actor.forward(
                rollout_output.model_inputs,
                mb_spec=self.config.mb_spec,
                post_hook=calc_logprobs,
            )
            if self.config.use_decoupled_loss:
                prox_logp = _logp
            else:
                # Overwrite the logp returned by the inference engine
                old_logp = _logp

        # Compute rewards using the reward function in synchronous RLVR pipeline.
        reward_score = rollout_output.rewards
        if self.config.group_reward_norm:
            for i in range(n_seqs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                r = reward_score[s]
                reward_score[s] = (r - r.mean()) / (r.std() + 1e-9)

        # Shift logprobs and mask for computing loss.
        ppo_loss_mask = rollout_output.prompt_mask.logical_not()
        ppo_loss_mask = torch.roll(ppo_loss_mask, shifts=-1)
        # Apply the mask to log probabilities.
        ref_logp *= ppo_loss_mask
        old_logp *= ppo_loss_mask

        # Compute KL-regularized rewards and GAEs.
        cu_seqlens = rollout_output.model_inputs["cu_seqlens"]
        seq_no_eos_mask = rollout_output.seq_no_eos_mask
        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_ctl,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            cu_seqlens=cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
        )
        advantages, _ = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=torch.zeros(
                input_ids.shape[0] + n_seqs,
                device=input_ids.device,
                dtype=torch.float32,
            ),
            rewards=rewards,
            short1cu_seqlens=cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform advantage normalization.
        if self.adv_norm:
            if self.group_adv_norm:
                n_samples = len(cu_seqlens) - 1
                assert n_samples % self.group_size == 0
                adv_list = []
                for i in range(0, n_samples, self.group_size):
                    adv_list.append(
                        masked_normalization(
                            advantages[cu_seqlens[i] : cu_seqlens[i + self.group_size]],
                            ppo_loss_mask[
                                cu_seqlens[i] : cu_seqlens[i + self.group_size]
                            ],
                            all_reduce=False,
                        )
                    )
                advantages = torch.cat(adv_list, 0)
            else:
                advantages = masked_normalization(advantages, ppo_loss_mask)

        # Prepare data to be splitted into mini-batches.
        ppo_global_batch = dict(
            **rollout_output.model_inputs,
            old_logp=old_logp,
            advantages=advantages,
            old_logp=old_logp,
            ppo_loss_mask=ppo_loss_mask,
        )
        input_lens = (
            rollout_output.model_inputs["cu_seqlens"][1:]
            - rollout_output.model_inputs["cu_seqlens"][:-1]
        )
        lens = input_lens.cpu().numpy().tolist()

        all_stats = []
        with stats_tracker.scope("ppo_actor"):
            ########## Logging code starts ##########
            result_denominators = {
                "correct_n_seqs": (reward_score > 0).bool(),
                "incorrect_n_seqs": (reward_score <= 0).bool(),
            }
            global_denominators = dict(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                n_tokens=torch.ones_like(ppo_loss_mask, dtype=torch.bool),
                n_valid_tokens=ppo_loss_mask.bool(),
                **result_denominators,
            )
            stats_tracker.denominator(**global_denominators)
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
            stats_tracker.stat(**stats, denominator="n_valid_tokens")

            seq_stats = dict(
                no_eos_ratios=seq_no_eos_mask.float(),
                task_reward=reward_score,
                prompt_len=ppo_loss_mask.logical_not.float(),
                seq_len=input_lens.float(),
            )
            stats_tracker.stat(**seq_stats, denominator="n_seqs")
            scalars = dict(
                mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
                eps_clip=self.config.eps_clip,
                use_prox_logp=prox_logp is not None,
            )
            if self.config.c_clip is not None:
                scalars["c_clip"] = self.config.c_clip
                scalars["use_dual_clip"] = 1
            else:
                scalars["use_dual_clip"] = 0
            if self.config.behav_imp_weight_cap is not None:
                scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
            stats_tracker.scalar(**scalars)

            global_stats = stats_tracker.export()
            for k in global_denominators:
                global_stats.pop(f"ppo_actor/{k}")
            ########## Logging code ends ##########

            for mb in dict_split_mbs(
                ppo_global_batch,
                mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
                lens=lens,
            ):
                model_inputs = {k: mb[k] for k in rollout_output.model_inputs}
                loss_fn = self._get_ppo_loss_fn(
                    old_logp=mb["old_logp"],
                    advantages=mb["advantages"],
                    ppo_loss_mask=mb["ppo_loss_mask"],
                    prox_logp=prox_logp if self.config.use_decoupled_loss else None,
                )
                train_stat = self.actor.train_batch(
                    model_inputs,
                    loss_fn=loss_fn,
                    mb_spec=self.config.mb_spec,
                    loss_weight_fn=lambda x: x.data["ppo_loss_mask"].count_nonzero(),
                    token_normalize_scope=self.token_normalize_scope,
                )
                stats_tracker.scalar(**train_stat)
                all_stats.append(stats_tracker.export())

        self.actor.inc_version()
        all_stats[0].update(global_stats)
        return all_stats

    def _get_ppo_loss_fn(self, old_logp, advantages, ppo_loss_mask, prox_logp):
        def ppo_loss(logits: torch.Tensor, input_data: Dict):
            """Loss function for ppo actor step, all inputs should be splitted into
            pipeline micro batches, returns loss and logging stats."""
            input_ids = input_data["input_ids"].squeeze(0)
            cu_seqlens = input_data["cu_seqlens"]

            logits = logits.squeeze(0).float()
            logits /= self.gconfig.temperature
            logprobs = gather_logprobs(logits, torch.roll(input_ids, shifts=-1))
            loss, ppo_stat = ppo_functional.actor_loss_fn(
                logprobs=logprobs,
                old_logprobs=old_logp,
                advantages=advantages,
                eps_clip=self.config.eps_clip,
                loss_mask=ppo_loss_mask,
                c_clip=self.config.c_clip,
                proximal_logprobs=prox_logp,
                behav_imp_weight_cap=self.config.behav_imp_weight_cap,
            )

            entropy = calc_entropy(logits=logits, cu_seqlens=cu_seqlens)

            # Log training statistics
            stats_tracker.denominator(
                n_tokens=torch.ones(
                    logits.shape[0], dtype=torch.bool, device=logits.device
                ),
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
                stats_tracker.denominator(
                    unclipped_behave_tokens=ppo_stat["behave_mask"]
                )
                stats_tracker.stat(
                    behave_imp_weight=ppo_stat["behave_imp_weight"],
                    behave_approx_kl=ppo_stat["behave_approx_kl"],
                    denominator="unclipped_behave_tokens",
                )
            vocab_min_logits = logits.detach().min(-1).values.float()
            vocab_max_logits = logits.detach().max(-1).values.float()
            stats_tracker.stat(
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
                denominator="n_tokens",
            )

            clip_mask = ppo_stat["clip_mask"]
            clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
            clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
            stats_tracker.stat(
                clipped_new_logp=clipped_new_logp,
                clipped_old_logp=clipped_old_logp,
                denominator="clipped_tokens",
            )
            return loss

        return ppo_loss

    def train(self):
        self._setup_models()
        self.create_train_dataloader()

        if self.config.async_training:
            self.rollout_controller.start_generate_loop()

        total_epochs = self.args.exp_ctrl.total_train_epochs
        global_step = 0
        for epoch in range(total_epochs):
            for step, data in enumerate(self.train_dataloader):
                self.rollout_controller.submit(data)
                if self.config.async_training:
                    # Submitted data will not actually be sent for rollout.
                    # The rollout controller over-subscribe the data to
                    # ensure that there are enough data being generated.
                    if global_step < self.args.rollout.max_head_offpolicyness + 1:
                        continue

                # Run rollout
                rollout_output = self._rollout_step()

                # Run RL training and update weights.
                self._train_step(rollout_output)

                # Synchronize weights to the client.
                self.actor.update_weights_to(self.llm_client)

                # IMPORTANT! Update the version for staleness control
                self.rollout_controller.set_version(self.actor.get_version())
                # TODO: update version through name resolve

                global_step += 1
