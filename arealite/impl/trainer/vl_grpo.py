# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0
import functools
from typing import List, Optional
from collections.abc import MutableSequence
import torch
from datasets import Dataset
from arealite.api.cli_args import (
    TrainerConfig,
    TrainingArgs,
    MicroBatchSpec,
)
from arealite.api.io_struct import Trajectory
from arealite import ppo_functional
from arealite.system.rollout_controller import RolloutController
from arealite.utils import (
    compute_varlen_position_indices,
    concat_padded_tensors,
    gather_logprobs,
    masked_normalization,
    split_dict_tensor_with_cu_seqlens,
    to_device,
    unpad_input,
)
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from realhf.base import logging
from .grpo import SpmdGRPOTrainer, grpo_loss_fn
from PIL.Image import Image as ImageObject
from realhf.base import logging, stats_tracker
logger = logging.getLogger("VL GRPO Trainer", "system")


class VL_SpmdGRPOTrainer(SpmdGRPOTrainer):
    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(args, trainer_config, train_dataset, valid_dataset, rollout_controller)
        self.actor_processor, self.actor_tokenizer = load_hf_processor_and_tokenizer(self.config.actor.path)
        
    def _train_step(self, trajs: List[Trajectory]):
        rollout = concat_padded_tensors([traj.data for traj in trajs])
        rollout = to_device(rollout, torch.cuda.current_device())

        # Marks which sequence does not has an EOS token, i.e.,
        # generation is truncated by the configured maximum generation length
        batch_tokens = rollout["input_ids"]
        images = [traj.images for traj in trajs]


        if isinstance(images, MutableSequence) and all(isinstance(i, MutableSequence) and all(isinstance(x, str) for x in i) for i in images):
            #paths/url to images
            #convert to double list
            tmp_images=[]
            for image_list in images:
                image_list = [ImageObject.open(image) for image in images]
                tmp_images.append(image_list)
            images = tmp_images

        assert all(isinstance(image, ImageObject) for image_list in images for image in image_list),(
            "All images should be PIL.Image objects, but got: "
            f"{[type(image) for image_list in images for image in image_list]}"
        )
        processed_inputs = self.actor_processor.image_processor(
                    images=images,
                    return_tensors="pt",
                )
        pixel_values = processed_inputs["pixel_values"]
        image_grid_thw = processed_inputs["image_grid_thw"]
        batch_sizes=image_grid_thw.shape[0]
        assert all(image_grid_thw_[0] == 1 for image_grid_thw_ in image_grid_thw), (
            "All data should have 1 image, but got: "
            f"{[image_grid_thw_[0] for image_grid_thw_ in image_grid_thw]}"
        )
        pixel_values = pixel_values.reshape(
            batch_sizes, -1, *pixel_values.shape[1:]
        )
        pixel_values = pixel_values.to(batch_tokens.device)
        image_grid_thw = image_grid_thw.to(batch_tokens.device)

        seq_no_eos_mask = (
            batch_tokens[:, -1] != self.actor_tokenizer.eos_token_id
        ).logical_and(batch_tokens[:, -1] != self.actor_tokenizer.pad_token_id)

        # Remove padding to use flash-attn
        attn_mask = rollout["attention_mask"]
        input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            rollout["input_ids"], attn_mask
        )
        position_ids = compute_varlen_position_indices(input_ids.shape[0], cu_seqlens)

        # Transformer forward input data
        model_inputs = dict(
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values.unsqueeze(1),
            image_grid_thw=image_grid_thw.unsqueeze(1),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_cache=False,
        )
        old_logp, *_ = unpad_input(rollout["logprobs"], attn_mask)
        prompt_mask, *_ = unpad_input(rollout["prompt_mask"], attn_mask)
        # Shift logprobs and mask for computing loss.
        loss_mask = prompt_mask.logical_not()
        loss_mask = torch.roll(loss_mask, shifts=-1)
        old_logp = torch.roll(old_logp, shifts=-1)

        input_ids = model_inputs["input_ids"].squeeze(0)
        n_seqs = seq_no_eos_mask.shape[0]
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
            return logprobs.unsqueeze(0)

        if self.ref is not None and self.config.kl_ctl != 0.0:
            ref_logp = self.ref.forward(
                model_inputs,
                mb_spec=self.config.mb_spec,
                post_hook=calc_logprobs,
            ).squeeze(0)
        else:
            ref_logp = torch.zeros_like(input_ids, dtype=torch.float32)

        # Recompute logprobs using the current actor model.
        prox_logp = None
        if self.config.recompute_logprob:
            _logp = self.actor.forward(
                model_inputs,
                mb_spec=self.config.mb_spec,
                post_hook=calc_logprobs,
            ).squeeze(0)
            if self.config.use_decoupled_loss:
                prox_logp = _logp
            else:
                # Overwrite the logp returned by the inference engine
                old_logp = _logp

        # Compute rewards using the reward function in synchronous RLVR pipeline.
        reward_score = rollout["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(reward_score, max=self.max_reward_clip)
        if self.config.group_reward_norm:
            for i in range(n_seqs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                r = reward_score[s]
                reward_score[s] = (r - r.mean()) / (r.std() + 1e-9)

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards and GAEs.
        cu_seqlens = model_inputs["cu_seqlens"]
        seq_no_eos_mask = seq_no_eos_mask
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
                            loss_mask[cu_seqlens[i] : cu_seqlens[i + self.group_size]],
                            all_reduce=False,
                        )
                    )
                advantages = torch.cat(adv_list, 0)
            else:
                advantages = masked_normalization(advantages, loss_mask)

        # Prepare data to be splitted into mini-batches.
        global_batch = dict(
            **model_inputs,
            old_logp=old_logp,
            advantages=advantages,
            loss_mask=loss_mask,
            prox_logp=prox_logp,
        )
        input_lens = model_inputs["cu_seqlens"][1:] - model_inputs["cu_seqlens"][:-1]

        all_stats = []
        with stats_tracker.scope("actor"):
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

            prompt_lens = []
            for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                prompt_lens.append(prompt_mask[s:e].sum())
            prompt_lens = torch.tensor(prompt_lens, device=reward_score.device)
            seq_stats = dict(
                no_eos_ratios=seq_no_eos_mask.float(),
                task_reward=reward_score,
                prompt_len=prompt_lens.float(),
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
                global_stats.pop(f"actor/{k}")
            ########## Logging code ends ##########

            mb_inputs = split_dict_tensor_with_cu_seqlens(
                global_batch,
                mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
            )
            for mb in mb_inputs.mbs:
                model_inputs = {k: mb[k] for k in model_inputs}
                train_stat = self.actor.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        temperature=self.gconfig.temperature,
                        eps_clip=self.config.eps_clip,
                        c_clip=self.config.c_clip,
                        behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                    ),
                    mb_spec=self.config.mb_spec,
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)
                all_stats.append(stats_tracker.export())
        all_stats[0].update(global_stats)
        return all_stats