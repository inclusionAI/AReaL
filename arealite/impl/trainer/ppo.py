from typing import Dict, Optional

import torch
import torch.distributed as dist
from datasets import Dataset

from arealite import ppo_functional
from arealite.api.cli_args import TrainerConfig, TrainingArgs
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import LLMRequest
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.trainer_api import Trainer
from arealite.impl.rollout_controller import RolloutController
from arealite.utils import (
    calc_entropy,
    compute_varlen_position_indices,
    concat_padded_tensors,
    gather_logprobs,
    masked_normalization,
    pad_sequences_to_tensors,
    to_device,
    unpad_input,
)
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import stats_tracker


class SpmdRlvrPPOTrainer(Trainer):

    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(
            args, trainer_config, train_dataset, valid_dataset, rollout_controller
        )
        if self.rollout_controller is None:
            raise ValueError("PPO Trainer requires a rollout controller.")

        self.config = config = trainer_config.ppo

        # Create actor model
        engine_factory = EngineFactory(args)
        self.actor = engine_factory.make_engine(config.actor)

        self.actor_tokenizer = load_hf_tokenizer(config.actor.path)

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
        self.group_size = self.config.group_size

    def _setup_models(self):
        # TODO: disable drop out for all models
        self.actor.init_distributed(self.config.ppo.actor_train)
        if self.ref is not None:
            self.ref.init_distributed()

    def _get_rollout_batch(self, data) -> Dict[str, torch.Tensor]:
        if self.config.use_async:
            # Wait until enough trajectories has been collected.
            trajs = self.rollout_controller.prepare_batch(
                batch_size=self.args.train_dataset.batch_size // dist.get_world_size()
            )
            data = concat_padded_tensors([traj.data for traj in trajs])
            return data

        # Run batched rollout by submitting requests to LLM servers
        reqs = [
            LLMRequest(input_ids=input_ids, gconfig=self.gconfig)
            for input_ids in data["input_ids"]
        ]
        resps = self.rollout_controller.generate_batch(self.llm_client, reqs)
        raw_data = []
        for resp in resps:
            input_len = len(resp.input_tokens)
            output_len = len(resp.output_tokens)
            x = dict(
                input_ids=resp.input_tokens + resp.output_tokens,
                logprobs=[0] * input_len + resp.output_logprobs,
                prompt_mask=[1] * input_len + [0] * output_len,
            )
            raw_data.append(x)

        # Padding will add an attention_mask field
        data = pad_sequences_to_tensors(raw_data)
        return data

    def train(self):
        self._setup_models()
        self.create_train_dataloader()

        if self.config.use_async:
            self.rollout_controller.start_run_episode_loop(self.train_dataloader)

        total_epochs = self.args.exp_ctrl.total_train_epochs
        gconfig = self.gconfig
        for epoch in range(total_epochs):
            for step, prompt in enumerate(self.train_dataloader):
                # Run generation or rollout to collect data
                rollout = self._get_rollout_batch(prompt)
                seqlens = [int(m.sum()) for m in rollout["attention_mask"]]
                total_seqlens = sum(seqlens)
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
                position_ids = compute_varlen_position_indices(
                    input_ids.shape[0], cu_seqlens
                )

                # Transformer forward input data
                # TODO: should pad input data before running forward
                input_data = dict(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=None,
                    position_ids=position_ids.unsqueeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_cache=False,
                )

                # Run reference model forward
                def calc_logprobs(logits, input_data):
                    logits = logits.float()
                    logits /= gconfig.temperature
                    logprobs = gather_logprobs(
                        logits, torch.roll(input_data["input_ids"], shifts=-1)
                    )
                    return logprobs

                if self.ref is not None and self.config.kl_ctl != 0.0:
                    ref_logp = self.ref.forward(
                        input_data,
                        mb_spec=self.config.mb_spec,
                        post_hook=calc_logprobs,
                    )
                else:
                    ref_logp = torch.zeros_like(input_ids, dtype=torch.float32)

                # Recompute logprobs using the current actor model.
                prox_logp = None
                if self.config.recompute_logprob:
                    _logp = self.actor.forward(
                        input_data,
                        mb_spec=self.config.mb_spec,
                        post_hook=calc_logprobs,
                    )
                    if self.config.use_decoupled_loss:
                        prox_logp = _logp
                    else:
                        # Overwrite the logp returned by the inference engine
                        old_logp = _logp

                # TODO: call compute reward functions here.
                reward_score = rollout["rewards"].float()
                ...

                # Shift logprobs and mask for computing loss.
                old_logp, *_ = unpad_input(rollout["logprobs"], attn_mask)
                prompt_mask, *_ = unpad_input(rollout["prompt_mask"], attn_mask)
                ppo_loss_mask = prompt_mask.logical_not()
                ppo_loss_mask = torch.roll(ppo_loss_mask, shifts=-1)
                # Apply the mask to log probabilities.
                ref_logp *= ppo_loss_mask
                old_logp *= ppo_loss_mask

                # Compute KL-regularized rewards and GAEs.
                short1cu_seqlens = cu_seqlens.clone()
                short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
                kl_rewards, rewards = ppo_functional.get_packed_rewards(
                    kl_ctl=self.kl_ctl,
                    clip_reward_value=self.max_reward_clip,
                    log_probs=old_logp,
                    ref_log_probs=ref_logp,
                    reward_score=reward_score,
                    short1cu_seqlens=short1cu_seqlens,
                    seq_no_eos_mask=seq_no_eos_mask,
                    mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
                )
                advantages, _ = ppo_functional.get_packed_advantages_and_returns(
                    gamma=self.discount,
                    lam=self.gae_lambda,
                    values=torch.zeros_like(input_ids, dtype=torch.float32),
                    rewards=rewards,
                    short1cu_seqlens=short1cu_seqlens,
                    seq_no_eos_mask=seq_no_eos_mask,
                )

                # Optionally perform advantage normalization.
                if self.adv_norm:
                    if self.group_adv_norm == False:
                        advantages = masked_normalization(advantages, ppo_loss_mask)
                    else:
                        n_samples = len(cu_seqlens) - 1
                        assert n_samples % self.group_size == 0
                        adv_list = []
                        for i in range(0, n_samples, self.group_size):
                            adv_list.append(
                                masked_normalization(
                                    advantages[
                                        short1cu_seqlens[i] : short1cu_seqlens[
                                            i + self.group_size
                                        ]
                                    ],
                                    ppo_loss_mask[
                                        short1cu_seqlens[i] : short1cu_seqlens[
                                            i + self.group_size
                                        ]
                                    ],
                                    all_reduce=False,
                                )
                            )
                        advantages = torch.cat(adv_list, 0)

                def ppo_loss_fn(
                    logits: torch.FloatTensor, input_data: Dict
                ) -> torch.Tensor:
                    """Loss function for ppo actor step, all inputs should be splitted into
                    pipeline micro batches, returns loss and logging stats."""
                    input_ids = input_data["input_ids"]
                    cu_seqlens = input_data["cu_seqlens"]

                    logits = logits.float()
                    logits /= gconfig.temperature
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

                # TODO: split minibatches
                # Prepare data to be splitted into mini-batches.
                stats = self.actor.train_batch(input_data, loss_fn=ppo_loss_fn)

                # Synchronize weights to the client.
                self.actor.update_weights_to(self.llm_client)
