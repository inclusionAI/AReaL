import functools
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    PROX_LOGP_METHODS_SKIP_FORWARD,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    dynamic_sampling,
    gather_logprobs,
    gather_logprobs_entropy,
    ppo_actor_loss_fn,
    reward_overlong_penalty,
)
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger(__name__)


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.group_size = config.group_size

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature
        self.dynamic_sampling = config.dynamic_sampling

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.behav_imp_weight_cap:
                logger.info(
                    f"  Importance weight cap: {config.behav_imp_weight_cap:.1f} "
                    "(filters out tokens with extreme weights)"
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info(f"  group_size: {config.group_size}")
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(
        self,
        data: dict[str, Any],
    ) -> torch.Tensor | None:
        # Determine if forward pass is needed based on prox_logp_method
        # - loglinear: Skip forward pass (use approximation)
        # - recompute/metrics: Do forward pass
        if self.config.use_decoupled_loss:
            if self.config.prox_logp_method in PROX_LOGP_METHODS_SKIP_FORWARD:
                return None  # Skip forward pass, use approximation
        else:
            # Standard PPO: follow recompute_logprob flag
            if not self.config.recompute_logprob:
                return None

        def calc_logprobs(logits, input_data):
            labels = input_data.get(
                "rolled_input_ids",
                torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
            )
            logprobs = gather_logprobs(logits, labels, self.temperature)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

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
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
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
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
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
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

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

        return data

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: dict[str, Any]) -> None:
        with stats_tracker.scope("dynamic_sampling"):
            if self.dynamic_sampling and len(data["rewards"]) % self.group_size == 0:
                data, sampling_stat = dynamic_sampling(data, self.group_size)
                stats_tracker.scalar(**sampling_stat)

        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
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
        ########## Logging code ends ##########

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

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
                        m2_threshold=self.m2_threshold,
                        importance_sampling_level=self.config.importance_sampling_level,
                        current_version=current_version,
                        prox_logp_method=self.config.prox_logp_method,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


class FSDPPPOActor(FSDPEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


class MegatronPPOActor(MegatronEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: dict,
    temperature: float,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    # Use rolled input_ids. Ulysses SP will roll input_ids in ulysses_prepare_inputs().
    labels = input_data.get(
        "rolled_input_ids",
        torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
    )
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    # Use full loss_mask. Ulysses SP will slice loss_mask in ulysses_prepare_inputs().
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    # Check if prox_logp is None (happens when compute_logp() was skipped)
    prox_logp_is_none = prox_logp_gt is None
    if prox_logp_is_none:
        # prox_logp is None, should only happen with loglinear method
        if prox_logp_method not in PROX_LOGP_METHODS_SKIP_FORWARD:
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        # For loglinear, we must have versions available to compute approximation
        if "versions" not in input_data:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{PROX_LOGP_METHOD_LOGLINEAR}' but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    logprobs, entropy = gather_logprobs_entropy(logits, labels, temperature)
    entropy = entropy.detach()

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    # Handle different prox_logp_method values
    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if (
            prox_logp_is_none
            and "versions" in input_data
            and current_version is not None
        ):
            versions = input_data["versions"]
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs.detach(),
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )
    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    # If m2_threshold is set, use M2PO loss function.
    if m2_threshold is not None:
        delta = old_logp - prox_logp
        m2 = delta * delta
        mask_flat = loss_mask.view(-1)
        m2_selected = m2.view(-1)[mask_flat]
        if m2_selected.numel() == 0:
            full_loss_mask = loss_mask
        else:
            sorted_m2, indices = torch.sort(m2_selected, descending=True)
            restored_indices = torch.argsort(indices)
            sorted_m2_loss_mask = get_m2po_loss_mask(
                sorted_m2=sorted_m2, m2_threshold=m2_threshold
            )
            m2_selected_mask = sorted_m2_loss_mask[restored_indices]
            m2_full_flat = torch.zeros_like(
                mask_flat, dtype=torch.bool, device=loss_mask.device
            )
            m2_full_flat[mask_flat] = m2_selected_mask
            full_loss_mask = m2_full_flat.view_as(loss_mask)
        loss_mask = full_loss_mask

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
        importance_sampling_level=importance_sampling_level,
        cu_seqlens=input_data.get("cu_seqlens"),
    )

    # Log training statistics
    stats_tracker.denominator(
        # NOTE: n_tokens must have shape [batch, seq] to match vocab stats below (lines 762-767).
        # Using torch.ones_like(loss_mask) ensures correct shape when this function is called
        # standalone (e.g., by recipe/AEnt or tests), not just from ppo_update() which already
        # registers n_tokens at line 401.
        n_tokens=torch.ones_like(loss_mask, dtype=torch.bool, device=logits.device),
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

    # Compute and log proximal approximation metrics
    # Requirements: versions available, metrics mode enabled, and has ground truth
    can_log_metrics = (
        "versions" in input_data
        and current_version is not None
        and prox_logp_method == PROX_LOGP_METHOD_METRICS
        and not prox_logp_is_none  # Only log metrics when we have ground truth
    )

    if can_log_metrics:
        versions = input_data["versions"]

        # Compute all approximation methods for comparison (method=None means compute all)
        approximations = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs.detach(),
            versions=versions,
            current_version=current_version,
            method=None,  # Compute all methods for metrics comparison
        )

        # Use the same mask as actual training (respects behav_imp_weight_cap)
        # If behave_mask exists, use it; otherwise use loss_mask
        approx_metrics_mask = stat.get("behave_mask", loss_mask)

        # Log metrics under compute_logp scope
        with stats_tracker.scope("compute_logp"):
            # Set denominator within this scope (using the capped mask)
            stats_tracker.denominator(n_valid_tokens=approx_metrics_mask.bool())

            # Log ground truth proximal logp
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

            # For each approximation method, compute and log metrics
            for method_name, approx_logp in approximations.items():
                # Absolute error (against ground truth)
                abs_error = torch.abs(prox_logp_gt - approx_logp).float()

                # Relative error (percentage)
                rel_error = torch.abs(
                    (prox_logp_gt - approx_logp) / (torch.abs(prox_logp_gt) + 1e-8)
                ).float()

                # Squared error (for variance computation)
                squared_error = ((prox_logp_gt - approx_logp) ** 2).float()

                # Importance weight errors
                # Ground truth: exp(log_p_proximal_gt - log_p_behave)
                behav_imp_weight_gt = torch.exp(prox_logp_gt - old_logp).float()
                # Approximated: exp(log_p_approx - log_p_behave)
                behav_imp_weight_approx = torch.exp(approx_logp - old_logp).float()
                imp_weight_error = torch.abs(
                    behav_imp_weight_gt - behav_imp_weight_approx
                ).float()
                imp_weight_rel_error = torch.abs(
                    (behav_imp_weight_gt - behav_imp_weight_approx)
                    / (behav_imp_weight_gt + 1e-8)
                ).float()

                # Log all metrics for this method
                stats_tracker.stat(
                    **{
                        f"{method_name}/approx_logp": approx_logp.float(),
                        f"{method_name}/abs_error": abs_error,
                        f"{method_name}/rel_error": rel_error,
                        f"{method_name}/squared_error": squared_error,
                        f"{method_name}/behav_imp_weight": behav_imp_weight_approx,
                        f"{method_name}/imp_weight_abs_error": imp_weight_error,
                        f"{method_name}/imp_weight_rel_error": imp_weight_rel_error,
                    },
                    denominator="n_valid_tokens",
                )

    # Log version difference metrics (sample staleness)
    # This does NOT require ground truth, only version information
    if "versions" in input_data and current_version is not None:
        versions = input_data["versions"]

        # Use the same mask as actual training (respects behav_imp_weight_cap)
        version_metrics_mask = stat.get("behave_mask", loss_mask)

        with stats_tracker.scope("version_stats"):
            # Set denominator within this scope
            stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

            # Compute version differences (sample staleness)
            v_proximal = current_version - 1
            v_theta = current_version
            v_behave = versions.float()

            # CRITICAL: Filter out prompt tokens (version < 0) and invalid tokens
            # Only consider generated tokens with valid versions
            valid_generated_mask = version_metrics_mask & (versions >= 0)

            # Sample staleness: how many versions old is this token
            # staleness = current_policy_version - token_generation_version
            sample_staleness_proximal = (
                v_proximal - v_behave
            )  # Relative to proximal policy
            sample_staleness_theta = v_theta - v_behave  # Relative to current policy

            # Only compute stats on valid generated tokens
            if valid_generated_mask.any():
                # Get staleness values for valid tokens only
                staleness_proximal_valid = sample_staleness_proximal[
                    valid_generated_mask
                ]
                staleness_theta_valid = sample_staleness_theta[valid_generated_mask]

                # Compute scalar statistics
                stats_tracker.scalar(
                    sample_staleness_proximal_avg=staleness_proximal_valid.float()
                    .mean()
                    .item(),
                    sample_staleness_proximal_max=staleness_proximal_valid.float()
                    .max()
                    .item(),
                    sample_staleness_proximal_min=staleness_proximal_valid.float()
                    .min()
                    .item(),
                    sample_staleness_theta_avg=staleness_theta_valid.float()
                    .mean()
                    .item(),
                    sample_staleness_theta_max=staleness_theta_valid.float()
                    .max()
                    .item(),
                    sample_staleness_theta_min=staleness_theta_valid.float()
                    .min()
                    .item(),
                    v_theta=v_theta,
                    v_proximal=v_proximal,
                    n_valid_generated_tokens=valid_generated_mask.sum().item(),
                )

    return loss


def get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask
