import functools
import warnings
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level geometric mean ratios and average advantages per sequence (GSPO).

    Args:
        log_ratio: Log of probability ratios (logprobs - proximal_logprobs)
        advantages: Per-token advantages
        loss_mask: Boolean mask indicating valid tokens
        cu_seqlens: Cumulative sequence lengths. Required for 1D tensors (packed format).
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            For a single sequence, use cu_seqlens=torch.tensor([0, seq_len]).

    Returns:
        ratio: Sequence-level importance sampling ratios (broadcast to all tokens)
        advantages: Sequence-averaged advantages (broadcast to all tokens)
            Note: We use mean instead of sum to keep gradient magnitude independent
            of sequence length. When multiplied by ratio and summed over tokens,
            this gives the correct total gradient contribution per sequence.
    """
    # Handle both 1D (packed) and 2D (padded) tensor shapes
    if log_ratio.ndim == 1:
        # For 1D tensors (packed format), cu_seqlens is required
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens is required for 1D tensors (packed format). "
                "In AReaL, 1D tensors are produced by pack_tensor_dict() and always have cu_seqlens. "
                "For a single sequence, use cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32)."
            )

        # Packed sequences: use cu_seqlens boundaries
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # Create sequence index for each token: [0,0,0,1,1,2,2,2,2,...]
        sequence_idx = torch.arange(
            batch_size, device=log_ratio.device
        ).repeat_interleave(seq_lengths)

        # Use scatter_add for vectorized summation per sequence (faster than Python loop)
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(
            batch_size, device=log_ratio.device, dtype=log_ratio.dtype
        ).scatter_add_(0, sequence_idx, masked_log_ratio)

        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(
            batch_size, device=advantages.device, dtype=advantages.dtype
        ).scatter_add_(0, sequence_idx, masked_advantages)

        valid_count_per_seq = (
            torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
            .scatter_add_(0, sequence_idx, loss_mask.int())
            .clamp(min=1)
        )

        # Compute sequence-level means
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(
            log_ratio.dtype
        )
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(
            advantages.dtype
        )

        # Broadcast sequence-level values back to token-level
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)

        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        # For 2D tensors (padded sequences)
        # Input shape: [batch_size, seq_len]
        # Compute mean log ratio over sequence length for each sample
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (
            loss_mask.sum(dim=1).clamp(min=1)
        )
        # Broadcast back to original shape: each sequence gets its own geometric mean ratio
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        # Apply mask
        ratio = torch.where(loss_mask, ratio, 0.0)

        # Average token advantages per sequence
        # This ensures gradient magnitude is independent of sequence length
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths).expand_as(
            log_ratio
        )

    return ratio, advantages


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    behav_imp_weight_cap: float | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.

    Args:
        importance_sampling_level: Level at which to compute importance sampling ratios.
            - 'token': Per-token ratios
            - 'sequence': Sequence-level geometric mean of per-token ratios (GSPO)
        cu_seqlens: Cumulative sequence lengths for packed sequences (1D tensors).
            Required when inputs are 1D and importance_sampling_level='sequence'.
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            Not needed for 2D padded inputs (sequences identified by batch dimension).
    """
    loss_mask_count = loss_mask.count_nonzero() or 1

    if importance_sampling_level == "sequence":
        # GSPO: Compute sequence-level geometric mean of probability ratios
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        # Standard PPO: per-token ratio
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)
    behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()
    behav_mask = (
        (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
        if behav_imp_weight_cap is not None
        else loss_mask
    )
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    pg_loss = pg_loss * behav_imp_weight
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if proximal_logprobs is not None:
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat


def sapo_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    loss_mask: torch.Tensor,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """SAPO (Soft Adaptive Policy Optimization) loss with asymmetric sigmoid gates.

    SAPO replaces PPO clipping with soft sigmoid gates, providing smooth gradients.
    Note: SAPO requires use_decoupled_loss=False.

    Args:
        logprobs: Current policy log probabilities
        old_logprobs: Old policy log probabilities
        advantages: Advantage values
        tau_pos: Temperature for positive advantages (higher = sharper gate)
        tau_neg: Temperature for negative advantages (higher = sharper gate)
        loss_mask: Mask for valid tokens
        importance_sampling_level: "token" or "sequence" level importance sampling
        cu_seqlens: Cumulative sequence lengths for sequence-level IS

    Returns:
        Tuple of (loss, statistics dict compatible with PPO)
    """
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("SAPO temperatures (tau_pos, tau_neg) must be positive.")
    loss_mask_count = loss_mask.count_nonzero() or 1
    advantages = advantages.detach()
    log_ratio = logprobs - old_logprobs

    if importance_sampling_level == "sequence":
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    # SAPO: Asymmetric sigmoid gates with 4/τ gradient normalization
    gate_pos = torch.sigmoid(tau_pos * (ratio - 1.0))
    gate_neg = torch.sigmoid(tau_neg * (ratio - 1.0))
    scale_pos = 4.0 / tau_pos
    scale_neg = 4.0 / tau_neg
    scaled_gate_pos = gate_pos * scale_pos
    scaled_gate_neg = gate_neg * scale_neg

    # Select gate based on advantage sign
    is_positive = advantages > 0
    soft_gate = torch.where(is_positive, scaled_gate_pos, scaled_gate_neg)

    # Compute loss
    pg_loss = -soft_gate * advantages
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count

    # Return stat dict compatible with PPO (fake clip_mask for logging compatibility)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),  # SAPO doesn't clip
        dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        # SAPO-specific stats (scaled gates for consistency)
        sapo_soft_gate=soft_gate.detach(),
        sapo_gate_pos=scaled_gate_pos.detach(),
        sapo_gate_neg=scaled_gate_neg.detach(),
    )

    return pg_loss, stat


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
    diff = torch.abs(x - y)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


def _mse_loss(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * (x - y) ** 2


def ppo_critic_loss_fn(
    value: torch.FloatTensor,
    old_value: torch.FloatTensor,
    target_value: torch.FloatTensor,
    value_eps_clip: float,
    loss_mask: torch.Tensor | None = None,
    loss_fn_type: str = "mse",
) -> tuple[torch.Tensor, dict]:
    """Compute PPO critic loss function given padded batch inputs.

    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.

    Args:
        value (torch.FloatTensor): Values. The position of the final token is not included.
            (The whole generated sequence is not a state.)
        old_value (torch.FloatTensor): Old values.
        target_value (torch.FloatTensor): Returns computed by GAE.
        value_eps_clip (float): Clip ratio.
        loss_mask (Optional[torch.Tensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.
        loss_fn_type (str, optional): Type of loss function. Defaults to 'mse'.

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    assert value.dtype == torch.float32
    assert old_value.dtype == torch.float32
    assert target_value.dtype == torch.float32

    if loss_fn_type == "huber":
        loss_fn = functools.partial(_huber_loss, delta=10.0)
    elif loss_fn_type == "mse":
        loss_fn = _mse_loss
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    if target_value.is_inference():
        target_value = target_value.clone()  # clone a inference tensor

    value_loss_original = loss_fn(value, target_value)

    value_clipped = old_value + (value - old_value).clamp(
        -value_eps_clip, value_eps_clip
    )

    value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    with torch.no_grad():
        clip_mask = value_loss_clipped.detach() > value_loss_original.detach()
        if loss_mask is not None:
            clip_mask.logical_and_(loss_mask)

        stat = dict(clip_mask=clip_mask, loss=value_loss.detach())

    if loss_mask is not None:
        value_loss = (
            torch.where(loss_mask, value_loss, 0).sum() / loss_mask.count_nonzero()
        )
    else:
        value_loss = value_loss.mean()

    return value_loss, stat


def dynamic_sampling(
    data: dict[str, Any], group_size: int
) -> tuple[dict[str, Any], dict[str, int]]:
    """Filter samples by group when all rewards in a group are equal.

    Assumes samples of the same group are adjacent in the batch.

    Returns a new dict containing only kept samples (mask applied on batch dim
    for all tensor values whose first dimension equals batch size), and a small
    stats dict.
    """
    rewards = data["rewards"]
    if not torch.is_tensor(rewards):
        raise TypeError("data['rewards'] must be a torch.Tensor")
    batch_size = rewards.shape[0]

    if group_size <= 0:
        warnings.warn("group_size <= 0; returning original data")
        return data, dict(n_group_kept=0, n_group_filtered=0)

    if batch_size % group_size != 0:
        warnings.warn(
            "The group size is not divisible by the batch size. Return the original data"
        )
        return data, dict(
            n_group_kept=batch_size // max(group_size, 1), n_group_filtered=0
        )

    # Calculate number of groups (must be divisible)
    num_groups = batch_size // group_size

    # Reshape rewards to (num_groups, group_size) for group-wise operations
    rewards_reshaped = rewards.view(num_groups, group_size)

    # Check if all elements in each group are equal to the first element
    all_equal = (rewards_reshaped == rewards_reshaped[:, 0:1]).all(dim=1)

    # Create mask for groups to keep (where not all rewards are equal)
    valid_groups = ~all_equal

    # Expand the group mask to individual samples
    mask = valid_groups.repeat_interleave(group_size)

    # In case all group is filtered out, return the original data (although not gradient in this case)
    if not mask.any():
        return data, dict(n_group_kept=0, n_group_filtered=num_groups)

    n_group_kept = int(valid_groups.sum().item())
    n_group_filtered = int(num_groups - n_group_kept)

    # Apply mask row-wise across tensors that share the same batch dimension
    filtered: dict[str, Any] = {}
    for k, v in data.items():
        if torch.is_tensor(v) and v.shape[:1] == (batch_size,):
            filtered[k] = v[mask]
        else:
            # keep untouched (e.g., scalars, metadata); caller should ensure consistency
            filtered[k] = v
    return filtered, dict(n_group_kept=n_group_kept, n_group_filtered=n_group_filtered)


# code modified from VERL: https://github.com/volcengine/verl/blob/main/verl/workers/reward_manager/dapo.py
def reward_overlong_penalty(
    data: dict[str, Any],
    overlong_tokens: int,
    overlong_penalty_factor: float,
    max_response_length: int,
) -> dict[str, Any]:
    reward_score = data["rewards"]
    input_ids = data["input_ids"]
    response_lengths = (data["loss_mask"].sum(dim=-1)).long()
    batch_size = input_ids.shape[0]
    for sample_idx in range(batch_size):
        reward_score_cur = reward_score[sample_idx]
        response_length_cur = response_lengths[sample_idx]
        expected_len = max_response_length - overlong_tokens
        exceed_len = response_length_cur - expected_len
        overlong_reward = min(
            -exceed_len / overlong_tokens * overlong_penalty_factor, 0
        )
        reward_score_cur += overlong_reward
        reward_score[sample_idx] = reward_score_cur

    data["rewards"] = reward_score
    return data
