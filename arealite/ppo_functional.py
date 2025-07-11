import functools
from typing import Dict, Optional, Tuple

import torch
import torch.distributed

from realhf.base import pkg_version


def actor_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    c_clip: Optional[float] = None,
    proximal_logprobs: Optional[torch.Tensor] = None,
    behav_imp_weight_cap: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    denorm_logprobs = (
        proximal_logprobs if proximal_logprobs is not None else old_logprobs
    )
    loss_mask_count = loss_mask.count_nonzero() or 1
    ratio = torch.where(loss_mask, torch.exp(logprobs - denorm_logprobs), 0)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
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
    if proximal_logprobs is not None:
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
        approx_kl=(logprobs - denorm_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if proximal_logprobs is not None:
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
    diff = torch.abs(x - y)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


def _mse_loss(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * (x - y) ** 2


def critic_loss_fn(
    value: torch.Tensor,
    old_value: torch.Tensor,
    target_value: torch.Tensor,
    value_eps_clip: float,
    loss_mask: torch.Tensor,
    loss_fn_type: str = "mse",
) -> Tuple[torch.Tensor, Dict]:
    if loss_fn_type == "huber":
        loss_fn = functools.partial(_huber_loss, delta=10.0)
    elif loss_fn_type == "mse":
        loss_fn = _mse_loss
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")
    value_loss_original = loss_fn(value, target_value)
    value_clipped = old_value + (value - old_value).clamp(
        -value_eps_clip, value_eps_clip
    )
    value_loss_clipped = loss_fn(value_clipped, target_value)
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    with torch.no_grad():
        clip_mask = value_loss_clipped.detach() > value_loss_original.detach()
        clip_mask.logical_and_(loss_mask)
        stat = dict(clip_mask=clip_mask, loss=value_loss.detach())
    value_loss = torch.where(loss_mask, value_loss, 0).sum() / loss_mask.count_nonzero()
    return value_loss, stat


@torch.no_grad()
def get_packed_rewards(
    kl_ctl: float,
    clip_reward_value: float,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    reward_score: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_no_eos_mask: torch.Tensor,
    mask_no_eos_with_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tot_rewards = -kl_ctl * (log_probs - ref_log_probs)
    tot_rewards[cu_seqlens[1:] - 1] = 0
    kl_rewards = tot_rewards.clone()
    reward_score = reward_score.clip(-clip_reward_value, clip_reward_value)
    indices = torch.clip(cu_seqlens[1:] - 2, min=0)
    if mask_no_eos_with_zero:
        tot_rewards[indices] += torch.where(seq_no_eos_mask, 0, reward_score)
    else:
        tot_rewards[indices] += reward_score
    return kl_rewards, tot_rewards


def pygae1d_nolp_misalign(
    rewards: torch.Tensor,
    values: torch.Tensor,
    cu_seqlens_: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens = cu_seqlens_.clone()
    cu_seqlens[1:] += torch.ones_like(cu_seqlens_[1:]).cumsum(0)
    bs = cu_seqlens_.shape[0] - 1
    assert values.shape[0] == rewards.shape[0] + bs
    advantages_reversed = []
    returns_reversed = []
    for i in reversed(range(bs)):
        v_offset = cu_seqlens[i]
        r_offset, r_end = cu_seqlens_[i], cu_seqlens_[i + 1]
        assert cu_seqlens[i + 1] - v_offset - 1 == r_end - r_offset
        lastgaelam = 0
        for t in reversed(range(r_end - r_offset)):
            nextvalues = values[v_offset + t + 1]
            if t == r_end - r_offset - 1:
                nextvalues *= bootstrap[i]
            delta = rewards[r_offset + t] + gamma * nextvalues - values[v_offset + t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            returns_reversed.append(lastgaelam + values[v_offset + t])
    advantages = torch.stack(advantages_reversed[::-1])
    returns = torch.stack(returns_reversed[::-1])
    return advantages, returns


def cugae1d_nolp_misalign_func(
    rewards: torch.Tensor,
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    truncate: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pkg_version.is_available("cugae"):
        from cugae import cugae1d_nolp_misalign_func as gae_1d_nolp_misalign
    else:
        from realhf._C.cugae import gae_1d_nolp_misalign
    assert len(rewards.shape) == len(values.shape) == len(cu_seqlens.shape) == 1
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == rewards.shape[0]
    return gae_1d_nolp_misalign(rewards, values, cu_seqlens, truncate, gamma, lam)


@torch.no_grad()
def get_packed_advantages_and_returns(
    gamma: float,
    lam: float,
    values: torch.Tensor,
    rewards: torch.Tensor,
    short1cu_seqlens: torch.Tensor,
    seq_no_eos_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rewards.get_device() == -1:
        return pygae1d_nolp_misalign(
            rewards, values, short1cu_seqlens, seq_no_eos_mask, gamma, lam
        )
    try:
        return cugae1d_nolp_misalign_func(
            rewards, values, short1cu_seqlens.int(), seq_no_eos_mask.bool(), gamma, lam
        )
    except ModuleNotFoundError:
        return pygae1d_nolp_misalign(
            rewards, values, short1cu_seqlens, seq_no_eos_mask, gamma, lam
        )
