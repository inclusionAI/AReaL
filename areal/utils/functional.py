import functools
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F

from areal.platforms import is_npu_available
from areal.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


def _gather_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


def _gather_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


# remove torch.compile due to npu problems
if not is_npu_available:
    _gather_logprobs = torch.compile(_gather_logprobs)
    _gather_logprobs_entropy = torch.compile(_gather_logprobs_entropy)


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]

    if batch_size <= chunk_size:
        return _gather_logprobs(logits, labels, temperature)

    log_probs_labels_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs = _gather_logprobs(chunk_logits, chunk_labels, temperature)

        log_probs_labels_list.append(chunk_log_probs)

    return torch.cat(log_probs_labels_list)


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]
    log_probs_labels_list = []
    entropy_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs, chunk_entropy = _gather_logprobs_entropy(
            chunk_logits, chunk_labels, temperature
        )

        log_probs_labels_list.append(chunk_log_probs)
        entropy_list.append(chunk_entropy)

    logprobs = torch.cat(log_probs_labels_list)
    entropy = torch.cat(entropy_list)

    if get_ulysses_sequence_parallel_world_size() > 1:
        sp_group = get_ulysses_sequence_parallel_group()
        logprobs = dist_F.all_gather(logprobs, group=sp_group)
        logprobs = torch.cat(logprobs, dim=-1)
        entropy = dist_F.all_gather(entropy, group=sp_group)
        entropy = torch.cat(entropy, dim=-1)

    return logprobs, entropy


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
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


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: Optional[float] = None,
    c_clip: Optional[float] = None,
    behav_imp_weight_cap: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.
    """
    loss_mask_count = loss_mask.count_nonzero() or 1
    ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)

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
    loss_mask: Optional[torch.Tensor] = None,
    loss_fn_type: str = "mse",
) -> Tuple[torch.Tensor, Dict]:
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
    data: Dict[str, Any], group_size: int
) -> Tuple[Dict[str, Any], Dict[str, int]]:
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
    filtered: Dict[str, Any] = {}
    for k, v in data.items():
        if torch.is_tensor(v) and v.shape[:1] == (batch_size,):
            filtered[k] = v[mask]
        else:
            # keep untouched (e.g., scalars, metadata); caller should ensure consistency
            filtered[k] = v
    return filtered, dict(n_group_kept=n_group_kept, n_group_filtered=n_group_filtered)


# code modified from VERL: https://github.com/volcengine/verl/blob/main/verl/workers/reward_manager/dapo.py
def reward_overlong_penalty(
    data: Dict[str, Any],
    overlong_tokens: int,
    overlong_penalty_factor: float,
    max_response_length: int,
) -> Dict[str, Any]:
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
