"""
Core functions to implement HGPO (History-based Group Policy Optimization) algorithms.
Self-contained in recipe for upstream verl-agent submission.
"""

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
from functools import reduce


def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def compute_step_discounted_returns(batch: DataProto, gamma: float):

    print("compute_step_discounted_returns")
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

def compute_hgpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    history_length: int,
    epsilon: float = 1e-6,
    mode: str = 'mean_std_norm',
    length_weight_alpha: float = 1.0,
    base_group: bool = False,
):
    """
    Compute HGPO step-level advantages.

    token_level_rewards: (bs, response_length)
    step_rewards: [N], scalar per sample
    base_group: If True, add episode_advantages as initial group to the aggregate_items weight advantage computation
    """
    step_advantages = hgpo_advantage_estimate(
        token_level_rewards,
        step_rewards,
        response_mask,
        anchor_obs,
        index,
        traj_index,
        history_length,
        epsilon=epsilon,
        mode=mode,
        length_weight_alpha=length_weight_alpha,
        base_group=base_group,
    )
    return step_advantages, step_advantages

def trajectory_advantages(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute trajectory-level advantage using mean-std normalization from GiGPO.
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages

def hgpo_advantage_estimate(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    history_length: int,            # max history steps to look at (excluding current), so max k = history_length + 1
    epsilon: float = 1e-6,
    mode: str = "mean_std_norm",    # "mean_std_norm" / "mean_norm"
    length_weight_alpha: float = 1.0,
    base_group: bool = False,
):
    device = response_mask.device
    response_length = response_mask.shape[-1]

    rewards_all = step_rewards.detach().to(device=device, dtype=torch.float32)
    N = anchor_obs.shape[0]
    all_step_adv = torch.zeros(N, device=device, dtype=torch.float32)

    # ----- optional base (trajectory-level) -----
    base_adv_scalars = None
    if base_group:
        base_adv = trajectory_advantages(
            token_level_rewards,
            response_mask,
            index,
            traj_index,
            epsilon=epsilon,
            remove_std=(mode == "mean_norm"),
            compute_mean_std_cross_steps=True,
        ).to(device=device, dtype=torch.float32)

        mask_sum = response_mask.sum(dim=1).clamp(min=1e-8)
        base_adv_scalars = (base_adv * response_mask).sum(dim=1) / mask_sum  # [N]

    # ----- aggregator: items are (k, adv), L=k+1 used for weight -----
    def aggregate_items(items):
        if not items:
            return torch.zeros((), device=device, dtype=torch.float32)

        ks = []
        advs = []
        for (k, a) in items:
            ks.append(k)
            advs.append(a.to(torch.float32))

        advs = torch.stack(advs, dim=0)
        ks_t = torch.tensor(ks, device=device, dtype=torch.float32)

        valid_mask = (advs != 0)
        if valid_mask.any():
            advs_v = advs[valid_mask]
            ks_v = ks_t[valid_mask]
            L = ks_v + 1
            w_raw = L ** length_weight_alpha
            w = w_raw / (w_raw.sum() + epsilon)
            items_val = (w * advs_v).sum()
        else:
            items_val = torch.zeros((), device=device, dtype=torch.float32)

        return items_val

    # ===================== main =====================
    for gid in np.unique(index):
        group_indices = np.flatnonzero(index == gid)
        if group_indices.size == 0:
            continue

        group_obs = anchor_obs[group_indices]
        group_traj_ids = traj_index[group_indices]

        group_idx_t = torch.as_tensor(group_indices, device=device, dtype=torch.long)
        group_rewards = rewards_all.index_select(0, group_idx_t)  # [G]

        uniq_traj, inv = np.unique(group_traj_ids, return_inverse=True)
        n_traj = len(uniq_traj)

        traj_positions = [[] for _ in range(n_traj)]
        for pos, t in enumerate(inv):
            traj_positions[t].append(pos)
        traj_positions = [np.asarray(p, dtype=np.int64) for p in traj_positions]

        traj_obs = [group_obs[p] for p in traj_positions]
        traj_rewards = [
            group_rewards.index_select(0, torch.as_tensor(p, device=device, dtype=torch.long))
            for p in traj_positions
        ]
        traj_gidx = [group_indices[p] for p in traj_positions]

        clusters = defaultdict(list)
        traj_h = []
        for ti in range(n_traj):
            traj_h.append([to_hashable(s) for s in traj_obs[ti]])

        max_k = history_length + 1

        for ti in range(n_traj):
            T = len(traj_h[ti])
            for si in range(T):
                k_upper = min(max_k, si + 1)
                for k in range(1, k_upper + 1):
                    seq = traj_h[ti][si - k + 1 : si + 1]
                    clusters[(k, tuple(seq))].append((ti, si))

        per_step_items = defaultdict(list)

        for (k, _key), members in clusters.items():
            nL = len(members)
            if nL <= 1:
                continue

            r = torch.stack([traj_rewards[ti][si] for (ti, si) in members], dim=0)
            mean = r.mean()
            std = r.std(unbiased=False)

            if mode == "mean_std_norm":
                a = (r - mean) / (std + epsilon)
            elif mode == "mean_norm":
                a = (r - mean)
            else:
                raise ValueError(f"Invalid mode: {mode}")

            for j, (ti, si) in enumerate(members):
                per_step_items[(ti, si)].append((k, a[j]))

        for ti in range(n_traj):
            T = len(traj_obs[ti])
            for si in range(T):
                global_idx = int(traj_gidx[ti][si])
                items = per_step_items.get((ti, si), [])

                if base_group and base_adv_scalars is not None:
                    base_item = (0, base_adv_scalars[global_idx])
                    items = [base_item] + items

                all_step_adv[global_idx] = aggregate_items(items)

    scores = all_step_adv.unsqueeze(-1).expand(-1, response_length) * response_mask
    return scores
