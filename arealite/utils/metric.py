import torch
from tensordict import TensorDict

from arealite.controller.utils import group_avg_torch
from realhf.base import logging, stats_tracker
from realhf.api.core.data_api import RL_TASKS

def get_old_logp(tensordict):
    prompt_mask = tensordict["prompt_mask"]
    seqlen = tensordict["seqlen"]
    logprobs = tensordict["logprobs"]
    batch, seq_len = logprobs.shape
    idx = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    valid_mask = (idx < seqlen.unsqueeze(1)) & (prompt_mask == 0)
    return logprobs[valid_mask]


def calc_training_data_metrics(padded: TensorDict):
    assert isinstance(padded, TensorDict)
    old_logp = get_old_logp(padded)
    old_p = torch.exp(old_logp)
    entropy = -torch.mean(old_p * old_logp)

    prompt_len = padded["prompt_mask"].sum(1)

    stats_tracker.scalar(**{"prompt_len": prompt_len.float().mean(),
                            "sglang_old_logp": old_logp.mean(),
                            "entropy": entropy.item()})

    # total reward
    keys = ["rewards", "seqlen"]
    for key in keys:
        tensor = padded[key]
        for value in tensor:
            stats_tracker.scalar(**{key: value})

    # task reward
    groups, rewards = group_avg_torch(
        padded["rewards"], padded["task_ids"])
    for group, reward in zip(groups, rewards):
        task_name = RL_TASKS[int(group)]
        stats_tracker.scalar(**{f"{task_name}_reward": float(reward)})

    batch_size = torch.tensor(padded.batch_size)

    # eos ratio
    no_eos_num = padded["seq_no_eos_mask"].sum()
    eos_ratio = (batch_size - no_eos_num) / batch_size
    stats_tracker.scalar(**{"eos_ratio": eos_ratio,
                            "eos_num": batch_size - no_eos_num})

    # seqlen of eos query
    eos_seqlen = padded["seqlen"][padded["seq_no_eos_mask"].bool().logical_not()].float().mean()
    stats_tracker.scalar(**{"eos_seqlen": eos_seqlen.item()})

def calc_training_data_version_metrics(padded: TensorDict, current_model_version: int):
    batch_size = torch.tensor(padded.batch_size)

    # stale metrics
    stale_metric = {
        "stale_num": 0
    }
    for b in range(batch_size.item()):
        query_start_version = padded["versions"][b][0].item()
        stale_version = current_model_version - query_start_version
        if stale_version > 0:
            stale_metric["stale_num"] += 1
            stale_metric_name = f"stale_{stale_version}_version_num"
            if stale_metric_name not in stale_metric:
                stale_metric[stale_metric_name] = 0
            stale_metric[stale_metric_name] += 1

    stale_metric["stale_ratio"] = stale_metric["stale_num"] / batch_size
    stats_tracker.scalar(**stale_metric)
