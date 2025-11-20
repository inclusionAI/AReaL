import torch

from realhf.api.core.data_api import RL_TASKS
from realhf.base import stats_tracker


def get_old_logp(tensordict: dict[str, torch.Tensor]):
    prompt_mask = tensordict["prompt_mask"]
    seqlen = tensordict["seqlen"]
    logprobs = tensordict["logprobs"]
    batch, seq_len = logprobs.shape
    idx = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    valid_mask = (idx < seqlen.unsqueeze(1)) & (prompt_mask == 0)
    return logprobs[valid_mask]


def calc_training_data_metrics(padded: dict[str, torch.Tensor]):
    old_logp = get_old_logp(padded)
    old_p = torch.exp(old_logp)
    entropy = -torch.mean(old_p * old_logp)

    prompt_len = padded["prompt_mask"].sum(1)

    stats_tracker.scalar(
        **{
            "prompt_len": prompt_len.float().mean(),
            "sglang_old_logp": old_logp.mean(),
            "entropy": entropy.item(),
        }
    )

    # total reward
    keys = ["rewards", "seqlen"]
    for key in keys:
        tensor = padded[key]
        for value in tensor:
            stats_tracker.scalar(**{key: value})

    # task reward
    groups, rewards = group_avg_torch(padded["rewards"], padded["task_ids"])
    for group, reward in zip(groups, rewards):
        task_name = RL_TASKS[int(group)]
        stats_tracker.scalar(**{f"{task_name}_reward": float(reward)})

    batch_size = torch.tensor(padded["rewards"].shape[0])

    # eos ratio
    no_eos_num = padded["seq_no_eos_mask"].sum()
    eos_ratio = (batch_size - no_eos_num) / batch_size
    stats_tracker.scalar(**{"eos_ratio": eos_ratio, "eos_num": batch_size - no_eos_num})

    # seqlen of eos query
    eos_seqlen = (
        padded["seqlen"][padded["seq_no_eos_mask"].bool().logical_not()].float().mean()
    )
    stats_tracker.scalar(**{"eos_seqlen": eos_seqlen.item()})


def calc_training_data_group_metrics(padded: dict[str, torch.Tensor], group_size: int):
    # 全对全错比例
    # sample 按分组放置在一起
    perfect_group_num = 0
    failed_group_num = 0

    group_by_task = {}
    batch_size = padded["rewards"].shape[0]
    task_num = batch_size // group_size
    for i in range(task_num):
        task_id = padded["task_ids"][i * group_size].item()
        if task_id not in group_by_task:
            group_by_task[task_id] = {"perfect": 0, "failed": 0, "total": 0}
        group_by_task[task_id]["total"] += 1
        group_rewards = padded["rewards"][i * group_size : (i + 1) * group_size]
        if (group_rewards > 0).all():
            perfect_group_num += 1
            group_by_task[task_id]["perfect"] += 1
        if (group_rewards <= 0).all():
            failed_group_num += 1
            group_by_task[task_id]["failed"] += 1

    stats_tracker.scalar(
        **{
            "perfect_group_ratio": perfect_group_num / task_num,
            "perfect_group_num": perfect_group_num,
            "failed_group_num": failed_group_num,
            "failed_group_ratio": failed_group_num / task_num,
        }
    )

    for task_id, v in group_by_task.items():
        task_name = RL_TASKS[int(task_id)]
        stats_tracker.scalar(
            **{
                f"{task_name}_perfect_group_ratio": v["perfect"] / v["total"],
                f"{task_name}_failed_group_ratio": v["failed"] / v["total"],
                f"{task_name}_perfect_group_num": v["perfect"],
                f"{task_name}_failed_group_num": v["failed"],
                f"{task_name}_total_group_num": v["total"],
            }
        )


def calc_training_data_version_metrics(
    padded: dict[str, torch.Tensor], current_model_version: int
):
    batch_size = torch.tensor(padded["rewards"].shape[0])

    # stale metrics
    stale_metric = {"stale_num": 0}
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


# same as avg(origin_tensor) group by (group_tensor)
def group_avg_torch(origin_tensor, group_tensor):
    unique_groups, inverse_indices = torch.unique(
        group_tensor, sorted=True, return_inverse=True
    )
    result = torch.zeros_like(unique_groups, dtype=torch.float)
    sum_per_group = result.scatter_add(0, inverse_indices, origin_tensor.float())
    counts = torch.bincount(inverse_indices, minlength=len(unique_groups)).float()
    avgs = torch.where(
        counts > 0, sum_per_group / counts, torch.zeros_like(sum_per_group)
    )
    return unique_groups, avgs
