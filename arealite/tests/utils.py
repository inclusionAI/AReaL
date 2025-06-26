import random

import torch

from arealite.api.io_struct import Trajectory, TrajStats


def mock_rollout_output(bs, n_samples):
    trajs = []
    min_seqlen, max_seqlen = 8, 16
    for _ in range(bs * n_samples):
        input_len = random.randint(min_seqlen, max_seqlen)
        prompt_len = random.randint(1, min_seqlen - 1)
        input_ids = torch.randint(0, 100, (input_len,))
        prompt_mask = torch.tensor([1] * prompt_len + [0] * (input_len - prompt_len))
        logprobs = -torch.randn(input_len).abs()
        versions = torch.zeros(input_len)
        traj = Trajectory(
            prompt=None,
            data=dict(
                input_ids=input_ids.unsqueeze(0),
                prompt_mask=prompt_mask.unsqueeze(0),
                logprobs=logprobs.unsqueeze(0),
                versions=versions.unsqueeze(0),
                rewards=torch.tensor([random.random()]),
            ),
            stats=TrajStats(
                start_time=0,
                total_reward=0,
                episode_length=1,
                info={},
            ),
        )
        trajs.append(traj)

    return trajs
