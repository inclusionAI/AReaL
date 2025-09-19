import torch

from areal.api.cli_args import (
    RemoteMegatronEngineConfig,
    TrainControllerConfig,
)
from areal.controller.train_controller import DistributedTrainController
from areal.dataset.distributed_batch_memory import DistributedBatchMemory
from areal.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from areal.scheduler.local import LocalScheduler


def main_grpo():
    scheduler = LocalScheduler({})

    actor = DistributedTrainController(
        RemoteMegatronEngine(
            RemoteMegatronEngineConfig(experiment_name="ff", trial_name="ff")
        ),
        TrainControllerConfig(),
        scheduler,
    )

    actor.initialize()
    rollout_res = torch.load("rollout_res.pt", weights_only=False)
    rollout_res = rollout_res.to("cpu").clone()
    print(f"[Trainer] rollout_res from file: {rollout_res}")
    rollout_res_dict = rollout_res.to_dict()
    for k, v in rollout_res_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[1] == 1:
            rollout_res_dict[k] = v.squeeze(1)
    dis_batch = DistributedBatchMemory(rollout_res_dict)
    stats = actor.train_distributed_batch(dis_batch)
    print(f"[Trainer] train exec success, stats: {stats}")


if __name__ == "__main__":
    main_grpo()
