import sys

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    RemoteMegatronEngineConfig,
    RolloutControllerConfig,
    TrainControllerConfig,
    TrainEngineConfig,
    load_expr_config,
)
from arealite.api.engine_api import InferenceEngine
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.dataset.utils import process_rl_dataset
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler
from arealite.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer


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
