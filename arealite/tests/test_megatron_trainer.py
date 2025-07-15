import sys
import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import load_expr_config, BaseExperimentConfig, InferenceEngineConfig, TrainEngineConfig, \
    RolloutControllerConfig, TrainControllerConfig, RemoteMegatronEngineConfig
from arealite.api.engine_api import InferenceEngine
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler
from arealite.dataset.utils import process_rl_dataset
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.workflow.rlvr import RLVRWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer


def main_grpo():
    scheduler = LocalScheduler({})

    actor = DistributedTrainController(
        RemoteMegatronEngine(RemoteMegatronEngineConfig(experiment_name="ff", trial_name="ff")),
        TrainControllerConfig(),
        scheduler,
    )

    # actor.initialize()
    rollout_res = torch.load("rollout_res.pt")
    print(f"rollout_res from file: {rollout_res}")
    rollout_res_dict = rollout_res.to_dict()
    dis_batch = DistributedBatchMemory(rollout_res_dict)
    # actor.train_distributed_batch()


if __name__ == "__main__":
    main_grpo()
