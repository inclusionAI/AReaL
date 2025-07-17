import sys
import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import load_expr_config, BaseExperimentConfig, InferenceEngineConfig, TrainEngineConfig, \
    RolloutControllerConfig, TrainControllerConfig, RemoteMegatronEngineConfig
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.workflow.rlvr import RLVRWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta

import os
import shutil


def clear_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def main_grpo():
    # init controller
    scheduler = LocalScheduler({})

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(InferenceEngineConfig(experiment_name="ff", trial_name="ff")),
        RolloutControllerConfig(),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteMegatronEngine(RemoteMegatronEngineConfig(experiment_name="ff", trial_name="ff")),
        TrainControllerConfig(),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize()

    dataset = load_dataset("json",
                           data_files="/storage/xukuan.xk/repos/antnlp/personal/llm/benchmark/orz_areal_train_32.jsonl")
    train_dataset = dataset['train']
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 8
    batch_data = []
    step_num = 2
    epoch_num = 1
    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            batch_data = []
            for _ in range(batch_size):
                batch = next(data_generator)
                batch_data.append(batch)

            # Update inference engine weights
            exp_name = "ff"
            trial_name = "ff"
            actor_cfg = WeightUpdateMeta(
                type="disk",
                path=f"/storage/openpsi/checkpoints/{exp_name}/{trial_name}/",
                alloc_mode=None,
                comm_backend=None,
            )
            rollout_cfg = WeightUpdateMeta(
                type="disk",
                path=f"/storage/openpsi/checkpoints/{exp_name}/{trial_name}/{step}",
                alloc_mode=None,
                comm_backend=None,
            )

            actor.upload_weights(actor_cfg)
            print(f"[Trainer] actor upload_weights success.")


if __name__ == "__main__":
    main_grpo()
