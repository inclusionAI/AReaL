import os
import shutil

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    RemoteMegatronEngineConfig,
    RolloutControllerConfig,
    TrainControllerConfig,
)
from areal.api.engine_api import WeightUpdateMeta
from areal.controller.rollout_controller import DistributedRolloutController
from areal.controller.train_controller import DistributedTrainController
from areal.dataset.distributed_batch_memory import DistributedBatchMemory
from areal.extension.asystem.math_reward import reward_fn
from areal.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from areal.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from areal.scheduler.asystem import AsystemScheduler
from areal.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer


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

    scheduler = AsystemScheduler(
        {
            "endpoint": "http://asystem-scheduler.asystem-my001-swift.svc.sigma-my001.ml01.sgp-ml.local:8081",
            "expr_name": "areal-test",
            "trial_name": "trial-0",
            "train_config": {
                "image": "xxx",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
            "rollout_config": {
                "image": "xxx",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
        }
    )

    rollout_config = InferenceEngineConfig(experiment_name="areal", trial_name="async")

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(rollout_config),
        RolloutControllerConfig(
            experiment_name="areal",
            trial_name="async",
            allocation_mode="gen:d1t8p1,train:d1t8p1",
        ),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteMegatronEngine(
            RemoteMegatronEngineConfig(experiment_name="areal", trial_name="async")
        ),
        TrainControllerConfig(
            experiment_name="areal",
            trial_name="async",
            allocation_mode="gen:d1t8p1,train:d1t8p1",
        ),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize()

    gconfig = GenerationHyperparameters(
        max_new_tokens=15360, greedy=False, n_samples=16
    )
    MODEL_PATH = "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/moe_lite_0428_base_32k_hgf"
    tokenizer = load_hf_tokenizer(MODEL_PATH)
    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
    )

    dataset = load_dataset(
        "json",
        data_files="/storage/xukuan.xk/repos/antnlp/personal/llm/benchmark/orz_areal_train.jsonl",
    )
    train_dataset = dataset["train"]
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 512
    batch_data = []
    step_num = 100
    epoch_num = 10
    for epoch in range(epoch_num):
        data_generator = iter(dataloader)

        # Submit batches
        if rollout_config.max_head_offpolicyness > 0:
            batch_data = []
            for i in range(rollout_config.max_head_offpolicyness * batch_size):
                batch = next(data_generator)
                batch_data.append(batch)
            rollout.submit(batch_data, workflow=workflow)

        for step in range(step_num):
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
            rollout.update_weights(rollout_cfg)

            # Submit new batch
            batch_data = []
            for _ in range(batch_size):
                batch = next(data_generator)
                batch_data.append(batch)
            rollout.submit(batch_data, workflow=workflow)

            # rollout
            rollout_res = rollout.wait(batch_size, timeout=7200)

            rollout_res = rollout_res.to("cpu").clone()

            rollout_res_dict = rollout_res.to_dict()
            for k, v in rollout_res_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[0] == 1:
                    rollout_res_dict[k] = v.squeeze(0)
            dis_batch = DistributedBatchMemory(rollout_res_dict)
            stats = actor.train_distributed_batch(dis_batch)
            print(
                f"[Trainer] train exec success, step: {step}, epoch: {epoch}, stats: {stats}"
            )


if __name__ == "__main__":
    main_grpo()
