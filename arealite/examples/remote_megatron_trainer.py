import resource
import sys
import time

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import load_expr_config, BaseExperimentConfig, InferenceEngineConfig, TrainEngineConfig, \
    RolloutControllerConfig, TrainControllerConfig, RemoteMegatronEngineConfig, StatsLoggerConfig, WandBConfig
from arealite.api.io_struct import FinetuneSpec
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.utils.stats_logger import StatsLogger
from arealite.workflow.rlvr import RLVRWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
import os
import shutil

from realhf.base import logging, stats_tracker

def clear_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

logger = logging.getLogger("Trainer")

def main_grpo():
    experiment_name = "arealite"
    trial_name = "helloworld"

    # init controller
    scheduler = AsystemScheduler({
        "endpoint": "http://asystem-scheduler.asystem-my001-swift.svc.sigma-my001.ml01.sgp-ml.local:8081",
        "expr_name": experiment_name,
        "trial_name": trial_name,
        "train": {
            "worker": {
                "image": "",
                "cmd": "",
                "cpu": 4,
                "memory": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
            "engine": {
                "image": "",
                "cmd": "",
                "cpu": 4,
                "memory": 20,
                "gpu": 1,
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
        },
        "rollout": {
            "worker": {
                "image": "",
                "cmd": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
            "engine": {
                "image": "",
                "cmd": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
        }
    })

    dataset = load_dataset("json",
                           data_files="/storage/xinyu.kxy/data/moe_lite_math_0527_merge_train_areal.jsonl")
    train_dataset = dataset['train']
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 64
    group_size = 8
    MODEL_PATH = "/storage/liuyongkang.lyk/output_models/moelite-32k-qwen3-640w-ep3-3e4-05250954/hf_ckpts/8604"
    max_prompt_len = 1024
    max_new_tokens = 15360
    step_num = 1145
    epoch_num = 10
    global_step = 0
    os.environ['WANDB_API_KEY'] = 'local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe'
    os.environ["WANDB_BASE_URL"] = "https://slurm.alipay.com"
    deploy_mode = "colocation"

    stats_logger = StatsLogger(StatsLoggerConfig(
        experiment_name=experiment_name, trial_name=trial_name,fileroot="/storage/openpsi/experiments",
        wandb=WandBConfig(
            mode="online",
        ),

    ), FinetuneSpec(total_train_epochs=epoch_num, dataset_size=step_num * batch_size ,train_batch_size=batch_size))
    stats_logger.info(f"total_epochs={epoch_num} step_per_epoch={step_num}")

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(InferenceEngineConfig(experiment_name=experiment_name, trial_name=trial_name)),
        RolloutControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode="gen:d4t8p1,train:d32t1p1"),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteMegatronEngine(RemoteMegatronEngineConfig(experiment_name=experiment_name, trial_name=trial_name)),
        TrainControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode="gen:d4t8p1,train:d32t1p1"),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize(colotion_with=rollout if deploy_mode == "colocation" else None)

    tokenizer = load_hf_tokenizer(MODEL_PATH)
    gconfig = GenerationHyperparameters(
        max_new_tokens=max_new_tokens, greedy=False, n_samples=group_size
    )

    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
    )

    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            with (
                stats_tracker.record_timing("e2e"),
                stats_tracker.scope("grpo_actor"),
            ):
                batch_data = []
                while len(batch_data) < batch_size:
                    batch = next(data_generator)
                    prompt = batch["prompt"] if isinstance(batch, dict) else batch[0]["prompt"]
                    tokenized = tokenizer(prompt, truncation=False, return_length=True)
                    if tokenized["length"][0] <= max_prompt_len:
                        batch_data.append(batch)
                    else:
                        logger.warning(f"Ignored prompt with length {tokenized['length'][0]} > {max_prompt_len}")

                if len(batch_data) < batch_size:
                    break


                rollout_cfg = WeightUpdateMeta(
                    type="disk",
                    path=f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}/{step}",
                    alloc_mode=None,
                    comm_backend=None,
                )

                with (
                    stats_tracker.record_timing("rollout_step"),
                    stats_tracker.scope("rollout"),
                ):
                    with(stats_tracker.record_timing("main")):
                        logger.info(f"start to rollout, step: {step}, epoch: {epoch}")
                        rollout_res = rollout.rollout(batch_data, workflow=workflow)
                        logger.info(f"rollout succeeded, step: {step}, epoch: {epoch}")

                    with(stats_tracker.record_timing("post_data_process")):
                        rollout_res_dict = rollout_res.to_dict()
                        for k, v in rollout_res_dict.items():
                            if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[0] == 1:
                                rollout_res_dict[k] = v.squeeze(0)
                        torch.set_printoptions(threshold=float('inf'))
                        logger.info(f"after rollout rewards: {rollout_res_dict["rewards"]}")
                        dis_batch = DistributedBatchMemory(rollout_res_dict)

                with (
                    stats_tracker.record_timing("train_step"),
                    stats_tracker.scope("train"),
                ):
                    logger.info(f"start to train, step: {step}, epoch: {epoch}")
                    actor.train_distributed_batch(dis_batch)
                    logger.info(f"train succeeded, step: {step}, epoch: {epoch}")

                with (
                    stats_tracker.record_timing("weights_update_step"),
                    stats_tracker.scope("weights_update"),
                ):
                    logger.info(f"start to update weight, step: {step}, epoch: {epoch}")
                    rollout.update_weights(rollout_cfg)
                    logger.info(f"update weight succeeded, step: {step}, epoch: {epoch}")

                metric = stats_tracker.export()
                stats_logger.commit(epoch, step, global_step, metric)

            global_step += 1


    stats_logger.close()



if __name__ == "__main__":
    main_grpo()


