import resource
import sys
import logging
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

from realhf.base import stats_tracker


def clear_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def main_grpo():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()]
    )
    experiment_name = "arealite"
    trial_name = "helloworld"

    # init controller
    scheduler = AsystemScheduler({
        "endpoint": "http://asystem-scheduler.asystem-my001-swift.svc.sigma-my001.ml01.sgp-ml.local:8081",
        "expr_name": experiment_name,
        "trial_name": trial_name,
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
    })

    dataset = load_dataset("json",
                           data_files="/storage/xukuan.xk/repos/antnlp/personal/llm/benchmark/orz_areal_train.jsonl")
    train_dataset = dataset['train']
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 64
    group_size = 8
    MODEL_PATH = "/storage/liuyongkang.lyk/output_models/moelite-32k-qwen3-640w-ep3-3e4-05250954/hf_ckpts/8604"
    max_new_tokens = 15360
    batch_data = []
    step_num = 100
    epoch_num = 10
    global_step = 0
    os.environ['WANDB_API_KEY'] = 'local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe'
    os.environ["WANDB_BASE_URL"] = "https://slurm.alipay.com"

    logger = StatsLogger(StatsLoggerConfig(
        experiment_name=experiment_name, trial_name=trial_name,fileroot="/storage/openpsi/experiments",
        wandb=WandBConfig(
            mode="online",
        ),

    ), FinetuneSpec(total_train_epochs=epoch_num, dataset_size=step_num * batch_size ,train_batch_size=batch_size))
    logger.info(f"total_epochs={epoch_num} step_per_epoch={step_num}")

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(InferenceEngineConfig(experiment_name=experiment_name, trial_name=trial_name)),
        RolloutControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode="sglang.d4t8p1+d32t1p1"),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteMegatronEngine(RemoteMegatronEngineConfig(experiment_name=experiment_name, trial_name=trial_name)),
        TrainControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode="sglang.d4t8p1+d32t1p1"),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize()
    
    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            batch_data = []
            for _ in range(batch_size):
                batch = next(data_generator)
                batch_data.append(batch)

            # Update inference engine weights

            # actor_cfg = WeightUpdateMeta(
            #     type="disk",
            #     path=f"/storage/openpsi/checkpoints/{exp_name}/{trial_name}/",
            #     alloc_mode=None,
            #     comm_backend=None,
            # )
            rollout_cfg = WeightUpdateMeta(
                type="disk",
                path=f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}/{step}",
                alloc_mode=None,
                comm_backend=None,
            )

            # actor.upload_weights(actor_cfg)
            # print("[Trainer] actor upload_weights success.")
            # rollout.update_weights(rollout_cfg)
            # print("[Trainer] rollout update_weights success.")
            # clear_dir(rollout_cfg.path)
            # print(f"[Trainer] clear update weights dir success: {rollout_cfg.path}")

            # synchronous rollout

            tokenizer = load_hf_tokenizer(MODEL_PATH)
            gconfig = GenerationHyperparameters(
                max_new_tokens=max_new_tokens, greedy=False, n_samples=group_size
            )

            workflow = RLVRWorkflow(
                reward_fn=reward_fn,
                gconfig=gconfig,
                tokenizer=tokenizer,
            )
            with (
                stats_tracker.record_timing("rollout_step"),
                stats_tracker.scope("grpo_actor"),
            ):
                # input_: List[Dict[str, tensor]]
                rollout_res = rollout.rollout(batch_data, workflow=workflow)
                print(f"[Trainer] rollout exec success, rollout_res: {rollout_res}")
                rollout_res = rollout_res.to("cpu").clone()

                # torch.save(rollout_res, "rollout_res.pt")
                rollout_res_dict = rollout_res.to_dict()
                for k, v in rollout_res_dict.items():
                    if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[0] == 1:
                        rollout_res_dict[k] = v.squeeze(0)
                        # print(f"[Trainer] dzq_debug rollout squeeze: key: {k}, shape: {rollout_res_dict[k].shape}")
                torch.set_printoptions(threshold=float('inf'))
                print(f"[Trainer] after rollout rewards: {rollout_res_dict["rewards"]}")
                dis_batch = DistributedBatchMemory(rollout_res_dict)

            print(f"debug: step1", flush=True)
            time.sleep(10)
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(f"内存使用: {mem_usage / 1024:.2f} MB")
            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("grpo_actor"),
            ):
                actor.train_distributed_batch(dis_batch)
                print(f"[Trainer] train exec success, step: {step}, epoch: {epoch}")

            print(f"debug: step2", flush=True)
            with (
                stats_tracker.record_timing("weights_update_step"),
                stats_tracker.scope("grpo_actor"),
            ):
                rollout.update_weights(rollout_cfg)
                print(f"[Trainer] rollout update_weights success. step: {step}, epoch: {epoch}")

            print(f"debug: step3", flush=True)
            metric = stats_tracker.export()
            logger.commit(epoch, step, global_step, metric)

            global_step += 1
            print(f"debug: step4", flush=True)


    logger.close()



if __name__ == "__main__":
    main_grpo()


