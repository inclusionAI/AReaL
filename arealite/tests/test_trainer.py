import sys
import logging
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
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()]
    )
    # init controller
    scheduler = AsystemScheduler({
        "endpoint": "http://asystem-scheduler.asystem-my001-swift.svc.sigma-my001.ml01.sgp-ml.local:8081",
        "expr_name": "arealite-test",
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
    })

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(InferenceEngineConfig(experiment_name="arealite", trial_name="sync")),
        RolloutControllerConfig(experiment_name="arealite", trial_name="sync", allocation_mode="sglang.d4t8p1+d32t1p1"),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteMegatronEngine(RemoteMegatronEngineConfig(experiment_name="arealite", trial_name="sync")),
        TrainControllerConfig(experiment_name="arealite", trial_name="sync", allocation_mode="sglang.d4t8p1+d32t1p1"),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize()

    dataset = load_dataset("json",
                           data_files="/storage/xukuan.xk/repos/antnlp/personal/llm/benchmark/orz_areal_train.jsonl")
    train_dataset = dataset['train']
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 16
    batch_data = []
    step_num = 100
    epoch_num = 10
    max_prompt_len = 1024
    MODEL_PATH = "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/moe_lite_0428_base_32k_hgf"
    tokenizer = load_hf_tokenizer(MODEL_PATH)
    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            batch_data = []
            for _ in range(batch_size):
                batch = next(data_generator)
                prompt = batch["prompt"] if isinstance(batch, dict) else batch[0]["prompt"]
                tokenized = tokenizer(prompt, truncation=False, return_length=True)
                if tokenized["length"][0] <= max_prompt_len:
                    batch_data.append(batch)
                else:
                    print(f"Ignored prompt with length {tokenized['length'][0]} > {max_prompt_len}")

            # Update inference engine weights
            exp_name = "arealite"
            trial_name = "sync"
            # actor_cfg = WeightUpdateMeta(
            #     type="disk",
            #     path=f"/storage/openpsi/checkpoints/{exp_name}/{trial_name}/",
            #     alloc_mode=None,
            #     comm_backend=None,
            # )
            rollout_cfg = WeightUpdateMeta(
                type="disk",
                path=f"/storage/openpsi/checkpoints/{exp_name}/{trial_name}/{step}",
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
            gconfig = GenerationHyperparameters(
                max_new_tokens=4096, greedy=False, n_samples=16
            )

            workflow = RLVRWorkflow(
                reward_fn=reward_fn,
                gconfig=gconfig,
                tokenizer=tokenizer,
            )

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
            print(f"[Trainer] debug1")
            stats = actor.train_distributed_batch(dis_batch)
            print(f"[Trainer] train exec success, step: {step}, epoch: {epoch}, stats: {stats}")

            rollout.update_weights(rollout_cfg)
            print("[Trainer] rollout update_weights success.")


if __name__ == "__main__":
    main_grpo()

