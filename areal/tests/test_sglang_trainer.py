import os
import shutil

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    RolloutControllerConfig,
)
from areal.controller.rollout_controller import DistributedRolloutController
from areal.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from areal.scheduler.local import LocalScheduler
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
    scheduler = LocalScheduler({})

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(
            InferenceEngineConfig(experiment_name="ff", trial_name="ff")
        ),
        RolloutControllerConfig(),
        scheduler,
    )

    # engine initialize
    rollout.initialize()

    dataset = load_dataset(
        "json",
        data_files="/storage/xukuan.xk/repos/antnlp/personal/llm/benchmark/orz_areal_train.jsonl",
    )
    train_dataset = dataset["train"]
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 16
    batch_data = []
    step_num = 1
    epoch_num = 1
    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            batch_data = []
            for _ in range(batch_size):
                batch = next(data_generator)
                batch_data.append(batch)

            # synchronous rollout
            gconfig = GenerationHyperparameters(
                max_new_tokens=1024, greedy=False, n_samples=8
            )
            MODEL_PATH = "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/moe_lite_0428_base_32k_hgf"
            tokenizer = load_hf_tokenizer(MODEL_PATH)
            workflow = RLVRWorkflow(
                reward_fn=lambda **kwargs: 1.0,
                gconfig=gconfig,
                tokenizer=tokenizer,
            )

            # input_: List[Dict[str, tensor]]
            rollout_res = rollout.rollout(batch_data, workflow=workflow)
            print(
                f"[Trainer] rollout exec success, type: {rollout_res}, rollout_res: {rollout_res}"
            )
            rollout_res = rollout_res.to("cpu").clone()
            rollout_res_dict = rollout_res.to_dict()
            for k, v in rollout_res_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[0] == 1:
                    rollout_res_dict[k] = v.squeeze(0)
                    print(f"[Trainer] after squeeze {k} {rollout_res_dict[k].shape}")


if __name__ == "__main__":
    main_grpo()
