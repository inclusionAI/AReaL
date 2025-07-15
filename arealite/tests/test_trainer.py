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
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.workflow.rlvr import RLVRWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer

def main_grpo():
    dataset = load_dataset("json",data_files="/storage/xukuan.xk/repos/antnlp/personal/llm/benchmark/orz_areal_train_32.jsonl")
    train_dataset = dataset['train']  # 取出train split
    # rollout_config, training_config = load_expr_config(sys.argv[1:])

    # Single-controller mode initialization


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
    # ref = TrainController(
    #     MegatronGRPOActor(training_config.ref),
    #     config.training_controller_config,
    #     scheduler,
    # )
    # SPMD mode initialization
    # rollout = RemoteSGLangEngine(rollout_config.engine)
    # actor = MegatronGRPOActor(training_config.actor)
    # ref = MegatronGRPOActor(training_config.ref)

    rollout.initialize()
    actor.initialize()
    # ref.initialize()

    # # Synchronous RL
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 4
    batch_data = []
    step_num = 2
    epoch_num = 1
    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            # get batch data
            batch_data = []
            for _ in range(batch_size):
                batch = next(data_generator)
                batch_data.append(batch)

            # Update inference engine weights
            # wcfg = actor.upload_weights(WeightUpdateMeta)
            # future = rollout.update_weights(wcfg)
            # actor.upload_weights(wcfg)
            # future.result()

            # synchronous rollout
            gconfig = GenerationHyperparameters(
                max_new_tokens=16, greedy=False, n_samples=1
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
            
            print(f"rollout_ exec success, {len(rollout_res)}")
            # or asynchronous rollout with filtering and off-policyness control
            # rollout_batch = rollout.prepare_batch(batch,
            #                                       workflow=MyRolloutWorkflow(rollout_config.workflow),
            #                                       should_accept=lambda x: x['rewards'].mean() > 0)

            # print(stats)

if __name__ == "__main__":
    main_grpo()
