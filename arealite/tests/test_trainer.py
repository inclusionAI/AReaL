import sys

from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import load_expr_config, BaseExperimentConfig
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler



def main_grpo():
    dataset = load_dataset("openai/gsm8k", split="train")

    # rollout_config, training_config = load_expr_config(sys.argv[1:])

    # Single-controller mode initialization


    scheduler = LocalScheduler({})
    config, _ = load_expr_config(sys.argv[1:], BaseExperimentConfig)
    config.
    rollout_config = rollout_config
    training_config = rollout_config

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(rollout_config.engine),
        rollout_config.controller,
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteMegatronEngine(training_config.actor),
        config.training_controller_config,
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

    # Synchronous RL
    dataloader = StatefulDataLoader(dataset)
    for epoch in range(5):
        data_generator = iter(dataloader)
        for prompt in range(10):
            prompt = next(data_generator)

            # Update inference engine weights
            wcfg = actor.upload_weights(WeightUpdateMeta)
            future = rollout.update_weights(wcfg)
            actor.upload_weights(wcfg)
            future.result()

            # synchronous rollout
            rollout_batch = rollout.rollout(batch, workflow=MyRolloutWorkflow(rollout_config.workflow))
            # or asynchronous rollout with filtering and off-policyness control
            # rollout_batch = rollout.prepare_batch(batch,
            #                                       workflow=MyRolloutWorkflow(rollout_config.workflow),
            #                                       should_accept=lambda x: x['rewards'].mean() > 0)


            print(stats)

if __name__ == "__main__":
    main_grpo()