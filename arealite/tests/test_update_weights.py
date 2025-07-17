from arealite.api.cli_args import InferenceEngineConfig, RolloutControllerConfig
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler
from arealite.api.engine_api import WeightUpdateMeta


def main_grpo():
    # init controller
    scheduler = LocalScheduler({})

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(InferenceEngineConfig(experiment_name="ff", trial_name="ff")),
        RolloutControllerConfig(),
        scheduler,
    )

    # engine initialize
    rollout.initialize()

    # Update inference engine weights
    rollout_cfg = WeightUpdateMeta(
        type="disk",
        path=f"/storage/openpsi/checkpoints/ff/ff/0",
        alloc_mode=None,
        comm_backend=None,
    )

    rollout.update_weights(rollout_cfg)
    print("[Trainer] rollout update_weights success.")


if __name__ == "__main__":
    main_grpo()
