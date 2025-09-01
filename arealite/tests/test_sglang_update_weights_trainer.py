import logging

from arealite.api.cli_args import InferenceEngineConfig, RolloutControllerConfig
from arealite.api.engine_api import WeightUpdateMeta
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.asystem import AsystemScheduler


def main_grpo():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # init controller
    scheduler = AsystemScheduler(
        {
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
        }
    )

    rollout = DistributedRolloutController(
        RemoteSGLangEngine(
            InferenceEngineConfig(experiment_name="arealite", trial_name="sync")
        ),
        RolloutControllerConfig(
            experiment_name="arealite",
            trial_name="sync",
            allocation_mode="gen:d4t8p1,train:d32t1p1",
        ),
        scheduler,
    )
    # engine initialize
    rollout.initialize()

    # Update inference engine weights
    exp_name = "arealite"
    trial_name = "sync"
    rollout_cfg = WeightUpdateMeta(
        type="disk",
        path=f"/storage/openpsi/checkpoints/{exp_name}/{trial_name}/0",
        alloc_mode=None,
        comm_backend=None,
    )
    print("[Trainer] rollout begin exec update_weights...")
    rollout.update_weights(rollout_cfg)
    print("[Trainer] rollout update_weights success...")


if __name__ == "__main__":
    main_grpo()
