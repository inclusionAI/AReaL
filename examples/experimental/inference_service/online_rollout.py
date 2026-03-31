"""Rollout-only online example via the inference_service gateway stack."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch


def main(args: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from areal.api.cli_args import PPOConfig, load_expr_config
    from areal.experimental.inference_service.controller.config import (
        GatewayControllerConfig,
    )
    from areal.experimental.inference_service.controller.controller import (
        GatewayInferenceController,
    )
    from areal.infra.rpc.rtensor import RTensor
    from areal.utils import logging
    from areal.utils.environ import is_single_controller

    logger = logging.getLogger("InferenceServiceOnlineTrain")

    config, _ = load_expr_config(args, PPOConfig)
    openai_cfg = config.rollout.openai
    if openai_cfg is None or openai_cfg.mode != "online":
        raise ValueError(
            "online_rollout.py requires rollout.openai.mode='online' for inference_service online training."
        )
    if not is_single_controller():
        raise NotImplementedError(
            "online_rollout.py requires single-controller execution (for example: scheduler.type=local)."
        )
    from areal.api.alloc_mode import ModelAllocation

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend)
    if rollout_alloc.backend == "vllm":
        raise NotImplementedError(
            "online_rollout.py currently supports only the SGLang generation backend."
        )

    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.scheduler.slurm import SlurmScheduler

    sched_type = config.scheduler.type
    if sched_type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif sched_type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise NotImplementedError(f"Unknown scheduler type: {sched_type}")

    ctrl_config = GatewayControllerConfig(
        tokenizer_path=config.tokenizer_path,
        model_path=config.actor.path,
        consumer_batch_size=config.rollout.consumer_batch_size,
        max_concurrent_rollouts=config.rollout.max_concurrent_rollouts,
        max_head_offpolicyness=config.rollout.max_head_offpolicyness,
        queue_size=config.rollout.queue_size,
        enable_rollout_tracing=config.rollout.enable_rollout_tracing,
        fileroot=config.rollout.fileroot,
        experiment_name=config.rollout.experiment_name,
        trial_name=config.rollout.trial_name,
        dump_to_file=False,
        backend=config.rollout.backend,
        scheduling_spec=config.rollout.scheduling_spec,
        setup_timeout=config.rollout.setup_timeout,
        request_timeout=config.rollout.request_timeout,
        openai=openai_cfg,
    )

    ctrl = GatewayInferenceController(config=ctrl_config, scheduler=scheduler)
    try:
        ctrl.initialize(
            role="rollout",
            server_args=asdict(config.sglang),
        )

        logger.info("Proxy gateway available at %s", ctrl.proxy_gateway_addr)

        data = [{} for _ in range(config.train_dataset.batch_size)]
        result = ctrl.rollout_batch(
            data=data,
            workflow=None,
            workflow_kwargs={"timeout": config.rollout.request_timeout},
        )

        localized_rewards = [RTensor.localize(traj)["rewards"] for traj in result]
        all_rewards = torch.cat(localized_rewards, dim=0)
        logger.info(
            "Rollout complete (%d trajectories), avg_reward=%.4f",
            len(result),
            all_rewards.mean().item(),
        )
    finally:
        ctrl.destroy()
        scheduler.delete_workers(None)


if __name__ == "__main__":
    main(sys.argv[1:])
