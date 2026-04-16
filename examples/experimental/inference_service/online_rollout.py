"""Rollout-only online example via the inference_service gateway stack."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


def main(args: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--external-url", default=None)
    parser.add_argument("--external-api-key", default=None)
    parser.add_argument("--external-model", default=None)
    parser.add_argument("--external-name", default="ext-model")
    ext_args, remaining = parser.parse_known_args(args)

    from areal.api.cli_args import PPOConfig, load_expr_config
    from areal.experimental.inference_service.controller.config import (
        GatewayControllerConfig,
    )
    from areal.experimental.inference_service.controller.controller import (
        GatewayInferenceController,
    )
    from areal.utils import logging
    from areal.utils.environ import is_single_controller

    logger = logging.getLogger("InferenceServiceOnlineTrain")

    config, _ = load_expr_config(remaining, PPOConfig)
    openai_cfg = config.rollout.openai
    if openai_cfg is None or openai_cfg.mode != "online":
        raise ValueError(
            "online_rollout.py requires rollout.openai.mode='online' for inference_service online training."
        )
    if not is_single_controller():
        raise NotImplementedError(
            "online_rollout.py requires single-controller execution (for example: scheduler.type=local)."
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

    is_external = ext_args.external_url is not None

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
        admin_api_key=openai_cfg.admin_api_key,
        turn_discount=openai_cfg.turn_discount,
        export_style=openai_cfg.export_style,
    )
    if is_external:
        ctrl_config.api_url = ext_args.external_url
        ctrl_config.provider_api_key = ext_args.external_api_key
        ctrl_config.model = ext_args.external_model or ext_args.external_name
        server_args = None
    else:
        from areal.api.alloc_mode import ModelAllocation

        rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
        if rollout_alloc.backend == "sglang":
            server_args = asdict(config.sglang)
        elif rollout_alloc.backend == "vllm":
            server_args = asdict(config.vllm)
        else:
            raise ValueError(f"Unsupported rollout backend: {rollout_alloc.backend}")

    ctrl = GatewayInferenceController(config=ctrl_config, scheduler=scheduler)
    try:
        ctrl.initialize(
            role="rollout",
            server_args=server_args,
        )

        logger.info("Proxy gateway available at %s", ctrl.proxy_gateway_addr)

        if is_external:
            logger.info(
                "External mode: url=%s model=%s name=%s",
                ext_args.external_url,
                ext_args.external_model,
                ext_args.external_name,
            )

        result = ctrl.rollout_batch(
            data=None,
            batch_size=config.train_dataset.batch_size,
            workflow=None,
        )

        if is_external:
            logger.info("Rollout complete (%d trajectories)", len(result))
            for i, traj in enumerate(result):
                interactions = traj.get("interactions", [])
                for j, interaction in enumerate(interactions):
                    logger.info(
                        "Trajectory %d, interaction %d:\n  request:  %s\n  response: %s",
                        i,
                        j,
                        interaction.get("request", "")[:300],
                        interaction.get("response", "")[:300],
                    )
        else:
            import torch

            from areal.infra.rpc.rtensor import RTensor

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
