"""Direct rollout driver for the tau2 agent-service workflow."""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from datasets import Dataset

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    AgentConfig,
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    TrainDatasetConfig,
    load_expr_config,
)
from areal.experimental.agent_service.controller import AgentController
from areal.experimental.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.utils import logging

logger = logging.getLogger("Tau2AgentServiceRollout")


def get_tau2_dataset(domain: str, type: str = "rl", split: str = "train") -> Dataset:
    from tau2.registry import registry

    assert type == "rl", "Only RL dataset is supported for now"
    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found for domain {domain}, available splits: {list(splits.keys())}"
        )
    task_ids = splits[split]
    dataset_items = [{"task_id": task_id, "split": split} for task_id in task_ids]
    if len(dataset_items) < 128:
        original_items = dataset_items.copy()
        while len(dataset_items) < 128:
            dataset_items.extend(original_items)
    return Dataset.from_list(dataset_items)


@dataclass
class Tau2AgentServiceRolloutConfig(BaseExperimentConfig):
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    model_path: str = ""
    econfig: dict[str, Any] = field(default_factory=dict)
    agent_service: AgentConfig = field(
        default_factory=lambda: AgentConfig(
            agent_cls_path="examples.experimental.agent_service.tau2.agent.Tau2Agent"
        )
    )
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)


async def _run_rollouts(
    workflow: Any,
    controller: RolloutControllerV2,
    dataloader: Any,
    *,
    max_batches: int | None = None,
) -> None:
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        keys = list(batch.keys())
        batch_size = len(batch[keys[0]])
        data_rows = [{k: batch[k][i] for k in keys} for i in range(batch_size)]

        results = await asyncio.gather(
            *(workflow.arun_episode(controller, row) for row in data_rows)
        )

        rewards: list[float] = []
        trajectories = 0
        for result in results:
            if not result:
                continue
            trajectories += 1
            last_id = next(reversed(result))
            reward = result[last_id].reward
            if reward is not None:
                rewards.append(float(reward))

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(
            "Batch %d: n_trajs=%d, rewards=%s, avg_reward=%.4f",
            batch_idx,
            trajectories,
            rewards,
            avg_reward,
        )
        batch_count += 1

    logger.info("Rollout complete (%d batches)", batch_count)


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(argv, Tau2AgentServiceRolloutConfig)
    rollout_cfg = deepcopy(config.rollout)
    rollout_cfg.model = config.model_path

    from examples.experimental.agent_service.tau2.workflow import (
        Tau2AgentServiceWorkflow,
    )
    from examples.tau2.utils import Tau2EnvConfig

    econfig = (
        Tau2EnvConfig(**config.econfig)
        if isinstance(config.econfig, dict)
        else config.econfig
    )
    train_dataset = get_tau2_dataset(
        domain=econfig.domain,
        type=config.train_dataset.type,
        split=config.train_dataset.path.split("/")[-1],
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=0,
    )

    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.scheduler.slurm import SlurmScheduler

    if config.scheduler.type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif config.scheduler.type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise NotImplementedError(f"Unknown scheduler type: {config.scheduler.type}")

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if rollout_alloc.backend == "sglang":
        server_args = asdict(config.sglang)
    elif rollout_alloc.backend == "vllm":
        server_args = asdict(config.vllm)
    else:
        raise ValueError(f"Unsupported rollout backend: {rollout_alloc.backend}")

    rollout_controller = RolloutControllerV2(config=rollout_cfg, scheduler=scheduler)
    rollout_controller.initialize(role="rollout", server_args=server_args)

    agent_controller = AgentController(config=config.agent_service, scheduler=scheduler)
    agent_controller.initialize()

    workflow = Tau2AgentServiceWorkflow(
        agent_controller=agent_controller,
        inference_gateway_addr=rollout_controller.proxy_gateway_addr,
        inference_admin_api_key=rollout_cfg.admin_api_key,
        inference_model=config.model_path,
        econfig=asdict(econfig),
        gen_args={
            "temperature": config.gconfig.temperature,
            "max_completion_tokens": config.gconfig.max_new_tokens,
        },
        timeout=600.0,
        discount=rollout_cfg.openai.turn_discount if rollout_cfg.openai else 1.0,
        export_style=(
            rollout_cfg.openai.export_style if rollout_cfg.openai else "individual"
        ),
    )

    max_batches_env = os.environ.get("AREAL_MAX_BATCHES")
    max_batches = int(max_batches_env) if max_batches_env is not None else None

    try:
        asyncio.run(
            _run_rollouts(
                workflow,
                rollout_controller,
                dataloader,
                max_batches=max_batches,
            )
        )
    finally:
        agent_controller.destroy()
        rollout_controller.destroy()
        scheduler.delete_workers(None)


if __name__ == "__main__":
    main(sys.argv[1:])
