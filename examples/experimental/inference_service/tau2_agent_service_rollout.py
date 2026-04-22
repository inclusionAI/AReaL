"""Rollout script for Tau2 using both agent service and inference service.

Uses the agent service to run the Tau2Agent (PydanticAI) while the
inference service provides model inference with RL data collection.

Usage:
    python3 examples/experimental/inference_service/tau2_agent_service_rollout.py \
        --config examples/experimental/inference_service/tau2_rollout.yaml
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

from datasets import Dataset

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    TrainDatasetConfig,
    load_expr_config,
)
from areal.experimental.agent_service.controller.config import (
    AgentServiceControllerConfig,
)
from areal.experimental.agent_service.controller.controller import (
    AgentServiceController,
)
from areal.experimental.inference_service.controller.config import (
    GatewayControllerConfig,
)
from areal.experimental.inference_service.controller.controller import (
    GatewayInferenceController,
)
from areal.utils import logging

logger = logging.getLogger("Tau2AgentServiceRollout")


@dataclass
class Tau2EnvConfig:
    domain: str = field(
        default="telecom",
        metadata={"help": "The tau2 domain name."},
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    add_thinking_tool: bool = field(
        default=False, metadata={"help": "Whether to add a thinking tool."}
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm_base_url: str | None = field(
        default=None, metadata={"help": "The base URL of the user LLM."}
    )
    user_llm: str | None = field(
        default=None, metadata={"help": "The user LLM model name."}
    )
    user_llm_args: dict | None = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format."}
    )


@dataclass
class AgentServiceConfig:
    agent_cls_path: str = field(
        default="examples.experimental.inference_service.tau2_agent.Tau2Agent",
        metadata={"help": "Import path for the agent-service agent class."},
    )
    num_pairs: int = field(
        default=1, metadata={"help": "Number of agent Worker+DataProxy pairs."}
    )
    admin_api_key: str = field(
        default="areal-agent-admin",
        metadata={"help": "Admin API key for agent service."},
    )


def get_tau2_dataset(domain: str, type: str = "rl", split: str = "train") -> Dataset:
    from tau2.registry import registry

    from examples.tau2.agent import _get_task

    assert type == "rl", "Only RL dataset is supported"
    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found for domain {domain}, "
            f"available: {list(splits.keys())}"
        )
    task_ids = splits[split]
    dataset_items = []
    for tid in task_ids:
        task = _get_task(domain=domain, task_id=tid, split=split)
        dataset_items.append(
            {
                "task_id": tid,
                "split": split,
                "prompt": str(task.user_scenario),
            }
        )
    if len(dataset_items) < 128:
        original = dataset_items.copy()
        while len(dataset_items) < 128:
            dataset_items.extend(original)
    dataset = Dataset.from_list(dataset_items)
    logger.info("Created dataset with %d items for %s/%s", len(dataset), domain, split)
    return dataset


@dataclass
class Tau2AgentServiceRolloutConfig(BaseExperimentConfig):
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    model_path: str = ""
    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    agent_service: AgentServiceConfig = field(default_factory=AgentServiceConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(argv, Tau2AgentServiceRolloutConfig)
    econfig = config.econfig
    rollout_cfg = config.rollout
    agent_svc_cfg = config.agent_service

    os.environ["TAU2_DOMAIN"] = econfig.domain

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

    openai_cfg = rollout_cfg.openai
    ctrl_config = GatewayControllerConfig(
        tokenizer_path=config.tokenizer_path,
        model_path=config.model_path,
        consumer_batch_size=rollout_cfg.consumer_batch_size,
        max_concurrent_rollouts=rollout_cfg.max_concurrent_rollouts,
        max_head_offpolicyness=rollout_cfg.max_head_offpolicyness,
        queue_size=rollout_cfg.queue_size,
        enable_rollout_tracing=rollout_cfg.enable_rollout_tracing,
        fileroot=rollout_cfg.fileroot,
        experiment_name=rollout_cfg.experiment_name,
        trial_name=rollout_cfg.trial_name,
        dump_to_file=rollout_cfg.dump_to_file,
        backend=rollout_cfg.backend,
        scheduling_spec=rollout_cfg.scheduling_spec,
        setup_timeout=rollout_cfg.setup_timeout,
        request_timeout=rollout_cfg.request_timeout,
        **(
            {
                "admin_api_key": openai_cfg.admin_api_key,
                "turn_discount": openai_cfg.turn_discount,
                "export_style": openai_cfg.export_style,
                "tool_call_parser": openai_cfg.tool_call_parser,
                "reasoning_parser": openai_cfg.reasoning_parser,
                "engine_max_tokens": openai_cfg.engine_max_tokens,
                "chat_template_type": openai_cfg.chat_template_type,
            }
            if openai_cfg
            else {}
        ),
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

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if rollout_alloc.backend == "sglang":
        server_args = asdict(config.sglang)
    elif rollout_alloc.backend == "vllm":
        server_args = asdict(config.vllm)
    else:
        raise ValueError(f"Unsupported rollout backend: {rollout_alloc.backend}")

    inf_ctrl = GatewayInferenceController(config=ctrl_config, scheduler=scheduler)
    inf_ctrl.initialize(role="rollout", server_args=server_args)

    logger.info("Inference service ready at %s", inf_ctrl.gateway_addr)

    agent_ctrl_config = AgentServiceControllerConfig(
        agent_cls_path=agent_svc_cfg.agent_cls_path,
        admin_api_key=agent_svc_cfg.admin_api_key,
        num_pairs=agent_svc_cfg.num_pairs,
        inference_addr=inf_ctrl.gateway_addr,
        inference_model=config.model_path,
        inference_api_key=ctrl_config.admin_api_key or "areal-admin-key",
    )

    agent_ctrl = AgentServiceController(config=agent_ctrl_config, scheduler=scheduler)
    agent_ctrl.initialize()

    logger.info("Agent service ready at %s", agent_ctrl.gateway_addr)

    econfig_dict = asdict(econfig)
    workflow_kwargs: dict[str, Any] = dict(
        agent_controller=agent_ctrl,
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=600.0,
    )

    try:
        logger.info("Starting rollout loop")
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            keys = list(batch.keys())
            batch_size = len(batch[keys[0]])
            data = [{k: batch[k][i] for k in keys} for i in range(batch_size)]

            result = inf_ctrl.rollout_batch(
                data=data,
                workflow="examples.experimental.inference_service.tau2_workflow.Tau2AgentServiceWorkflow",
                workflow_kwargs=workflow_kwargs,
            )
            if result:
                import torch

                from areal.infra.rpc.rtensor import RTensor

                batch_rewards = []
                for traj in result:
                    local_traj = RTensor.localize(traj)
                    batch_rewards.append(local_traj["rewards"])
                all_rewards = torch.cat(batch_rewards, dim=0)
                logger.info(
                    "Batch %d: n_trajs=%d, rewards=%s, avg_reward=%.4f",
                    batch_idx,
                    len(result),
                    all_rewards,
                    all_rewards.mean().item(),
                )
            else:
                logger.warning("Batch %d: empty result (all rejected?)", batch_idx)
            batch_count += 1
        logger.info("Rollout complete (%d batches)", batch_count)
    finally:
        agent_ctrl.destroy()
        inf_ctrl.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
