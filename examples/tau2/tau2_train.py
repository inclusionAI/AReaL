"""Tau2 training script for RL training with proxy server mode.

This script demonstrates how to train an agent on tau2-bench using
AReaL's proxy server mode with the archon backend.
"""

import os
import sys
from dataclasses import dataclass, field

from datasets import Dataset
from loguru import logger as loguru_logger
from tau2.registry import registry

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils import logging
from areal.utils.stats_logger import StatsLogger

from .tau2_agent import Tau2AgentWorkflow
from .tau2_utils import Tau2EnvConfig

logger = logging.getLogger("Tau2Train")


# ================================ dataset ================================
def get_tau2_dataset(
    domain: str,
    type: str = "rl",
    split: str = "train",
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: The tau2 domain name, e.g., 'retail', 'airline', 'telecom'
        split: Dataset split (e.g., 'train', 'test')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries
    """
    assert type == "rl", "Only RL dataset is supported for now"

    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found in {splits}, available splits: {splits.keys()}"
        )
    task_ids = splits[split]

    dataset_items = [{"task_id": task_id, "split": split} for task_id in task_ids]
    dataset = Dataset.from_list(dataset_items)
    return dataset


@dataclass
class Tau2PPOConfig(PPOConfig):
    """PPO config extended with tau2-specific settings."""

    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation."},
    )


def main(args):
    config, _ = load_expr_config(args, Tau2PPOConfig)
    domain = config.econfig.domain

    # Remove the logging of loguru logger in tau2-bench package.
    loguru_logger.remove()
    loguru_logger.add(
        os.path.join(StatsLogger.get_log_path(config.stats_logger), "tau2.log"),
        level="INFO",
    )

    # Create dataset and dataloaders
    train_dataset = get_tau2_dataset(
        domain=domain,
        type=config.train_dataset.type,
        split=config.train_dataset.path.split("/")[-1],
    )
    valid_dataset = None
    if config.do_eval and config.valid_dataset is not None:
        valid_dataset = get_tau2_dataset(
            domain=domain,
            type=config.valid_dataset.type,
            split=config.valid_dataset.path.split("/")[-1],
        )

    # Create workflow with tau2-specific data preparation
    workflow = Tau2AgentWorkflow()

    # Prepare workflow kwargs with tau2-specific config
    workflow_kwargs = {
        "econfig": {
            "domain": config.econfig.domain,
            "max_steps": config.econfig.max_steps,
            "add_thinking_tool": config.econfig.add_thinking_tool,
            "solo_mode": config.econfig.solo_mode,
            "user_llm_base_url": config.econfig.user_llm_base_url,
            "user_llm": config.econfig.user_llm,
            "user_llm_args": config.econfig.user_llm_args,
            "turn_discount": config.econfig.turn_discount,
            "invalid_format_penalty": config.econfig.invalid_format_penalty,
        },
        "gconfig": {
            "n_samples": config.gconfig.n_samples,
            "max_new_tokens": config.gconfig.max_new_tokens,
            "min_new_tokens": config.gconfig.min_new_tokens,
            "max_tokens": config.gconfig.max_tokens,
            "greedy": config.gconfig.greedy,
            "top_p": config.gconfig.top_p,
            "top_k": config.gconfig.top_k,
            "temperature": config.gconfig.temperature,
        },
    }

    # Prepare eval workflow kwargs (same as train but different temperature)
    eval_workflow_kwargs = None
    if config.do_eval:
        eval_workflow_kwargs = workflow_kwargs.copy()
        eval_workflow_kwargs["gconfig"] = workflow_kwargs["gconfig"].copy()
        eval_workflow_kwargs["gconfig"]["temperature"] = 0.6  # Lower temp for eval

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow,
            eval_workflow=workflow if config.do_eval else None,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
