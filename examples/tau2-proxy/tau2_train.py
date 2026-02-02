"""Tau2 training script for RL training with proxy server and archon backend.

This script demonstrates how to train an agent on tau2-bench using
AReaL's single-controller mode with proxy server and archon backend.

Usage:
    python3 tau2_train.py scheduler.type=slurm

The key differences from the SPMD mode (tau2/tau2_train.py):
1. Uses AgentWorkflow instead of RolloutWorkflow
2. Uses proxy server for OpenAI-compatible API
3. Compatible with archon backend (archon handles training, proxy handles inference)
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

# Add examples/tau2-proxy to path for local imports when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau2_agent import Tau2ProxyAgentWorkflow
from tau2_utils import Tau2EnvConfig

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

    # Prepare econfig dict for workflow
    # Use experiment log path as default trajectory save directory
    trajectory_save_dir = config.econfig.trajectory_save_dir
    if trajectory_save_dir is None:
        trajectory_save_dir = os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "trajectories"
        )

    econfig_dict = {
        "domain": config.econfig.domain,
        "max_steps": config.econfig.max_steps,
        "add_thinking_tool": config.econfig.add_thinking_tool,
        "solo_mode": config.econfig.solo_mode,
        "user_llm_base_url": config.econfig.user_llm_base_url,
        "user_llm": config.econfig.user_llm,
        "user_llm_args": config.econfig.user_llm_args,
        "turn_discount": config.econfig.turn_discount,
        "invalid_format_penalty": config.econfig.invalid_format_penalty,
        "save_trajectories": config.econfig.save_trajectories,
        "trajectory_save_dir": trajectory_save_dir,
    }

    # Create AgentWorkflow for proxy server mode
    # In single-controller mode, the proxy server handles OpenAI API translation
    # The workflow receives base_url and http_client in run()
    workflow = Tau2ProxyAgentWorkflow(
        gconfig=config.gconfig,
        econfig=econfig_dict,
    )

    # Create eval workflow if needed
    eval_workflow = None
    if config.do_eval:
        eval_gconfig = config.gconfig.new(temperature=0.6)
        eval_workflow = Tau2ProxyAgentWorkflow(
            gconfig=eval_gconfig,
            econfig=econfig_dict,
        )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
