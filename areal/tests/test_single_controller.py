import os
import sys
from typing import Dict

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    GRPOConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.api.io_struct import FinetuneSpec
from areal.api.scheduler_api import ScheduleStrategy
from areal.controller.batch import DistributedBatchMemory
from areal.controller.train_controller import DistributedTrainController
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.reward.gsm8k_reward import gsm8k_reward_fn
from areal.scheduler.local import LocalScheduler
from areal.utils import logging, name_resolve
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("trainer_rollout_controller")


def main(args):
    cfg, _ = parse_cli_args(sys.argv[1:])
    config = to_structured_cfg(cfg, config_cls=GRPOConfig)
    config = OmegaConf.to_object(config)
    name_resolve.reconfigure(config.cluster.name_resolve)
    config: GRPOConfig
    AllocationMode.from_str(config.allocation_mode)

    scheduler = LocalScheduler(cfg)

    ####################### init rollout controller ###############################
    # inf_engine = RemoteSGLangEngine(config.rollout)
    # rollout_controller = DistributedRolloutController(
    #     inf_engine, config.rollout, scheduler
    # )
    # logger.info("Rollout controller begin initialize...")
    # rollout_controller.initialize(config.allocation_mode)
    # logger.info("Rollout controller initialize success...")

    train_engine = FSDPPPOActor(config=config.actor)

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=1024,  # dummy value
        train_batch_size=config.train_dataset.batch_size,
    )
    actor = DistributedTrainController(train_engine, config.actor, scheduler)
    actor.initialize(config.allocation_mode, ft_spec, ScheduleStrategy(), group_size=1)
    return
    ###################################################################################
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=0,
        world_size=1,
        split="train",
        max_length=config.train_dataset.max_length,
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=1,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    data_generator = cycle_dataloader(train_dataloader)
    batch_data = []
    for _ in range(config.train_dataset.batch_size):
        batch = next(data_generator)
        batch_data.append(batch)

    logger.info(f"get data batch: {batch_data[0]}")

    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    batch_data = DistributedBatchMemory.from_list(batch_data)
    logger.info("build distributed_batch_memory success.")
    rollout_res = rollout_controller.rollout_batch(batch_data, workflow)
    logger.info(f"rollout success, rollout_res: {rollout_res.get_data()}.")

    def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
        """Mock loss function for testing."""
        return torch.mean(logits)

    actor.train_batch(
        batch_data, mock_loss_fn, loss_weight_fn=lambda x: x["cu_seqlens"][-1]
    )

    print("[wht debug] train batch done.")

    actor.step_lr_scheduler()

    print("[wht debug] all step_lr_scheduler done.")


if __name__ == "__main__":
    main(sys.argv[1:])
