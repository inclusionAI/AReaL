import os
import sys

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    GRPOConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.scheduler.local import LocalScheduler
from areal.utils import name_resolve, logging
from areal.utils.data import cycle_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow
from areal.controller.rollout_controller import DistributedRolloutController
from areal.controller.batch import DistributedBatchMemory
from areal.reward.gsm8k_reward import gsm8k_reward_fn
from areal.engine.ppo.actor import FSDPPPOActor
from areal.api.io_struct import FinetuneSpec
from areal.controller.train_controller import DistributedTrainController
from areal.api.scheduler_api import ScheduleStrategy

logger = logging.getLogger("trainer_rollout_controller")

cfg, _ = parse_cli_args(sys.argv[1:])

from omegaconf import OmegaConf

# config, _ = load_expr_config(sys.argv[1:])
config = to_structured_cfg(cfg, config_cls=GRPOConfig)
config = OmegaConf.to_object(config)
name_resolve.reconfigure(config.cluster.name_resolve)
config: GRPOConfig
allocation_mode = AllocationMode.from_str(config.allocation_mode)
parallel_strategy = allocation_mode.train

scheduler = LocalScheduler(cfg)

####################### init controller ###############################
# init rollout controller
inf_engine = RemoteSGLangEngine(config.rollout)
inf_config = config.rollout
rollout_controller = DistributedRolloutController(inf_engine, config.rollout, scheduler)
logger.info("Rollout controller begin initialize...")
rollout_controller.initialize(cfg, None)
logger.info("Rollout controller initialize success...")

# init train controller
actor = None
enable_training = False
if enable_training:
    train_engine = FSDPPPOActor(config=config.actor)
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=1024,  # dummy value
        train_batch_size=config.train_dataset.batch_size,
    )
    actor = DistributedTrainController(train_engine, config.actor, scheduler)
    actor.initialize(config.allocation_mode, ft_spec, ScheduleStrategy(), group_size=1)
###################################################################################

tokenizer = load_hf_tokenizer(config.tokenizer_path)
workflow = RLVRWorkflow(
    reward_fn=gsm8k_reward_fn,
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    enable_thinking=False,
    dump_dir=os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated"),
)

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
    batch_size=config.train_dataset.batch_size,
    shuffle=config.train_dataset.shuffle,
    num_workers=config.train_dataset.num_workers,
    collate_fn=lambda x: x,
    drop_last=config.train_dataset.drop_last,
)
data_generator = cycle_dataloader(train_dataloader)
batch_data = next(data_generator)
logger.info(f"get data batch: {batch_data[0]}")

batch_data = DistributedBatchMemory.from_list(batch_data)
logger.info("build distributed_batch_memory success. begin rollout_batch...")
rollout_res = rollout_controller.rollout_batch(batch_data, workflow)
logger.info(f"rollout success, rollout_res: {rollout_res.get_data()}.")