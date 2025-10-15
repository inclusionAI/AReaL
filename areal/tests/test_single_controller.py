import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    GRPOConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.api.io_struct import FinetuneSpec
from areal.api.scheduler_api import ScheduleStrategy
from areal.controller.train_controller import DistributedTrainController
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.scheduler.local import LocalScheduler
from areal.utils import name_resolve
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

# init_config = {}

create_workers_config, _ = parse_cli_args(sys.argv[1:])

from omegaconf import OmegaConf

# config, _ = load_expr_config(sys.argv[1:])
config = to_structured_cfg(create_workers_config, config_cls=GRPOConfig)
config = OmegaConf.to_object(config)
name_resolve.reconfigure(config.cluster.name_resolve)
config: GRPOConfig
# seeding.set_random_seed(config.seed, key=f"trainer{rank}")
allocation_mode = AllocationMode.from_str(config.allocation_mode)
parallel_strategy = allocation_mode.train


shcheduler = LocalScheduler(create_workers_config)
shcheduler.create_workers("rollout", create_workers_config)
shcheduler.create_workers("actor", create_workers_config)

rollout_workers = shcheduler.get_workers("rollout", timeout=300)
actor_workers = shcheduler.get_workers("actor", timeout=300)

train_controller = DistributedTrainController
print("[wht debug] rollout workers:", rollout_workers)
print("[wht debug] actor workers:", actor_workers)

time.sleep(20)


rollout = RemoteSGLangEngine(config.rollout)
with ThreadPoolExecutor(max_workers=len(rollout_workers)) as executor:

    def create_engine_and_init(worker_id):
        print(f"[wht debug] start create rollout engine and init {worker_id}")
        shcheduler.create_engine(
            worker_id, rollout, train_data_parallel_size=parallel_strategy.dp_size
        )
        print(f"[wht debug] end create rollout engine and init {worker_id}")

    futures = []
    for i in range(len(rollout_workers)):
        futures.append(executor.submit(create_engine_and_init, rollout_workers[i].id))

    for future in futures:
        future.result()

ft_spec = FinetuneSpec(
    total_train_epochs=config.total_train_epochs,
    dataset_size=1024,  # dummy value
    train_batch_size=config.train_dataset.batch_size,
)

train_engine = FSDPPPOActor(config=config.actor)
actor = DistributedTrainController(train_engine, config.actor, shcheduler)
actor.initialize(config.allocation_mode, ft_spec, ScheduleStrategy(), group_size=1)

# with ThreadPoolExecutor(max_workers=len(actor_workers)) as executor:
#
#     def create_engine_and_init(worker_id):
#         print(f"[wht debug] start create actor engine and init {worker_id}")
#         shcheduler.create_engine(
#             worker_id, actor, None, ft_spec, parallel_strategy=parallel_strategy
#         )
#         print(f"[wht debug] end create actor engine and init {worker_id}")
#
#     futures = []
#     for i in range(len(actor_workers)):
#         futures.append(executor.submit(create_engine_and_init, actor_workers[i].id))
#
#     for future in futures:
#         future.result()

print("[wht debug] all engines created and initialized.")


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
    batch_size=config.train_dataset.batch_size,
    shuffle=config.train_dataset.shuffle,
    num_workers=config.train_dataset.num_workers,
    collate_fn=lambda x: x,
    drop_last=config.train_dataset.drop_last,
)
data_generator = cycle_dataloader(train_dataloader)
data = next(data_generator)

print(f"[wht debug] get data batch: {data[0]}")

from areal.reward.gsm8k_reward import gsm8k_reward_fn

workflow = RLVRWorkflow(
    reward_fn=gsm8k_reward_fn,
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    enable_thinking=False,
    dump_dir=os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated"),
)

batch = None
with ThreadPoolExecutor(max_workers=len(rollout_workers)) as executor:

    def call_rollout(worker_id, data):
        try:
            batch = shcheduler.call_engine(
                worker_id,
                "rollout_batch",
                data,
                workflow=workflow,
                should_accept=lambda sample: True,
            )
            print(f"[wht debug] rollout {worker_id} done, got batch: {batch}")
            return batch
        except Exception as e:
            print(f"[wht debug] rollout {worker_id} failed, error: {e}")
            raise e

    futures = []
    for i in range(len(rollout_workers)):
        futures.append(executor.submit(call_rollout, rollout_workers[i].id, data))
    for future in futures:
        r = future.result()
        print(f"[wht debug] rollout result: {r}")
        batch = r

print("[wht debug] all rollout done.")

assert not config.actor.use_decoupled_loss and not config.actor.recompute_logprob


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


actor.train_batch(batch, mock_loss_fn, loss_weight_fn=lambda x: x["cu_seqlens"][-1])

print("[wht debug] train batch done.")

actor.step_lr_scheduler()

print("[wht debug] all step_lr_scheduler done.")
