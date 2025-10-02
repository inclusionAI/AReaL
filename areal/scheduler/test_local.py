import sys
import os
import areal
import time
from areal.scheduler.local import LocalScheduler

from areal.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    RecoverConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils import seeding, stats_tracker
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.api.alloc_mode import AllocationMode
from areal.platforms import current_platform
from areal.utils import name_resolve, pkg_version
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)

from concurrent.futures import ThreadPoolExecutor


# init_config = {}

create_workers_config, _ = parse_cli_args(sys.argv[1:])

from omegaconf import MISSING, DictConfig, OmegaConf
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

print("[wht debug] rollout workers:", rollout_workers)
print("[wht debug] actor workers:", actor_workers)

time.sleep(20)


rollout = RemoteSGLangEngine(config.rollout)
with ThreadPoolExecutor(max_workers=len(rollout_workers)) as executor:
    def create_engine_and_init(worker_id):
        print(f"[wht debug] start create rollout engine and init {worker_id}")
        shcheduler.create_engine(worker_id, rollout, train_data_parallel_size=parallel_strategy.dp_size)
        print(f"[wht debug] end create rollout engine and init {worker_id}")

    futures = []
    for i in range(len(rollout_workers)):
        futures.append(executor.submit(create_engine_and_init, rollout_workers[i].id))

    for future in futures:
        future.result()

ft_spec = FinetuneSpec(
    total_train_epochs=config.total_train_epochs,
    dataset_size=1024, # dummy value
    train_batch_size=config.train_dataset.batch_size,
)

actor = FSDPPPOActor(config=config.actor)
with ThreadPoolExecutor(max_workers=len(actor_workers)) as executor:
    def create_engine_and_init(worker_id):
        print(f"[wht debug] start create actor engine and init {worker_id}")
        shcheduler.create_engine(worker_id, actor, None, ft_spec, parallel_strategy=parallel_strategy)
        print(f"[wht debug] end create actor engine and init {worker_id}")

    futures = []
    for i in range(len(actor_workers)):
        futures.append(executor.submit(create_engine_and_init, actor_workers[i].id))

    for future in futures:
        future.result()

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

from areal.scheduler.gsm8k_reward import gsm8k_reward_fn
workflow = RLVRWorkflow(
    reward_fn=gsm8k_reward_fn,
    gconfig=config.gconfig,
    tokenizer=tokenizer,
    enable_thinking=False,
    dump_dir=os.path.join(
        StatsLogger.get_log_path(config.stats_logger), "generated"
    ),
)

with ThreadPoolExecutor(max_workers=len(rollout_workers)) as executor:
    def call_rollout(worker_id, data):
        batch = shcheduler.call_engine(worker_id, "rollout_batch", 3, data, workflow=workflow, should_accept=lambda sample: True)
        print(f"[wht debug] rollout {worker_id} done, got batch: {batch}")
        return batch
    
    futures = []
    for i in range(len(rollout_workers)):
        futures.append(executor.submit(call_rollout, rollout_workers[i].id, data))
    for future in futures:
        r = future.result()
        print(f"[wht debug] rollout result: {r}")

import time
time.sleep(1000)


