import sys

import areal
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

print("rollout workers:", rollout_workers)
print("actor workers:", actor_workers)

# rollout = RemoteSGLangEngine(config.rollout)
# shcheduler.create_engine("rollout", rollout, config.rollout)
# rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
