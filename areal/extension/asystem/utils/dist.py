from datetime import timedelta

import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("Dist Util")


def init_distributed_if_needed(
    init_config: dict, world_size: int, rank: int, backend: str = "gloo"
):
    """Initialize distributed environment if not already initialized"""
    if dist.is_initialized():
        logger.info(f"Distributed already initialized: rank={dist.get_rank()}")
        return

    logger.info("Initializing PyTorch distributed...")
    master_ip = init_config.get("master_ip")
    master_port = init_config.get("master_port")
    init_method = f"tcp://{master_ip}:{master_port}"

    logger.info(f"Init distributed with backend={backend}, init_method={init_method}")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(minutes=10),
    )
    logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")
