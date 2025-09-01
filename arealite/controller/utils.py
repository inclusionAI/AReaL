import time

import torch
from requests.exceptions import ConnectionError

from realhf.base import logging

logger = logging.getLogger("ControllerUtil")


def create_engine_with_retry(
    create_engine_func, *args, max_retries=60, retry_delay=10, **kwargs
):
    logger.info(
        f"create_engine_with_retry debug: {args}, {max_retries}, {retry_delay}, {kwargs}"
    )
    retries = 0
    while retries < max_retries:
        try:
            return create_engine_func(*args, **kwargs)
        except ConnectionError as e:
            logger.info(f"worker is not ready, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retries += 1
        except Exception as e:
            logger.error(f"Connection failed: {e}. unknown exception")
            raise e

    raise RuntimeError("Failed to connect to remote service after maximum retries.")


# same as avg(origin_tensor) group by (group_tensor)
def group_avg_torch(origin_tensor, group_tensor):
    unique_groups, inverse_indices = torch.unique(
        group_tensor, sorted=True, return_inverse=True
    )
    result = torch.zeros_like(unique_groups, dtype=torch.float)
    sum_per_group = result.scatter_add(0, inverse_indices, origin_tensor.float())
    counts = torch.bincount(inverse_indices, minlength=len(unique_groups)).float()
    avgs = torch.where(
        counts > 0, sum_per_group / counts, torch.zeros_like(sum_per_group)
    )
    return unique_groups, avgs
