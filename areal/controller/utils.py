import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from requests.exceptions import ConnectionError

from areal.api.scheduler_api import Scheduler, Worker
from areal.utils import logging
from areal.utils.http import wait_future_ordered

logger = logging.getLogger("ControllerUtil")


def create_engine_with_retry(
    create_engine_func, max_retries=60, retry_delay=10, *args, **kwargs
):
    """
    Attempts to create an engine with retry logic.
    :param create_engine_func: Callable function for creating the engine.
    :param max_retries: Maximum number of retries before giving up.
    :param retry_delay: Seconds to wait between retries.
    :param args: Positional arguments to pass to create_engine_func.
    :param kwargs: Keyword arguments to pass to create_engine_func.
    :return: Engine instance created by create_engine_func.
    :raises RuntimeError: If maximum retries are reached and connection still fails.
    """
    logger.info(
        f"Create engine with retry: {max_retries}, {retry_delay}, {args}, {kwargs}"
    )
    retries = 0
    while retries < max_retries:
        try:
            return create_engine_func(*args, **kwargs)
        except ConnectionError as e:
            logger.info(
                f"Worker is not ready, exception: {e}, retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
            retries += 1
        except Exception as e:
            logger.error(f"Connection failed: {e}. unknown exception")
            raise e

    raise RuntimeError("Failed to connect to remote service after maximum retries.")


def rpc_call(
    scheduler: Scheduler,
    workers: List[Worker],
    method: str,
    batches: Optional[List[Any]] = None,
    *args,
    **kwargs,
) -> List[Any]:
    """
    Utility method: Perform concurrent RPC calls to multiple workers.
    :param scheduler: Scheduler object with a call_engine(worker_id, method, *args, **kwargs) method.
    :param workers: List of worker instances. Each worker must have an 'id' attribute.
    :param method: Name of the method to invoke on each worker.
    :param batches: Optional list of batches, each batch is passed to the corresponding worker.
                   If provided, its length must match the number of workers.
    :param args: Positional arguments to pass to call_engine.
    :param kwargs: Keyword arguments to pass to call_engine.
    :return: List of results returned in the order of workers.
    :raises ValueError: If the batches parameter is provided but its length does not match the number of workers.
    :raises RuntimeError: If any exception occurs during RPC execution.
    """

    if batches is not None and len(batches) != len(workers):
        raise ValueError(
            f"Batches length ({len(batches)}) must match workers count ({len(workers)})"
        )
    logger.info(f"Start to rpc call, method: {method}")

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = []
        for i, worker in enumerate(workers):
            # 构建调用参数
            if batches is not None:
                # 当有batch参数时：将batch作为第一位置参数
                worker_args = (batches[i],) + args
                future = executor.submit(
                    scheduler.call_engine, worker.id, method, *worker_args, **kwargs
                )
            else:
                future = executor.submit(
                    scheduler.call_engine, worker.id, method, *args, **kwargs
                )
            futures.append(future)
        try:
            results = wait_future_ordered(futures, exit_on_exception=True)
        except Exception as e:
            raise RuntimeError(f"{method} failed, error: {e}")

    return results
