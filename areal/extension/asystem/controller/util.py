import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch

from areal.utils import stats_tracker


def execute_parallel_tasks(workers, scheduler, method_name, *args):
    """Execute tasks in parallel across all workers.

    This is a helper function to reduce code duplication when executing
    the same method on all workers with identical parameters.

    Parameters
    ----------
    workers : list
        List of worker objects
    scheduler : Scheduler
        Scheduler instance for async calls
    method_name : str
        Name of the method to call on each worker's engine
    *args, **kwargs
        Arguments to pass to the method

    Returns
    -------
    list
        Results from all workers

    Raises
    ------
    RuntimeError
        If any worker fails to execute the task
    """
    tasks = [
        scheduler.async_call_engine(worker.id, method_name, *args, _should_bcast=False)
        for worker in workers
    ]

    try:
        # 创建新的事件循环并运行所有任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                asyncio.gather(*tasks, return_exceptions=False)
            )
            return result
        finally:
            try:
                loop.close()
            except Exception:
                pass
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise RuntimeError(f"{method_name} failed, error: {e}")


def calc_metrics(batch_inputs):
    # seqlen std
    seqlens = [td["seqlen"].sum().item() for td in batch_inputs]
    seqlen_std = torch.tensor(seqlens).float().std().item()
    stats_tracker.scalar(**{"seqlen_std": seqlen_std})
