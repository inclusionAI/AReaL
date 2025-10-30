import traceback
import os
import signal
from concurrent.futures import Future, as_completed
from realhf.base import logging

logger = logging.getLogger("Utils")


def wait_future_ordered(futures: list[Future], exit_on_exception: bool = False) -> list:
    results = [None] * len(futures)
    future_index_map = {future: i for i, future in enumerate(futures)}
    for future in as_completed(futures):
        index = future_index_map[future]
        try:
            results[index] = future.result()
        except Exception as e:
            logger.warning(f"Exception caught when waiting for future: {e}")
            logger.warning(traceback.format_exc())
            if exit_on_exception:
                logger.info("Exiting due to exception in future.")
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                raise e
    return results