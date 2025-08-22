import time

from realhf.base import logging
from requests.exceptions import ConnectionError
logger = logging.getLogger("ControllerUtil")

def create_engine_with_retry(
        create_engine_func,
        *args,
        max_retries=60,
        retry_delay=10,
        **kwargs
):
    logger.info(f"create_engine_with_retry debug: {args}, {max_retries}, {retry_delay}, {kwargs}")
    retries = 0
    while retries < max_retries:
        try:
            create_engine_func(*args, **kwargs)
            return
        except ConnectionError as e:
            logger.info(f"worker is not ready, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retries += 1
        except Exception as e:
            logger.error(f"Connection failed: {e}. unknown exception")
            raise e

    raise RuntimeError("Failed to connect to remote service after maximum retries.")
