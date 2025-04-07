import asyncio
import logging
import os
import random
import time
from enum import Enum
from statistics import median
from typing import Any, Dict
import aiohttp

try:
    from realhf.base import logging, constants
except Exception:
    import logging

    constants = None

logger = logging.getLogger("function call")

FUNCTIONCALL_SERVICE_DOMAIN = os.getenv(
    "FUNCTIONCALL_SERVICE_DOMAIN",
    "",
)


class Language(Enum):
    PYTHON = 0
    JAVA = 1
    CPP = 2
    C = 3
    MATH = 4
    SQL = 5
    GO = 6
    NODEJS = 7
    CSHARP = 8
    TYPESCRIPT = 9
    JAVASCRIPT = 10

    def __str__(self):
        return f"{self.name.lower()}"


def calculate_percentile(elapsed_times, percentile):
    sorted_times = sorted(elapsed_times)
    index = int(len(sorted_times) * (percentile / 100))
    return sorted_times[min(index, len(sorted_times) - 1)]


def has_system_error(response_json):
    for result in response_json.get("results", []):
        if result.get("errorType", "") == "SystemError":
            return True, result
    return False, None


async def async_invoke_function(
    session: aiohttp.ClientSession,
    url: str,
    timeout: aiohttp.ClientTimeout,
    payload: Dict[str, Any] = None,
    max_retries: int = 100,
    initial_retry_interval: float = 0.5,
    max_retry_interval: float = 10.0,
):
    if payload is None:
        payload = {}

    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                url,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(
                        f"HTTP Error {response.status}: {text} : {response.headers}"
                    )

                try:
                    response_json = await response.json()

                    exist, err_info = has_system_error(response_json)
                    if exist:
                        raise Exception(
                            f'SystemError detected, uid: {response_json.get("uid")}, err: {err_info}'
                        )

                    return response_json, response.headers
                except aiohttp.ContentTypeError as e:
                    raise Exception("Invalid JSON response") from e

        except asyncio.TimeoutError as e:
            logger.warning(
                f"Request timeout after {timeout}s, URL: {url}, Headers: {session.headers}"
            )
            break

        except Exception as e:
            logger.error(
                f"Async invocation failed on attempt {retries + 1}:{str(e)}, URL: {url}, Headers: {session.headers}"
            )

        retries += 1
        if retries > max_retries:
            return None, None

        sleep_time = min(
            initial_retry_interval * (2**retries) + random.uniform(0, 5),
            max_retry_interval,
        )
        await asyncio.sleep(sleep_time)


async def batch_function_call_async(payload_list, url, timeout, concurrency=1500):
    connector = aiohttp.TCPConnector(
        limit=concurrency,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_task(payload):
            if not payload:
                return None
            async with semaphore:
                st = time.monotonic()
                result = await async_invoke_function(session, url, timeout, payload)
                return result, time.monotonic() - st

        tasks = [limited_task(payload) for payload in payload_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = results if results else []
        data_list = []
        elapsed_times = []
        max_elapsed = -1
        max_elapsed_header = None
        for (data, header), elapsed in results:
            if elapsed > max_elapsed:
                max_elapsed = elapsed
                max_elapsed_header = header
            data_list.append(data)
            elapsed_times.append(elapsed)
            # logger.debug(f"functioncall took {elapsed:.4f} seconds, header: {header}.)")

        p50 = median(elapsed_times)
        p90 = calculate_percentile(elapsed_times, 90)
        p99 = calculate_percentile(elapsed_times, 99)
        logger.info(
            f"Longest functioncall took {max_elapsed:.4f} seconds, header: {max_elapsed_header}, timeout: {timeout}, connector: {id(connector)}, Active connections: {len(connector._conns)}, p50: {p50}, p90: {p90}, p99: {p99}"
        )

        return data_list


def get_runtime_name(runtime, language):
    if runtime:
        return runtime
    else:
        return str(language).lower() + "-default"


def caculate_concurrency():
    concurrency_for_one_exp = 3000
    try:
        dp = constants.data_parallel_world_size()
    except Exception as e:
        dp = 16
    return concurrency_for_one_exp // dp


def batch_function_call(payload_list, task_type, timeout):
    start_time = time.time()
    url = f"{FUNCTIONCALL_SERVICE_DOMAIN}/apis/functioncalls"

    concurrency = caculate_concurrency()
    logger.info(
        f"Batch function call start, task type: {task_type}, request count: {len(payload_list)}, time: {time.ctime(start_time)} ms, concurrency: {concurrency}"
    )
    result = asyncio.run(
        batch_function_call_async(payload_list, url, timeout, concurrency=concurrency)
    )
    execution_time = time.time() - start_time
    logger.info(
        f"Batch function call done, task type: {task_type}, batch size: {len(payload_list)}, cost: {execution_time * 1000:.0f} ms"
    )
    return result
