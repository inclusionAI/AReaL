import asyncio
from typing import Any, Dict, Optional

import aiohttp

from realhf.base import logging

DEFAULT_RETRIES = 1
DEFAULT_REQUEST_TIMEOUT = 3600


logger = logging.getLogger(__file__)


def get_default_connector():
    return aiohttp.TCPConnector(limit=0, use_dns_cache=False, force_close=True)


async def arequest_with_retry(
    addr: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    session: aiohttp.ClientSession | None = None,
    method: str = "POST",
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    retry_delay: float = 1.0,
    verbose=False,
) -> Dict:
    timeout = timeout or DEFAULT_REQUEST_TIMEOUT
    last_exception = None
    max_retries = max_retries or DEFAULT_RETRIES
    base_url = f"http://{addr}"
    url = f"{base_url}{endpoint}"

    timeo = aiohttp.ClientTimeout(
        total=timeout,
        sock_connect=timeout,
        connect=timeout,
    )
    if session is None:
        _session = aiohttp.ClientSession(
            timeout=timeo,
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
    else:
        _session = session

    for attempt in range(max_retries):
        try:
            if verbose:
                logger.info("enter client session, start sending requests")
            if method.upper() == "GET":
                ctx = _session.get(url, timeout=timeo)
            elif method.upper() == "POST":
                ctx = _session.post(url, json=payload, timeout=timeo)
            elif method.upper() == "PUT":
                ctx = _session.put(url, json=payload, timeout=timeo)
            elif method.upper() == "DELETE":
                ctx = _session.delete(url, timeout=timeo)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            async with ctx as response:
                if verbose:
                    logger.info("http requests return")
                response.raise_for_status()
                res = await response.json()
                if verbose:
                    logger.info("get http result")
                if session is None:
                    await _session.close()
                return res
        except (
            aiohttp.ClientError,
            aiohttp.ClientResponseError,
            asyncio.TimeoutError,
        ) as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    if session is None:
        await _session.close()
    raise RuntimeError(
        f"Failed after {max_retries} retries each. "
        f"Payload: {payload}. Addr: {addr}. Endpoint: {endpoint}. "
        f"Last error: {last_exception}"
    )
