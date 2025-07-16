import asyncio
from typing import Any, Dict, Optional
from PIL import Image
import aiohttp

DEFAULT_RETRIES = 1
DEFAULT_REQUEST_TIMEOUT = 3600


async def arequest_with_retry(
    addr: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    method: str = "POST",
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    retry_delay: float = 1.0,
) -> aiohttp.ClientResponse:
    timeout = timeout or DEFAULT_REQUEST_TIMEOUT
    last_exception = None
    max_retries = max_retries or DEFAULT_RETRIES
    base_url = f"http://{addr}"
    url = f"{base_url}{endpoint}"

    for attempt in range(max_retries):

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=timeout,
                    sock_connect=timeout,
                )
            ) as session:
                if method.upper() == "GET":
                    response = await session.get(url)
                elif method.upper() == "POST":
                    response = await session.post(url, data=payload)
                elif method.upper() == "PUT":
                    response = await session.put(url, data=payload)
                elif method.upper() == "DELETE":
                    response = await session.delete(url)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                response.raise_for_status()
                return await response.json()
        except (
            aiohttp.ClientError,
            aiohttp.ClientResponseError,
            asyncio.TimeoutError,
        ) as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    raise RuntimeError(
        f"Failed after {max_retries} retries each. " f"Last error: {last_exception}"
    )
