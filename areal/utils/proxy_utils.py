import asyncio
import os
from collections.abc import Awaitable, Callable

import aiohttp
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_exponential,
)

from areal.utils import logging

logger = logging.getLogger("ArealOpenAI Proxy Utils")


def ensure_end_with_slash(url: str) -> str:
    if not url.endswith("/"):
        return url + "/"
    return url


# Based on sglang.srt.entrypoints.http_server.validate_json_request
async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict | BaseModel | None = None,
    total_timeout: int = 10,
) -> dict:
    timeout = aiohttp.ClientTimeout(total=total_timeout)

    if payload is None:
        payload = {}
    elif isinstance(payload, BaseModel):
        payload = payload.model_dump()

    async with session.post(url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()


def should_retry(exception: Exception):
    """Check if exception is a retryable HTTP error (503, 502, 429, etc.)"""
    if isinstance(exception, aiohttp.ClientResponseError):
        return exception.status in [504, 503, 502, 429, 408]
    if isinstance(exception, aiohttp.ClientConnectionError):
        return True
    elif isinstance(exception, asyncio.TimeoutError):
        return True
    return False


def log_retry(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    level = "warning"
    if isinstance(exception, aiohttp.ClientResponseError):
        if exception.status in [429]:
            # debug for 503
            level = "debug"

    exception_message = (
        "Timeout" if isinstance(exception, asyncio.TimeoutError) else str(exception)
    )
    message = f"Retry #{retry_state.attempt_number} due to: {exception_message}"
    if level == "warning":
        logger.warning(message)
    else:
        logger.debug(message)


def get_retry_strategy(
    allowed_attempt: int, multiplier: float = 0.5, min: float = 0.5, max: float = 5
):
    return retry(
        retry=retry_if_exception(should_retry),
        wait=wait_exponential(multiplier=multiplier, min=min, max=max),
        stop=stop_never if allowed_attempt < 0 else stop_after_attempt(allowed_attempt),
        reraise=True,
        # before_sleep=log_retry,
    )


@get_retry_strategy(allowed_attempt=10)
async def post_json_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict | BaseModel | None = None,
    total_timeout: float = 10,
) -> dict:
    return await post_json(session, url, payload, total_timeout)


RL_START_SESSION_PATHNAME = "rl/start_session"
RL_END_SESSION_PATHNAME = "rl/end_session"
RL_SET_REWARD_PATHNAME = "rl/set_reward"
RL_FINISH_TASK_PATHNAME = "rl/finish_task"
CHAT_COMPLETIONS_PATHNAME = "chat/completions"
RESPONSES_PATHNAME = "responses"


class AReaLStartSessionRequest(BaseModel):
    task_id: str
    init_from_session_id: str | None = None


class AReaLSetRewardRequest(BaseModel):
    interaction_id: str | None = None
    reward: float


class AReaLFinishTaskRequest(BaseModel):
    task_id: str
    put_to_queue: bool = True


async def _set_reward(
    http_session: aiohttp.ClientSession,
    interaction_id: str | None,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
):
    payload = AReaLSetRewardRequest(interaction_id=interaction_id, reward=reward)
    await post_json_with_retry(http_session, url=url, payload=payload)


async def set_interaction_reward(
    http_session: aiohttp.ClientSession,
    interaction_id: str,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
):
    await _set_reward(
        http_session, interaction_id=interaction_id, reward=reward, url=url
    )


async def set_last_interaction_reward(
    http_session: aiohttp.ClientSession,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
):
    await _set_reward(http_session, interaction_id=None, reward=reward, url=url)


async def run_and_submit_rewards(
    func: Callable[..., Awaitable[float | dict[str, float]]],
    data: dict,
    base_url: str | None = None,
    pathname: str = RL_SET_REWARD_PATHNAME,
):
    """Run a coroutine function and submit rewards to the proxy server.
    The function should return a float or a dictionary of interaction_id to reward.

    Args:
        func: The coroutine function to run.
        data: The data to pass to the function.
        base_url:
            The base URL of the proxy server. If not provided, the OPENAI_BASE_URL
            environment variable will be used.
        pathname:
            The pathname to set the reward. If not provided, the RL_SET_REWARD_PATHNAME
            will be used.
    """
    if base_url is None:
        base_url = os.environ["OPENAI_BASE_URL"]
    base_url = ensure_end_with_slash(base_url)

    def _get_float_reward(reward: float | int):
        if isinstance(reward, float) or isinstance(reward, int):
            return float(reward)
        else:
            raise ValueError(
                f"Reward must be a float or an integer, got {type(reward)}"
            )

    async with aiohttp.ClientSession(base_url) as session:
        rewards = await func(data)

        if isinstance(rewards, dict):
            for interaction_id, reward in rewards.items():
                await set_interaction_reward(
                    session,
                    interaction_id=interaction_id,
                    reward=_get_float_reward(reward),
                    url=pathname,
                )
        else:
            await set_last_interaction_reward(
                session,
                reward=_get_float_reward(rewards),
                url=pathname,
            )
