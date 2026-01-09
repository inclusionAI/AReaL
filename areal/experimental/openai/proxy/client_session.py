import asyncio
import os
from collections.abc import Awaitable, Callable
from types import TracebackType

import aiohttp
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_exponential,
)

from areal.utils.http import ensure_end_with_slash
from areal.utils.logging import getLogger

from .server import (
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    AReaLSetRewardRequest,
    AReaLStartSessionRequest,
)

logger = getLogger("OpenAIProxyClient")


class OpenAIProxyClientSession(aiohttp.ClientSession):
    def __init__(self, base_url: str, task_id: str, *args, **kwargs):
        base_url = ensure_end_with_slash(base_url)
        super().__init__(base_url, *args, **kwargs)

        self.base_url = base_url
        self.task_id = task_id
        self.session_id = None

    @property
    def session_url(self) -> str:
        return str(self._build_url(self.session_id))

    async def set_reward(self, completion_id: str, reward: float):
        """Set reward for a specific completion/response by its ID."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_interaction_reward(
            self,
            interaction_id=completion_id,
            reward=reward,
            url=f"{self.session_id}/{RL_SET_REWARD_PATHNAME}",
        )

    async def set_last_reward(self, reward: float):
        """Set reward for the most recent completion/response."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_last_interaction_reward(
            self, reward=reward, url=f"{self.session_id}/{RL_SET_REWARD_PATHNAME}"
        )

    async def __aenter__(self) -> "OpenAIProxyClientSession":
        await super().__aenter__()
        data = await _start_session(
            self,
            f"{RL_START_SESSION_PATHNAME}",
            payload=AReaLStartSessionRequest(task_id=self.task_id),
        )
        self.session_id = data["session_id"]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if exc_type is not None:
            await super().__aexit__(exc_type, exc_val, exc_tb)
            return

        if self.session_id is None:
            raise ValueError("Session ID is not set")

        await post_json_with_retry(self, f"{self.session_id}/{RL_END_SESSION_PATHNAME}")
        await super().__aexit__(exc_type, exc_val, exc_tb)


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


async def _set_reward(
    http_session: aiohttp.ClientSession,
    interaction_id: str | None,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
):
    payload = AReaLSetRewardRequest(interaction_id=interaction_id, reward=reward)
    try:
        await post_json_with_retry(http_session, url=url, payload=payload)
    except aiohttp.ClientResponseError as e:
        if e.status == 400:
            logger.error(f"[error code {e.status}] Error setting reward: {e.message}")
        else:
            raise e


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


@get_retry_strategy(allowed_attempt=-1)
async def _start_session(*args, **kwargs):
    return await post_json(*args, **kwargs)


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
