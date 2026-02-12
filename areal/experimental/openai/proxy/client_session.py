from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING

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

from areal.infra.utils.http import ensure_end_with_slash
from areal.utils.logging import getLogger

from .server import (
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    SetRewardRequest,
    StartSessionRequest,
    deserialize_interactions,
)

if TYPE_CHECKING:
    from ..types import InteractionWithTokenLogpReward

logger = getLogger("OpenAIProxyClient")


class OpenAIProxyClient:
    """Client session for interacting with the OpenAI proxy server.

    This class manages RL session lifecycle (start/end session) and provides
    methods for setting rewards and exporting interactions. It uses composition
    rather than inheritance - an aiohttp.ClientSession must be passed in.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The HTTP session to use for requests
    base_url : str
        Base URL of the proxy server
    task_id : str
        Unique identifier for this task

    Example
    -------
    ```python
    async with aiohttp.ClientSession() as http_session:
        proxy_client = OpenAIProxyClient(
            session=http_session,
            base_url="http://localhost:8000",
            task_id="task-1",
        )
        async with proxy_client:
            # Make OpenAI API calls using proxy_client.session_url
            await proxy_client.set_last_reward(1.0)

        # After context exit, export interactions
        interactions = await proxy_client.export_interactions()
    ```
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        task_id: str,
    ):
        self._session = session
        self.base_url = ensure_end_with_slash(base_url)
        self.task_id = task_id
        self.session_id: str | None = None

    @property
    def session_url(self) -> str:
        """Return the full URL for this session."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        return f"{self.base_url}{self.session_id}/"

    async def set_reward(self, completion_id: str, reward: float):
        """Set reward for a specific completion/response by its ID."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_interaction_reward(
            self._session,
            interaction_id=completion_id,
            reward=reward,
            url=f"{self.base_url}{self.session_id}/{RL_SET_REWARD_PATHNAME}",
        )

    async def set_last_reward(self, reward: float):
        """Set reward for the most recent completion/response."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_last_interaction_reward(
            self._session,
            reward=reward,
            url=f"{self.base_url}{self.session_id}/{RL_SET_REWARD_PATHNAME}",
        )

    async def export_interactions(
        self,
        discount: float = 1.0,
        style: str = "individual",
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Export interactions for this session via HTTP.

        This method should be called after the session context exits
        (i.e., after `__aexit__` has ended the RL session), since
        `/export_trajectories` waits for the session to finish.

        Parameters
        ----------
        discount : float
            Discount factor for reward propagation
        style : str
            Export style ("individual" or "merged")

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary mapping interaction IDs to their data
        """

        if self.session_id is None:
            raise ValueError("Session ID is not set")

        url = f"{self.base_url}export_trajectories"
        payload = {
            "session_id": self.session_id,
            "discount": discount,
            "style": style,
        }
        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return deserialize_interactions(data["interactions"])

    async def __aenter__(self) -> OpenAIProxyClient:
        """Start the RL session via HTTP request."""
        data = await _start_session(
            self._session,
            url=f"{self.base_url}{RL_START_SESSION_PATHNAME}",
            payload=StartSessionRequest(task_id=self.task_id),
        )
        self.session_id = data["session_id"]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """End the RL session via HTTP request.

        Always attempts to end the session, even on exception, to avoid
        leaving zombie sessions on the server.
        """
        if self.session_id is None:
            return  # Session was never started

        # Always try to end the session, even on exception
        try:
            await post_json_with_retry(
                self._session,
                url=f"{self.base_url}{self.session_id}/{RL_END_SESSION_PATHNAME}",
            )
        except Exception as e:
            # Raised errors will be properly handled by OpenAIProxyWorkflow
            logger.warning(f"Failed to end session {self.session_id}: {e}")
            raise


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

    exception_message = (
        "Timeout" if isinstance(exception, asyncio.TimeoutError) else str(exception)
    )
    message = f"Retry #{retry_state.attempt_number} due to: {exception_message}"
    logger.warning(message)


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
    payload = SetRewardRequest(interaction_id=interaction_id, reward=reward)
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
