import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParams

from areal.api.engine_api import InferenceEngine
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.client import ArealOpenAI
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports, gethostip

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward


class SessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id

        self._completed = False
        self._completions = InteractionCache()
        self._completed_event = asyncio.Event()

        self._start_time = time.time()
        self._end_time = None

    def finish(self):
        self._completed = True
        self._end_time = time.time()
        self._completed_event.set()

    @property
    def completions(self):
        return self._completions

    async def wait_for_finish(self):
        await self._completed_event.wait()

    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, "InteractionWithTokenLogpReward"]:
        if len(self.completions) == 0:
            return {}
        self.completions.apply_reward_discount(turn_discount=discount)
        return self.completions.export_interactions(style=style)


RL_START_SESSION_PATHNAME = "rl/start_session"
RL_END_SESSION_PATHNAME = "rl/end_session"
RL_SET_REWARD_PATHNAME = "rl/set_reward"
CHAT_COMPLETIONS_PATHNAME = "chat/completions"
RESPONSES_PATHNAME = "responses"


class AReaLStartSessionRequest(BaseModel):
    task_id: str


class AReaLSetRewardRequest(BaseModel):
    interaction_id: str | None = None
    reward: float


@dataclass
class _AppSharedData:
    model: ArealOpenAI
    session_cache: dict[str, SessionData]
    active_sessions: Queue[str]
    capacity: int
    lock: threading.Lock


def build_app(
    model: ArealOpenAI,
    session_cache: dict[str, SessionData],
    active_sessions: Queue[str],
    lock: threading.Lock,
    logger: logging.Logger,
):
    app = FastAPI()

    app.state.shared_data = _AppSharedData(
        model=model,
        session_cache=session_cache,
        active_sessions=active_sessions,
        capacity=0,
        lock=lock,
    )

    def get_shared_data() -> _AppSharedData:
        return app.state.shared_data

    def get_model() -> ArealOpenAI:
        state = get_shared_data()
        if state.model is None:
            raise HTTPException(status_code=500, detail="model client not found")
        return state.model

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/grant_capacity")
    def grant_capacity():
        state = get_shared_data()
        with state.lock:
            state.capacity += 1
            return {"capacity": state.capacity}

    @app.post(f"/{RL_START_SESSION_PATHNAME}")
    async def start_session(request: AReaLStartSessionRequest):
        """Start a new session or initialize from an existing session."""
        task_id = request.task_id
        state = get_shared_data()
        with state.lock:
            if state.capacity <= 0:
                raise HTTPException(
                    status_code=429,
                    detail="No available capacity to start a new session",
                )

            idx = 0
            while (session_id := f"{task_id}-{idx}") in state.session_cache:
                idx += 1

            queue = state.active_sessions
            try:
                queue.put_nowait(session_id)
            except Full:
                raise HTTPException(status_code=503, detail="Queue is full")

            state.capacity -= 1
            session_cache = state.session_cache
            session_cache[session_id] = SessionData(session_id=session_id)

        return {"session_id": session_id}

    @app.post(f"/{{session_id}}/{RL_END_SESSION_PATHNAME}")
    async def end_session(session_id: str):
        state = get_shared_data()
        if session_id not in state.session_cache:
            raise HTTPException(status_code=400, detail="Session not found")

        state.session_cache[session_id].finish()
        return {"message": "success"}

    @app.post(f"/{{session_id}}/{RL_SET_REWARD_PATHNAME}")
    async def set_reward(request: AReaLSetRewardRequest, session_id: str):
        interaction_id = request.interaction_id
        reward = request.reward

        state = get_shared_data()
        if session_id not in state.session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )

        completions = state.session_cache[session_id].completions
        if interaction_id is None:
            # take the last interaction id
            if len(completions) == 0:
                logger.error(f"No interactions in session {session_id}")
                raise HTTPException(
                    status_code=400, detail="No interactions in session"
                )
            interaction_id = completions.last_interaction_id
        elif interaction_id not in completions:
            logger.error(
                f"Interaction {interaction_id} not found in session {session_id}"
            )
            raise HTTPException(
                status_code=400, detail=f"Interaction {interaction_id} not found"
            )
        state.session_cache[session_id].completions.set_reward(interaction_id, reward)
        return {"message": "success"}

    async def _call_client_create(
        create_fn: Callable[..., Awaitable[ChatCompletion | Response]],
        request: dict[str, Any] | BaseModel,
        session_id: str,
        extra_ignored_args: list[str] | None = None,
    ) -> ChatCompletion | Response:
        state = get_shared_data()
        session_cache = state.session_cache
        if session_id not in session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )

        sig = inspect.signature(create_fn)
        areal_client_ignored_args = ["model"] + (extra_ignored_args or [])
        areal_client_disallowed_args = ["areal_cache"]
        areal_client_allowed_args = list(
            k
            for k in sig.parameters.keys()
            if k not in areal_client_ignored_args
            and k not in areal_client_disallowed_args
        )

        kwargs = (
            request.model_dump() if isinstance(request, BaseModel) else dict(request)
        )
        dropped_args = []
        for k, v in kwargs.items():
            if k not in areal_client_allowed_args:
                dropped_args.append((k, v))

        for k, _ in dropped_args:
            del kwargs[k]

        def _is_default_value(k: str, v: Any) -> bool:
            if isinstance(request, BaseModel):
                return v == type(request).model_fields[k].default
            return False

        dropped_non_default_args = [
            (k, v)
            for k, v in dropped_args
            if k not in areal_client_ignored_args and not _is_default_value(k, v)
        ]
        if len(dropped_non_default_args):
            dropped_args_str = "\n".join(
                [f"  {k}: {v}" for k, v in dropped_non_default_args]
            )
            if logger is not None:
                logger.warning(
                    f"dropped unsupported non-default arguments for areal client:\n"
                    f"{dropped_args_str}"
                )

        if "temperature" not in kwargs:
            kwargs["temperature"] = 1.0
            if logger is not None:
                logger.warning("temperature not set in request, defaulting to 1.0")
        if "top_p" not in kwargs:
            kwargs["top_p"] = 1.0
            if logger is not None:
                logger.warning("top_p not set in request, defaulting to 1.0")

        try:
            return await create_fn(
                areal_cache=session_cache[session_id].completions, **kwargs
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        f"/{{session_id}}/{CHAT_COMPLETIONS_PATHNAME}",
        dependencies=[Depends(validate_json_request)],
    )
    async def chat_completions(
        request: CompletionCreateParams, session_id: str
    ) -> ChatCompletion:
        model = get_model()
        return await _call_client_create(
            create_fn=model.chat.completions.create,
            request=request,
            session_id=session_id,
        )

    @app.post(
        f"/{{session_id}}/{RESPONSES_PATHNAME}",
        dependencies=[Depends(validate_json_request)],
    )
    async def responses(request: ResponseCreateParams, session_id: str) -> Response:
        model = get_model()
        return await _call_client_create(
            create_fn=model.responses.create,
            request=request,
            session_id=session_id,
        )

    return app


_PROXY_SERVER_MAX_QSIZE = 1024


class OpenAIProxyServer:
    def __init__(
        self,
        model: ArealOpenAI,
        server_log_level: int = logging.INFO,
        uvicorn_log_level: int = logging.WARNING,
    ):
        self.port = find_free_ports(1)[0]
        self.model = model

        self.session_cache: dict[str, SessionData] = {}
        self.active_sessions: Queue[str] = Queue(maxsize=_PROXY_SERVER_MAX_QSIZE)
        self.lock = threading.Lock()
        self.logger = getLogger("OpenAIProxy", level=server_log_level)

        self.app = build_app(
            model=model,
            session_cache=self.session_cache,
            active_sessions=self.active_sessions,
            lock=self.lock,
            logger=self.logger,
        )

        self.host_ip = gethostip()
        self._localhost = "0.0.0.0"
        self.server_config = uvicorn.Config(
            self.app, host=self._localhost, port=self.port, log_level=uvicorn_log_level
        )
        self.server = uvicorn.Server(self.server_config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def engine(self) -> InferenceEngine:
        return self.model.engine

    @property
    def public_addr(self) -> str:
        return f"http://{self.host_ip}:{self.port}"

    @property
    def local_addr(self) -> str:
        return f"http://{self._localhost}:{self.port}"

    def _check_health(self, timeout: int = 20) -> bool:
        try:
            response = requests.get(f"{self.local_addr}/health", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def wait_until_ready(self, timeout: int = 20):
        while not self._check_health(timeout):
            self.logger.info(
                f"Waiting for OpenAI proxy server at {self.public_addr} to be ready..."
            )
            time.sleep(1)
        self.logger.info(f"OpenAI proxy server at {self.public_addr} is ready!")

    def start(self, wait_until_ready: bool = True, timeout: int = 20):
        self.logger.info(f"Starting OpenAI proxy server on {self.public_addr}")
        self.thread.start()
        if wait_until_ready:
            self.wait_until_ready(timeout)

    def close(self):
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.thread.is_alive():
                self.logger.warning(
                    "Proxy server thread did not stop gracefully within timeout"
                )
        self.logger.info("OpenAI proxy server stopped.")

    async def grant_capacity(self):
        resp = requests.post(f"{self.local_addr}/grant_capacity")
        resp.raise_for_status()
        await asyncio.sleep(0)

    async def fetch_next_session(self) -> str:
        while True:
            try:
                return self.active_sessions.get_nowait()
            except Empty:
                await asyncio.sleep(0.1)

    async def wait_for_session(
        self, session_id: str, discount: float = 1.0, style: str = "individual"
    ) -> SessionData:
        if session_id not in self.session_cache:
            raise KeyError(f"Session {session_id} not found")
        # Wait for session to be completed using event
        await self.session_cache[session_id].wait_for_finish()
        session = self.session_cache.pop(session_id)
        return session.export_interactions(discount=discount, style=style)

    def set_reward(self, session_id: str, completion_id: str, reward: float):
        """Set reward for a specific completion/response by its ID."""
        if session_id not in self.session_cache:
            raise KeyError(f"Session {session_id} not found")
        session = self.session_cache[session_id]
        session.completions.set_reward(completion_id, reward)

    def set_last_reward(self, session_id: str, reward: float):
        """Set reward for the most recent completion/response."""
        if session_id not in self.session_cache:
            raise KeyError(f"Session {session_id} not found")
        session = self.session_cache[session_id]
        session.completions.set_last_reward(reward)


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
