import asyncio
import os
import threading
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field

import aiohttp
import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sglang.srt.entrypoints.http_server import validate_json_request
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from openai.types.chat.chat_completion import ChatCompletion

from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging
from areal.utils.network import gethostip

logger = logging.getLogger("ArealOpenAI Proxy")


@retry(
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    stop=stop_after_attempt(10),
    reraise=True,
)
async def _post_json_with_retry(
    session: aiohttp.ClientSession, url: str, payload: dict
) -> dict:
    timeout = aiohttp.ClientTimeout(total=10)
    async with session.post(url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()


class AReaLStartSessionRequest(BaseModel):
    init_from_session_id: str | None = None


class AReaLEndSessionRequest(BaseModel):
    session_id: str
    final_reward: float | None = None


class AReaLSetRewardRequest(BaseModel):
    session_id: str
    completion_id: str
    reward: float


class SessionData:
    def __init__(
        self,
        id: str,
        completed: bool = False,
        completions: OrderedDict[str, InteractionWithTokenLogpReward] | None = None,
        final_reward: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        messages: list[dict] | None = None,
    ):
        self.id = id
        self.completed = completed
        self.completions = completions or OrderedDict()
        self.final_reward = final_reward
        self.start_time = start_time or time.time()
        self.end_time = end_time
        self.messages = messages or []

    def export_completions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        # copied from client code, to be moved to InteractionWithTokenLogpReward
        if len(self.completions) == 0:
            return {}

        # print(f"session {self.id}, length of completions: {len(self.completions)}")
        comp_time_sequence = list(
            reversed([comp for _, comp in self.completions.items()])
        )
        current_reward = self.final_reward
        # Check if the last-created completion has a reward set
        if comp_time_sequence:
            for i, comp in enumerate(comp_time_sequence):
                if i == 0:
                    if comp.reward is not None and current_reward is not None:
                        logger.warning(
                            "Both final reward and the most recent completion have a reward set. "
                            "Only the final reward will be used."
                        )
                    elif comp.reward is None and current_reward is None:
                        logger.warning(
                            "The most recent completion does not have a reward set. "
                            "All completions will have None reward."
                        )
                    current_reward = current_reward or comp.reward or 0.0
                else:
                    delta_reward = comp.reward or 0.0
                    current_reward = current_reward * discount + delta_reward

                comp.reward = current_reward

        completions = self.completions

        if style == "concat":
            raise NotImplementedError("Concat style is not implemented yet")
            # for comp in completions.values():
            #     if comp.chat_template_type != "concat":
            #         raise ValueError(
            #             "Cannot export completions in 'concat' style when "
            #             'comp.chat_template_type != "concat" for any completion. '
            #             "This is because when applying chat template using some tokenizers, "
            #             "there might be some tokens added or removed (e.g. think tokens), "
            #             "making it impossible to construct the conversation tree. "
            #             "Please use 'individual' style instead."
            #         )

            # def _is_prefix(a: list[dict], b: list[dict]) -> bool:
            #     # True if a is a strict prefix of b
            #     if len(a) >= len(b):
            #         return False
            #     for i in range(len(a)):
            #         if a[i] != b[i]:
            #             return False
            #     return True

            # # Precompute normalized messages
            # meta = {}
            # for cid, comp in completions.items():
            #     meta[cid] = {
            #         "norm_msgs": comp.messages or [],
            #         "obj": comp,
            #     }

            # # 1) Construct parent-child relationships using longest prefix rule
            # # Sort potential children by (message length asc, created asc) so parents are available
            # ordered = sorted(
            #     meta.items(),
            #     key=lambda kv: (
            #         len(kv[1]["norm_msgs"]),
            #         kv[1]["obj"].completion.created,
            #     ),
            # )

            # # Reset parents before rebuilding
            # for _, info in ordered:
            #     info["obj"].parent = None

            # for child_id, child_info in ordered:
            #     child_msgs = child_info["norm_msgs"]
            #     best_parent = None
            #     best_len = -1
            #     for parent_id, parent_info in ordered:
            #         if parent_id == child_id:
            #             continue
            #         parent_msgs = parent_info["norm_msgs"]
            #         if _is_prefix(parent_msgs, child_msgs):
            #             plen = len(parent_msgs)
            #             # choose the longest prefix
            #             if plen > best_len:
            #                 best_parent = parent_info["obj"]
            #                 best_len = plen
            #     child_info["obj"].parent = best_parent

            # # Build children mapping to find leaf nodes.
            # children_map: dict[str, list[InteractionWithTokenLogpReward]] = defaultdict(
            #     list
            # )
            # for _, info in meta.items():
            #     obj: InteractionWithTokenLogpReward = info["obj"]
            #     if obj.parent is not None:
            #         children_map[obj.parent.completion.id].append(obj)

            # # Return only leaf nodes (nodes without children)
            # parents_with_children = set(children_map.keys())
            # leaf_only: dict[str, InteractionWithTokenLogpReward] = {}
            # for cid, info in meta.items():
            #     obj = info["obj"]
            #     if obj.completion.id not in parents_with_children:
            #         leaf_only[cid] = obj
            # return leaf_only
        elif style == "individual":
            return dict(**completions)
        else:
            raise ValueError(f"Invalid export completions style {style}")


@dataclass
class SharedData:
    client: ArealOpenAI | None = field(default=None)
    session_cache: dict[str, SessionData] = field(default_factory=dict)


def build_app(
    client: ArealOpenAI | None = None,
    session_cache: dict[str, SessionData] | None = None,
):
    app = FastAPI()

    session_cache = session_cache if session_cache is not None else {}
    app.state.shared_data = SharedData(client=client, session_cache=session_cache)

    def get_shared_data() -> SharedData:
        return app.state.shared_data

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/rl/start_session")
    async def start_session(request: AReaLStartSessionRequest):
        """Start a new session or initialize from an existing session."""
        init_from_session_id = request.init_from_session_id

        session_id = uuid.uuid4().hex  # assume unique
        session_cache = get_shared_data().session_cache

        if init_from_session_id is not None:
            if init_from_session_id not in session_cache:
                raise HTTPException(status_code=400, detail="Session not found")
            session_cache[session_id] = deepcopy(session_cache[init_from_session_id])
            return {"session_id": session_id}

        session_cache[session_id] = SessionData(
            id=session_id,
            completed=False,
            completions=OrderedDict(),
            final_reward=None,
            start_time=time.time(),
            end_time=None,
            messages=[],
        )
        return {"session_id": session_id}

    @app.post("/v1/rl/end_session")
    async def end_session(request: AReaLEndSessionRequest):
        session_id = request.session_id
        final_reward = request.final_reward

        state = get_shared_data()
        if session_id not in state.session_cache:
            raise HTTPException(status_code=400, detail="Session not found")
        state.session_cache[session_id].final_reward = final_reward
        state.session_cache[session_id].completed = True
        return {"message": "success"}

    @app.post("/v1/rl/set_reward")
    async def set_reward(request: AReaLSetRewardRequest):
        session_id = request.session_id
        completion_id = request.completion_id
        reward = request.reward

        state = get_shared_data()
        if session_id not in state.session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )
        completions = state.session_cache[session_id].completions
        if completion_id not in completions:
            raise HTTPException(
                status_code=400, detail=f"Completion {completion_id} not found"
            )
        state.session_cache[session_id].completions[completion_id].reward = reward
        return {"message": "success"}

    @app.post(
        "/v1/{session_id}/chat/completions",
        dependencies=[Depends(validate_json_request)],
    )
    async def chat_completions(
        request: ChatCompletionRequest, session_id: str
    ) -> ChatCompletion:
        state = get_shared_data()
        session_cache = state.session_cache

        if state.client is None:
            raise HTTPException(
                status_code=500,
                detail=f"Client not found. request: {request.model_dump()}",
            )

        if session_id not in session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )

        kwargs = request.model_dump(exclude={"session_id"})

        # TODO: inspect the client create method to get the allowed arguments
        areal_client_allowed_args = [
            "messages",
            "frequency_penalty",
            "max_completion_tokens",
            "max_tokens",
            "metadata",
            "stop",
            "store",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
            "extra_body",
        ]

        dropped_args = []
        for k, v in kwargs.items():
            if k not in areal_client_allowed_args:
                dropped_args.append((k, v))

        for k, _ in dropped_args:
            del kwargs[k]

        dropped_non_default_args = [
            (k, v)
            for k, v in dropped_args
            if v != ChatCompletionRequest.model_fields[k].default
        ]
        if len(dropped_non_default_args):
            dropped_args_str = "\n".join(
                [f"  {k}: {v}" for k, v in dropped_non_default_args]
            )
            logger.warning(
                f"dropped unsupported non-default arguments for areal client:\n"
                f"{dropped_args_str}"
            )

        if "temperature" not in kwargs:
            kwargs["temperature"] = 1.0
            logger.warning("temperature not set in request, defaulting to 1.0")
        elif kwargs["temperature"] != 1.0:
            logger.warning(
                f"temperature is set to {kwargs['temperature']} in request, "
                f"we suggest using 1.0 for RL tasks"
            )

        if "top_p" not in kwargs:
            kwargs["top_p"] = 1.0
            logger.warning("top_p not set in request, defaulting to 1.0")
        elif kwargs["top_p"] != 1.0:
            logger.warning(
                f"top_p is set to {kwargs['top_p']} in request, "
                f"we suggest using 1.0 for RL tasks"
            )

        try:
            completion: ChatCompletion = await state.client.chat.completions.create(
                areal_completion_cache=session_cache[session_id].completions, **kwargs
            )
            return completion
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # TODO: add response support

    return app


class ProxyServer:
    def __init__(
        self,
        port: int,
        *,
        client: ArealOpenAI | None = None,
        session_cache: dict[str, SessionData] | None = None,
        log_level: str = "warning",
    ):
        self.port = port
        self.session_cache = session_cache if session_cache is not None else {}
        self.app = build_app(client, self.session_cache)
        self.host_ip = gethostip()
        self._localhost = "0.0.0.0"
        self.server_config = uvicorn.Config(
            self.app, host=self._localhost, port=self.port, log_level=log_level
        )
        self.server = uvicorn.Server(self.server_config)
        self.thread = threading.Thread(target=self.server.run)

    @property
    def public_addr(self) -> str:
        return f"http://{self.host_ip}:{self.port}"

    @property
    def local_addr(self) -> str:
        return f"http://{self._localhost}:{self.port}"

    def check_health(self, timeout: int = 20) -> bool:
        try:
            response = requests.get(f"{self.local_addr}/health", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def wait_until_ready(self, timeout: int = 20):
        while not self.check_health(timeout):
            logger.info(f"Waiting for server {self.public_addr} to be ready...")
            time.sleep(1)
        logger.info(f"Server {self.public_addr} is ready!")

    def start(self, wait_until_ready: bool = True, timeout: int = 20):
        logger.info(f"Starting proxy server on {self.public_addr}")
        self.thread.start()
        if wait_until_ready:
            self.wait_until_ready(timeout)

    def close(self):
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.thread.is_alive():
                logger.warning(
                    "Proxy server thread did not stop gracefully within timeout"
                )
        logger.info("Proxy server stopped.")

    async def get_completions(
        self, session_ids: list[str], discount: float = 1.0, style: str = "individual"
    ) -> dict[str, InteractionWithTokenLogpReward]:
        completions = {}
        finished_index = 0
        while finished_index < len(session_ids):
            session_id = session_ids[finished_index]
            if session_id not in self.session_cache:
                print(f"all session ids: {list(self.session_cache.keys())}")
                print(f"requested session ids: {session_ids}")
                raise ValueError(f"Session {session_id} not found")

            if self.session_cache[session_id].completed:
                completions.update(
                    self.session_cache[session_id].export_completions(
                        discount=discount, style=style
                    )
                )
                finished_index += 1
            else:
                await asyncio.sleep(0.1)
        return completions


def insert_session_id_to_kwargs(kwargs: dict, session_id: str) -> None:
    # add self.session_id to extra_body
    extra_body = kwargs.get("extra_body", {})
    if extra_body is None:
        extra_body = {}

    # Ensure extra_body is a dict and add session_id
    if not isinstance(extra_body, dict):
        extra_body = {}

    extra_body["session_id"] = session_id
    kwargs["extra_body"] = extra_body


class ProxySession:
    def __init__(self, base_url: str = None):
        assert base_url is not None
        self.stripped_base_url = base_url.rstrip("/")
        self.session_id = None
        self.final_reward = None
        self.http_session = aiohttp.ClientSession()
        self.ori_openai_base_url = None
        self.ori_openai_api_key = None

    async def set_reward(self, reward: float, completion_id: str | None = None):
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        if completion_id is not None:
            payload = AReaLSetRewardRequest(
                session_id=self.session_id, completion_id=completion_id, reward=reward
            ).model_dump()
            await _post_json_with_retry(
                self.http_session,
                f"{self.stripped_base_url}/rl/set_reward",
                payload=payload,
            )
        else:
            self.final_reward = reward

    def __enter__(self):
        raise TypeError("Use async with instead")

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    async def __aenter__(self):
        data = await _post_json_with_retry(
            self.http_session,
            f"{self.stripped_base_url}/rl/start_session",
            payload={},
        )
        self.session_id = data["session_id"]

        self.ori_openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.ori_openai_api_key = os.getenv("OPENAI_API_KEY")

        os.environ["OPENAI_BASE_URL"] = f"{self.stripped_base_url}/{self.session_id}"
        os.environ["OPENAI_API_KEY"] = f"{self.session_id}"
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            await self.http_session.close()
            return

        if self.session_id is None:
            raise ValueError("Session ID is not set")

        payload = AReaLEndSessionRequest(
            session_id=self.session_id, final_reward=self.final_reward
        ).model_dump()
        await _post_json_with_retry(
            self.http_session,
            f"{self.stripped_base_url}/rl/end_session",
            payload=payload,
        )
        await self.http_session.close()

        os.environ["OPENAI_BASE_URL"] = self.ori_openai_base_url
        os.environ["OPENAI_API_KEY"] = self.ori_openai_api_key


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=8000)
#     args = parser.parse_args()

#     server = ProxyServer(args.port)
#     server.start(wait_until_ready=True)

#     async def run_client():
#         async with ProxyClient(f"{server.public_addr}/v1") as client:
#             await client.set_reward(1.0)

#     asyncio.run(run_client())
