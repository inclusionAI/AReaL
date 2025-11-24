import asyncio
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import aiofiles
import aiofiles.os
from transformers import PreTrainedTokenizerFast

from realhf.api.core.data_api import load_hf_tokenizer

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy import (
    ProxyServer,
    ProxySession,
)
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports

logger = logging.getLogger(__name__)


def sync_run_task(
    data, proxy_addr, run_agent_return_reward: Callable[[Any], Awaitable[float]]
):
    async def run_task(data, proxy_addr, run_agent_return_reward: Callable):
        async with ProxySession(base_url=proxy_addr) as session:
            session_id = session.session_id
            try:
                reward = await run_agent_return_reward(data)
            except Exception as e:
                logger.warning(f"Error in sync_run_task: {e}")
                reward = 0.0

            await session.set_reward(reward)

        return None, session_id, reward

    return asyncio.run(
        run_task(
            data=data,
            proxy_addr=proxy_addr,
            run_agent_return_reward=run_agent_return_reward,
        )
    )


# singleton
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class SingletonProcessPoolExecutor(ProcessPoolExecutor):
    pass


@singleton
class ProxyServerSingletonWarpper:
    def __init__(
        self,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        tool_call_parser: str | None = None,
        chat_template_type: str = "hf",
    ):
        self.client = ArealOpenAI(
            engine=engine,
            tokenizer=tokenizer,
            tool_call_parser=tool_call_parser,
            chat_template_type=chat_template_type,
        )

        free_port = find_free_ports(1)[0]
        self.proxy_server = ProxyServer(port=free_port, client=self.client)
        self.proxy_server.start(wait_until_ready=True)

    def get_client(self):
        return self.client

    def get_proxy_server(self):
        return self.proxy_server


class ProxyRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        tool_call_parser: str = "qwen25",
        chat_template_type: str = "concat",
        # proxy_server: ProxyServer,
        # run_agent_return_reward: Callable[[Any], Awaitable[float]],
        run_agent_return_reward_path: str = None,
        process_pool_executor_size: int = 128,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        export_style: str = "individual",
    ):
        # self.proxy_server = proxy_server
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.chat_template_type = chat_template_type
        self.api_version = "v1"
        self.n_samples = gconfig.n_samples
        self.rollout_stat_scope = rollout_stat_scope
        self.process_pool_executor = SingletonProcessPoolExecutor(
            max_workers=process_pool_executor_size
        )
        self.gconfig = gconfig
        assert (
            run_agent_return_reward_path is not None
        ), "run_agent_return_reward_path must not be None"
        self.run_agent_return_reward = import_from_string(
            ".".join([run_agent_return_reward_path, "run_agent_return_reward"]),
        )
        self.dump_dir = dump_dir
        self.export_style = export_style

    async def arun_episode(self, engine: InferenceEngine, data):
        if isinstance(self.tokenizer, str):
            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        self.proxy_server_warpper = ProxyServerSingletonWarpper(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            chat_template_type=self.chat_template_type,
        )
        self.proxy_server = self.proxy_server_warpper.get_proxy_server()
        futures = [
            self.process_pool_executor.submit(
                sync_run_task,
                data,
                f"{self.proxy_server.public_addr}/{self.api_version}",
                self.run_agent_return_reward,
            )
            for _ in range(self.n_samples)
        ]
        results = await asyncio.gather(
            *[asyncio.wrap_future(future) for future in futures]
        )
        error_message, session_ids, rewards = zip(*results)
        if any(error_message):
            for msg in error_message:
                if msg is not None:
                    logger.error(f"Error in run_agent: {msg}")
            raise RuntimeError("One or more tasks failed in run_agent.")

        for reward in rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        completions = await self.proxy_server.get_completions(
            session_ids=session_ids, style=self.export_style, discount=0.9
        )

        if self.dump_dir is not None:
            for session_id, completion in completions.items():
                version = completion.model_response.output_versions[-1]

                dump_path = os.path.join(self.dump_dir, str(version))
                await aiofiles.os.makedirs(dump_path, exist_ok=True)
                # Get the unique identifier for this prompt
                qid = None
                for key in ["query_id", "id", "qid"]:
                    qid = data.get(key, None)
                    if qid is not None:
                        break
                qid = qid + f"_{session_id}" if qid is not None else session_id

                # Dump rollout to file
                file_path = os.path.join(dump_path, f"{qid}.txt")
                async with aiofiles.open(file_path, "a") as f:
                    info = "\n".join([f"completion is: {completion}"])
                    await f.write(info + "\n")

        return completions
