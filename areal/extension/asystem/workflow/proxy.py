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
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging, stats_tracker
from areal.utils.data import (
    concat_padded_tensors,
)
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports

logger = logging.getLogger(__name__)


def sync_run_task(
    data,
    proxy_addr,
    run_agent_return_reward: Callable[[Any], Awaitable[float | tuple[float, dict]]],
):
    async def run_task(data, proxy_addr, run_agent_return_reward: Callable):
        async with ProxySession(base_url=proxy_addr) as session:
            session_id = session.session_id
            try:
                res = await run_agent_return_reward(data)
                if isinstance(res, tuple):
                    assert len(res) == 2
                    reward, stats = res
                    assert isinstance(stats, dict)
                else:
                    reward = res
                    stats = {}
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Error in sync_run_task: {e}")
                raise e

            await session.set_reward(reward)

        return None, session_id, reward, stats

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
        engine_max_tokens: int | None = None,
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        chat_template_type: str = "hf",
    ):
        self.client = ArealOpenAI(
            engine=engine,
            tokenizer=tokenizer,
            engine_max_tokens=engine_max_tokens,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
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
        reasoning_parser: str = "qwen3",
        chat_template_type: str = "concat",
        # proxy_server: ProxyServer,
        # run_agent_return_reward: Callable[[Any], Awaitable[float]],
        run_agent_return_reward_path: str | dict[str, str] = None,
        process_pool_executor_size: int = 128,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        export_style: str = "individual",
    ):
        # self.proxy_server = proxy_server
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.chat_template_type = chat_template_type
        self.api_version = "v1"
        self.n_samples = gconfig.n_samples
        self.rollout_stat_scope = rollout_stat_scope
        self.process_pool_executor = SingletonProcessPoolExecutor(
            max_workers=process_pool_executor_size
        )
        self.gconfig = gconfig
        assert run_agent_return_reward_path is not None, (
            "run_agent_return_reward_path must not be None"
        )
        if isinstance(run_agent_return_reward_path, dict):
            self.run_agent_return_reward = {}
            for task_type, path in run_agent_return_reward_path.items():
                self.run_agent_return_reward[task_type] = import_from_string(
                    ".".join([path, "run_agent_return_reward"]),
                )
        elif isinstance(run_agent_return_reward_path, str):
            self.run_agent_return_reward = import_from_string(
                ".".join([run_agent_return_reward_path, "run_agent_return_reward"]),
            )
        else:
            raise ValueError(
                f"run_agent_return_reward_path must be a dict or str, but got {type(run_agent_return_reward_path)}"
            )
        self.dump_dir = dump_dir
        self.export_style = export_style

    async def arun_episode(self, engine: InferenceEngine, data):
        if isinstance(self.tokenizer, str):
            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        self.proxy_server_warpper = ProxyServerSingletonWarpper(
            engine=engine,
            tokenizer=self.tokenizer,
            engine_max_tokens=self.gconfig.max_tokens,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            chat_template_type=self.chat_template_type,
        )
        self.proxy_server = self.proxy_server_warpper.get_proxy_server()

        if isinstance(self.run_agent_return_reward, dict):
            task_type = data.get("task", "default")
            run_agent_return_reward = self.run_agent_return_reward.get(
                task_type,
                self.run_agent_return_reward.get("default", None),
            )
            if run_agent_return_reward is None:
                raise ValueError(
                    f"No run_agent_return_reward found for task_type {task_type}"
                )
        else:
            run_agent_return_reward = self.run_agent_return_reward

        futures = [
            self.process_pool_executor.submit(
                sync_run_task,
                data,
                f"{self.proxy_server.public_addr}/{self.api_version}",
                run_agent_return_reward,
            )
            for _ in range(self.n_samples)
        ]
        results = await asyncio.gather(
            *[asyncio.wrap_future(future) for future in futures]
        )
        error_message, session_ids, rewards, all_stats = zip(*results)
        if any(error_message):
            for msg in error_message:
                if msg is not None:
                    logger.error(f"Error in run_agent: {msg}")
            raise RuntimeError("One or more tasks failed in run_agent.")

        for reward, stats in zip(rewards, all_stats):
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward, **stats)

        completions: dict[str, list[InteractionWithTokenLogpReward]] = {}

        for session_id in session_ids:
            completion_dict = await self.proxy_server.get_completions(
                session_ids=[session_id], style=self.export_style, discount=0.9
            )
            completions[session_id] = list(completion_dict.values())

            print(
                f"session_id: {session_id}, completions num: {len(completions[session_id])}"
            )

        if self.dump_dir is not None:
            for session_id, completion_list in completions.items():
                if len(completion_list) == 0:
                    continue
                version = completion_list[0].model_response.output_version

                dump_path = os.path.join(
                    os.environ.get("LOG_DIR", ""), self.dump_dir, str(version)
                )
                await aiofiles.os.makedirs(dump_path, exist_ok=True)
                # Get the unique identifier for this prompt
                qid = None
                for key in ["query_id", "id", "qid", "task_id"]:
                    qid = data.get(key, None)
                    if qid is not None:
                        break
                qid = str(qid)
                qid = qid + f"_{session_id}" if qid is not None else session_id

                info = f"\n=== Completion Session ID: {session_id} ===\n"
                for i, completion in enumerate(completion_list):
                    info += f"Completion {i + 1}\n"
                    info += "=======Input Messages=======\n"
                    for message in completion.messages:
                        role = message.get("role", "unknown")
                        content = message.get("content", "")
                        info += f"role[{role}]: {content}\n"
                        if "tool_calls" in message:
                            info += f"\t[tool_calls]: {message['tool_calls']}\n"

                    if completion.is_completion:
                        info += f"=======Completion=======\n{completion.completion}\n"
                    else:
                        info += f"=======Response=======\n{completion.response}\n"
                    info += f"=======Reward=======\n{completion.reward}\n"
                    info += f"=======Input Tokens=======\n{completion.model_response.input_tokens}\n"
                    info += f"=======Output Tokens=======\n{completion.model_response.output_tokens}\n"
                    info += "=========================\n\n"

                # Dump rollout to file
                file_path = os.path.join(dump_path, f"{qid}.txt")
                async with aiofiles.open(file_path, "a") as f:
                    await f.write(info + "\n")

        trajs = []
        for session_id, completion_list in completions.items():
            assert all(
                isinstance(v, InteractionWithTokenLogpReward) for v in completion_list
            ), completion_list

            trajs.extend([v.to_tensor_dict() for v in completion_list])

            if len(trajs) > 0:
                print(f"[wht debug], trajs[0] keys: {trajs[0].keys()}, trajs[0] is {trajs[0]}")
        results = concat_padded_tensors(trajs)
        return results
