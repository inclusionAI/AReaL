import asyncio
import json
import os
import shlex
import sys
import threading
import traceback
import uuid
from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GenerationHyperparameters, PPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import StepInfo
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.experimental.openai.proxy import (
    ProxyServer,
    ProxySession,
    ensure_end_with_slash,
)
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils import logging, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.dataloader import create_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("GSM8K GRPO Proxy Example")


class ProxyWorkflow(RolloutWorkflow):
    def __init__(
        self,
        proxy_server: ProxyServer,
        rollout_stat_scope: str = "rollout",
        export_style: str = "concat",
        dump_dir: str | None = None,
    ):
        self.proxy_server = proxy_server
        self.rollout_stat_scope = rollout_stat_scope
        self.export_style = export_style
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine: InferenceEngine, data):
        # task_index = data["_index"]
        # task_uuid = data["_uuid"]
        # logger.info("waiting for session ids")
        session_ids = await self.proxy_server.get_session_ids()
        rewards, completions = await self.proxy_server.get_results(
            session_ids, style=self.export_style
        )
        for reward in rewards.values():
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        return completions


def inject_index(
    dataset: Dataset, key: str = "_index", include_uuid: bool = False
) -> Dataset:
    def inject_fn(x: dict, idx: int) -> dict:
        result = {**x, key: idx}
        if include_uuid:
            result["_uuid"] = uuid.uuid4().hex
        return result

    return dataset.map(inject_fn, with_indices=True)


class AgentWorkflow:
    @abstractmethod
    async def arun_episode(self, task_id: str, data: dict):
        raise NotImplementedError()


def run_fn(func: Callable, extra_envs: dict, *args, **kwargs) -> float:
    for key, value in extra_envs.items():
        os.environ[key] = value
    try:
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(func(*args, **kwargs))
        else:
            return func(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        raise e


class GSM8kAgentWorkflow(AgentWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        base_url: str,
        max_concurrent_processes: int,
        process_mode: str = "process_pool",
        agent_module_path: str | None = None,
        run_agent_cmd: str | None = None,
        agent_run_args: dict | None = None,
    ):
        self.group_size = gconfig.n_samples
        gconfig.n_samples = 1
        self.gconfig = gconfig
        self.base_url = ensure_end_with_slash(base_url)
        # self.process_semaphore = asyncio.Semaphore(max_concurrent_processes)

        if process_mode not in ["process_pool", "subprocess"]:
            raise ValueError(f"Invalid process mode: {process_mode}")
        self.process_mode = process_mode
        if self.process_mode == "process_pool":
            self.process_pool = ProcessPoolExecutor(
                max_workers=max_concurrent_processes
            )
            if agent_module_path is None:
                raise ValueError(
                    "agent_module_path is required when process_mode is process_pool"
                )
            self.agent_func = import_from_string(
                ".".join([agent_module_path, "run_and_submit"])
            )
        else:
            if run_agent_cmd is None:
                if agent_module_path is None:
                    raise ValueError(
                        "run_agent_cmd or agent_module_path is required when process_mode is subprocess"
                    )
                run_agent_cmd = f"{sys.executable} -m {agent_module_path}"
            self.run_agent_cmd = run_agent_cmd
        self.agent_run_args = agent_run_args or {}

    async def _run_episode(self, task_id: str, data: dict):
        # logger.info(f"task_id: {task_id}")
        process_data = {
            "gconfig": asdict(self.gconfig),
            "agent_run_args": self.agent_run_args,
            "task_id": task_id,
            "messages": data["messages"],
            "answer": data["answer"],
        }

        # async with self.process_semaphore:
        async with ProxySession(base_url=self.base_url, task_id=task_id) as session:
            extra_envs = {
                "OPENAI_BASE_URL": session.session_url,
                "OPENAI_API_KEY": "dummy",  # os.environ["OPENAI_API_KEY"],
                "AREAL_SESSION_ID": session.session_id,
                "AREAL_TASK_ID": task_id,
            }

            # logger.info(f"Running task {task_id} with semaphore")
            if self.process_mode == "process_pool":
                await asyncio.wrap_future(
                    self.process_pool.submit(
                        run_fn,
                        func=self.agent_func,
                        extra_envs=extra_envs,
                        data=process_data,
                    )
                )
            else:
                # NOTE: running Python subprocess can be slow, use process pool instead
                # Only use subprocess for agents implemented in other languages, like Typescript or Go
                logger.info(f"Running command: {self.run_agent_cmd}")
                process = await asyncio.create_subprocess_exec(
                    *shlex.split(self.run_agent_cmd),
                    env={**os.environ.copy(), **extra_envs},
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                json_str = json.dumps(process_data, ensure_ascii=False)
                stdout, stderr = await process.communicate(
                    input=json_str.encode("utf-8")
                )
                if process.returncode != 0:
                    raise RuntimeError(f"Subprocess Error:\n{stderr.decode()}")
                # logger.info(f"Subprocess Output:\n{stdout.decode()}")

    async def arun_episode(self, task_id: str, data: dict):
        await asyncio.gather(
            *[self._run_episode(task_id, data) for _ in range(self.group_size)]
        )
        await ProxyServer.finish_task(task_id, base_url=self.base_url)


async def prepare_batch(
    workflow: AgentWorkflow,
    data: list[dict],
    step_info: StepInfo,
) -> list[str]:
    async def run_task(item: dict):
        # task_id should be unique for each run
        task_id = (
            f"epoch-{step_info.epoch}-step-{step_info.epoch_step}-task-{item['_index']}"
        )
        # task_id = f"uuid.uuid4().hex"  # alternatively, use uuid
        await workflow.arun_episode(task_id, item)
        return task_id

    task_ids = await asyncio.gather(*[run_task(item) for item in data])
    return task_ids


@dataclass
class ProxyPPOConfig(PPOConfig):
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser that used by ProxyServer."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Export style for the proxy server.",
            "choices": ["individual", "concat"],
        },
    )
    process_mode: str = field(
        default="process_pool",
        metadata={
            "help": "Process mode for running the agent. process_pool: run the agent in a process pool, subprocess: run the agent in a subprocess.",
            "choices": ["process_pool", "subprocess"],
        },
    )
    agent_module_path: str | None = field(
        default="examples.experimental.proxy.gsm8k_agent",
        metadata={"help": "Module path for the agent definition."},
    )
    run_agent_cmd: str | None = field(
        default=None,
        metadata={
            "help": "Command to run the agent. If not provided, it will be constructed from agent_module_path."
        },
    )
    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the agent."},
    )


def main(args):
    config, _ = load_expr_config(args, ProxyPPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    train_dataset = inject_index(train_dataset, include_uuid=True)

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = inject_index(valid_dataset, include_uuid=True)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        rank = trainer.actor.data_parallel_rank
        world_size = trainer.actor.data_parallel_world_size
        chat_template_type = "hf" if config.export_style == "individual" else "concat"
        cpu_count = os.cpu_count() or 64
        num_processes = config.rollout.max_concurrent_rollouts or cpu_count
        if config.process_mode == "subprocess":
            # divide by 4 to avoid overloading the CPU, adjust based on your needs
            num_processes = min(cpu_count, num_processes) // 4

        # Agent loop for training and eval
        train_finished = False

        async def run_agent(
            agent_train_dataloader: StatefulDataLoader,
            agent_workflow: AgentWorkflow,
            name: str = "train",
        ):
            data_generator = cycle_dataloader(agent_train_dataloader)
            global_step = 0
            steps_per_epoch = len(agent_train_dataloader)

            while not train_finished:
                epoch = global_step // steps_per_epoch
                step = global_step % steps_per_epoch
                step_info = StepInfo(
                    global_step=global_step,
                    epoch=epoch,
                    epoch_step=step,
                    steps_per_epoch=steps_per_epoch,
                )
                batch_data = next(data_generator)
                await prepare_batch(agent_workflow, batch_data, step_info)
                global_step += 1
                logger.info(
                    f"rank {rank}, {global_step} rollout steps completed for {name} agent"
                )

        def _get_server_workflow_and_agent_thread(
            rollout: InferenceEngine, dataset: Dataset, is_eval: bool = False
        ):
            # Prepare proxy server, workflow and agent thread
            name = "train" if not is_eval else "eval"
            gconfig = (
                config.gconfig.new(temperature=0.6, n_samples=1)
                if is_eval
                else config.gconfig
            )
            rollout_stat_scope = "rollout" if not is_eval else "eval-rollout"
            dump_dir = "generated" if not is_eval else "generated-eval"

            server = ProxyServer(
                rollout=rollout,
                tokenizer=tokenizer,
                tool_call_parser=config.tool_call_parser,
                chat_template_type=chat_template_type,
                name=f"{name} proxy server",
            )
            server.start(wait_until_ready=True)
            workflow = ProxyWorkflow(
                proxy_server=server,
                rollout_stat_scope=rollout_stat_scope,
                export_style=config.export_style,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger), dump_dir
                ),
            )
            agent_dataloader = create_dataloader(
                dataset,
                rank=rank,
                world_size=world_size,
                dataset_config=config.train_dataset,
            )
            agent_workflow = GSM8kAgentWorkflow(
                gconfig=gconfig,
                base_url=f"{server.public_addr}/v1",
                max_concurrent_processes=num_processes // world_size,
                agent_module_path=config.agent_module_path,
                run_agent_cmd=config.run_agent_cmd,
                agent_run_args=config.agent_run_args,
                process_mode=config.process_mode,
            )
            agent_thread = threading.Thread(
                target=asyncio.run,
                args=(run_agent(agent_dataloader, agent_workflow, name=name),),
            )
            agent_thread.start()
            return server, workflow, agent_thread

        proxy_server, workflow, agent_thread = _get_server_workflow_and_agent_thread(
            trainer.rollout, train_dataset, is_eval=False
        )
        eval_proxy_server, eval_workflow, eval_agent_thread = (
            _get_server_workflow_and_agent_thread(
                trainer.eval_rollout, valid_dataset, is_eval=True
            )
        )
        # Start training and eval loop
        trainer.train(workflow, eval_workflow)

        # Stop agent threads and close proxy servers
        train_finished = True
        agent_thread.join(timeout=10)
        eval_agent_thread.join(timeout=10)
        proxy_server.close()
        eval_proxy_server.close()


if __name__ == "__main__":
    main(sys.argv[1:])
