import asyncio
import os
import sys
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import aiofiles
import numpy as np
from datasets import Dataset
from loguru import logger as loguru_logger
from tau2.registry import registry
from tau2_utils import Tau2EnvConfig, Tau2RunInfo

from areal.api.cli_args import (
    GenerationHyperparameters,
    PPOConfig,
    load_expr_config,
)
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.proxy import (
    ProxyServer,
    ProxySession,
    ensure_end_with_slash,
)
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Tau2 Example")


# ================================ dataset ================================
def get_tau2_dataset(
    domain: str,
    type: str = "rl",
    split: str = "train",
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: The tau2 domain name, e.g., 'retail', 'airline', 'telecom'
        split: Dataset split (e.g., 'train', 'test')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries
    """
    assert type == "rl", "Only RL dataset is supported for now"
    # TODO: support SFT dataset

    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found in {splits}, available splits: {splits.keys()}"
        )
    task_ids = splits[split]
    # print(f"domain: {domain}, split: {split}, task_ids: {task_ids}")

    dataset_items = [{"task_id": task_id, "split": split} for task_id in task_ids]
    dataset = Dataset.from_list(dataset_items)
    return dataset


def run_fn(func: Callable, extra_envs: dict, *args, **kwargs) -> Any:
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


class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        proxy_server: ProxyServer,
        gconfig: GenerationHyperparameters,
        econfig: Tau2EnvConfig,
        base_url: str,
        max_concurrent_processes: int,
        agent_module_path: str,
        agent_run_args: dict | None = None,
        rollout_stat_scope: str = "rollout",
        export_style: str = "individual",
        max_tokens_per_mb: int = 32768,
        dump_dir: str | None = None,
    ):
        self.proxy_server = proxy_server
        self.group_size = gconfig.n_samples
        self.gconfig = gconfig.new(n_samples=1)
        self.econfig = econfig
        self.base_url = ensure_end_with_slash(base_url)
        self.process_pool = ProcessPoolExecutor(max_workers=max_concurrent_processes)
        self.agent_func = import_from_string(
            ".".join([agent_module_path, "run_and_submit"])
        )
        self.agent_run_args = agent_run_args or {}
        self.rollout_stat_scope = rollout_stat_scope
        self.export_style = export_style
        self.max_tokens_per_mb = max_tokens_per_mb
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_episode(self, task_id: str, data: dict) -> Any:
        process_data = {
            "gconfig": asdict(self.gconfig),
            "econfig": asdict(self.econfig),
            "agent_run_args": self.agent_run_args,
            **data,
        }

        async with ProxySession(base_url=self.base_url, task_id=task_id) as session:
            extra_envs = {
                "OPENAI_BASE_URL": session.session_url,
                "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "AREAL_SESSION_ID": session.session_id,
                "AREAL_TASK_ID": task_id,
            }
            return await asyncio.wrap_future(
                self.process_pool.submit(
                    run_fn,
                    func=self.agent_func,
                    extra_envs=extra_envs,
                    data=process_data,
                )
            )

    async def arun_episode(self, engine: InferenceEngine, data):
        task_id = uuid.uuid4().hex  # use a unique task id for each run
        run_infos: list[Tau2RunInfo] = await asyncio.gather(
            *[self._run_episode(task_id, data) for _ in range(self.group_size)]
        )
        # the queue is prepared for separated agent and trainer mode, should not be used in this example
        await ProxyServer.finish_task(
            task_id, base_url=self.base_url, put_to_queue=False
        )

        session_ids = [f"{task_id}-{i}" for i in range(self.group_size)]
        rewards, completions = await self.proxy_server.get_results(
            session_ids, style=self.export_style
        )

        # log stats
        for reward in rewards.values():
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        for info in run_infos:
            stats_tracker.get(self.rollout_stat_scope).scalar(
                steps_count=len(info.messages),
                orchestrator_error=int(info.error is not None),
            )

            def add_to_stats(name: str, times: list[float]):
                if len(times):
                    for key in ["mean", "max", "min", "std", "sum"]:
                        stats_tracker.get(self.rollout_stat_scope).scalar(
                            **{f"{name}_time/{key}": getattr(np.array(times), key)()}
                        )

            add_to_stats("agent", info.agent_time)
            add_to_stats("user", info.user_time)

        # Dump info to file
        version = engine.get_version()
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

        if "task_id" in data:
            real_task_id = data["task_id"][:120] + "-" + task_id
        else:
            real_task_id = task_id

        assert len(run_infos) == len(rewards), (
            len(run_infos),
            len(rewards),
            self.group_size,
        )

        for i, info in enumerate(run_infos):
            try:
                json_path = os.path.join(
                    self.dump_dir, str(version), f"{real_task_id}-{i}.json"
                )
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(info.model_dump_json())

                file_path = os.path.join(
                    self.dump_dir, str(version), f"{real_task_id}-{i}.txt"
                )
                async with aiofiles.open(file_path, "a") as f:
                    await f.write(str(info))
            except Exception as e:
                logger.error(f"Error dumping rollout to file: {e}")

        return completions


@dataclass
class Tau2PPOConfig(PPOConfig):
    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser that used by ArealOpenAI client."},
    )
    export_style: str = field(
        default="individual",
        metadata={
            "help": "Style for exporting completion results from the proxy server."
        },
    )
    agent_module_path: str | None = field(
        default="examples.tau2.tau2_agent",
        metadata={"help": "Module path for the agent definition."},
    )
    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the agent."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation."},
    )


def main(args):
    config, _ = load_expr_config(args, Tau2PPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    domain = config.econfig.domain

    # remove the logging of loguru logger in tau2-bench package.
    loguru_logger.remove()
    loguru_logger.add(
        os.path.join(StatsLogger.get_log_path(config.stats_logger), "tau2.log"),
        level="INFO",
    )

    # Create dataset and dataloaders
    train_dataset = get_tau2_dataset(
        domain=domain,
        type=config.train_dataset.type,
        split=config.train_dataset.path.split("/")[-1],
    )
    valid_dataset = get_tau2_dataset(
        domain=domain,
        type=config.valid_dataset.type,
        split=config.valid_dataset.path.split("/")[-1],
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        world_size = trainer.actor.data_parallel_world_size
        chat_template_type = "hf" if config.export_style == "individual" else "concat"
        cpu_count = os.cpu_count() or 64
        num_processes = config.rollout.max_concurrent_rollouts or cpu_count

        def _get_server_and_workflow(rollout: InferenceEngine, is_eval: bool = False):
            name = "train" if not is_eval else "eval"
            temperature = 0.6 if is_eval else 1.0
            rollout_stat_scope = "rollout" if not is_eval else "eval-rollout"
            dump_dir = "generated" if not is_eval else "generated-eval"

            server = ProxyServer(
                rollout=rollout,
                tokenizer=tokenizer,
                tool_call_parser=config.tool_call_parser,
                chat_template_type=chat_template_type,
                name=f"{name} proxy server",
                max_total_tokens=config.gconfig.max_tokens,
            )
            server.start(wait_until_ready=True)
            workflow = Tau2Workflow(
                proxy_server=server,
                gconfig=config.gconfig.new(temperature=temperature),
                econfig=config.econfig,
                base_url=f"{server.public_addr}/v1",
                max_concurrent_processes=num_processes // world_size,
                agent_module_path=config.agent_module_path,
                agent_run_args=config.agent_run_args,
                rollout_stat_scope=rollout_stat_scope,
                export_style=config.export_style,
                max_tokens_per_mb=config.actor.mb_spec.max_tokens_per_mb,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger),
                    dump_dir,
                ),
            )
            return server, workflow

        proxy_server, workflow = _get_server_and_workflow(
            trainer.rollout, is_eval=False
        )

        if config.do_eval:
            eval_proxy_server, eval_workflow = _get_server_and_workflow(
                trainer.eval_rollout, is_eval=True
            )
        else:
            eval_proxy_server, eval_workflow = None, None

        trainer.train(workflow, eval_workflow, granularity=1)
        proxy_server.close()
        if eval_proxy_server is not None:
            eval_proxy_server.close()


if __name__ == "__main__":
    main(sys.argv[1:])
