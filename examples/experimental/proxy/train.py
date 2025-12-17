import asyncio
import os
import sys
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

from areal.api.cli_args import GenerationHyperparameters, PPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
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

logger = logging.getLogger("GSM8K GRPO Proxy Example")


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


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


class ProxyWorkflow(RolloutWorkflow):
    def __init__(
        self,
        proxy_server: ProxyServer,
        gconfig: GenerationHyperparameters,
        base_url: str,
        max_concurrent_processes: int,
        agent_module_path: str,
        agent_run_args: dict | None = None,
        rollout_stat_scope: str = "rollout",
        export_style: str = "concat",
        dump_dir: str | None = None,
    ):
        self.proxy_server = proxy_server
        self.group_size = gconfig.n_samples
        self.gconfig = gconfig.new(n_samples=1)
        self.base_url = ensure_end_with_slash(base_url)
        self.process_pool = ProcessPoolExecutor(max_workers=max_concurrent_processes)
        self.agent_func = import_from_string(
            ".".join([agent_module_path, "run_and_submit"])
        )
        self.agent_run_args = agent_run_args or {}
        self.rollout_stat_scope = rollout_stat_scope
        self.export_style = export_style
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_episode(self, task_id: str, data: dict):
        process_data = {
            "gconfig": asdict(self.gconfig),
            "agent_run_args": self.agent_run_args,
            "messages": data["messages"],
            "answer": data["answer"],
        }
        async with ProxySession(base_url=self.base_url, task_id=task_id) as session:
            extra_envs = {
                "OPENAI_BASE_URL": session.session_url,
                "OPENAI_API_KEY": "dummy",  # os.environ["OPENAI_API_KEY"],
                "AREAL_SESSION_ID": session.session_id,
                "AREAL_TASK_ID": task_id,
            }
            await asyncio.wrap_future(
                self.process_pool.submit(
                    run_fn,
                    func=self.agent_func,
                    extra_envs=extra_envs,
                    data=process_data,
                )
            )

    async def arun_episode(self, engine: InferenceEngine, data):
        task_id = uuid.uuid4().hex
        await asyncio.gather(
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
        for reward in rewards.values():
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        return completions


@dataclass
class ProxyPPOConfig(PPOConfig):
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser that used by ProxyServer."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Style for exporting completion results from the proxy server.",
            "choices": ["individual", "concat"],
        },
    )
    agent_module_path: str | None = field(
        default="examples.experimental.proxy.gsm8k_agent",
        metadata={"help": "Module path for the agent definition."},
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

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
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
                gconfig=gconfig,
                base_url=f"{server.public_addr}/v1",
                max_concurrent_processes=num_processes // world_size,
                agent_module_path=config.agent_module_path,
                agent_run_args=config.agent_run_args,
                rollout_stat_scope=rollout_stat_scope,
                export_style=config.export_style,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger),
                    dump_dir,
                ),
            )
            return server, workflow

        proxy_server, workflow = _get_server_and_workflow(
            trainer.rollout, is_eval=False
        )
        eval_proxy_server, eval_workflow = _get_server_and_workflow(
            trainer.eval_rollout, is_eval=True
        )

        trainer.train(workflow, eval_workflow)
        proxy_server.close()
        eval_proxy_server.close()


if __name__ == "__main__":
    main(sys.argv[1:])
