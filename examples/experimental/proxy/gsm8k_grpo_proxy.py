"""
NOTE: this example is under development and in experimental stage, the interface are subject to change.
"""

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import aiofiles
import aiofiles.os
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy import (
    ProxyServer,
    ProxySession,
)
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.dynamic_import import import_from_string
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("GSM8K GRPO Proxy Example")


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


# pickle used by ProcessPoolExecutor can not serialize a local function, so we need a global function
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


class ProxyRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        proxy_server: ProxyServer,
        run_agent_return_reward: Callable[[Any], Awaitable[float]],
        process_pool_executor: ProcessPoolExecutor = None,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        export_style: str = "individual",
    ):
        self.proxy_server = proxy_server
        self.api_version = "v1"
        self.n_samples = gconfig.n_samples
        self.rollout_stat_scope = rollout_stat_scope
        self.process_pool_executor = process_pool_executor
        self.gconfig = gconfig
        self.run_agent_return_reward = run_agent_return_reward
        self.dump_dir = dump_dir
        self.export_style = export_style

    async def arun_episode(self, engine: InferenceEngine, data):
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


@dataclass
class ProxyAgentConfig(GRPOConfig):
    tool_call_parser: str = field(
        default="qwen25",
    )

    agent_process_pool_size: int = field(
        default=256,
        metadata={"help": "Number of parallel processes for running agents."},
    )

    agent_module_path: str = field(
        default="examples.any_agents.agent.math.math_agent",
        metadata={"help": "Module path for the agent definition."},
    )

    export_style: str = field(
        default="concat",
        metadata={"help": "Export style for the proxy server."},
    )


def main(args):
    config, _ = load_expr_config(args, ProxyAgentConfig)

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    client = ArealOpenAI(
        engine=rollout,
        tokenizer=tokenizer,
        tool_call_parser=config.tool_call_parser,
        chat_template_type="concat" if config.export_style == "concat" else "hf",
    )

    free_port = find_free_ports(1)[0]
    proxy_server = ProxyServer(port=free_port, client=client)
    proxy_server.start(wait_until_ready=True)

    all_addresses = [None for _ in range(actor.data_parallel_world_size)]
    dist.all_gather_object(
        all_addresses, proxy_server.public_addr, group=actor.data_parallel_group
    )
    logger.info(f"Found {len(all_addresses)} proxy servers: {all_addresses}")
    dist.barrier(group=actor.cpu_group)

    process_pool_executor = ProcessPoolExecutor(
        max_workers=config.agent_process_pool_size
    )

    run_agent_return_reward = import_from_string(
        ".".join([config.agent_module_path, "run_agent_return_reward"]),
    )

    workflow = ProxyRLVRWorkflow(
        gconfig=config.gconfig,
        rollout_stat_scope="rollout",
        proxy_server=proxy_server,
        run_agent_return_reward=run_agent_return_reward,
        process_pool_executor=process_pool_executor,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        export_style=config.export_style,
    )
    eval_workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=1,  # for multi-turn rollouts, granularity must be 1
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Upload statistics to the logger (e.g., wandb)
        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    proxy_server.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
