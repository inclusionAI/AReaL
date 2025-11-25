import asyncio
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch.distributed as dist
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    load_expr_config,
)
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import MegatronPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

try:  # Package-style relative import (works if executed via -m with package context)
    from .react_agent import MultiTurnReactAgent  # type: ignore
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from react_agent import MultiTurnReactAgent  # type: ignore

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"ASearcher-Reasoning @ {worker_id}")


def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


class TongyiDeepResearchReactWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_llm_calls_per_run: int = 100,
        judge_engine: RemoteSGLangEngine | None = None,
    ):
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.judge_client = ArealOpenAI(engine=judge_engine, tokenizer=tokenizer)
        self.agent = MultiTurnReactAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_llm_calls_per_run=max_llm_calls_per_run,
            max_total_tokens=max_tokens,
            judge_client=self.judge_client,
        )

    async def arun_episode(self, engine, data):
        # Get the unique identifier for this prompt
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = str(qid) or uuid.uuid4().hex
        data["qid"] = qid

        # path to save trajs
        version = engine.get_version()
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            save_traj_path = os.path.join(
                self.dump_dir, str(version), f"{qid}_{{traj_id}}.json"
            )

        clients = [
            ArealOpenAI(
                engine=engine, tokenizer=self.tokenizer, chat_template_type="concat"
            )
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        all_stats = await asyncio.gather(
            *[
                self.agent.make_trajectory(
                    data=data,
                    client=clients[i],
                    save_path=save_traj_path.format(traj_id=i),
                )
                for i in range(self.n_trajs)
            ]
        )
        for stats in all_stats:
            stats_tracker.get(self.rollout_stat_scope).scalar(**stats)

        completions_with_rewards = {}
        for client in clients:
            completion_with_rewards = client.export_completions(style="concat")
            assert len(completion_with_rewards) == 1
            completions_with_rewards.update(completion_with_rewards)
        assert len(all_stats) == self.n_trajs
        assert len(completions_with_rewards) == self.n_trajs
        return completion_with_rewards


@dataclass
class AgentRLConfig(GRPOConfig):
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    max_llm_calls_per_run: int = field(
        default=100,
        metadata={
            "help": "Maximum number of LLM calls per trajectory. By default max_llm_calls_per_run=100."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )
    # Logging Agent Trajectories
    log_agent_stats: bool = field(
        default=False,
        metadata={"help": "Log stats for agent trajectories"},
    )
    log_agent_stats_keys: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys of log stats for agent trajectories"},
    )
    judge_engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)


def get_search_dataset(dataset_path, tokenizer):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return dataset


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    # Initialize train engine
    actor = MegatronPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataloader = create_dataloader(
        get_search_dataset(config.train_dataset.path, tokenizer),
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    # Initialize judge inference engine
    judge_engine = RemoteSGLangEngine(config.judge_engine)
    # NOTE: judge engine should not have off-policyness control.
    judge_engine.config.max_head_offpolicyness = int(1e12)
    judge_engine.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_disk(
        config.experiment_name, config.trial_name, config.cluster.fileroot
    )

    actor.initialize(
        None, ft_spec, parallel_strategy=parallel_strategy, seed=config.seed
    )
    actor.connect_engine(rollout, weight_update_meta)

    # Create rollout workflow
    workflow = TongyiDeepResearchReactWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        n_trajs=config.n_trajs,
        max_tokens=config.max_tokens_per_trajectory,
        max_llm_calls_per_run=config.max_llm_calls_per_run,
        judge_engine=judge_engine,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    # Recover
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
                granularity=config.n_trajs,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("actor update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            judge_engine.set_version(global_step + 1)

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

        # Upload statistics to the logger (e.g., wandb)
        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    rollout.destroy()
    judge_engine.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
