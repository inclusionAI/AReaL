import asyncio
import hashlib
import itertools
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import List

import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    StepInfo,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.data import broadcast_tensor_container
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.redistributor import redistribute
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

# sys.path.append("/storage/openpsi/users/xushusheng.xss/projects/ASearcher-Lite@0908")
# sys.path.append(str(Path(__file__).resolve().parents[2]))
from .react_agent import MultiTurnReactAgent

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
        dump_dir: str | None = None,
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_llm_calls_per_run: int = 100,
        judge_engine: RemoteSGLangEngine | None = None,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
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

        # check for generated qid when resuming
        if self.dump_dir is not None:
            import glob

            _pattern = os.path.join(self.dump_dir, "*", f"{qid}.jsonl")
            if len(glob.glob(_pattern)) > 0:
                logger.info(f"{qid} is already trained on")
                return None

        # path to save trajs
        version = engine.get_version()
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            os.path.join(self.dump_dir, str(version), f"{qid}/ID.json")

        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        judge_client = self.judge_client

        # Collect trajectories
        await asyncio.gather(
            *[
                self.agent.make_trajectory(
                    data=data,
                    client=client,
                    judge_client=judge_client,
                )
                for i in range(self.n_trajs)
            ]
        )

        result = client.export_completions(turn_discount=0.0, return_leaf_only=True)
        assert (
            len(result) == self.n_trajs
        ), f"Expected {self.n_trajs} trajectories, but got {len(result)}"
        return result


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
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys of log stats for agent trajectories"},
    )
    judge_engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)


def get_search_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_search_dataset(
            config.train_dataset.path,
            tokenizer,
            actor.data_parallel_rank,
            actor.data_parallel_world_size,
        ),
        batch_size=config.train_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
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
    judge_engine.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    actor.initialize(None, ft_spec)
    ref = None

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.

    weight_update_meta = WeightUpdateMeta.from_disk(
        config.experiment_name, config.trial_name, config.cluster.fileroot
    )

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
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
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
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

    data_generator = itertools.cycle(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        print(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

        with stats_tracker.record_timing("rollout"):
            batch = None
            if actor.is_data_parallel_head():
                if config.async_training:
                    batch = rollout.prepare_batch(
                        train_dataloader,
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                else:
                    batch = rollout.rollout_batch(
                        next(data_generator),
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                batch = batch.to(actor.device)
                batch = redistribute(batch, group=actor.data_parallel_group).data
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

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
            if config.log_agent_stats:
                agent_denominator = (batch["begin_of_trajectory"] > 0).bool()
                stats_tracker.denominator(agent=agent_denominator)
                stats_tracker.stat(
                    **{k: batch[k].float() for k in config.log_agent_stats_keys},
                    denominator="agent",
                )

            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("actor update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            current_platform.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

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

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
