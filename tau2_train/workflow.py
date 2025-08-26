import itertools
import asyncio
import os
import uuid

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

from tau2_train.domains.airline.environment import get_environment, get_tasks
from tau2_train.data_model.tasks import Task
from tau2_train.agent import LLMAgent
from tau2_train.orchestrator import Orchestrator
from tau2_train.user_simulator import UserSimulator
from tau2_train.evaluator.evaluator import evaluate_simulation

import sys
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging

import json
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import random

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)

from dataclasses import dataclass, field

from torchdata.stateful_dataloader import StatefulDataLoader

from loguru import logger

from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker

from areal.api.io_struct import (
    FinetuneSpec,
    WeightUpdateMeta,
    AllocationMode,
    StepInfo
)


worker_id = uuid.uuid4().hex[:4]
logger = logging.getLogger(f"Tau @ {worker_id}")

class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        user_model: str = "/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
        max_num_turns: int = 128,
        n_trajs: int = 1,
        dump_dir: str | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.user_model = user_model
        self.dump_dir = dump_dir
        self.max_num_turns = max_num_turns
        self.n_trajs = n_trajs
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def collect_agent_trajectory(self, engine, task):
        
        environment = get_environment()
        agent = LLMAgent(engine, self.tokenizer, self.gconfig, environment.get_policy(), environment.get_tools())
        user = UserSimulator(
            instructions=task.user_scenario,
            llm=self.user_model
        )

        orchestrator = Orchestrator(
            "airline",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
        )

        simulation = await orchestrator.run()

        reward_info = evaluate_simulation(
            task=task,
            simulation=simulation,
            evaluation_type="all",
        )
        
        messagaes = orchestrator.get_trajectory()
        traj_records = agent.records
        
        # TODO  calculate reward
        reward = reward_info.reward
        ##
        return messagaes, traj_records, reward

    async def arun_episode(self, engine: InferenceEngine, data=None):
        

        if data is None:
            tasks = get_tasks()
            task = random.choice(tasks)
        else:
 
            task = Task.model_validate(data)
                
        trajs = await asyncio.gather(*[self.collect_agent_trajectory(engine, task) for _ in range(self.n_trajs)])
        version = engine.get_version()

        results = []
        for i, (messagaes, traj_records, reward) in enumerate(trajs):
        
            for j, record in enumerate(traj_records):

                seq = record.input_tokens + record.output_tokens
                logprobs = [0.0] * record.input_len + record.output_logprobs
                loss_mask = [0] * record.input_len + [1] * record.output_len
                versions = [-1] * record.input_len + record.output_versions

                res = dict(
                    # unsqueeze to add an additional batch dimension
                    input_ids=torch.tensor(seq).unsqueeze(0),
                    loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                    logprobs=torch.tensor(logprobs).unsqueeze(0),
                    versions=torch.tensor(versions).unsqueeze(0),
                    attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                    # reward
                    rewards=torch.tensor([float(reward)]),
                )
                if len(loss_mask) <= 32768:
                    results.append(TensorDict(res, batch_size=[1]))
        
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{data['id']}_{uuid.uuid4().hex}.jsonl"), "w"
            ) as f:
                for i, (messages, _, score) in enumerate(trajs):
                    f.write(json.dumps(dict(messages=messages, reward=score, traj_idx=i)) + "\n")

        return concat_padded_tensors(results)

@dataclass
class TauRLConfig(GRPOConfig):
    max_turns: int = field(
        default=10,
        metadata={
            "help": "maximum number of turns for search agent"
        }
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        }
    )

def get_tau2_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, TauRLConfig)
    config: TauRLConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_tau2_dataset(config.train_dataset.path, tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
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
    rollout.initialize(None, ft_spec)

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    print("-" * 100, os.environ["LOCAL_RANK"], flush=True)
    actor.initialize(None, ft_spec)
    ref = None

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_nccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = Tau2Workflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_num_turns=config.max_turns,
        n_trajs=config.n_trajs,
    )

   # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
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

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        from areal.utils.redistributor import redistribute
        batch = redistribute(batch).data

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

        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        # with stats_tracker.record_timing("eval"):

        #     def evaluate_fn():
        #         rollout.pause()
        #         cnt = 0
        #         for data in valid_dataloader:
        #             for item in data:
        #                 eval_rollout.submit(item, workflow)
        #                 cnt += 1
        #         batch = eval_rollout.wait(cnt, timeout=None)
        #         rewards = batch["rewards"].float().to(actor.device)
        #         with stats_tracker.scope("grpo-eval"):
        #             stats_tracker.denominator(
        #                 n_seqs=torch.ones(
        #                     rewards.shape[0],
        #                     device=rewards.device,
        #                     dtype=torch.bool,
        #                 )
        #             )
        #             stats_tracker.stat(task_reward=rewards, denominator="n_seqs")
        #         rollout.resume()

        #     evaluator.evaluate(
        #         evaluate_fn,
        #         epoch,
        #         step,
        #         global_step,
        #     )

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

        stats_logger.commit(epoch, step, global_step, stats)

    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
