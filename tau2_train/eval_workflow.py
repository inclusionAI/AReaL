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
import numpy as np

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors, broadcast_tensor_container
from realhf.base import logging
from areal.utils.redistributor import redistribute

import json
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import random
import copy

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)

from dataclasses import dataclass, field, asdict

from torchdata.stateful_dataloader import StatefulDataLoader

from loguru import logger
logger.remove()

from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
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

import yaml


worker_id = uuid.uuid4().hex[:4]
logger = logging.getLogger(f"Tau @ {worker_id}")

class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        user_model: str = "/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
        user_api_key: str = "empty",
        user_base_url: str = "",
        max_num_turns: int = 128,
        max_context_length: int = 32768,
        n_trajs: int = 1,
        reward_type: str = "all",
        dump_dir: str | None = None,
        eval_model: str | None = None,
        eval_api_key: str | None = None,
        eval_base_url: str | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_num_turns = max_num_turns
        self.n_trajs = n_trajs
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        self.max_context_length = max_context_length

        self.user_model = user_model
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url

        self.eval_model = eval_model or user_model
        self.eval_api_key = eval_api_key or user_api_key
        self.eval_base_url = eval_base_url or user_base_url

        self.reward_type = reward_type

    async def collect_agent_trajectory(self, engine, task):
        
        environment = get_environment()
        agent = LLMAgent(
            engine, 
            self.tokenizer, 
            self.gconfig, 
            environment.get_policy(), 
            environment.get_tools(),
            max_context_length=self.max_context_length - 100
        )
        user = UserSimulator(
            instructions=task.user_scenario,
            llm=self.user_model,
            api_key=self.user_api_key,
            base_url=self.user_base_url,
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
            llm=self.eval_model,
            api_key=self.eval_api_key,
            base_url=self.eval_base_url
        )
        
        messagaes = orchestrator.get_trajectory()
        traj_records = agent.records
        
        # TODO  calculate reward

        if self.reward_type == "db":
            reward = reward_info.db_check.db_reward if reward_info.db_check is not None else 0
        elif self.reward_type == "all":
            try:
                reward = reward_info.db_check.db_reward
                if len(reward_info.action_checks) > 0:
                    reward *= np.mean([x.action_reward for x in reward_info.action_checks])
                if len(reward_info.nl_assertions) > 0:
                    reward *= np.mean([x.met for x in reward_info.nl_assertions])

                print(
                "[debug] reward info: ", task.id,
                reward_info.db_check.db_reward,
                [x.action_reward for x in reward_info.action_checks],
                [x.met for x in reward_info.nl_assertions]
            )

            except Exception as e:
                print("[debug] reward info: ", e, reward_info)
                reward = 0
        else:
            raise NotImplementedError

        return messagaes, traj_records, reward

    async def arun_episode(self, engine: InferenceEngine, raw_data=None):
        
        data = copy.deepcopy(raw_data)
        if data is None:
            tasks = get_tasks()
            task = random.choice(tasks)
        else:
            data['evaluation_criteria'] = json.loads(data['evaluation_criteria'])
            task = Task.model_validate(data)

        trajs = await asyncio.gather(*[self.collect_agent_trajectory(engine, task) for _ in range(self.n_trajs)])
        version = engine.get_version()

        results = []
        rewards = []
        for i, (messagaes, traj_records, reward) in enumerate(trajs):
            rewards.append(reward)
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
                if len(loss_mask) <= self.max_context_length:
                    results.append(TensorDict(res, batch_size=[1]))
        
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{data['id']}_{uuid.uuid4().hex}.jsonl"), "w"
            ) as f:
                for i, (messages, _, score) in enumerate(trajs):
                    f.write(json.dumps(dict(messages=messages, reward=score, traj_idx=i)) + "\n")

        # if np.max(rewards) == 0:
        #     return None
        # if np.min(rewards) == 1:
        #     return None

        if len(results) == 0:
            return None
        else:
            return concat_padded_tensors(results)

@dataclass
class TauRLConfig(GRPOConfig):
    max_turns: int = field(
        default=32,
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
    user_model: str = field(
        default="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
        metadata={
            "help": "Model name for user simulation."
        }
    )

    user_api_key: str = field(
        default="empty",
        metadata={
            "help": "api_key for user simulation model"
        }
    )

    user_base_url: str = field(
        default="http://33.180.160.63:30000/v1/",
        metadata={
            "help": "base_url for user simulation model"
        }
    )

    max_context_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum context length of the trained model"
        }
    )
    
    reward_type: str = field(
        default="all",
        metadata={
            "help": "Reward type for training or evaluation. Options: db, all"
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

    with open(os.path.join(StatsLogger.get_log_path(config.stats_logger), "config.yaml"), 'w') as file:
        print("save config.yaml in", os.path.join(StatsLogger.get_log_path(config.stats_logger)))
        yaml.dump(asdict(config), file, default_flow_style=False, sort_keys=True)

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
        get_tau2_dataset(config.train_dataset.path, tokenizer, actor.data_parallel_rank, actor.data_parallel_world_size),
        batch_size=config.train_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )

    valid_dataloader = StatefulDataLoader(
        get_tau2_dataset(config.valid_dataset.path, tokenizer, actor.data_parallel_rank, actor.data_parallel_world_size),
        batch_size=1,
        shuffle=False,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(copy.deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    actor.initialize(None, ft_spec)
    ref = None

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    
    # weight_update_meta = [
    #     WeightUpdateMeta.from_fsdp_nccl(
    #         AllocationMode.from_str(config.allocation_mode), actor
    #     )
    # ]
    # dist.broadcast_object_list(weight_update_meta, src=0)
    # weight_update_meta = weight_update_meta[0]

    weight_update_meta = WeightUpdateMeta.from_disk(
            config.experiment_name,
            config.trial_name,
            config.cluster.fileroot
        )

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)


    eval_workflow = Tau2Workflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
        max_num_turns=config.max_turns,
        n_trajs=config.n_trajs,
        user_model=config.user_model,
        user_api_key=config.user_api_key,
        user_base_url=config.user_base_url,
        max_context_length=config.max_context_length,
        reward_type="db"
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    # for global_step in range(0, 1):
    global_step = 0
    epoch = 0
    step = 0

    rollout.pause()
    dist.barrier(device_ids=[actor.device.index])
    with stats_tracker.record_timing("eval"):
        def evaluate_fn():
            # Stats are logged in the workflow
            # and will be exported later
            cnt = 0
            for data in valid_dataloader:
                for item in data:
                    print(f"[rank: {rank}], task_id: {item['id']}")
                    eval_rollout.submit(item, eval_workflow)
                    cnt += 1
            eval_rollout.wait(cnt, timeout=None)
        if actor.is_data_parallel_head():
            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )
    dist.barrier(device_ids=[actor.device.index])
    rollout.resume()
    current_platform.synchronize()


    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
