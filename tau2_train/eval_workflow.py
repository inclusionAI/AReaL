import os
import sys
import json
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

from tau2_train.workflow_megatron import Tau2Workflow, get_tau2_dataset, TauRLConfig


import yaml
import numpy as np
from loguru import logger
logger.remove()

from dataclasses import dataclass, field, asdict
from copy import deepcopy
import uuid
import asyncio
import functools
import colorama
from concurrent.futures import ProcessPoolExecutor

import torch

from megatron.core import parallel_state as mpu
from torch import distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config, GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta, ModelRequest
from areal.api.alloc_mode import MegatronParallelStrategy, ParallelStrategy
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.ppo.actor import MegatronPPOActor
from areal.platforms import current_platform
from areal.utils import seeding, logging, stats_tracker
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger


def main(args):
    config, _ = load_expr_config(args, TauRLConfig)
    config: TauRLConfig

    with open(os.path.join(StatsLogger.get_log_path(config.stats_logger), "config.yaml"), 'w') as file:
        print("save config.yaml in", os.path.join(StatsLogger.get_log_path(config.stats_logger)))
        yaml.dump(asdict(config), file, default_flow_style=False, sort_keys=True)

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    dist.init_process_group(backend='gloo')

    # Create dataset and dataloaders
    valid_dataloader = StatefulDataLoader(
        # get_tau2_dataset(config.valid_dataset.path, tokenizer, mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()),
        get_tau2_dataset(config.valid_dataset.path, tokenizer, dist.get_rank(), dist.get_world_size()),
        batch_size=1,
        shuffle=False,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=1000 * 64,
        train_batch_size=64,
    )

    # Initialize inference engine
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize(train_data_parallel_size=dist.get_world_size())

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
        user_llm_args=asdict(config.user_llm_args) if config.user_llm_args is not None else None,
        max_context_length=config.max_context_length,
        reward_type="db",
        dynamic_filtering=False,
    )

    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    dist.barrier()
    with stats_tracker.record_timing("eval"):

        def evaluate_fn():

            cnt = 0
            for data in valid_dataloader:
                for item in data:
                    eval_rollout.submit(item, eval_workflow)
                    cnt += 1
            eval_rollout.wait(cnt, timeout=None)
        evaluate_fn()
    dist.barrier()

    stats_logger.close()
    eval_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
