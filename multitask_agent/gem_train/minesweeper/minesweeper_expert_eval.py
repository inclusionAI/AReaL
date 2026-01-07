import os
import sys
import re
from dataclasses import dataclass, field
import uuid
import asyncio
import json
import copy

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from datasets import load_dataset
import numpy as np
import gem

from areal.api.cli_args import GRPOConfig, load_expr_config, GenerationHyperparameters
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    StepInfo,
    WeightUpdateMeta,
    ModelRequest,
)
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import MegatronPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.platforms import current_platform
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.perf_tracer import Category, trace_perf
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

import multitask_agent.gem_train.minesweeper_with_template
import multitask_agent.gem_train.sudoku_with_template
from multitask_agent.gem_train.gem_train import GEMWorkflow, GEMConfig
from multitask_agent.gem_train.minesweeper.cpp_solver import CPPSolver

logger = logging.getLogger("GEM grpo")


class MinesweeperSolverWorkflow(GEMWorkflow):
    async def collect_agent_trajectory(self, engine, template=None):
        solver = CPPSolver()
        env = gem.make(self.env_name)
        if template is not None:
            # print(f"debug====set template")
            env.set_template(template)
        obs, info = env.reset()
        _obs = obs + info.get("suffix", "")
        messages = [{"role": "user", "content": _obs}]
        rewards = [0.0]
        versions = [[]]
        tokens = [None]

        # print(f"debug====obs: {_obs}")
        traj_rid = uuid.uuid4().hex
        while True:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            # req = ModelRequest(
            #     rid=traj_rid,
            #     input_ids=input_ids,
            #     gconfig=self.gconfig.new(
            #         n_samples=1,
            #     ),
            # )
            # if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
            #     break
            # resp = await engine.agenerate(req)
            # completion_str: str = self.tokenizer.decode(resp.output_tokens)
            completion_str: str = solver.act(env.rows, env.cols, _obs)
            tmp = completion_str.split("</think>")
            thinking = tmp[0] + "</think>"
            if len(tmp) < 2:
                action = ""
            else:
                action = tmp[1]
            flag = True
            while flag:
                action = action.rstrip()
                flag = False
                for stop in ["<|im_end|>", self.tokenizer.eos_token]:
                    if action.endswith(stop):
                        action = action[: -len(stop)]
                        flag = True
                        break
            messages.append(
                {"role": "assistant", "content": action, "thinking": thinking}
            )
            # print(f"debug===action: {completion_str}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            _obs = next_obs + info.get("suffix", "")
            # print(f"debug====obs: {_obs}")
            # TODO: tool
            messages.append({"role": "user", "content": _obs})
            rewards.extend([reward, 0.0])
            versions.extend([[], []])
            tokens.extend(
                [
                    {
                        "input_tokens": input_ids,
                        "output_tokens": [],
                        "output_logprobs": [],
                    },
                    None,
                ]
            )

            obs = next_obs
            if terminated or truncated:
                break

        assert len(messages) == len(rewards), (len(messages), len(rewards))
        return messages, rewards, versions, tokens


def main(args):
    config, _ = load_expr_config(args, GEMConfig)
    config: GEMConfig
    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    dist.init_process_group(backend="gloo")

    # Configure performance tracer
    if config.perf_tracer is not None:
        perf_tracer.configure(config.perf_tracer, rank=rank)

    # Create dataset and dataloaders
    # train_dataset = get_boba_math_dataset(config.train_dataset.path, tokenizer)
    valid_dataset = list(range(128))
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        dataset_config=config.train_dataset,
    )

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=1000 * 64,
        train_batch_size=64,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.config.max_head_offpolicyness = int(1e12)
    rollout.initialize(train_data_parallel_size=dist.get_world_size())

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = MinesweeperSolverWorkflow(
        config.gconfig,
        tokenizer,
        config.env_name,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        n_trajs=config.n_trajs,
        max_tokens=config.max_traj_tokens,
        dynamic_filtering=False,
    )

    # Run training.
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    dist.barrier()
    with stats_tracker.record_timing("eval"):
        cnt = 0
        for data in valid_dataloader:
            for item in data:
                rollout.submit(item, workflow)
                cnt += 1
        rollout.wait(cnt)
    dist.barrier()

    stats_logger.close()
    rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
