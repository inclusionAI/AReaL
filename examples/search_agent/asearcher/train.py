import itertools
import asyncio
import os
import sys
import uuid
import json
import numpy as np
import uuid
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from areal.utils.hf_utils import load_hf_tokenizer
from dataclasses import dataclass, field
from typing import List

import hashlib

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
    InferenceEngineConfig,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import logging, stats_tracker
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer import PPOTrainer
from areal.core import workflow_context
from areal.api.alloc_mode import AllocationMode

from reasoning_agent import run_agent
from utils.search_tool import SearchToolBox

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"ASearcher-Reasoning @ {worker_id}")

def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


class ASearcherReasoningWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: str,
        dump_dir: str | None = None,
        max_turns: int = 128,
        force_turns: int = 4,
        search_client_type: str = "async-online-search-access",
        topk: int = 10,
        max_tokens: int = 30000,
        judge_engine: RemoteSGLangEngine | None = None,
    ):
        self.gconfig = gconfig.new(n_samples=1),
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.force_turns = force_turns
        self.max_turns = max_turns
        self.topk = topk
        self.search_client_type = search_client_type

        self.toolbox = SearchToolBox(dataset_path=dataset_path, reward_type="F1", topk=self.topk, search_client_type=self.search_client_type, use_jina=True)
        self.judge_client = ArealOpenAI(engine=judge_engine, tokenizer= tokenizer)
    
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
        save_trajs_path = None
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            save_trajs_path = os.path.join(self.dump_dir, str(version), f"{qid}/{{ID}}.json")

        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        judge_client = self.judge_client

        # Collect trajectories 
        completions, reward, stats = await run_agent(
            client=client, 
            judge_client=judge_client,
            tokenizer=self.tokenizer,
            data=data,
            toolbox=self.toolbox,
            max_turns=self.max_turns,
            force_turns=self.force_turns,
            topk=self.topk,
            force_valid=True,
            max_tokens=self.max_tokens,
            save_path=save_trajs_path.format(ID=qid) if save_trajs_path is not None else None,
        )

        # Set advantages to all completions
        for comp in completions:
            client.set_reward(comp.id, reward)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward) 
        stats_tracker.get(workflow_context.stat_scope()).scalar(**stats)
        
        completions_with_rewards = client.export_interactions(style="individual")
        return completions_with_rewards

@dataclass
class AgentRLConfig(GRPOConfig):
    max_turns: int = field(
        default=128,
        metadata={
            "help": "maximum number of turns for search agent"
        }
    )
    force_turns: int = field(
        default=4,
        metadata={
            "help": "minimum number of turns for search agent"
        }
    )
    search_client_type: str = field(
        default="async-online-search-access",
        metadata={
            "help": "Type of tool (async-online-search-access/async-search-access). By default we use 'async-online-search-access'"
        }
    )
    topk: int = field(
        default=10,
        metadata={
            "help": "search returns the top-k results. Default top_k=10"
        }
    )
    dump_dir: str | None = field(
        default=None,
        metadata={
            "help": "directory to dump agent trajectories"
        }
    )
    # Logging Agent Trajectories
    judge_engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)


def get_search_dataset(dataset_path):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return dataset

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_search_dataset(config.train_dataset.path)

    # Initialize judge engine
    # Parse allocation mode.
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    judge_engine = RemoteSGLangEngine(config.judge_engine)
    judge_engine.initialize(train_data_parallel_size=allocation_mode.train.dp_size)

    with PPOTrainer(
        config=config,
        train_dataset=train_dataset,
    ) as trainer:
        workflow = ASearcherReasoningWorkflow(
            gconfig=config.gconfig,
            tokenizer=tokenizer,
            dataset_path=config.train_dataset.path,
            dump_dir=config.dump_dir,
            max_turns=config.max_turns,
            force_turns=config.force_turns,
            search_client_type=config.search_client_type,
            topk=config.topk,
            max_tokens=min(config.actor.mb_spec.max_tokens_per_mb, 32768),
            judge_engine=judge_engine,
        )
        trainer.train(workflow=workflow)

if __name__ == "__main__":
    main(sys.argv[1:])
