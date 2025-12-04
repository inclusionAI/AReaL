import asyncio
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    load_expr_config,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer import PPOTrainer
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
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
            completion_with_rewards = client.export_interactions(style="concat")
            # FIXME: sometimes len(completion_with_rewards) > 1, needs to figure out why
            assert len(completion_with_rewards) == 1, len(completion_with_rewards)
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

    # NOTE: Ensure config.actor.group_size == config.n_trajs for proper batching
    # The trainer uses config.actor.group_size as granularity in prepare_batch
    assert config.actor.group_size == config.n_trajs, (
        f"config.actor.group_size ({config.actor.group_size}) must equal config.n_trajs ({config.n_trajs})"
    )

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset
    train_dataset = get_search_dataset(config.train_dataset.path, tokenizer=tokenizer)

    judge_engine = RemoteSGLangEngine(config.judge_engine)
    try:
        # NOTE: judge engine should not have off-policyness control.
        judge_engine.config.max_head_offpolicyness = int(1e12)
        judge_engine.initialize()

        # Create trainer (no valid_dataset for this example)
        with PPOTrainer(config, train_dataset, valid_dataset=None) as trainer:
            # Create rollout workflow
            workflow = TongyiDeepResearchReactWorkflow(
                gconfig=config.gconfig,
                tokenizer=trainer.tokenizer,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger), "generated"
                ),
                n_trajs=config.n_trajs,
                max_tokens=config.max_tokens_per_trajectory,
                max_llm_calls_per_run=config.max_llm_calls_per_run,
                judge_engine=judge_engine,
            )

            # Run training
            trainer.train(workflow, eval_workflow=None)
    finally:
        # Cleanup judge engine
        judge_engine.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
