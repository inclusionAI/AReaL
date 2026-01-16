"""
Rollout-only example using RemoteSGLangEngine and DistRolloutCoordinator.

This script demonstrates how to run rollout without training, iterating
through the entire dataset using prepare_batch.

Usage:
    torchrun --nproc_per_node=1 examples/rollout_only/rollout.py \
        --config examples/math/gsm8k_grpo.yaml \
        actor.path=Qwen/Qwen2.5-1.5B
"""

import os
import sys
from dataclasses import dataclass, field

import torch.distributed as dist

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.engine_api import TrainEngine
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats


class MockTrainEngine(TrainEngine):
    """Minimal TrainEngine mock for rollout-only usage.

    Provides CPU-based data parallel group for DistRolloutCoordinator
    without actual training capabilities.
    """

    def __init__(self):
        self._data_parallel_group = None
        self._context_and_model_parallel_group = None
        self._cpu_group = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

    @property
    def data_parallel_group(self):
        if self._data_parallel_group is None:
            self._data_parallel_group = dist.new_group()
        return self._data_parallel_group

    @property
    def context_and_model_parallel_group(self):
        if self._context_and_model_parallel_group is None:
            # For rollout-only, context/model parallel is just self
            self._context_and_model_parallel_group = dist.new_group([self._rank])
        return self._context_and_model_parallel_group

    @property
    def cpu_group(self):
        if self._cpu_group is None:
            self._cpu_group = dist.new_group()
        return self._cpu_group

    def is_data_parallel_head(self) -> bool:
        """All ranks are data parallel heads in this mock."""
        return True

    def current_data_parallel_head(self) -> int:
        """Return current rank as the data parallel head."""
        return self._rank

    @property
    def device(self):
        return "cpu"


@dataclass
class RolloutOnlyConfig(GRPOConfig):
    """Configuration for rollout-only execution."""

    max_batches: int = field(
        default=-1,
        metadata={"help": "Maximum number of batches to process. -1 for all."},
    )


def main(args):
    config, _ = load_expr_config(args, RolloutOnlyConfig)

    # Initialize distributed
    dist.init_process_group("gloo")
    group = dist.new_group()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"rollout{rank}")

    # Create dataset and dataloader (same as gsm8k_rl_mt.py)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    train_dataloader = create_dataloader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        dataset_config=config.train_dataset,
    )

    # Initialize inference engine
    config.rollout.max_head_offpolicyness = int(1e12)  # Disable staleness control
    rollout_engine = RemoteSGLangEngine(config.rollout)
    rollout_engine.initialize()

    # Create mock train engine and coordinator
    mock_train_engine = MockTrainEngine()
    coordinator = DistRolloutCoordinator(
        rollout_engine=rollout_engine,
        train_engine=mock_train_engine,
    )

    workflow_kwargs = dict(
        reward_fn="examples.multi_turn_math.gsm8k_rl_mt.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        export_style="individual",
        max_turns=8,
    )
    # Run rollout through entire dataset
    batch_count = 0
    total_samples = 0

    print(f"[Rank {rank}] Starting rollout...")

    for batch_idx, data in enumerate(train_dataloader):
        if config.max_batches > 0 and batch_count >= config.max_batches:
            break

        batch = coordinator.prepare_batch(
            dataloader=train_dataloader,
            workflow="examples.multi_turn_math.gsm8k_rl_mt.MultiturnRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            group_size=config.gconfig.n_samples,
        )

        batch_size = batch.get("input_ids", batch.get("packed_input_ids")).shape[0]
        total_samples += batch_size
        batch_count += 1

        if rank == 0 and batch_count % 10 == 0:
            print(f"Processed {batch_count} batches, {total_samples} samples")

    # Export and print statistics
    rollout_stats = stats_tracker.export_all(reduce_group=group)
    if rank == 0:
        print(f"\nRollout completed!")
        print(f"Total batches: {batch_count}")
        print(f"Total samples: {total_samples}")
        print(f"\nStatistics:\n{tabulate_stats(rollout_stats)}")

    # Cleanup
    rollout_engine.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
