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
        self._initialized = False
        self._version = 0

    # === Process group properties (used by DistRolloutCoordinator) ===

    @property
    def data_parallel_group(self):
        if self._data_parallel_group is None:
            self._data_parallel_group = dist.new_group()
        return self._data_parallel_group

    @property
    def data_parallel_rank(self) -> int:
        return self._rank

    @property
    def data_parallel_world_size(self) -> int:
        return self._world_size

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

    # === Initialization methods (stubs) ===

    def create_process_group(self, parallel_strategy=None):
        pass

    def initialize(self, *args, **kwargs):
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    # === Version management (stubs) ===

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    # === Training methods (stubs - not used in rollout-only) ===

    def train(self, mode: bool = True):
        pass

    def update_weights(self, meta):
        pass

    def connect_engine(self, engine, meta):
        pass

    def rollout_batch(self, data, workflow, workflow_kwargs=None, group_size=1):
        raise NotImplementedError("MockTrainEngine does not support rollout_batch")

    def prepare_batch(
        self, dataloader, workflow, workflow_kwargs=None,
        should_accept_fn=None, group_size=1, dynamic_bs=False
    ):
        raise NotImplementedError("MockTrainEngine does not support prepare_batch")

    def save(self, meta):
        pass

    def load(self, meta):
        pass

    def optimizer_zero_grad(self):
        pass

    def optimizer_step(self):
        return {"update_successful": True, "grad_norm": 0.0, "lr": 0.0}

    def lr_scheduler_step(self):
        pass

    def forward_backward_batch(self, mb_list, process_output_fn, forward_only=False):
        pass

    def train_batch(self, input_, loss_fn, loss_weight_fn):
        return {}

    def eval_batch(self, input_, loss_fn, loss_weight_fn):
        return None

    def forward_batch(self, input_, output_seqlens=None, aggregate_fn=None):
        return None

    def export_stats(self):
        return {}

    def onload(self):
        pass

    def offload(self):
        pass

    def get_device_stats(self):
        from areal.api.io_struct import DeviceRuntimeInfo
        return DeviceRuntimeInfo()

@dataclass
class MultiTurnGRPOConfig(GRPOConfig):
    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Export style for the completions. By default export_style=concat."
        },
    )
    max_batches: int = field(
        default=8,
        metadata={"help": "Maximum number of batches to rollout. -1 for all."},
    )


def main(args):
    config, _ = load_expr_config(args, MultiTurnGRPOConfig)

    # Initialize distributed
    dist.init_process_group("gloo")
    group = dist.new_group()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"rollout{rank}")

    # Create dataset and dataloader (same as gsm8k_rl_mt.py)
    # Trim dataset here, otherwise workflow executor indefinitely submit data
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    batch_size = config.train_dataset.batch_size // world_size
    max_num_prompts = config.max_batches * batch_size
    train_dataset = train_dataset.select(range(max_num_prompts))

    train_dataloader = create_dataloader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        dataset_config=config.train_dataset,
    )

    # Initialize inference engine
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

    for step, data in enumerate(train_dataloader):
        if config.max_batches > 0 and batch_count >= config.max_batches:
            break

        batch = coordinator.prepare_batch(
            dataloader=train_dataloader,
            workflow="examples.multi_turn_math.gsm8k_rl_mt.MultiturnRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            group_size=config.gconfig.n_samples,
        )

        batch_shape = batch.get("input_ids", batch.get("packed_input_ids")).shape
        batch_size = batch_shape[0]
        print(f"[Rank {rank}] output batch shape: {batch_shape}")
        total_samples += batch_size
        batch_count += 1

        mock_train_engine.set_version(step + 1)
        rollout_engine.set_version(step + 1)

        if batch_count % 10 == 0:
            print(f"[Rank {rank}] Processed {batch_count} batches, {total_samples} samples")

    # Export and print statistics
    rollout_stats = stats_tracker.export_all(reduce_group=group)
    print(f"\n[Rank {rank}] Rollout completed!")
    print(f"[Rank {rank}] Total batches: {batch_count}")
    print(f"[Rank {rank}] Total samples: {total_samples}")
    print(f"\n[Rank {rank}] Statistics:\n{tabulate_stats(rollout_stats)}")

    # Cleanup
    rollout_engine.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
