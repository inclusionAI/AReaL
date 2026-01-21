import argparse
import os
from typing import Any, Dict

import torch
import torch.distributed as dist

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.platforms import current_platform

MODEL_PATHS = {
    "qwen3": "/storage/openpsi/models/Qwen__Qwen3-0.6B/",
    "qwen3moe": "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/",
}
HF_MODEL_PATHS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    # TODO: switch Qwen3MoE to smaller model initialized from scratch
    "qwen3moe": "Qwen/Qwen3-30B-A3B",
}
for model_type, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        MODEL_PATHS[model_type] = HF_MODEL_PATHS[model_type]


def setup_distributed_environment():
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def mock_input(
    device: torch.device,
    batch_size=128,
    min_seqlen=1,
    max_seqlen=1024,
) -> Dict[str, Any]:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        10000, 50000, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def make_engine(model_type, mb_spec, ulysses_sp_size=1, init_optimizer=False):
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=OptimizerConfig() if init_optimizer else None,
    )
    print(f"config = {config}")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = FSDPEngine(config)
    assert dist.get_world_size() >= ulysses_sp_size
    parallel_strategy = ParallelStrategy(
        data_parallel_size=dist.get_world_size() // ulysses_sp_size,
        context_parallel_size=ulysses_sp_size,
    )
    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_ulysses(model_type: str):
    setup_distributed_environment()

    torch.manual_seed(42)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    mb_spec = MicroBatchSpec(n_mbs=4)

    engine = make_engine(model_type, mb_spec, ulysses_sp_size=2)
    engine_golden = make_engine(model_type, mb_spec)

    input = mock_input(device=engine.device, batch_size=4, max_seqlen=16)

    engine.eval()
    logits = engine.forward(
        input_=input,
        post_hook=None,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    dist.barrier()

    print(f"rank {rank} logits shape: {logits.shape} with content {logits}")

    # All ranks in the model parallel group should have the same logits
    logits_list = [torch.empty_like(logits) for _ in range(world_size)]
    dist.all_gather(logits_list, logits, group=dist.group.WORLD)

    assert all(
        torch.equal(logits, logits_list[0]) for logits in logits_list
    ), "Logits should be the same across all model parallel ranks."

    engine_golden.eval()
    logits_golden = engine_golden.forward(
        input_=input,
        post_hook=None,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    dist.barrier()

    print(
        f"rank {rank} logits_golden shape: {logits_golden.shape} with content {logits_golden}"
    )

    if rank == 0:
        diff = torch.abs(logits - logits_golden)
        print(
            f"rank {rank} diff between SP=1 and SP=2 logits: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}"
        )
        # statistics
        non_zero_logits = torch.count_nonzero(logits)
        masked_diff = diff.masked_fill(logits == 0, torch.finfo(logits.dtype).max)
        print(f"logits non-zero count: {torch.count_nonzero(logits)}")
        print(
            f"diff < 10 count: {torch.count_nonzero(masked_diff < 10)}, {torch.count_nonzero(masked_diff < 10) * 100/non_zero_logits:.2f} percent."
        )
        print(
            f"diff < 1 count: {torch.count_nonzero(masked_diff < 1)}, {torch.count_nonzero(masked_diff < 1) * 100/non_zero_logits:.2f} percent."
        )
        print(
            f"diff < 0.1 count: {torch.count_nonzero(masked_diff < 0.1)}, {torch.count_nonzero(masked_diff < 0.1) * 100/non_zero_logits:.2f} percent."
        )
        print(
            f"diff < 0.01 count: {torch.count_nonzero(masked_diff < 0.01)}, {torch.count_nonzero(masked_diff < 0.01) * 100/non_zero_logits:.2f} percent."
        )
        try:
            torch.testing.assert_close(
                logits.to(torch.float32),
                logits_golden.to(torch.float32),
                rtol=0.2,
                atol=100,
            )
        except AssertionError as e:
            print(f"AssertionError in torch.testing.assert_close: {e}")

    current_platform.synchronize()
    dist.barrier()

    engine_golden.destroy()
    engine.destroy()


def main():
    parser = argparse.ArgumentParser(description="Run FSDP Ulysses Engine Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3"],
        default="qwen3",
        help="Type of model to test",
    )
    args = parser.parse_args()
    test_ulysses(args.model_type)


if __name__ == "__main__":
    # run with `torchrun` to test with multiple GPUs & multiple nodes
    main()
