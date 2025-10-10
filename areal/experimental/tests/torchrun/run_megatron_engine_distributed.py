import argparse
import copy
import os
import tempfile
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from transformers import AutoTokenizer

from areal.api.alloc_mode import AllocationMode
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.experimental.api.cli_args import (
    ExperimentalTrainEngineConfig as TrainEngineConfig,
)
from areal.experimental.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
)
from areal.experimental.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import seeding
from areal.utils.data import broadcast_tensor_container

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


def write_result(out: str, succ: bool):
    with open(out, "w") as f:
        if succ:
            f.write("Passed")
        else:
            f.write("Failed")


def all_gather_logits(logits, input_data):
    """Gather logits across model parallel group and concatenate them."""
    logits_list = [
        torch.empty_like(logits)
        for _ in range(mpu.get_tensor_model_parallel_world_size())
    ]
    dist.all_gather(logits_list, logits, group=mpu.get_tensor_model_parallel_group())
    logits = torch.cat(logits_list, dim=-1)
    return logits


def mock_input(
    batch_size=128,
    min_seqlen=1,
    max_seqlen=1024,
    device=current_platform.device_type,
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


def make_engine(model_type, allocation_mode, mb_spec, init_optimizer=False):
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=OptimizerConfig() if init_optimizer else None,
        megatron=MegatronEngineConfig(),
    )
    alloc_mode = AllocationMode.from_str(allocation_mode)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    return engine


def make_fsdp_engine(model_type, allocation_mode, mb_spec, init_optimizer=False):
    engine_config = TrainEngineConfig(
        experiment_name=f"test",
        trial_name="test",
        mb_spec=mb_spec,
        path=MODEL_PATHS[model_type],
        optimizer=OptimizerConfig() if init_optimizer else None,
    )
    alloc_mode = AllocationMode.from_str(allocation_mode)
    # ignore other parallel strategy (not supported in fsdp)
    alloc_mode.train.data_parallel_size = (
        alloc_mode.train.world_size // alloc_mode.train.context_parallel_size
    )
    engine = FSDPEngine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(parallel_strategy=alloc_mode.train)
    engine.initialize(None, ft_spec)
    return engine


def test_forward(model_type: str, alloc_mode: str, output: Optional[str] = None):
    rank = int(os.environ["RANK"])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(model_type, alloc_mode, mb_spec)
    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")
    bcasted_input = broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )
    logits = engine.forward(
        input_=bcasted_input,
        post_hook=all_gather_logits,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )

    print(f"final rank {rank} result shape: {logits.shape}")
    print(f"final rank {rank} result: {logits}")

    # All ranks in the model parallel group should have the same logits
    dist.barrier()
    model_parallel_group = mpu.get_model_parallel_group()
    model_parallel_world_size = len(dist.get_process_group_ranks(model_parallel_group))
    logits_list = [torch.empty_like(logits) for _ in range(model_parallel_world_size)]
    dist.all_gather(logits_list, logits, group=model_parallel_group)

    assert all(
        torch.equal(logits, logits_list[0]) for logits in logits_list
    ), "Logits should be the same across all model parallel ranks."

    # make FSDP engine, and check if the difference between FSDP and megatron engine
    fsdp_engine = make_fsdp_engine("qwen3", alloc_mode, mb_spec)
    fsdp_logits = fsdp_engine.forward(
        input_=input_,
        post_hook=None,
        aggregate_fn=lambda xs: torch.cat(xs, dim=0),
    )
    print(
        f"rank {rank} logits.shape={logits.shape} fsdp_logits.shape={fsdp_logits.shape}"
    )
    # only compare results on data parallel head
    failed = False
    if engine.is_data_parallel_head():
        diff = torch.abs(logits - fsdp_logits)
        print(
            f"rank {rank} diff between megatron and fsdp logits: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}"
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
                fsdp_logits.to(torch.float32),
                rtol=0.2,
                atol=100,
            )
        except AssertionError as e:
            failed = True
            print(f"AssertionError in torch.testing.assert_close: {e}")

    current_platform.synchronize()
    dist.barrier()
    fsdp_engine.destroy()
    engine.destroy()
    engine.destroy_process_groups()

    print(f"Test: test_forward(model_type={model_type}, alloc_mode={alloc_mode}) Done.")
    if rank == 0 and output is not None:
        write_result(output, not failed)


def mock_loss_fn(logits: torch.Tensor, input_data) -> torch.Tensor:
    """Mock loss function for testing."""
    from megatron.core import tensor_parallel

    labels = input_data["input_ids"]
    logprobs = -tensor_parallel.vocab_parallel_cross_entropy(
        vocab_parallel_logits=logits, target=labels
    )
    return torch.mean(logprobs)


def test_train(model_type: str, alloc_mode: str, output: Optional[str] = None):
    print(f"running train test: model_type={model_type} alloc_mode={alloc_mode}")
    rank = int(os.environ["RANK"])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(model_type, alloc_mode, mb_spec, init_optimizer=True)
    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")
    bcasted_input = broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )

    train_result = engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    print(f"final rank {rank} train_result: {train_result}")
    current_platform.synchronize()
    dist.barrier()
    engine.destroy()
    engine.destroy_process_groups()

    if rank == 0 and output is not None:
        write_result(output, True)
    print(f"Test: test_train(model_type={model_type}, alloc_mode={alloc_mode}) Done.")


def test_train_dcp_save_load(
    model_type: str, alloc_mode: str, output: Optional[str] = None
):

    print(
        f"running test_train_dcp_save_load(model_type={model_type} alloc_mode={alloc_mode})"
    )
    rank = int(os.environ["RANK"])

    base_dir = tempfile.gettempdir()
    path = os.path.join(base_dir, "megatron_engine_train_dcp_test")
    if rank == 0:
        os.makedirs(path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(model_type, alloc_mode, mb_spec, init_optimizer=True)

    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")
    bcasted_input = broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )

    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    # train step 1
    train_result = engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    print(f"final rank {rank} train_result: {train_result}")

    current_platform.synchronize()
    dist.barrier()

    # save checkpoint for recover
    engine.save(save_load_meta)

    # train step 2
    engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))

        for p in engine.model.parameters():
            p.data.zero_()

        # recover
        engine.load(save_load_meta)

    engine.train()
    # train step 2 after recover
    engine.train_batch(
        input_=bcasted_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    current_platform.synchronize()
    dist.barrier()

    with torch.no_grad():
        engine.eval()
        succ = True
        for name, param in engine.model.named_parameters():
            if not torch.allclose(param, params[name]):
                diff = torch.abs(params[name] - param)
                print(
                    f"rank {rank} diff of {name}: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}, count(diff)={torch.count_nonzero(diff)}"
                )
                succ = False
        assert succ, "Weights should be same after recover"

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()
    engine.destroy_process_groups()

    if output:
        write_result(output, True)

    print(
        f"Test: test_train_dcp_save_load(model_type={model_type}, alloc_mode={alloc_mode}) Done."
    )


def test_simple_dcp_save_load(
    model_type: str, alloc_mode: str, output: Optional[str] = None
):

    print(
        f"running test_simple_dcp_save_load(model_type={model_type} alloc_mode={alloc_mode})"
    )
    rank = int(os.environ["RANK"])

    base_dir = tempfile.gettempdir()
    path = os.path.join(base_dir, "megatron_engine_simple_dcp_test")
    if rank == 0:
        os.makedirs(path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_type])

    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(model_type, alloc_mode, mb_spec, init_optimizer=True)

    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_ = mock_input(batch_size=16, max_seqlen=128, device=engine.device)
    print(f"rank {rank} is_data_parallel_head()={engine.is_data_parallel_head()}")

    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))
        engine.save(save_load_meta)

        for p in engine.model.parameters():
            p.data.zero_()

        engine.load(save_load_meta)

        succ = True
        for name, param in engine.model.named_parameters():
            if not torch.allclose(param, params[name]):
                diff = torch.abs(params[name] - param)
                print(
                    f"rank {rank} diff of {name}: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}, count(diff)={torch.count_nonzero(diff)}"
                )
                succ = False
        assert succ, "Weights should be same after recover"

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()
    engine.destroy_process_groups()

    if output:
        write_result(output, True)

    print(
        f"Test: test_simple_dcp_save_load(model_type={model_type}, alloc_mode={alloc_mode}) Done."
    )


def main():
    parser = argparse.ArgumentParser(description="Run Megatron Engine Distributed Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3", "qwen3moe"],
        default="qwen3",
        help="Type of model to test",
    )
    parser.add_argument(
        "--allocation_mode",
        type=str,
        default="d1p2t2c2",
        help="Allocation mode for the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output result",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["forward", "train", "simple_dcp_save_load", "train_dcp_save_load"],
        default="forward",
        help="Type of test to run: 'forward' or 'train'",
    )
    args = parser.parse_args()

    print(args)
    if args.test_type == "train":
        test_train(args.model_type, args.allocation_mode, output=args.output)
    elif args.test_type == "forward":
        test_forward(args.model_type, args.allocation_mode, output=args.output)
    elif args.test_type == "simple_dcp_save_load":
        test_simple_dcp_save_load(
            args.model_type, args.allocation_mode, output=args.output
        )
    elif args.test_type == "train_dcp_save_load":
        test_train_dcp_save_load(
            args.model_type, args.allocation_mode, output=args.output
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    # run with `torchrun` to test with multiple GPUs & multiple nodes
    main()
