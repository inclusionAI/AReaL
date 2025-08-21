import pytest

from areal.experimental.api.io_struct import (
    AllocationMode,
    AllocationType,
    InvalidAllocationModeError,
    ParallelStrategy,
)


def test_colocate():
    alloc_mode_str = "d2p2t1"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    train_ps = alloc_mode.train
    assert ParallelStrategy.parallelism_eq(
        train_ps,
        ParallelStrategy(
            tensor_parallel_size=1, data_parallel_size=2, pipeline_parallel_size=2
        ),
    )
    assert train_ps.world_size == 4, alloc_mode_str

    alloc_mode_str = "d2p2t4e2c4"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    train_ps = alloc_mode.train
    assert ParallelStrategy.parallelism_eq(
        train_ps,
        ParallelStrategy(
            tensor_parallel_size=4,
            data_parallel_size=2,
            pipeline_parallel_size=2,
            context_parallel_size=4,
            expert_parallel_size=2,
            expert_tensor_parallel_size=4,
        ),
    )
    assert train_ps.world_size == 64, alloc_mode_str
    assert train_ps.expert_data_parallel_size == 4, alloc_mode_str

    alloc_mode_str = "d4p2t2c2/d2p2t4e2"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    train_ps = alloc_mode.train
    assert ParallelStrategy.parallelism_eq(
        train_ps,
        ParallelStrategy(
            tensor_parallel_size=2,
            data_parallel_size=4,
            pipeline_parallel_size=2,
            context_parallel_size=2,
            expert_parallel_size=2,
            expert_tensor_parallel_size=4,
        ),
    )
    assert train_ps.world_size == 32, alloc_mode_str
    assert train_ps.expert_data_parallel_size == 2, alloc_mode_str

    alloc_mode_str = "d2p2t1c4/d2p2t1e2"
    with pytest.raises(InvalidAllocationModeError):
        train_ps = AllocationMode.from_str(alloc_mode_str)


def test_decoupled_train():
    alloc_mode_str = "vllm.d2p2t2+d2p2t2"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
    assert alloc_mode.gen_backend == "vllm"
    train_ps = alloc_mode.train
    assert ParallelStrategy.parallelism_eq(
        train_ps,
        ParallelStrategy(
            tensor_parallel_size=2,
            data_parallel_size=2,
            pipeline_parallel_size=2,
        ),
    )
    assert train_ps.world_size == 8, alloc_mode_str
    gen_ps = alloc_mode.gen
    assert ParallelStrategy.parallelism_eq(
        gen_ps,
        ParallelStrategy(
            tensor_parallel_size=2,
            data_parallel_size=2,
            pipeline_parallel_size=2,
        ),
    )
    assert gen_ps.world_size == 8, alloc_mode_str

    alloc_mode_str = "sglang.d4p2t2+d2p2t2c2/d2p2t2e2"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
    assert alloc_mode.gen_backend == "sglang"
    train_ps = alloc_mode.train
    assert ParallelStrategy.parallelism_eq(
        train_ps,
        ParallelStrategy(
            tensor_parallel_size=2,
            data_parallel_size=2,
            pipeline_parallel_size=2,
            context_parallel_size=2,
            expert_parallel_size=2,
            expert_tensor_parallel_size=2,
        ),
    )
    assert train_ps.world_size == 16, alloc_mode_str
    gen_ps = AllocationMode.from_str(alloc_mode_str).gen
    assert ParallelStrategy.parallelism_eq(
        gen_ps,
        ParallelStrategy(
            tensor_parallel_size=2, data_parallel_size=4, pipeline_parallel_size=2
        ),
    )
    assert gen_ps.world_size == 16, alloc_mode_str


def test_decoupled_eval():
    alloc_mode_str = "sglang.d4p1t1+eval"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_EVAL
    assert alloc_mode.gen_backend == "sglang"
    gen_ps = alloc_mode.gen
    assert ParallelStrategy.parallelism_eq(
        gen_ps,
        ParallelStrategy(
            tensor_parallel_size=1, data_parallel_size=4, pipeline_parallel_size=1
        ),
    )
    assert gen_ps.world_size == 4, alloc_mode_str


def test_llm_server_only():
    alloc_mode_str = "sglang.d4p2t2"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY
    assert alloc_mode.gen_backend == "sglang"
    gen_ps = alloc_mode.gen
    assert ParallelStrategy.parallelism_eq(
        gen_ps,
        ParallelStrategy(
            tensor_parallel_size=2, data_parallel_size=4, pipeline_parallel_size=2
        ),
    )
    assert gen_ps.world_size == 16, alloc_mode_str
