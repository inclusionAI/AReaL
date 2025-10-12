import pytest

from areal.api.alloc_mode import (
    AllocationMode,
    AllocationType,
    AllocationValidationError,
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
            expert_tensor_parallel_size=1,
        ),
    )
    assert train_ps.world_size == 64, alloc_mode_str
    assert train_ps.expert_data_parallel_size == 16, alloc_mode_str

    # Test with and without parentheses
    alloc_mode_str = "attn:d4p2t2c2|ffn:d2p2t4e2"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    assert (
        alloc_mode.train_backend == "megatron"
    )  # only megatron allows different attention-ffn parallelism
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

    # Test with parentheses - should produce same result
    alloc_mode_str = "(attn:d4p2t2c2|ffn:d2p2t4e2)"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    assert alloc_mode.train_backend == "megatron"
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

    # Test different pipeline parallel sizes error
    alloc_mode_str = "(attn:d2p2t1c4|ffn:d2p4t1e2)"
    with pytest.raises(AllocationValidationError) as exc_info:
        AllocationMode.from_str(alloc_mode_str)
    assert (
        "Pipeline parallel size for attention and FFN modules must be identical"
        in str(exc_info.value)
    )

    # Test different world sizes error (through mismatched dimensions)
    alloc_mode_str = "(attn:d4p2t1c1|ffn:d2p2t2e2)"
    with pytest.raises(InvalidAllocationModeError) as exc_info:
        AllocationMode.from_str(alloc_mode_str)
    assert (
        "World size for expert modules and attention modules must be identical"
        in str(exc_info.value)
    )


def test_decoupled_train():
    # Test inference backend with dot notation
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

    # Test inference backend with colon notation
    alloc_mode_str = "sglang:d4p2t2+megatron:(attn:d2p2t2c2|ffn:d2p2t2e2)"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
    assert alloc_mode.gen_backend == "sglang"
    assert alloc_mode.train_backend == "megatron"

    # Test without explicit megatron backend (should auto-detect to None for hybrid MoE)
    alloc_mode_str = "sglang.d4p2t2+(attn:d2p2t2c2|ffn:d2p2t2e2)"
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
    # Test with dot notation
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

    # Test with colon notation (same result as dot notation)
    alloc_mode_str = "sglang:d4p1t1+eval"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_EVAL
    assert alloc_mode.gen_backend == "sglang"
    assert alloc_mode.gen.world_size == 4, alloc_mode_str


def test_llm_server_only():
    # Test with dot notation
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

    # Test with colon notation (same result as dot notation)
    alloc_mode_str = "sglang:d4p2t2"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY
    assert alloc_mode.gen_backend == "sglang"
    assert alloc_mode.gen.world_size == 16, alloc_mode_str


def test_training_backends():
    # Test explicit training backends

    # FSDP training only
    alloc_mode_str = "fsdp:d4"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    assert alloc_mode.gen_backend is None
    assert alloc_mode.train_backend == "fsdp"
    train_ps = alloc_mode.train
    assert train_ps.data_parallel_size == 4
    assert train_ps.world_size == 4, alloc_mode_str

    # Megatron training only
    alloc_mode_str = "megatron:d2p2t1"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    assert alloc_mode.gen_backend is None
    assert alloc_mode.train_backend == "megatron"
    train_ps = alloc_mode.train
    assert train_ps.data_parallel_size == 2
    assert train_ps.pipeline_parallel_size == 2
    assert train_ps.world_size == 4, alloc_mode_str

    # Decoupled with different backends
    alloc_mode_str = "sglang:d2+fsdp:d4"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
    assert alloc_mode.gen_backend == "sglang"
    assert alloc_mode.train_backend == "fsdp"
    gen_ps = alloc_mode.gen
    train_ps = alloc_mode.train
    assert gen_ps.data_parallel_size == 2
    assert train_ps.data_parallel_size == 4
    assert gen_ps.world_size == 2, alloc_mode_str
    assert train_ps.world_size == 4, alloc_mode_str

    # Complex decoupled with different backends
    alloc_mode_str = "vllm:d2t4+megatron:d2p2t1"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
    assert alloc_mode.gen_backend == "vllm"
    assert alloc_mode.train_backend == "megatron"
    gen_ps = alloc_mode.gen
    train_ps = alloc_mode.train
    assert gen_ps.data_parallel_size == 2
    assert gen_ps.tensor_parallel_size == 4
    assert train_ps.data_parallel_size == 2
    assert train_ps.pipeline_parallel_size == 2
    assert gen_ps.world_size == 8, alloc_mode_str
    assert train_ps.world_size == 4, alloc_mode_str


def test_modern_syntax_with_backends():
    # Test modern hybrid MoE syntax with explicit backends

    # Megatron with hybrid MoE (FSDP doesn't support complex parallelism)
    alloc_mode_str = "megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.COLOCATE
    assert alloc_mode.gen_backend is None
    assert alloc_mode.train_backend == "megatron"
    train_ps = alloc_mode.train
    assert train_ps.tensor_parallel_size == 2
    assert train_ps.data_parallel_size == 4
    assert train_ps.expert_tensor_parallel_size == 4
    assert train_ps.world_size == 32, alloc_mode_str

    # Decoupled with inference backend and training hybrid MoE
    # Use the known working pattern from earlier tests: (attn:d4p2t2c2|ffn:d2p2t4e2)
    alloc_mode_str = "sglang:d4p1t2+megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)"
    alloc_mode = AllocationMode.from_str(alloc_mode_str)
    assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
    assert alloc_mode.gen_backend == "sglang"
    assert alloc_mode.train_backend == "megatron"
    gen_ps = alloc_mode.gen
    train_ps = alloc_mode.train
    assert gen_ps.data_parallel_size == 4
    assert gen_ps.tensor_parallel_size == 2
    assert train_ps.data_parallel_size == 4  # from attention
    assert train_ps.context_parallel_size == 2  # from attention
    assert train_ps.expert_parallel_size == 2  # from expert
    assert train_ps.expert_tensor_parallel_size == 4  # from expert
    assert gen_ps.world_size == 8, alloc_mode_str
    assert train_ps.world_size == 32, alloc_mode_str
