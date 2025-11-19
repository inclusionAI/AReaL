"""Integration tests for offload functionality in FSDP and Megatron engines using TMS mode.

Tests actual memory release/resume behavior and transfer speed, not just API calls.
"""

import os
import time

import pytest
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils.network import find_free_ports
from areal.utils.tms_utils import get_tms_env_vars

MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"


def get_gpu_memory_allocated_gb() -> float:
    """Get currently allocated GPU memory in GB."""
    device = current_platform.current_device()
    allocated = current_platform.device_memory_used(device)
    return allocated / (1024**3)


def estimate_pcie_bandwidth_gbps() -> float:
    # PCIe 5.0 x16: 64 GB/s one direction, 128 GB/s duplex directions
    return 64.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fsdp_engine_with_offload():
    """Create FSDP engine with TMS offload enabled."""
    tms_env_vars = get_tms_env_vars()

    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(find_free_ports(1)[0]),
            **tms_env_vars,
        }
    )
    torch_memory_saver.hook_mode = "preload"

    config = TrainEngineConfig(
        experiment_name="test_offload",
        trial_name="fsdp_tms",
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )

    engine = FSDPEngine(config)
    engine.create_process_group()
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.initialize(addr=None, ft_spec=ft_spec)
    engine.config.offload_train = True

    print("FSDP engine initialized with offload_train=True")

    try:
        yield engine
    finally:
        engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture
def megatron_engine_with_offload():
    """Create Megatron engine with TMS offload enabled."""
    tms_env_vars = get_tms_env_vars()
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(find_free_ports(1)[0]),
            **tms_env_vars,
        }
    )

    config = TrainEngineConfig(
        experiment_name="test_offload",
        trial_name="megatron",
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(),
    )

    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    engine.config.offload_train = True

    print(f"Megatron engine initialized with offload_train={config.offload_train}")

    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


# =============================================================================
# FSDP Offload Tests
# =============================================================================


def test_fsdp_offload_and_restore(fsdp_engine_with_offload):
    """Test FSDP TMS offload releases memory and restore recovers it correctly.

    This test validates:
    1. Memory is released during offload
    2. Transfer speed is reasonable
    3. Memory is restored correctly after wake_up
    """
    engine = fsdp_engine_with_offload

    # Initialize optimizer state with a dummy training step
    print("[TMS] Running dummy training step to initialize optimizer...")

    # Measure initial memory
    current_platform.synchronize()
    initial_memory_gb = get_gpu_memory_allocated_gb()
    print(f"[TMS] Initial GPU memory: {initial_memory_gb:.2f} GB")

    # === Test Offload ===
    start_time = time.perf_counter()
    engine.sleep()
    offload_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_offload_gb = get_gpu_memory_allocated_gb()
    memory_released_gb = initial_memory_gb - memory_after_offload_gb

    print(
        f"[TMS] After offload: {memory_after_offload_gb:.2f} GB "
        f"(released {memory_released_gb:.2f} GB in {offload_time:.3f}s)"
    )

    # Assert memory was released
    assert memory_released_gb > 0.1, (
        f"Expected memory release, but only {memory_released_gb:.2f} GB was released"
    )

    # Calculate and verify transfer speed
    if offload_time > 0:
        offload_speed_gbps = memory_released_gb / offload_time
        print(f"[TMS] Offload speed: {offload_speed_gbps:.2f} GB/s")

        # Speed should be reasonable
        pcie_bandwidth = estimate_pcie_bandwidth_gbps()
        pcie_bandwidth_threshold = pcie_bandwidth * 0.7
        assert offload_speed_gbps > pcie_bandwidth_threshold, (
            f"Offload speed {offload_speed_gbps:.2f} GB/s is too slow "
            f"(expected > {pcie_bandwidth_threshold:.2f} GB/s)"
        )

    # === Test Restore ===
    start_time = time.perf_counter()
    engine.wake_up()
    restore_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_restore_gb = get_gpu_memory_allocated_gb()
    memory_restored_gb = memory_after_restore_gb - memory_after_offload_gb

    print(
        f"[TMS] After restore: {memory_after_restore_gb:.2f} GB "
        f"(restored {memory_restored_gb:.2f} GB in {restore_time:.3f}s)"
    )

    # Memory should be restored to approximately initial level (within 10% tolerance)
    memory_diff = abs(memory_after_restore_gb - initial_memory_gb)
    tolerance = initial_memory_gb * 0.1
    assert memory_diff < tolerance, (
        f"Memory not restored correctly: initial={initial_memory_gb:.2f} GB, "
        f"after_restore={memory_after_restore_gb:.2f} GB, diff={memory_diff:.2f} GB"
    )

    # Calculate restore speed
    if restore_time > 0 and memory_restored_gb > 0:
        restore_speed_gbps = memory_restored_gb / restore_time
        print(f"[TMS] Restore speed: {restore_speed_gbps:.2f} GB/s")

        # Restore speed should also be reasonable
        assert restore_speed_gbps > 0.5, (
            f"Restore speed {restore_speed_gbps:.2f} GB/s is too slow"
        )


# =============================================================================
# Megatron Offload Tests
# =============================================================================


def test_megatron_offload_and_restore(megatron_engine_with_offload):
    """Test Megatron offload releases memory and restore recovers it correctly.

    This test validates:
    1. Memory is released during offload
    2. Transfer speed is reasonable (> 0.5 GB/s)
    3. Memory is restored correctly after wake_up
    """
    engine = megatron_engine_with_offload

    # Measure initial memory
    current_platform.synchronize()
    initial_memory_gb = get_gpu_memory_allocated_gb()
    print(f"[Megatron] Initial GPU memory: {initial_memory_gb:.2f} GB")

    # === Test Offload ===
    start_time = time.perf_counter()
    engine.sleep()
    offload_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_offload_gb = get_gpu_memory_allocated_gb()
    memory_released_gb = initial_memory_gb - memory_after_offload_gb

    print(
        f"[Megatron] After offload: {memory_after_offload_gb:.2f} GB "
        f"(released {memory_released_gb:.2f} GB in {offload_time:.3f}s)"
    )

    # Assert memory was released
    assert memory_released_gb > 0.1, (
        f"Expected memory release, but only {memory_released_gb:.2f} GB was released"
    )

    # Calculate and verify transfer speed
    if offload_time > 0:
        offload_speed_gbps = memory_released_gb / offload_time
        print(f"[Megatron] Offload speed: {offload_speed_gbps:.2f} GB/s")

        assert offload_speed_gbps > 0.5, (
            f"Offload speed {offload_speed_gbps:.2f} GB/s is too slow"
        )

    # === Test Restore ===
    start_time = time.perf_counter()
    engine.wake_up()
    restore_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_restore_gb = get_gpu_memory_allocated_gb()
    memory_restored_gb = memory_after_restore_gb - memory_after_offload_gb

    print(
        f"[Megatron] After restore: {memory_after_restore_gb:.2f} GB "
        f"(restored {memory_restored_gb:.2f} GB in {restore_time:.3f}s)"
    )

    # Memory should be restored to approximately initial level
    memory_diff = abs(memory_after_restore_gb - initial_memory_gb)
    tolerance = initial_memory_gb * 0.1
    assert memory_diff < tolerance, (
        f"Memory not restored correctly: initial={initial_memory_gb:.2f} GB, "
        f"after_restore={memory_after_restore_gb:.2f} GB, diff={memory_diff:.2f} GB"
    )

    # Calculate restore speed
    if restore_time > 0 and memory_restored_gb > 0:
        restore_speed_gbps = memory_restored_gb / restore_time
        print(f"[Megatron] Restore speed: {restore_speed_gbps:.2f} GB/s")

        assert restore_speed_gbps > 0.5, (
            f"Restore speed {restore_speed_gbps:.2f} GB/s is too slow"
        )
