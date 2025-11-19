"""Integration tests for offload functionality in FSDP and Megatron engines using TMS."""

import os
import sys
import time

import pytest
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils.network import find_free_ports
from areal.utils.tms_utils import get_tms_env_vars

_TMS_RESTARTED_MARKER = "_AREAL_TMS_RESTARTED"


def _should_restart_with_tms():
    """Check if we need to restart the process with TMS LD_PRELOAD."""

    if _TMS_RESTARTED_MARKER in os.environ:
        return False

    tms_env = get_tms_env_vars()
    return tms_env["LD_PRELOAD"] != os.environ.get("LD_PRELOAD", "")


def _restart_with_tms():
    """Restart the current process with TMS LD_PRELOAD environment variables."""
    tms_env = get_tms_env_vars()

    print("Restarting with TMS LD_PRELOAD environment...")

    # Update environment and re-execute
    new_env = os.environ.copy()
    new_env.update(tms_env)
    new_env[_TMS_RESTARTED_MARKER] = "1"

    # Use os.execve to replace current process
    try:
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)
    except OSError as e:
        print(f"Failed to restart with TMS environment: {e}", file=sys.stderr)
        sys.exit(1)


# Runs at module import time to ensure LD_PRELOAD is set before any CUDA operations
if _should_restart_with_tms():
    _restart_with_tms()

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
    # Use a conservative threshold for testing
    return 1.0


def _test_offload_and_onload(
    engine,
    engine_name: str,
    min_memory_release_gb: float = 0.1,
    memory_tolerance: float = 0.1,
    warmup_rounds: int = 3,
):
    """Common test logic for offload and onload.

    Parameters
    ----------
    engine : FSDPEngine | MegatronEngine
        Engine instance to test
    engine_name : str
        Name for logging (e.g., "FSDP", "Megatron")
    min_memory_release_gb : float
        Minimum expected memory release in GB
    memory_tolerance : float
        Tolerance ratio for memory restoration check
    warmup_rounds : int
        Number of warmup offload/restore cycles
    """
    # Measure initial memory
    current_platform.synchronize()
    initial_memory_gb = get_gpu_memory_allocated_gb()
    print(f"[{engine_name}] Initial GPU memory: {initial_memory_gb:.2f} GB")

    # Warm up
    print(f"[{engine_name}] Running {warmup_rounds} warmup cycles...")
    for _ in range(warmup_rounds):
        engine.offload()
        engine.onload()
    current_platform.synchronize()

    # === Test Offload ===
    start_time = time.perf_counter()
    engine.offload()
    offload_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_offload_gb = get_gpu_memory_allocated_gb()
    memory_released_gb = initial_memory_gb - memory_after_offload_gb

    print(
        f"[{engine_name}] After offload: {memory_after_offload_gb:.2f} GB "
        f"(released {memory_released_gb:.2f} GB in {offload_time:.3f}s)"
    )

    # Assert memory was released
    assert memory_released_gb > min_memory_release_gb, (
        f"Expected memory release > {min_memory_release_gb:.2f} GB, "
        f"but only {memory_released_gb:.2f} GB was released"
    )

    # Calculate and verify transfer speed
    if offload_time > 0:
        offload_speed_gbps = memory_released_gb / offload_time
        print(f"[{engine_name}] Offload speed: {offload_speed_gbps:.2f} GB/s")

        # Speed should be reasonable
        pcie_bandwidth = estimate_pcie_bandwidth_gbps()
        pcie_bandwidth_threshold = pcie_bandwidth * 0.4
        assert offload_speed_gbps > pcie_bandwidth_threshold, (
            f"Offload speed {offload_speed_gbps:.2f} GB/s is too slow "
            f"(expected > {pcie_bandwidth_threshold:.2f} GB/s)"
        )

    # === Test Onload ===
    start_time = time.perf_counter()
    engine.onload()
    onload_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_onload_gb = get_gpu_memory_allocated_gb()
    memory_restored_gb = memory_after_onload_gb - memory_after_offload_gb

    print(
        f"[{engine_name}] After onload: {memory_after_onload_gb:.2f} GB "
        f"(restored {memory_restored_gb:.2f} GB in {onload_time:.3f}s)"
    )

    # Memory should be restored to approximately initial level
    memory_diff = abs(memory_after_onload_gb - initial_memory_gb)
    tolerance = initial_memory_gb * memory_tolerance
    assert memory_diff < tolerance, (
        f"Memory not restored correctly: initial={initial_memory_gb:.2f} GB, "
        f"after_onload={memory_after_onload_gb:.2f} GB, diff={memory_diff:.2f} GB"
    )

    # Calculate onload speed
    if onload_time > 0 and memory_restored_gb > 0:
        onload_speed_gbps = memory_restored_gb / onload_time
        print(f"[{engine_name}] Onload speed: {onload_speed_gbps:.2f} GB/s")

        # Onload speed should also be reasonable
        assert onload_speed_gbps > pcie_bandwidth_threshold, (
            f"Onload speed {onload_speed_gbps:.2f} GB/s is too slow "
            f"(expected > {pcie_bandwidth_threshold:.2f} GB/s)"
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fsdp_engine_with_offload():
    """Create FSDP engine with TMS offload enabled."""
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(find_free_ports(1)[0]),
        }
    )

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
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(find_free_ports(1)[0]),
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
# Offload Tests
# =============================================================================


@pytest.mark.parametrize(
    "engine_fixture,engine_name",
    [
        ("fsdp_engine_with_offload", "FSDP"),
        ("megatron_engine_with_offload", "Megatron"),
    ],
)
def test_engine_offload_and_onload(request, engine_fixture, engine_name):
    """Test engine offload releases memory and onload recovers it correctly.

    This test validates:
    1. Memory is released during offload
    2. Transfer speed is reasonable
    3. Memory is restored correctly after onload

    Parametrized to test both FSDP and Megatron engines.
    """
    engine = request.getfixturevalue(engine_fixture)
    _test_offload_and_onload(
        engine=engine,
        engine_name=engine_name,
        min_memory_release_gb=0.1,
        memory_tolerance=0.1,
        warmup_rounds=3,
    )
