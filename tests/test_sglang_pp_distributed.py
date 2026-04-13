"""Distributed tests for sglang PP pipeline parallelism.

These tests verify that per-PP-rank NCCL weight update groups work correctly
in a multi-GPU distributed setting for all three training engine types:
Megatron, FSDP, and Archon.

Test matrix:
  - test_pp2_tp2_weight_group_init:   4 GPUs, PP=2 TP=2 (per engine)
  - test_dp2_pp2_tp2_weight_sync:     8 GPUs, DP=2 PP=2 TP=2 (per engine)
  - test_pp1_backward_compatible:     4 GPUs, PP=1 DP=2 TP=2 (per engine)

Requires 4-8 GPUs to run.
"""
import subprocess

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


# ---------------------------------------------------------------------------
# Engine types for parametrization
# ---------------------------------------------------------------------------

ENGINE_TYPES = ["megatron", "fsdp", "archon"]


# ---------------------------------------------------------------------------
# Helper: run a test via torchrun
# ---------------------------------------------------------------------------

def _run_test_with_torchrun(
    alloc_mode: str,
    test_type: str,
    output: str,
    gen_pp_size: int = 1,
    engine_type: str = "megatron",
):
    """Launch the distributed test worker under torchrun.

    Args:
        alloc_mode: Backend allocation string (e.g. "megatron:d1p2t2").
        test_type: One of "group_init" or "weight_sync".
        output: Path to file where the worker writes "Passed" or "Failed".
        gen_pp_size: Inference-side PP size (used by the worker to construct
            a gen_allocation for validation).
        engine_type: One of "megatron", "fsdp", "archon".
    """
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_sglang_pp_weight_sync.py",
        f"--backend={alloc_mode}",
        f"--output={output}",
        f"--test_type={test_type}",
        f"--gen_pp_size={gen_pp_size}",
        f"--engine_type={engine_type}",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"torchrun failed (returncode={e.returncode}):\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        )
    except subprocess.TimeoutExpired:
        pytest.fail("torchrun timed out after 600s")

    with open(output) as f:
        content = f.read().strip()
    assert content == "Passed", f"Test worker reported: {content}"


# ===================================================================== #
#  PP=2, TP=2 group initialization (4 GPUs)                             #
# ===================================================================== #

@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp2_tp2_weight_group_init(tmp_path_factory, engine_type):
    """Test per-PP-rank NCCL group initialization with PP=2, TP=2.

    Verifies:
      - Each PP rank gets a distinct group name ("update_weight_group_{pp_rank}").
      - is_pipeline_parallel_head detection is correct for each rank.
      - Per-PP-rank world sizes are computed correctly.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    alloc_mode = f"{engine_type}:d1p2t2"
    output = tmp_path_factory.mktemp("test_output") / f"pp2_tp2_group_init_{engine_type}.out"
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="group_init",
        output=str(output),
        gen_pp_size=2,
        engine_type=engine_type,
    )


# ===================================================================== #
#  DP=2, PP=2, TP=2 weight sync (8 GPUs)                               #
# ===================================================================== #

@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_dp2_pp2_tp2_weight_sync(tmp_path_factory, engine_type):
    """End-to-end weight sync test with DP=2, PP=2, TP=2 (8 GPUs).

    Verifies:
      - PP ranks hold different parameter name sets.
      - Weight sync completes without errors.
    """
    if current_platform.device_count() < 8:
        pytest.skip("Requires 8 GPUs for DP=2, PP=2, TP=2")

    alloc_mode = f"{engine_type}:d2p2t2"
    output = tmp_path_factory.mktemp("test_output") / f"dp2_pp2_tp2_sync_{engine_type}.out"
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="weight_sync",
        output=str(output),
        gen_pp_size=2,
        engine_type=engine_type,
    )


# ===================================================================== #
#  PP=1 backward compatibility (4 GPUs)                                 #
# ===================================================================== #

@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp1_backward_compatible(tmp_path_factory, engine_type):
    """Verify PP=1 still works correctly (backward compatibility).

    Verifies:
      - Single weight update group is created.
      - Weight sync completes without errors.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    alloc_mode = f"{engine_type}:d2p1t2"
    output = tmp_path_factory.mktemp("test_output") / f"pp1_backward_{engine_type}.out"
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="weight_sync",
        output=str(output),
        gen_pp_size=1,
        engine_type=engine_type,
    )
