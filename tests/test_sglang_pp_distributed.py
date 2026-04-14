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
import os
import signal
import subprocess
import sys
import threading

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


# ---------------------------------------------------------------------------
# Engine types for parametrization
# ---------------------------------------------------------------------------

ENGINE_TYPES = ["megatron", "fsdp", "archon"]


# ---------------------------------------------------------------------------
# Helper: build a valid allocation string per engine type
# ---------------------------------------------------------------------------

def _get_alloc_mode(engine_type: str, dp: int, pp: int, tp: int) -> str:
    """Build a valid allocation mode string for the given engine type.

    FSDP training does not support pipeline parallelism (PP > 1).
    When PP > 1 is requested for FSDP, the PP dimension is folded into DP
    so that the total GPU count remains the same (dp * pp * tp).
    The inference-side PP is controlled separately via ``gen_pp_size``.
    """
    if engine_type == "fsdp" and pp > 1:
        return f"fsdp:d{dp * pp}t{tp}"
    return f"{engine_type}:d{dp}p{pp}t{tp}"


# ---------------------------------------------------------------------------
# Helper: stream subprocess output in a background thread
# ---------------------------------------------------------------------------

def _reader_thread(pipe, collected, prefix):
    """Read lines from *pipe*, print with a prefix, and append to *collected*."""
    try:
        for line in pipe:
            sys.stdout.write(f"  [{prefix}] {line}")
            sys.stdout.flush()
            collected.append(line)
    except ValueError:
        pass  # pipe closed


# ---------------------------------------------------------------------------
# Helper: kill an entire process group (best-effort)
# ---------------------------------------------------------------------------

def _kill_process_tree(proc):
    """Send SIGKILL to the process group rooted at *proc*.

    torchrun spawns multiple worker processes.  Killing only the main
    process may leave workers alive, holding GPU memory and NCCL state.
    Using ``start_new_session=True`` when spawning the process ensures
    the entire tree shares a single process group that can be killed
    together.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


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

    tag = f"{engine_type}/{test_type}"
    print(f"\n{'=' * 60}", flush=True)
    print(f"[torchrun] {tag}  alloc={alloc_mode}  gpus={n_gpus}", flush=True)
    print(f"[torchrun] cmd: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ``start_new_session`` creates a dedicated process group so that
    # ``_kill_process_tree`` can reliably terminate ALL torchrun workers
    # on failure or timeout, preventing leaked GPU resources from blocking
    # subsequent tests.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    collected: list[str] = []
    reader = threading.Thread(
        target=_reader_thread, args=(proc.stdout, collected, tag), daemon=True
    )
    reader.start()

    try:
        returncode = proc.wait(timeout=300)
        reader.join(timeout=5)

        if returncode != 0:
            tail = "".join(collected[-50:])
            pytest.fail(
                f"torchrun failed (returncode={returncode}):\n{tail}"
            )
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        reader.join(timeout=5)
        tail = "".join(collected[-50:])
        pytest.fail(f"torchrun timed out after 300s. Last output:\n{tail}")
    except Exception:
        # Ensure cleanup on any unexpected error.
        _kill_process_tree(proc)
        raise

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

    alloc_mode = _get_alloc_mode(engine_type, dp=1, pp=2, tp=2)
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

    alloc_mode = _get_alloc_mode(engine_type, dp=2, pp=2, tp=2)
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

    alloc_mode = _get_alloc_mode(engine_type, dp=2, pp=1, tp=2)
    output = tmp_path_factory.mktemp("test_output") / f"pp1_backward_{engine_type}.out"
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="weight_sync",
        output=str(output),
        gen_pp_size=1,
        engine_type=engine_type,
    )
