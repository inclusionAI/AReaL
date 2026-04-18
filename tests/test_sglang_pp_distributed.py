"""End-to-end distributed tests for sglang PP pipeline parallelism.

These tests verify that the per-PP-rank NCCL weight update groups work
correctly by running actual training loops (``gsm8k_rl.py``) with all
three training engines and ``sglang PP=2`` on the inference side.

Each test mirrors a command the author has already validated manually:

  - **Megatron**: actor.backend="megatron:d1p2t2", rollout.backend="sglang:d1p2t2"
  - **FSDP**:     actor.backend="fsdp:d2p1t2",     rollout.backend="sglang:d1p2t2"
  - **Archon**:   actor.backend="archon:d2p1t2",    rollout.backend="sglang:d1p2t2"

All tests use 4 GPUs and ``total_train_steps=2`` for fast execution.

Test matrix (9 tests = 3 engines × 3 test types):
  - test_pp_e2e_train:           Run 2 training steps end-to-end (per engine).
  - test_pp_weight_group_init:   Verify per-PP-rank group naming / head detection
                                 via torchrun worker (per engine).
  - test_pp_weight_sync:         Verify PP weight sync arithmetic / distributed
                                 broadcast via torchrun worker (per engine).

Requires 4 GPUs to run.
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
# Engine configurations (mirrors the user's working commands exactly)
# ---------------------------------------------------------------------------

ENGINE_CONFIGS = {
    "megatron": {
        "config_yaml": "examples/math/gsm8k_grpo_megatron.yaml",
        "actor_backend": "megatron:d1p2t2",
        "rollout_backend": "sglang:d1p2t2",
    },
    "fsdp": {
        "config_yaml": "examples/math/gsm8k_grpo.yaml",
        "actor_backend": "fsdp:d2p1t2",
        "rollout_backend": "sglang:d1p2t2",
    },
    "archon": {
        "config_yaml": "examples/math/gsm8k_grpo.yaml",
        "actor_backend": "archon:d2p1t2",
        "rollout_backend": "sglang:d1p2t2",
    },
}

ENGINE_TYPES = list(ENGINE_CONFIGS.keys())


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

    The training script spawns sglang servers and torchrun workers.
    Killing only the main process may leave children alive.
    ``start_new_session=True`` ensures the entire tree shares a single
    process group that can be killed together.
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
# Helper: run a subprocess with streaming output, timeout, and cleanup
# ---------------------------------------------------------------------------

def _run_subprocess(cmd, tag, timeout=600):
    """Run *cmd* in a subprocess with streaming output and reliable cleanup.

    Args:
        cmd: Command list to pass to ``Popen``.
        tag: Short label used as prefix in log output.
        timeout: Maximum seconds to wait (default: 600).

    Raises:
        pytest.fail: On non-zero exit code or timeout.
    """
    print(f"\n{'=' * 72}", flush=True)
    print(f"[{tag}] cmd: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 72}", flush=True)

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
        returncode = proc.wait(timeout=timeout)
        reader.join(timeout=10)

        if returncode != 0:
            tail = "".join(collected[-80:])
            pytest.fail(
                f"[{tag}] process failed (returncode={returncode}):\n{tail}"
            )
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        reader.join(timeout=10)
        tail = "".join(collected[-80:])
        pytest.fail(f"[{tag}] timed out after {timeout}s. Last output:\n{tail}")
    except Exception:
        _kill_process_tree(proc)
        raise


# ===================================================================== #
#  E2E training test (gsm8k_rl.py with 2 steps, per engine)            #
# ===================================================================== #

@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp_e2e_train(tmp_path_factory, engine_type):
    """End-to-end training test: run gsm8k_rl.py for 2 steps.

    This mirrors the user's working commands exactly:
      - megatron: gsm8k_grpo_megatron.yaml, actor=megatron:d1p2t2, rollout=sglang:d1p2t2
      - fsdp:     gsm8k_grpo.yaml,          actor=fsdp:d2p1t2,     rollout=sglang:d1p2t2
      - archon:   gsm8k_grpo.yaml,          actor=archon:d2p1t2,   rollout=sglang:d1p2t2

    Verifies:
      - sglang server starts with PP=2 successfully.
      - Training engine creates per-PP-rank weight update groups.
      - At least 2 training steps complete without error.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    cfg = ENGINE_CONFIGS[engine_type]
    tmp_dir = str(tmp_path_factory.mktemp(f"e2e_{engine_type}"))

    cmd = [
        "python3", "examples/math/gsm8k_rl.py",
        "--config", cfg["config_yaml"],
        f"scheduler.type=local",
        f"actor.backend={cfg['actor_backend']}",
        f"rollout.backend={cfg['rollout_backend']}",
        f"total_train_steps=2",
        f"total_train_epochs=1",
        f"cluster.fileroot={tmp_dir}",
        f"cluster.name_resolve.nfs_record_root={tmp_dir}/name_resolve",
        f"saver.freq_steps=null",
        f"saver.freq_epochs=null",
        f"evaluator.freq_steps=null",
        f"evaluator.freq_epochs=null",
        f"recover.mode=disabled",
        f"stats_logger.wandb.mode=disabled",
    ]

    _run_subprocess(cmd, tag=f"e2e/{engine_type}", timeout=600)


# ===================================================================== #
#  Torchrun-based protocol tests (group_init / weight_sync)             #
# ===================================================================== #

def _get_alloc_mode(engine_type: str) -> str:
    """Return the training-side allocation mode from the engine config."""
    return ENGINE_CONFIGS[engine_type]["actor_backend"]


def _run_test_with_torchrun(
    alloc_mode: str,
    test_type: str,
    output: str,
    gen_pp_size: int = 2,
    engine_type: str = "megatron",
):
    """Launch the distributed test worker under torchrun.

    Args:
        alloc_mode: Training engine allocation string (e.g. "megatron:d1p2t2").
        test_type: One of "group_init" or "weight_sync".
        output: Path to file where the worker writes "Passed" or "Failed".
        gen_pp_size: Inference-side PP size (always 2 for sglang:d1p2t2).
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
    _run_subprocess(cmd, tag=tag, timeout=300)

    with open(output) as f:
        content = f.read().strip()
    assert content == "Passed", f"Test worker reported: {content}"


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp_weight_group_init(tmp_path_factory, engine_type):
    """Verify per-PP-rank NCCL group naming, head detection, and world sizes.

    Uses a lightweight torchrun worker that validates the protocol-level
    invariants all three engines share, without instantiating actual
    training engines (avoids heavy dependencies like mbridge, flash_attn).

    Checks:
      - Group name follows ``update_weight_group_{pp_rank}`` convention.
      - PP head detection logic is correct for each engine type.
      - Per-PP world size computation: ``gen_world_size // gen_pp_size``.
      - ``build_init_weights_group_request`` payload has correct fields.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    alloc_mode = _get_alloc_mode(engine_type)
    output = str(
        tmp_path_factory.mktemp("test_output")
        / f"group_init_{engine_type}.out"
    )
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="group_init",
        output=output,
        gen_pp_size=2,
        engine_type=engine_type,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp_weight_sync(tmp_path_factory, engine_type):
    """Verify allocation parsing, PP arithmetic, and distributed broadcast.

    Uses a lightweight torchrun worker that validates:
      - Allocation string parses to correct dp/pp/tp dimensions.
      - FSDP PP-DP folding: training PP is folded into DP.
      - Simulated layer partitioning across PP ranks (non-overlapping).
      - NCCL all-reduce and per-PP-rank broadcast complete successfully.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    alloc_mode = _get_alloc_mode(engine_type)
    output = str(
        tmp_path_factory.mktemp("test_output")
        / f"weight_sync_{engine_type}.out"
    )
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="weight_sync",
        output=output,
        gen_pp_size=2,
        engine_type=engine_type,
    )
