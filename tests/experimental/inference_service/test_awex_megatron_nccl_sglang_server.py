"""Controller tests for awex Megatron <-> AReaL SGLang NCCL integration.

Pattern follows Archon distributed tests:
1) A generic torchrun script contains SPMD logic.
2) This file acts as controller that launches torchrun with different parallel
   allocations and verifies the output marker file.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from tests.experimental.inference_service.integration_utils import get_test_model_path

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_awex_sglang_torchrun(
    n_gpus: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    output: str,
):
    model_path = get_test_model_path()
    max_trials = 4
    for trial in range(1, max_trials + 1):
        port = find_free_ports(1)[0]
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master_port={port}",
            "tests/experimental/inference_service/torchrun/run_awex_megatron_sglang_nccl.py",
            f"--dp-size={dp_size}",
            f"--tp-size={tp_size}",
            f"--pp-size={pp_size}",
            f"--model-path={model_path}",
            f"--output={output}",
            "--health-timeout=240",
            "--rpc-timeout=240",
        ]

        print(
            f"[controller] trial {trial}/{max_trials} launching: {' '.join(cmd)}",
            flush=True,
        )
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                timeout=1200,
            )
            break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            if trial >= max_trials:
                if isinstance(exc, subprocess.TimeoutExpired):
                    pytest.fail(
                        f"Torchrun timed out after {exc.timeout}s on trial {trial}/{max_trials}: {' '.join(cmd)}"
                    )
                else:
                    pytest.fail(
                        f"Torchrun failed with exit code {exc.returncode} on trial {trial}/{max_trials}. "
                        f"See streamed logs above. Command: {' '.join(cmd)}"
                    )
            print(
                f"[controller] trial {trial}/{max_trials} failed ({type(exc).__name__}); retrying with a fresh master_port",
                flush=True,
            )

    with open(output, encoding="utf-8") as f:
        result = f.read().strip()
    assert result == "Passed", f"Integration failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_awex_megatron_sglang_nccl_1gpu(tmp_path_factory):
    """Single-rank baseline (dp=1,tp=1,pp=1)."""
    if current_platform.device_count() < 1:
        pytest.skip("This test requires at least 1 GPU")
    output = tmp_path_factory.mktemp("test_output") / "awex_sglang_nccl_1gpu.out"
    _run_awex_sglang_torchrun(1, 1, 1, 1, str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_awex_megatron_sglang_nccl_dp2(tmp_path_factory):
    """DP2 allocation (dp=2,tp=1,pp=1)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "awex_sglang_nccl_dp2.out"
    _run_awex_sglang_torchrun(2, 2, 1, 1, str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_awex_megatron_sglang_nccl_tp2(tmp_path_factory):
    """TP2 allocation (dp=1,tp=2,pp=1)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "awex_sglang_nccl_tp2.out"
    _run_awex_sglang_torchrun(2, 1, 2, 1, str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_awex_megatron_sglang_nccl_pp2(tmp_path_factory):
    """PP2 allocation (dp=1,tp=1,pp=2)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "awex_sglang_nccl_pp2.out"
    _run_awex_sglang_torchrun(2, 1, 1, 2, str(output))
