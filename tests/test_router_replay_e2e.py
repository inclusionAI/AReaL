"""End-to-end tests for Router Replay (R3) on MoE models.

Launches the distributed runner under ``torchrun`` with 4–8 GPUs and exercises
the R3 pipeline on Moonlight-16B-A3B (fallback: Qwen3-30B-A3B).  Each test
spawns a dedicated process group so they are safe to run sequentially.

The test surface mirrors SkyRL's ``tests/backends/skyrl_train/gpu/gpu_ci/
megatron/test_router_replay.py`` in three ways:

1. ``patch_plumbing``: lightweight sanity for the RouterReplay instance
   registration count.
2. ``forward_replay``: synthetic ``routed_experts`` side-channel through
   ``engine.forward``.
3. ``forward_backward``: full ``engine.train_batch`` round with non-zero
   ``advantages`` / ``rollout_expert_indices`` (SkyRL's
   ``test_forward_backward`` equivalent) — verifies the training loss is
   finite and non-zero when R3 is enabled.

These tests are marked ``slow``/``multi_gpu`` and will be skipped in CI
by default; run with ``pytest -m multi_gpu -k r3_e2e``.
"""

from __future__ import annotations

import subprocess

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_e2e(
    model_type: str,
    alloc_mode: str,
    test_type: str,
    output: str,
    timeout_sec: int = 1800,
):
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_router_replay_distributed.py",
                f"--model_type={model_type}",
                f"--backend={alloc_mode}",
                f"--test_type={test_type}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"R3 E2E subprocess failed.\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        )
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"R3 E2E test failed: {result}"


# ---------------------------------------------------------------------------
# 4-GPU: Moonlight single-stage (TP=4, EP=4, PP=1)
# ---------------------------------------------------------------------------


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_moonlight_patch_plumbing(tmp_path_factory):
    """R3: RouterReplay instance count must match local MoE layer count.

    Exercises TP=4/EP=4 on 4 GPUs, single PP stage.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Moonlight R3 patch plumbing requires >= 4 GPUs")
    out = tmp_path_factory.mktemp("r3") / "moonlight_patch.out"
    _run_e2e(
        model_type="moonlight",
        alloc_mode="megatron:(attn:d1p1t4|ffn:d1p1t1e4)",
        test_type="patch_plumbing",
        output=str(out),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_moonlight_forward_replay(tmp_path_factory):
    """R3: forward with side-channelled routed_experts.

    Exercises TP=4/EP=4 on 4 GPUs; runs ``engine.forward()`` once with a
    synthetic routed_experts tensor and verifies the side-channel is
    consumed and the R3 state is cleared at the end.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Moonlight R3 forward replay requires >= 4 GPUs")
    out = tmp_path_factory.mktemp("r3") / "moonlight_forward.out"
    _run_e2e(
        model_type="moonlight",
        alloc_mode="megatron:(attn:d1p1t4|ffn:d1p1t1e4)",
        test_type="forward_replay",
        output=str(out),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_moonlight_forward_backward(tmp_path_factory):
    """R3 full forward_backward (SkyRL ``test_forward_backward`` parity).

    Runs ``engine.train_batch`` with a synthetic ``rollout_expert_indices``
    tensor and non-zero ``advantages`` / ``rollout_logprobs``.  Asserts the
    loss is finite, non-zero, and that all RouterReplay instances have had
    their action cleared upon completion.  Exercises TP=4/EP=4 on 4 GPUs.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Moonlight R3 forward_backward requires >= 4 GPUs")
    out = tmp_path_factory.mktemp("r3") / "moonlight_fb.out"
    _run_e2e(
        model_type="moonlight",
        alloc_mode="megatron:(attn:d1p1t4|ffn:d1p1t1e4)",
        test_type="forward_backward",
        output=str(out),
    )


# ---------------------------------------------------------------------------
# 8-GPU: Moonlight with PP=2 + TP=4 + EP=4 (reference config)
# ---------------------------------------------------------------------------


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_moonlight_pp2_tp4_ep4(tmp_path_factory):
    """R3 patch plumbing under PP=2 + TP=4 + EP=4 (8 GPUs).

    Mirrors ``examples/math/moonlight_16b_a3b_gsm8k_grpo_megatron.yaml``.
    Verifies that per-PP-rank RouterReplay instance counts still match
    the local MoE-layer count Megatron builds.
    """
    if current_platform.device_count() < 8:
        pytest.skip("Moonlight R3 PP=2 config requires 8 GPUs")
    out = tmp_path_factory.mktemp("r3") / "moonlight_pp2_patch.out"
    _run_e2e(
        model_type="moonlight",
        alloc_mode="megatron:(attn:d1p2t4|ffn:d1p2t1e4)",
        test_type="patch_plumbing",
        output=str(out),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_moonlight_pp2_tp4_ep4_forward_backward(tmp_path_factory):
    """R3 full train_batch under PP=2 + TP=4 + EP=4 (8 GPUs).

    The SkyRL R3 ``max_parallelism`` case (TP=2/PP=2/CP=2/EP=4) isn't
    directly portable because AReaL's allocation syntax doesn't encode
    CP on the FFN side; we use the 8-GPU AReaL reference layout
    (PP=2, TP=4, EP=4) instead and verify the full forward-backward
    round-trip.
    """
    if current_platform.device_count() < 8:
        pytest.skip("Moonlight R3 PP=2 forward_backward requires 8 GPUs")
    out = tmp_path_factory.mktemp("r3") / "moonlight_pp2_fb.out"
    _run_e2e(
        model_type="moonlight",
        alloc_mode="megatron:(attn:d1p2t4|ffn:d1p2t1e4)",
        test_type="forward_backward",
        output=str(out),
    )


# ---------------------------------------------------------------------------
# 4-GPU: Qwen3-30B-A3B fallback (runs if Moonlight weights are unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_qwen3moe_fallback(tmp_path_factory):
    """Fallback path: Qwen3-30B-A3B MoE when Moonlight is unavailable.

    Same 4-GPU TP=2 CP=2 EP=4 layout as ``test_qwen3moe_expert_parallel``
    in ``test_megatron_engine_distributed.py``.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Qwen3 MoE R3 requires >= 4 GPUs")
    out = tmp_path_factory.mktemp("r3") / "qwen3moe_fallback.out"
    _run_e2e(
        model_type="qwen3moe",
        alloc_mode="megatron:(attn:d1p1t2c2|ffn:d1p1t1e4)",
        test_type="patch_plumbing",
        output=str(out),
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_r3_e2e_qwen3moe_forward_backward(tmp_path_factory):
    """Qwen3-30B-A3B MoE R3 full train_batch fallback (4 GPUs)."""
    if current_platform.device_count() < 4:
        pytest.skip("Qwen3 MoE R3 forward_backward requires >= 4 GPUs")
    out = tmp_path_factory.mktemp("r3") / "qwen3moe_fb.out"
    _run_e2e(
        model_type="qwen3moe",
        alloc_mode="megatron:(attn:d1p1t2c2|ffn:d1p1t1e4)",
        test_type="forward_backward",
        output=str(out),
    )
