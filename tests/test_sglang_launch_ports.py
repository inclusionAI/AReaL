# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from areal.engine import sglang_remote


@pytest.fixture
def cuda_platform(monkeypatch):
    """Exercise CUDA namespace rules without requiring a physical accelerator."""
    monkeypatch.setattr(
        sglang_remote,
        "current_platform",
        SimpleNamespace(device_control_env_var="CUDA_VISIBLE_DEVICES"),
    )


@pytest.mark.parametrize(
    "communication_args",
    [
        {},
        {"nccl_port": None},
        {"nccl_port": 19001},
        {"dist_init_addr": "localhost:19002"},
    ],
)
def test_launch_server_allocates_only_unspecified_communication_port(
    monkeypatch, communication_args
):
    build_cmd = MagicMock(return_value=["sglang-server"])
    monkeypatch.setattr(sglang_remote.SGLangConfig, "build_cmd_from_args", build_cmd)
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", MagicMock())
    choose_port = MagicMock(return_value=[19003])
    monkeypatch.setattr(sglang_remote, "find_free_ports", choose_port)
    args = {"port": 19000, **communication_args}

    sglang_remote.SGLangBackend().launch_server(args)

    launched_args = build_cmd.call_args.args[0]
    if communication_args.get("nccl_port") or communication_args.get("dist_init_addr"):
        choose_port.assert_not_called()
        for key, value in communication_args.items():
            assert launched_args[key] == value
    else:
        choose_port.assert_called_once_with(1, exclude_ports={19000})
        assert launched_args["nccl_port"] == 19003
    assert launched_args["port"] == 19000


@pytest.mark.parametrize("isolated", [False, True])
def test_colocated_cyclic_worker_order_assigns_all_64_ranks(
    monkeypatch, isolated, cuda_platform
):
    """Two TP4 servers per node keep distinct ranks under cyclic Slurm placement."""
    from areal.engine.awex.plugins.sglang_plugin import _resolve_transfer_rank

    build_cmd = MagicMock(return_value=["sglang-server"])
    popen = MagicMock()
    monkeypatch.setattr(sglang_remote.SGLangConfig, "build_cmd_from_args", build_cmd)
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", popen)
    monkeypatch.delenv("AWEX_TRANSFER_RANK", raising=False)
    monkeypatch.delenv("AREAL_SGLANG_BASE_GPU_ID", raising=False)
    ranks = []

    for worker_index in range(16):
        node_id, local_id = worker_index % 8, worker_index // 8
        monkeypatch.setenv("SLURM_LOCALID", str(local_id))
        visible_devices = (
            ",".join(str(local_id * 4 + i) for i in range(4)) if isolated else ""
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
        sglang_remote.SGLangBackend().launch_server(
            {
                "nccl_port": 19000,
                "awex_colocate_mode": True,
                "_awex_gpus_per_server": 4,
                "base_gpu_id": (worker_index % 2) * 4,
            }
        )
        args = build_cmd.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        assert args["base_gpu_id"] == (0 if isolated else local_id * 4)
        assert env["AWEX_TRANSFER_RANK_BASE"] == str(local_id * 4)
        monkeypatch.setenv("AWEX_TRANSFER_RANK_BASE", env["AWEX_TRANSFER_RANK_BASE"])
        for tp_rank in range(4):
            ranks.append(
                _resolve_transfer_rank(
                    infer_world_size=64,
                    gpu_id=args["base_gpu_id"] + tp_rank,
                    node_id=node_id,
                    nnodes=8,
                    instance_world_size=4,
                    server_base_gpu_id=args["base_gpu_id"],
                )
            )

    assert sorted(ranks) == list(range(64))


@pytest.mark.parametrize(
    ("env", "base", "fallback", "expected"),
    [
        ({"CUDA_VISIBLE_DEVICES": "2,5,6,7"}, 0, 4, 4),
        ({"CUDA_VISIBLE_DEVICES": "2,5,6,7", "SLURM_LOCALID": "1"}, 0, 0, 4),
        ({"AREAL_SGLANG_BASE_GPU_ID": "4"}, 4, 0, 4),
        ({}, 4, 4, 4),
    ],
)
def test_colocated_transfer_base_preserves_device_namespace(
    env, base, fallback, expected, cuda_platform
):
    """Device isolation and explicit placement keep a dense transfer coordinate."""
    assert (
        sglang_remote._resolve_colocated_transfer_rank_base(env, 4, base, fallback)
        == expected
    )


def test_colocated_transfer_base_rejects_negative_isolated_slurm_slot(cuda_platform):
    """An isolated namespace must not hide an invalid Slurm task slot."""
    with pytest.raises(ValueError, match="SLURM_LOCALID must be non-negative"):
        sglang_remote._resolve_colocated_transfer_rank_base(
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_LOCALID": "-1"}, 4, 0, 0
        )


def test_separated_launch_preserves_explicit_transfer_base(monkeypatch):
    """The colocate placement fix does not replace separated worker ranks."""
    popen = MagicMock()
    monkeypatch.setattr(
        sglang_remote.SGLangConfig, "build_cmd_from_args", lambda args: ["server"]
    )
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", popen)

    sglang_remote.SGLangBackend().launch_server(
        {"nccl_port": 19000, "_awex_transfer_rank_base": 12}
    )

    assert popen.call_args.kwargs["env"]["AWEX_TRANSFER_RANK_BASE"] == "12"
