import subprocess

import pytest

from areal.utils.network import find_free_ports


def _run_test_with_torchrun(n_gpus: int):
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/torchrun/run_fsdp_ulysses_train_batch.py",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr.decode()}")


@pytest.mark.two_gpu
def test_fsdp_ulysses_train_batch_2gpu(tmp_path_factory):
    _run_test_with_torchrun(2)


@pytest.mark.one_gpu
def test_fsdp_ulysses_train_batch_1gpu(tmp_path_factory):
    _run_test_with_torchrun(1)
