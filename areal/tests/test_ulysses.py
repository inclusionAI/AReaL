import subprocess

import pytest

from areal.utils.network import find_free_ports


def _run_test_with_torchrun():
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={2}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/torchrun/run_ulysses.py",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr.decode()}")


@pytest.mark.two_gpu
def test_ulysses(tmp_path_factory):
    _run_test_with_torchrun()
