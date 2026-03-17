import os
import socket
import subprocess
import sys

import pytest


pytest.importorskip("torch")


def _can_bind(host: str, family: int) -> bool:
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
        return True
    except OSError:
        return False


def _get_free_port(host: str, family: int) -> int:
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def test_ipv6_tcp_init_method_end_to_end(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Run this test on Linux hosts")
    if not socket.has_ipv6:
        pytest.skip("IPv6 is not supported on this host")

    if not _can_bind("::1", socket.AF_INET6):
        pytest.skip("IPv6 loopback (::1) is not available on this host")

    master_port = _get_free_port("::1", socket.AF_INET6)
    init_port = _get_free_port("::1", socket.AF_INET6)
    output = tmp_path / "ipv6_tcp_init_method.out"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--nnodes=1",
                "--master-addr=::1",
                f"--master-port={master_port}",
                "tests/torchrun/run_ipv6_tcp_init_method.py",
                "--init-host=::1",
                f"--init-port={init_port}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
            timeout=60,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"torchrun failed: {e.stderr}\n{e.stdout}")

    assert output.read_text().strip() == "Passed"


def test_ipv4_tcp_init_method_end_to_end(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Run this test on Linux hosts")
    if not _can_bind("127.0.0.1", socket.AF_INET):
        pytest.skip("IPv4 loopback (127.0.0.1) is not available on this host")

    master_port = _get_free_port("127.0.0.1", socket.AF_INET)
    init_port = _get_free_port("127.0.0.1", socket.AF_INET)
    output = tmp_path / "ipv4_tcp_init_method.out"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--nnodes=1",
                "--master-addr=127.0.0.1",
                f"--master-port={master_port}",
                "tests/torchrun/run_ipv6_tcp_init_method.py",
                "--init-host=127.0.0.1",
                f"--init-port={init_port}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
            timeout=60,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"torchrun failed: {e.stderr}\n{e.stdout}")

    assert output.read_text().strip() == "Passed"
