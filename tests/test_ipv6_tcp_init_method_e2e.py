import importlib.util
import socket
import sys
from datetime import timedelta
from pathlib import Path

import pytest


pytest.importorskip("torch")
torch = pytest.importorskip("torch")


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


def _load_network_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "areal" / "utils" / "network.py"
    spec = importlib.util.spec_from_file_location("areal_utils_network", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _worker_init_pg_and_allreduce(
    rank: int,
    world_size: int,
    init_method: str,
    output_path: str,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=30),
    )
    try:
        x = torch.tensor([rank + 1], dtype=torch.int64)
        torch.distributed.all_reduce(x)
        expected = world_size * (world_size + 1) // 2
        ok = int(x.item()) == expected
        if rank == 0:
            with open(output_path, "w") as f:
                f.write(
                    "Passed"
                    if ok
                    else f"Failed: got={int(x.item())} expected={expected}"
                )
        if not ok:
            raise RuntimeError(
                f"all_reduce mismatch: got={int(x.item())} expected={expected}"
            )
    finally:
        torch.distributed.destroy_process_group()


def _run_e2e(*, host: str, family: int, tmp_path, output_name: str) -> None:
    network = _load_network_module()
    output = tmp_path / output_name

    last_err: Exception | None = None
    for _ in range(5):
        port = _get_free_port(host, family)
        init_method = f"tcp://{network.format_host_for_url(host)}:{port}"
        try:
            torch.multiprocessing.start_processes(
                _worker_init_pg_and_allreduce,
                args=(2, init_method, str(output)),
                nprocs=2,
                start_method="spawn",
                join=True,
            )
            break
        except Exception as e:
            last_err = e
    else:
        raise AssertionError(f"distributed init failed after retries: {repr(last_err)}")

    assert output.read_text().strip() == "Passed"


def test_ipv6_tcp_init_method_end_to_end(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Run this test on Linux hosts")
    if not socket.has_ipv6:
        pytest.skip("IPv6 is not supported on this host")

    if not _can_bind("::1", socket.AF_INET6):
        pytest.skip("IPv6 loopback (::1) is not available on this host")
    _run_e2e(
        host="::1",
        family=socket.AF_INET6,
        tmp_path=tmp_path,
        output_name="ipv6_tcp_init_method.out",
    )


def test_ipv4_tcp_init_method_end_to_end(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Run this test on Linux hosts")
    if not _can_bind("127.0.0.1", socket.AF_INET):
        pytest.skip("IPv4 loopback (127.0.0.1) is not available on this host")
    _run_e2e(
        host="127.0.0.1",
        family=socket.AF_INET,
        tmp_path=tmp_path,
        output_name="ipv4_tcp_init_method.out",
    )
