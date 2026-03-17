import socket

import pytest


def _can_bind_ipv4_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


def _can_bind_ipv6_loopback() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ip_stack() -> dict[str, bool]:
    ipv4 = _can_bind_ipv4_loopback()
    ipv6 = _can_bind_ipv6_loopback()
    return {"ipv4": ipv4, "ipv6": ipv6}
