import random
import socket
from typing import List, Set


def gethostname():
    return socket.gethostname()


def gethostip():
    return socket.gethostbyname(socket.gethostname())


def find_free_ports(
    count: int, port_range: tuple = (1024, 65535), exclude_ports: Set[int] | None = None
) -> List[int]:
    """
    Find multiple free ports within a specified range.

    Args:
        count: Number of free ports to find
        port_range: Tuple of (min_port, max_port) to search within
        exclude_ports: Set of ports to exclude from search

    Returns:
        List of free port numbers

    Raises:
        ValueError: If unable to find requested number of free ports
    """
    if exclude_ports is None:
        exclude_ports = set()

    min_port, max_port = port_range
    free_ports = []
    attempted_ports = set()

    # Calculate available port range
    available_range = max_port - min_port + 1 - len(exclude_ports)

    if count > available_range:
        raise ValueError(
            f"Cannot find {count} ports in range {port_range}. "
            f"Only {available_range} ports available."
        )

    max_attempts = count * 10  # Reasonable limit to avoid infinite loops
    attempts = 0

    while len(free_ports) < count and attempts < max_attempts:
        # Generate random port within range
        port = random.randint(min_port, max_port)

        # Skip if port already attempted or excluded
        if port in attempted_ports or port in exclude_ports:
            attempts += 1
            continue

        attempted_ports.add(port)

        if is_port_free(port):
            free_ports.append(port)

        attempts += 1

    if len(free_ports) < count:
        raise ValueError(
            f"Could only find {len(free_ports)} free ports "
            f"out of {count} requested after {max_attempts} attempts"
        )

    return sorted(free_ports)


def is_port_free(port: int) -> bool:
    """
    Check if a port is free by attempting to bind to it.

    Args:
        port: Port number to check

    Returns:
        True if port is free, False otherwise
    """
    # Check TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        sock.close()
    except OSError:
        return False

    # Check UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", port))
        sock.close()
        return True
    except OSError:
        return False
