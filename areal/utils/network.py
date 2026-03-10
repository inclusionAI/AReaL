import random
import socket


def gethostname():
    return socket.gethostname()


def is_ipv6_address(ip: str) -> bool:
    """Return True if *ip* is an IPv6 address string."""
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except OSError:
        return False


def format_addr(host: str, port: int) -> str:
    """Format host:port, wrapping IPv6 addresses in brackets as required by URLs."""
    if is_ipv6_address(host):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def gethostip(probe_host: str = "8.8.8.8", probe_port: int = 80) -> str:
    """
    Find the local IP address for outbound traffic. Tries IPv4 first, then falls back
    to IPv6 for IPv6-only environments.

    Args:
        probe_host: Remote IPv4 address used to trigger route selection, default to Google
                    Public DNS IP.
        probe_port: Remote port used for the UDP probe.

    Returns:
        The selected local IP address as a string (IPv4 or IPv6)

    Raises:
        RuntimeError: If no suitable IP address can be determined
    """
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except socket.gaierror:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((probe_host, probe_port))
            return sock.getsockname()[0]
    except OSError:
        pass

    # IPv6 fallback for IPv6-only environments
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6)
        for info in infos:
            ip = info[4][0]
            if ip and not ip.startswith("::1"):
                return ip
    except socket.gaierror:
        pass

    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
            sock.connect(("2001:4860:4860::8888", 80))
            return sock.getsockname()[0]
    except OSError as e:
        raise RuntimeError("Could not determine host IP") from e


def find_free_ports(
    count: int, port_range: tuple = (1024, 65535), exclude_ports: set[int] | None = None
) -> list[int]:
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
    Check if a port is free by attempting to bind to it on both IPv4 and IPv6.

    Args:
        port: Port number to check

    Returns:
        True if port is free on all address families, False otherwise
    """
    for family in (socket.AF_INET, socket.AF_INET6):
        for sock_type in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            try:
                sock = socket.socket(family, sock_type)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.bind(("", port))
                sock.close()
            except OSError:
                try:
                    sock.close()
                except Exception:
                    pass
                return False
    return True
