# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import fcntl
import os
import socket
import time
from contextlib import closing
from functools import wraps

from realhf.base import constants, logging, name_resolve, names

logger = logging.getLogger(__name__)


def gethostname():
    return socket.gethostname()


def gethostip():
    return socket.gethostbyname(socket.gethostname())


def find_free_port(
    low=1,
    high=65536,
    exclude_ports=None,
    experiment_name="port",
    trial_name="port",
    lockfile_root=constants.PORT_LOCKFILE_ROOT,
):
    # TODO: user random sampling instead of bind
    """Find a free port within the specified range, excluding certain ports."""

    ports_name = names.used_ports(experiment_name, trial_name, gethostip())

    free_port = None
    os.makedirs(lockfile_root, exist_ok=True)
    lockfile = os.path.join(lockfile_root, gethostip())
    while True:
        with open(lockfile, "w") as fd:
            # This will block until lock is acquired
            fcntl.flock(fd, fcntl.LOCK_EX)
            used_ports = list(map(int, name_resolve.get_subtree(ports_name)))
            assert len(used_ports) == len(set(used_ports))
            if exclude_ports is None:
                exclude_ports = set(used_ports)
            else:
                exclude_ports = exclude_ports.union(set(used_ports))
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("", 0))
                    port = s.getsockname()[1]
                    if low <= port <= high and port not in exclude_ports:
                        name_resolve.add_subentry(ports_name, str(port))
                        logger.info(f"Found free port {port}")
                        free_port = port
                        break
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        time.sleep(0.05)
    return free_port


def find_multiple_free_ports(
    count,
    low=1,
    high=65536,
    experiment_name="port",
    trial_name="port",
    lockfile_root=constants.PORT_LOCKFILE_ROOT,
):
    """Find multiple mutually exclusive free ports."""
    free_ports = set()
    for _ in range(count):
        port = find_free_port(
            low=low,
            high=high,
            exclude_ports=free_ports,
            experiment_name=experiment_name,
            trial_name=trial_name,
            lockfile_root=lockfile_root,
        )
        free_ports.add(port)
    return list(free_ports)
