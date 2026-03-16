"""Thin wrapper around ``sglang.launch_server`` that ignores AReaL-specific
CLI arguments appended by :class:`LocalScheduler`.

The LocalScheduler unconditionally appends ``--experiment-name``,
``--trial-name``, ``--role``, ``--worker-index``, ``--name-resolve-type``,
``--nfs-record-root``, ``--etcd3-addr``, and ``--fileroot`` to every worker
command.  ``sglang.launch_server`` rejects these as unrecognised arguments.

This module strips those keys before delegating to SGLang.
"""

from __future__ import annotations

import sys

_AREAL_KEYS = frozenset(
    {
        "--experiment-name",
        "--trial-name",
        "--role",
        "--worker-index",
        "--name-resolve-type",
        "--nfs-record-root",
        "--etcd3-addr",
        "--fileroot",
    }
)


def _strip_areal_args(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in _AREAL_KEYS:
            skip_next = True
            continue
        filtered.append(arg)
    return filtered


if __name__ == "__main__":
    import os

    cleaned = _strip_areal_args(sys.argv[1:])
    from sglang.launch_server import prepare_server_args, run_server
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(cleaned)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
