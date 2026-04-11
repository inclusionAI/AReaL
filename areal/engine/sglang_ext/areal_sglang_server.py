"""AReaL wrapper for sglang.launch_server.

This module is the entry point used by ``python -m`` when launching
sglang servers from within AReaL.  It applies runtime monkey-patches
(e.g. per-PP-rank NCCL group support) **before** sglang initialises
its workers, then delegates to the upstream ``sglang.launch_server``
logic.

Usage (automatically invoked by AReaL's launcher infrastructure)::

    python -m areal.engine.sglang_ext.areal_sglang_server [sglang args ...]
"""

import logging
import os
import sys
import time
import warnings

_diag_logger = logging.getLogger("areal.diag.sglang_server")

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def _apply_patches() -> None:
    """Apply AReaL-specific patches to sglang before server start."""
    _diag_logger.info(
        f"[DIAG] _apply_patches() called (PID={os.getpid()})"
    )
    from areal.patches.sglang_pp_weight_update import apply_sglang_pp_patch

    apply_sglang_pp_patch()
    _diag_logger.info(
        f"[DIAG] apply_sglang_pp_patch() completed (PID={os.getpid()})"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(name)s] %(asctime)s PID=%(process)d %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    _diag_logger.info(
        f"[DIAG] areal_sglang_server __main__ starting (PID={os.getpid()})"
    )
    # Apply AReaL patches before sglang initialises any workers.
    _apply_patches()

    _diag_logger.info(f"[DIAG] Preparing server args... (PID={os.getpid()})")
    server_args = prepare_server_args(sys.argv[1:])
    _diag_logger.info(
        f"[DIAG] Server args prepared: "
        f"pp_size={getattr(server_args, 'pp_size', 'N/A')} "
        f"tp_size={getattr(server_args, 'tp_size', 'N/A')} "
        f"(PID={os.getpid()})"
    )

    try:
        # Delegate to sglang's server runner.
        from sglang.launch_server import run_server

        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
