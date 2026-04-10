"""AReaL wrapper for sglang.launch_server.

This module is the entry point used by ``python -m`` when launching
sglang servers from within AReaL.  It applies runtime monkey-patches
(e.g. per-PP-rank NCCL group support) **before** sglang initialises
its workers, then delegates to the upstream ``sglang.launch_server``
logic.

Usage (automatically invoked by AReaL's launcher infrastructure)::

    python -m areal.engine.sglang_ext.areal_sglang_server [sglang args ...]
"""

import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def _apply_patches() -> None:
    """Apply AReaL-specific patches to sglang before server start."""
    from areal.patches.sglang_pp_weight_update import apply_sglang_pp_patch

    apply_sglang_pp_patch()


if __name__ == "__main__":
    # Apply AReaL patches before sglang initialises any workers.
    _apply_patches()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        # Delegate to sglang's server runner.
        from sglang.launch_server import run_server

        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
