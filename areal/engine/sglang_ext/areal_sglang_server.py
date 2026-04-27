# SPDX-License-Identifier: Apache-2.0

"""AReaL wrapper for sglang server launch.

This module is the entry point used by ``python -m`` when launching
sglang servers from within AReaL, which uses
compositional bridges to extend sglang's scheduler
with AReaL-specific capabilities.

Usage (automatically invoked by AReaL's launcher infrastructure)::

    python -m areal.engine.sglang_ext.areal_sglang_server [sglang args ...]
"""

import logging
import os
import sys

logger = logging.getLogger("areal.engine.sglang_ext")


if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(asctime)s PID=%(process)d %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("areal").setLevel(logging.DEBUG)
    logger.info("areal_sglang_server starting (PID=%d)", os.getpid())

    server_args = prepare_server_args(sys.argv[1:])

    try:
        # ---- BEGIN AREAL ----
        # Use the customized launch_server that wires up compositional bridges
        # (AwexSchedulerBridge, PPSchedulerBridge) instead of monkey-patches.
        from areal.experimental.inference_service.sglang.launch_server import (
            areal_launch_server,
        )

        areal_launch_server(server_args)
        # ---- END AREAL ----
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
