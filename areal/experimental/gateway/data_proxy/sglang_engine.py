"""Lightweight SGLang wrapper engine for the gateway controller.

This engine is loaded onto RPC server workers by the
``GatewayRolloutController``.  It **only** implements
``launch_server`` / ``teardown_server`` so that the worker can:

1. Start an SGLang subprocess on its allocated GPU(s).
2. Expose the ``/fork`` endpoint (inherited from the RPC server)
   so that data-proxy processes can be forked from the same node.

All other inference traffic is routed through the
gateway HTTP stack and never touches this engine.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from areal.api.cli_args import SGLangConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import LocalInfServerInfo
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging
from areal.utils.network import find_free_ports, gethostip

logger = logging.getLogger("GatewaySGLangEngine")


class GatewaySGLangEngine(InferenceEngine):
    """Minimal ``InferenceEngine`` that only launches / tears down SGLang.

    Designed to be created on an RPC-server worker via
    ``scheduler.create_engine``.  The worker's ``/fork`` endpoint remains
    available for spawning co-located data-proxy processes.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._server_info: LocalInfServerInfo | None = None
        self._initialized = False

    # -- InferenceEngine required properties --------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    # -- Server lifecycle ---------------------------------------------------

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch an SGLang server subprocess on this worker's GPU(s).

        Parameters
        ----------
        server_args : dict[str, Any]
            Arguments produced by ``SGLangConfig.build_args`` (model path,
            TP size, dtype, etc.).  ``host`` and ``port`` are filled in
            automatically if not already set.

        Returns
        -------
        LocalInfServerInfo
            Connection info for the launched server.
        """
        if "host" not in server_args or not server_args["host"]:
            server_args["host"] = gethostip()
        if "port" not in server_args or not server_args["port"]:
            server_args["port"] = find_free_ports(1)[0]

        cmd = SGLangConfig.build_cmd_from_args(server_args)
        logger.info("Launching SGLang: %s", " ".join(cmd))

        # Replace generic "python" with the current interpreter
        if cmd[0].startswith("python"):
            cmd[0] = sys.executable

        process = subprocess.Popen(cmd)  # noqa: S603

        self._server_info = LocalInfServerInfo(
            host=server_args["host"],
            port=server_args["port"],
            process=process,
        )
        self._initialized = True
        logger.info(
            "SGLang server launched at %s:%s (pid %s)",
            server_args["host"],
            server_args["port"],
            process.pid,
        )
        return self._server_info

    def teardown_server(self) -> None:
        """Kill the SGLang subprocess if it is still running."""
        if self._server_info is not None and self._server_info.process is not None:
            if self._server_info.process.poll() is None:
                kill_process_tree(self._server_info.process.pid, graceful=True)
                logger.info("SGLang server terminated")
            self._server_info = None
        self._initialized = False

    def initialize(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """No-op — SGLang is started via ``launch_server``."""

    def destroy(self) -> None:
        self.teardown_server()
