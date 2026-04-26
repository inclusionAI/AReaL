# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin wrapper around ``sglang.launch_server`` that installs AReaL's R3
monkey patches before the upstream server boots.

Usage (transparent replacement for ``python3 -m sglang.launch_server ...``)::

    python3 -m areal.infra.launcher.sglang_launch_server --model-path ...

Only the R3 patch is installed; all other CLI behaviour is delegated to
``sglang.launch_server`` unchanged so this entrypoint stays drop-in safe
when R3 is not used.
"""

from __future__ import annotations

import logging
import os
import sys


def _install_areal_patches() -> None:
    """Install AReaL monkey patches that must be active in the SGLang
    server process (scheduler/tokenizer manager/HTTP server)."""
    try:
        from areal.infra.launcher.sglang_r3_patch import apply_sglang_r3_patch

        apply_sglang_r3_patch()
    except Exception:  # pragma: no cover - defensive
        logging.getLogger(__name__).exception(
            "[R3] Failed to install AReaL SGLang patches; server will "
            "start without R3 wire-format fixes."
        )


def main() -> None:
    _install_areal_patches()

    # Delegate to upstream launcher.  We keep argv intact (including
    # argv[0] mangling done by ``python3 -m``) because
    # ``sglang.launch_server`` uses ``sys.argv[1:]`` directly.
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    server_args = prepare_server_args(sys.argv[1:])

    # Same dispatch as ``sglang/launch_server.py``.
    try:
        if getattr(server_args, "grpc_mode", False):
            import asyncio

            from sglang.srt.entrypoints.grpc_server import serve_grpc

            asyncio.run(serve_grpc(server_args))
        elif getattr(server_args, "encoder_only", False):
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
        else:
            from sglang.srt.entrypoints.http_server import launch_server

            launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
