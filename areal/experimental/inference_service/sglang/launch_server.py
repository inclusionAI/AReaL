# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Adapted from sglang.srt.entrypoints.http_server.launch_server
# (SGLang commit pinned in this repo).
#
# AReaL additions are between # ---- BEGIN AREAL ---- / # ---- END AREAL ----
# markers. Everything else mirrors the upstream launch_server flow.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys


def areal_launch_server(server_args) -> None:
    import uvicorn
    from sglang.srt.entrypoints.engine import (
        _launch_subprocesses,
        run_detokenizer_process,
    )
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _GlobalState,
        add_prometheus_track_response_middleware,
        app,
        app_has_admin_force_endpoints,
        set_global_state,
        set_uvicorn_logging_configs,
    )
    from sglang.srt.managers.multi_tokenizer_mixin import (
        monkey_patch_uvicorn_multiprocessing,
        write_data_for_multi_tokenizer,
    )
    from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
        parse_remote_instance_transfer_engine_info_from_scheduler_infos,
    )

    # ---- BEGIN AREAL ----
    from areal.experimental.inference_service.sglang.awex import (
        register_awex_endpoints,
    )
    from areal.experimental.inference_service.sglang.rpc_proxy import RpcProxy
    from areal.experimental.inference_service.sglang.scheduler import (
        areal_run_scheduler_process,
        create_result_ipc,
    )
    # ---- END AREAL ----

    try:
        from sglang.srt.entrypoints.engine import init_tokenizer_manager
    except ImportError:
        init_tokenizer_manager = None

    # ---- BEGIN AREAL ----
    result_ipc = create_result_ipc()
    # ---- END AREAL ----

    tokenizer_manager, template_manager, scheduler_infos, port_args = (
        _launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=init_tokenizer_manager,
            # ---- BEGIN AREAL ----
            run_scheduler_process_func=areal_run_scheduler_process,
            # ---- END AREAL ----
            run_detokenizer_process_func=run_detokenizer_process,
        )
    )

    # ---- BEGIN AREAL ----
    if tokenizer_manager is None:
        return
    # ---- END AREAL ----

    remote_instance_transfer_engine_info = (
        parse_remote_instance_transfer_engine_info_from_scheduler_infos(scheduler_infos)
    )

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
            remote_instance_transfer_engine_info=remote_instance_transfer_engine_info,
        )
    )

    # ---- BEGIN AREAL ----
    rpc_proxy = RpcProxy(port_args, result_ipc)
    register_awex_endpoints(app, rpc_proxy)
    # ---- END AREAL ----

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    if server_args.tokenizer_worker_num == 1:
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=None,
            execute_warmup_func=_execute_server_warmup,
        )

        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
            )
    else:
        app.is_single_tokenizer_mode = False
        write_data_for_multi_tokenizer(port_args, server_args, scheduler_infos[0])

    try:
        set_uvicorn_logging_configs(server_args)

        if server_args.tokenizer_worker_num == 1:
            uvicorn.run(
                app,
                host=server_args.host,
                port=server_args.port,
                root_path=server_args.fastapi_root_path,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
            )
        else:
            monkey_patch_uvicorn_multiprocessing()
            uvicorn.run(
                "sglang.srt.entrypoints.http_server:app",
                host=server_args.host,
                port=server_args.port,
                root_path=server_args.fastapi_root_path,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
                workers=server_args.tokenizer_worker_num,
            )
    finally:
        # ---- BEGIN AREAL ----
        rpc_proxy.close()
        # ---- END AREAL ----


if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()
    server_args = prepare_server_args(sys.argv[1:])

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
