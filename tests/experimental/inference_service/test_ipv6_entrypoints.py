from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch


def test_data_proxy_main_formats_ipv6_serving_addr():
    from areal.experimental.inference_service.data_proxy import (
        __main__ as data_proxy_main,
    )

    args = argparse.Namespace(
        host="::1",
        port=8082,
        backend_addr="http://backend",
        backend_type="sglang",
        tokenizer_path="mock-tokenizer",
        log_level="info",
        request_timeout=120.0,
        set_reward_finish_timeout=0.0,
        admin_api_key="admin-key",
        callback_server_addr="http://[::1]:19000",
    )

    with (
        patch.object(
            data_proxy_main.argparse.ArgumentParser,
            "parse_known_args",
            return_value=(args, []),
        ),
        patch.object(data_proxy_main, "create_app") as mock_create_app,
        patch.object(data_proxy_main.uvicorn, "run") as mock_run,
    ):
        mock_create_app.return_value = MagicMock()

        data_proxy_main.main()

    config = mock_create_app.call_args.args[0]
    assert config.serving_addr == "[::1]:8082"
    mock_run.assert_called_once()


def test_guard_main_registers_ipv6_worker_addr():
    from areal.experimental.inference_service.guard import __main__ as guard_main

    args = argparse.Namespace(
        port=0,
        host="::1",
        experiment_name="test-exp",
        trial_name="test-trial",
        role="guard",
        worker_index=0,
        name_resolve_type="nfs",
        nfs_record_root="/tmp/areal/name_resolve",
        etcd3_addr="localhost:2379",
        fileroot=None,
    )
    mock_server = MagicMock()
    mock_server.socket.getsockname.return_value = ("::1", 18080, 0, 0)
    mock_server.serve_forever.side_effect = KeyboardInterrupt

    with (
        patch.object(
            guard_main.argparse.ArgumentParser,
            "parse_known_args",
            return_value=(args, []),
        ),
        patch.object(guard_main, "make_server", return_value=mock_server),
        patch.object(guard_main.name_resolve, "reconfigure"),
        patch.object(guard_main.name_resolve, "add") as mock_add,
        patch.object(guard_main.names, "worker_discovery", return_value="worker-key"),
        patch.object(guard_main.guard_app, "cleanup_forked_children"),
    ):
        guard_main.main()

    mock_add.assert_called_once_with("worker-key", "[::1]:18080", replace=True)
    mock_server.shutdown.assert_called_once()
