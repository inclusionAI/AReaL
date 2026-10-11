# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from concurrent.futures import Future
from threading import Event, Thread

import pytest
from flask import Flask
from torch.distributed.checkpoint.api import CheckpointException

from areal.infra.rpc.guard import engine_blueprint as eb
from areal.infra.rpc.guard.app import GuardState


class _EngineBaseError(BaseException):
    pass


@pytest.fixture
def engine_worker(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(eb, "_engine_thread", None)
    monkeypatch.setattr(eb, "_engine_work_queue", None)
    yield
    worker = eb._engine_thread
    queue = eb._engine_work_queue
    if worker is not None and worker.is_alive():
        queue.put(None)
        worker.join(timeout=5)
        assert not worker.is_alive()


@pytest.mark.parametrize(
    "error",
    [
        CheckpointException("checkpoint save failed", {}),
        _EngineBaseError("engine failed"),
        SystemExit("engine requested exit"),
        KeyboardInterrupt("engine interrupted"),
    ],
    ids=["checkpoint", "custom_base_exception", "system_exit", "keyboard_interrupt"],
)
def test_engine_base_exception_completes_future_and_queued_work(
    engine_worker: None, error: BaseException
) -> None:
    eb._init_engine_thread()
    started = Event()
    release = Event()
    failed: Future = Future()
    queued: Future = Future()

    def fail() -> None:
        started.set()
        assert release.wait(timeout=5)
        raise error

    eb._engine_work_queue.put((fail, (), {}, failed, "save_checkpoint"))
    try:
        assert started.wait(timeout=5)
        eb._engine_work_queue.put((lambda: "queued", (), {}, queued, "queued_task"))
    finally:
        release.set()

    with pytest.raises(RuntimeError, match=type(error).__name__) as caught:
        failed.result(timeout=5)
    assert caught.value.__cause__ is error
    assert queued.result(timeout=5) == "queued"
    assert eb._engine_thread.is_alive()
    assert eb._submit_to_engine_thread("next_task", lambda: "next") == "next"


@pytest.mark.parametrize(
    "error", [ValueError("bad payload"), RuntimeError("save failed")]
)
def test_engine_exception_preserves_identity_and_worker(
    engine_worker: None, error: Exception
) -> None:
    eb._init_engine_thread()
    failed: Future = Future()

    def fail() -> None:
        raise error

    eb._engine_work_queue.put((fail, (), {}, failed, "ordinary_failure"))
    with pytest.raises(type(error)) as caught:
        failed.result(timeout=5)
    assert caught.value is error
    assert eb._submit_to_engine_thread("next_task", lambda: "next") == "next"


def test_checkpoint_exception_returns_http_error_and_next_call_succeeds(
    engine_worker: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Engine:
        def save_checkpoint(self) -> None:
            raise CheckpointException("checkpoint save failed", {})

        def echo(self) -> str:
            return "ok"

    monkeypatch.setattr(eb, "_engines", {"test": Engine()})
    app = Flask(__name__)
    app.config["guard_state"] = GuardState()
    app.register_blueprint(eb.engine_bp)
    responses = []

    def request_save() -> None:
        with app.test_client() as client:
            responses.append(
                client.post(
                    "/call",
                    json={
                        "engine_name": "test",
                        "method": "save_checkpoint",
                        "rpc_meta": {"broadcast": False},
                    },
                )
            )

    # Bound the HTTP regression test even if the engine leaves its Future pending.
    request_thread = Thread(target=request_save, daemon=True)
    request_thread.start()
    request_thread.join(timeout=5)
    assert not request_thread.is_alive(), "Checkpoint RPC did not return"
    assert len(responses) == 1
    response = responses[0]
    assert response.status_code == 500
    assert "CheckpointException" in response.get_json()["error"]
    with app.test_client() as client:
        response = client.post(
            "/call",
            json={
                "engine_name": "test",
                "method": "echo",
                "rpc_meta": {"broadcast": False},
            },
        )
    assert response.status_code == 200
    assert response.get_json()["result"] == "ok"
    assert eb._engine_thread.is_alive()
