# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
import dataclasses
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pytest

from realhf.api.core.config import ModelName
from realhf.api.core.system_api import GserverManager as GserverManagerConfig
from realhf.api.core.system_api import WorkerInformation
from realhf.base import constants, name_resolve, names, network, testing
from realhf.base.names import gen_servers
from realhf.system.gserver_manager import GserverManager

N_SERVERS = 2

UPDATE_WEIGHTS_CALL_COUNT = defaultdict(int)


@dataclass
class UpdateWeightFromDiskReqInput:
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None


@pytest.fixture
def mock_servers():
    from http import HTTPStatus
    from threading import Thread

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import ORJSONResponse, PlainTextResponse

    ports = network.find_multiple_free_ports(N_SERVERS)

    # Create mock server responses
    servers = []
    jobs = []
    should_run = True

    def run_app1():
        app = FastAPI()

        @app.get("/metrics")
        async def metrics1():
            n_requests = random.choice(list(range(4)))
            return PlainTextResponse(
                f"sglang:num_running_reqs {float(n_requests)}\nother_metric 123"
            )

        @app.post("/update_weights_from_disk/")
        async def update_weights_from_disk1(req: UpdateWeightFromDiskReqInput):
            UPDATE_WEIGHTS_CALL_COUNT[ports[0]] += 1
            return ORJSONResponse(
                {"success": True, "message": "Weights updated successfully"},
                status_code=HTTPStatus.OK,
            )

        config = uvicorn.Config(
            app,
            host="localhost",
            port=ports[0],
            log_level="info",
        )
        server = uvicorn.Server(config)
        servers.append(server)
        while should_run:
            server.run()

    def run_app2():
        app = FastAPI()

        @app.get("/metrics")
        async def metrics2():
            n_requests = random.choice(list(range(4)))
            return PlainTextResponse(
                f"sglang:num_running_reqs {float(n_requests)}\nother_metric 123"
            )

        @app.post("/update_weights_from_disk/")
        async def update_weights_from_disk2(req: UpdateWeightFromDiskReqInput):
            UPDATE_WEIGHTS_CALL_COUNT[ports[1]] += 1
            return ORJSONResponse(
                {"success": True, "message": "Weights updated successfully"},
                status_code=HTTPStatus.OK,
            )

        config = uvicorn.Config(
            app,
            host="localhost",
            port=ports[1],
            log_level="info",
        )
        server = uvicorn.Server(config)
        servers.append(server)
        while should_run:
            server.run()

    job1 = Thread(target=run_app1)
    job2 = Thread(target=run_app2)
    jobs = [job1, job2]
    for job in jobs:
        job.start()

    yield ports
    should_run = False
    for server in servers:
        server.should_exit = True
    for job in jobs:
        job.join()


@pytest.fixture
def gserver_manager(mock_servers):
    testing.clear_name_resolve()
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    # Register mock servers in name resolve
    server_urls = [f"http://localhost:{mock_servers[i]}/" for i in range(N_SERVERS)]
    name = gen_servers(testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME)
    name_resolve.add_subentry(name, server_urls[0])
    name_resolve.add_subentry(name, server_urls[1])

    # Mock requests.get for metrics endpoint
    m = GserverManager()
    config = GserverManagerConfig(
        model_name=ModelName("default", 0),
        n_servers=N_SERVERS,
        schedule_policy="round_robin",
        worker_info=WorkerInformation(
            experiment_name=testing._DEFAULT_EXPR_NAME,
            trial_name=testing._DEFAULT_TRIAL_NAME,
            worker_type="gserver_manager",
            worker_count=1,
            worker_index=0,
        ),
    )
    m._configure(config)
    # launch the server
    m._poll()
    yield m
    m.exit()


@pytest.mark.asyncio
async def test_schedule_policy(gserver_manager):
    # Test round-robin scheduling
    from realhf.api.core.model_api import GenReqMeta

    req_meta = GenReqMeta(
        prompt_len=100,
        group_size=2,
        new_token_budget=1024,
        predicted_new_tokens=None,
    )
    idx1 = gserver_manager._round_robin_schedule(req_meta)
    assert idx1 == 0
    idx2 = gserver_manager._round_robin_schedule(req_meta)
    assert idx2 == 1
    idx3 = gserver_manager._round_robin_schedule(req_meta)
    assert idx3 == 0


@pytest.mark.asyncio
async def test_weight_update(gserver_manager):
    from fastapi.testclient import TestClient

    from realhf.api.core.model_api import GenReqMeta

    client = TestClient(gserver_manager.app)

    # Set up a new parameter version
    name = names.model_version(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME, "default"
    )
    name_resolve.add(name, "1")
    global UPDATE_WEIGHTS_CALL_COUNT
    UPDATE_WEIGHTS_CALL_COUNT.clear()

    req_meta = GenReqMeta(
        prompt_len=100,
        group_size=2,
        new_token_budget=1024,
        predicted_new_tokens=None,
    )

    client.post("/schedule_request", json=dataclasses.asdict(req_meta))
    assert gserver_manager._last_param_realloc_step == 1
    assert len(UPDATE_WEIGHTS_CALL_COUNT) == N_SERVERS
    for v in UPDATE_WEIGHTS_CALL_COUNT.values():
        assert v == 1

    # weights updated, no more weights update
    client.post("/schedule_request", json=dataclasses.asdict(req_meta))
    assert gserver_manager._last_param_realloc_step == 1
    assert len(UPDATE_WEIGHTS_CALL_COUNT) == N_SERVERS
    for v in UPDATE_WEIGHTS_CALL_COUNT.values():
        assert v == 1
    UPDATE_WEIGHTS_CALL_COUNT.clear()

    # Test with no version set
    name_resolve.delete(name)


def test_server_lifecycle(gserver_manager):
    # Test that the server starts and stops properly
    assert gserver_manager.thread is not None
    assert gserver_manager.thread.is_alive()

    # Test exit hook
    gserver_manager.exit()
    time.sleep(0.1)  # Give thread time to stop
    assert not gserver_manager.thread.is_alive()


@pytest.mark.asyncio
async def test_http_server_endpoints(gserver_manager):
    # Test the FastAPI endpoints
    from fastapi.testclient import TestClient

    from realhf.api.core.model_api import GenReqMeta

    client = TestClient(gserver_manager.app)

    # Test schedule_request endpoint
    req_meta = GenReqMeta(
        prompt_len=100,
        group_size=2,
        new_token_budget=1024,
        predicted_new_tokens=None,
    )

    # Test round-robin behavior through the endpoint
    responses = set()
    for _ in range(10):
        response = client.post("/schedule_request", json=dataclasses.asdict(req_meta))
        assert response.status_code == 200
        responses.add(response.json()["url"])

    # Should have used all available servers
    assert responses == set(gserver_manager.server_urls)


def test_unique_server_urls(gserver_manager):
    # Ensure server URLs are unique
    assert len(set(gserver_manager.server_urls)) == len(gserver_manager.server_urls)
