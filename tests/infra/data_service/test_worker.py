from __future__ import annotations

# pyright: reportMissingImports=false
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from datasets import Dataset

from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

DATASET_ID = "test-train"


@pytest.fixture
def config() -> DataWorkerConfig:
    return DataWorkerConfig(
        host="127.0.0.1",
        port=0,
        rank=0,
        world_size=1,
        dataloader_num_workers=0,
    )


def _make_mock_dataset(n: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"sample_{i}" for i in range(n)],
            "label": list(range(n)),
        }
    )


def _load_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_path": "test/dataset",
        "dataset_type": "rl",
        "seed": 42,
        "shuffle": False,
    }
    payload.update(overrides)
    return payload


@pytest_asyncio.fixture
async def client(config: DataWorkerConfig):
    app = create_worker_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def loaded_client(config: DataWorkerConfig):
    with (
        patch("areal.infra.data_service.worker.app._get_custom_dataset") as mock_get,
        patch(
            "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
        ) as mock_load,
    ):
        ds = _make_mock_dataset(8)
        mock_get.return_value = ds
        mock_load.return_value = (None, None)

        app = create_worker_app(config)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/datasets/load", json=_load_payload())
            assert resp.status_code == 200
            yield c


@pytest.mark.asyncio
class TestWorkerHealth:
    async def test_health_returns_200(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["datasets"] == 0

    async def test_health_shows_dataset_count(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["datasets"] == 1


@pytest.mark.asyncio
class TestDatasetLoading:
    async def test_load_dataset_returns_steps_per_epoch(self, config: DataWorkerConfig):
        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset"
            ) as mock_get,
        ):
            ds = _make_mock_dataset(20)
            mock_get.return_value = ds

            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as c:
                resp = await c.post("/datasets/load", json=_load_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_per_epoch"] > 0
        assert data["dataset_size"] == 20

    async def test_load_dataset_duplicate_409(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post("/datasets/load", json=_load_payload())
        assert resp.status_code == 409

    async def test_unload_dataset_removes(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post(
            "/datasets/unload", json={"dataset_id": DATASET_ID}
        )
        assert resp.status_code == 200

        health = await loaded_client.get("/health")
        assert health.status_code == 200
        assert health.json()["datasets"] == 0

    async def test_unload_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post("/datasets/unload", json={"dataset_id": "unknown"})
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestSampleFetch:
    async def test_fetch_samples_returns_data(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0, 1]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["samples"]) == 2

    async def test_fetch_samples_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/v1/samples/fetch",
            json={"dataset_id": "unknown", "indices": [0]},
        )
        assert resp.status_code == 404

    async def test_fetch_samples_returns_distinct_items(
        self, loaded_client: httpx.AsyncClient
    ):
        resp = await loaded_client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0, 1, 2]},
        )
        assert resp.status_code == 200
        samples = resp.json()["samples"]
        assert len(samples) == 3
        assert samples[0] != samples[1]


@pytest.mark.asyncio
class TestEpochReset:
    async def test_epoch_reset_updates_epoch(self, loaded_client: httpx.AsyncClient):
        reset = await loaded_client.post(
            "/epoch/reset", json={"dataset_id": DATASET_ID, "epoch": 1}
        )
        assert reset.status_code == 200
        assert reset.json()["epoch"] == 1

    async def test_epoch_reset_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/epoch/reset", json={"dataset_id": "unknown", "epoch": 1}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestStatePersistence:
    async def test_state_save_creates_file(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        resp = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert resp.status_code == 200
        out = resp.json()
        assert out["status"] == "ok"
        assert (tmp_path / "worker_0.pkl").exists()

    async def test_state_load_restores(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        save = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert save.status_code == 200

        load = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert load.status_code == 200
        assert load.json()["status"] == "ok"

    async def test_state_load_missing_file_404(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        missing = tmp_path / "does-not-exist"
        resp = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(missing)}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestTensorShardEndpoints:
    async def test_data_clear_returns_ok(self, client: httpx.AsyncClient):
        resp = await client.delete("/data/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tensor_shards"] == 0

    async def test_data_shard_not_found_404(self, client: httpx.AsyncClient):
        resp = await client.get("/data/nonexistent")
        assert resp.status_code == 404
