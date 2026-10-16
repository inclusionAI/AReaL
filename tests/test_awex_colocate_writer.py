# SPDX-License-Identifier: Apache-2.0

import weakref
from types import SimpleNamespace

import pytest
import torch

from areal.engine.awex import colocate_writer
from areal.engine.awex.colocate_writer import AwexWeightPublisher

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for AWEX IPC exports"
)


@pytest.mark.parametrize(
    "completion",
    [True, {"ok": False, "error": "reader failed"}, TimeoutError("reader timeout")],
)
def test_colocate_export_lifetime_follows_reader_ack(monkeypatch, completion):
    """Free packed GPU storage before write_finished only after a successful ACK."""
    from awex.util import tensor_util

    model = torch.nn.Linear(8, 8, bias=False, device="cuda")
    engine = SimpleNamespace(model=[model], cpu_group=object())
    residency = SimpleNamespace(release_memory=lambda tags: None)
    publisher = AwexWeightPublisher(engine, residency=residency)
    publisher._ip_address = "127.0.0.1"
    publisher._physical_gpu_id = 0
    publisher._logical_train_rank = 0
    publisher._num_infer_engines = 1
    publisher._timeout_s = 1
    publisher._prepare_residency_for_publish = lambda: None
    publisher._lazy_initialize = lambda: None
    publisher._convert_parameters = lambda: {"weight": model.weight.detach().clone()}
    exports = []
    signals = []

    def serialize(payload):
        assert all(ref() is None for ref in exports)
        exports[:] = [weakref.ref(tensor) for tensor in payload[0]]
        return b"payload"

    class MetaServer:
        def add_object_to_set(self, key, rank):
            pass

        def put_object(self, key, value):
            if key.startswith("write_finished"):
                assert publisher._pending_ipc_export is None
                assert all(ref() is None for ref in exports)
            signals.append(key)

        def get_object(self, key, timeout):
            assert key.startswith("weights_update_finished")
            assert all(ref() is not None for ref in exports)
            if isinstance(completion, Exception):
                raise completion
            return completion

        def delete_if_exists(self, key):
            pass

        def wait_set_until_size(self, key, size, timeout):
            assert publisher._pending_ipc_export is None
            assert all(ref() is None for ref in exports)
            signals.append(key)

    publisher._meta_server_client = MetaServer()
    monkeypatch.setattr(tensor_util, "cuda_ipc_serialize", serialize)
    monkeypatch.setattr(colocate_writer.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(colocate_writer.dist, "barrier", lambda group: None)

    if completion is True:
        for version in range(3):
            publisher.execute_colocate_weight_update(version)
            publisher.finish_colocate_weight_update(training_world_size=1)
        assert signals.count("finished_weights_update_engines") == 3
    else:
        error_type = TimeoutError if isinstance(completion, Exception) else RuntimeError
        error_text = (
            "reader timeout" if isinstance(completion, Exception) else "reader failed"
        )
        with pytest.raises(error_type, match=error_text):
            publisher.execute_colocate_weight_update(0)
        assert publisher._pending_ipc_export is not None
        assert all(ref() is not None for ref in exports)
        assert not any(key.startswith("write_finished") for key in signals)
