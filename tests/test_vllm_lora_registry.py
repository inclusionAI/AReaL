from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "areal"
    / "engine"
    / "vllm_ext"
    / "lora_registry.py"
)
_SPEC = spec_from_file_location("test_lora_registry", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_lora_registry = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_lora_registry)

apply_pending_lora_registry_update = (
    _lora_registry.apply_pending_lora_registry_update
)
cache_pending_lora_registry_update = (
    _lora_registry.cache_pending_lora_registry_update
)


class DummyLoRARequest:
    def __init__(
        self,
        *,
        lora_name: str,
        lora_int_id: int,
        path: str,
        base_model_name: str,
    ):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.path = path
        self.base_model_name = base_model_name


def test_apply_pending_lora_registry_update_rekeys_existing_request():
    app_state = SimpleNamespace(
        openai_serving_models=SimpleNamespace(
            lora_requests={
                "lora-gsm8k-v0": DummyLoRARequest(
                    lora_name="lora-gsm8k-v0",
                    lora_int_id=7,
                    path="/tmp/lora-v0",
                    base_model_name="base-model",
                )
            }
        )
    )

    cache_pending_lora_registry_update(
        app_state,
        lora_name="lora-gsm8k-v1",
        lora_int_id=7,
        base_model_name="base-model",
    )

    assert apply_pending_lora_registry_update(app_state) is True

    lora_requests = app_state.openai_serving_models.lora_requests
    assert "lora-gsm8k-v0" not in lora_requests
    assert "lora-gsm8k-v1" in lora_requests
    updated = lora_requests["lora-gsm8k-v1"]
    assert updated.lora_name == "lora-gsm8k-v1"
    assert updated.lora_int_id == 7
    assert updated.path == "/tmp/lora-v0"
    assert updated.base_model_name == "base-model"
    assert not hasattr(app_state, "_areal_pending_lora_registry_update")


def test_apply_pending_lora_registry_update_returns_false_without_matching_request():
    app_state = SimpleNamespace(
        openai_serving_models=SimpleNamespace(lora_requests={})
    )

    cache_pending_lora_registry_update(
        app_state,
        lora_name="lora-gsm8k-v1",
        lora_int_id=99,
        base_model_name="base-model",
    )

    assert apply_pending_lora_registry_update(app_state) is False
    assert hasattr(app_state, "_areal_pending_lora_registry_update")
