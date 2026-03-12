from __future__ import annotations

from typing import Any

_PENDING_LORA_UPDATE_ATTR = "_areal_pending_lora_registry_update"


def cache_pending_lora_registry_update(
    app_state: Any,
    *,
    lora_name: str,
    lora_int_id: int,
    base_model_name: str,
) -> None:
    """Cache the next LoRA registry rename for the API server.

    The XCCL LoRA update flow sends adapter metadata first and performs the
    actual in-place update in a follow-up request. The OpenAI serving layer
    needs that metadata later so it can re-key ``lora_requests`` from the old
    versioned adapter name to the new one once the update succeeds.
    """

    setattr(
        app_state,
        _PENDING_LORA_UPDATE_ATTR,
        {
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "base_model_name": base_model_name,
        },
    )


def clear_pending_lora_registry_update(app_state: Any) -> None:
    if hasattr(app_state, _PENDING_LORA_UPDATE_ATTR):
        delattr(app_state, _PENDING_LORA_UPDATE_ATTR)


def apply_pending_lora_registry_update(app_state: Any) -> bool:
    """Apply the cached LoRA registry rename to ``openai_serving_models``.

    Returns ``True`` when an existing adapter entry was found and updated.
    Returns ``False`` when no cached update exists or the current server state
    does not expose a matching LoRA request.
    """

    pending = getattr(app_state, _PENDING_LORA_UPDATE_ATTR, None)
    if not pending:
        return False

    serving_models = getattr(app_state, "openai_serving_models", None)
    if serving_models is None:
        return False

    lora_requests = getattr(serving_models, "lora_requests", None)
    if not isinstance(lora_requests, dict):
        return False

    target_request = None
    keys_to_remove: list[str] = []
    for key, request in list(lora_requests.items()):
        if getattr(request, "lora_int_id", None) == pending["lora_int_id"]:
            if target_request is None:
                target_request = request
            if key != pending["lora_name"]:
                keys_to_remove.append(key)

    if target_request is None:
        return False

    for key in keys_to_remove:
        lora_requests.pop(key, None)

    target_request.lora_name = pending["lora_name"]
    if pending["base_model_name"]:
        target_request.base_model_name = pending["base_model_name"]
    lora_requests[pending["lora_name"]] = target_request
    clear_pending_lora_registry_update(app_state)
    return True
