"""SGLang server-side monkey patches required for AReaL's R3 (Router Replay).

Background
----------
When ``skip_tokenizer_init=True`` (which AReaL forces whenever R3 is enabled,
see ``rl_trainer.py``), the SGLang *scheduler* bypasses the
``DetokenizerManager`` and sends ``BatchTokenIDOutput`` directly to
``TokenizerManager``.  The side effect is that the ``routed_experts`` tensor
is **not** base64-encoded by ``DetokenizerManager._extract_routed_experts``
anymore -- it reaches ``TokenizerManager._handle_batch_output`` still as a
raw ``torch.Tensor``.

``TokenizerManager`` then attaches the tensor verbatim to
``meta_info["routed_experts"]`` and lets FastAPI/``ORJSONResponse``
serialize the whole response.  FastAPI's ``jsonable_encoder`` does not
know how to encode ``torch.Tensor``; it silently returns an **empty
dict** (``{}``) instead of raising.  The client then receives

    meta_info["routed_experts"] == {}

and hits ``TypeError: int() argument must be a string, a bytes-like
object or a real number, not 'dict'`` when it tries
``np.asarray(routed_experts, dtype=np.int32)``.  Because the error is
swallowed in ``parse_generation_response``, ``routed_experts`` becomes
``None`` and the downstream ``RemoteInfEngine`` raises::

    RuntimeError: Requested return_routed_experts=True but received None
                  from SGLang

This module installs a monkey patch on
``sglang.srt.managers.tokenizer_manager.TokenizerManager._handle_batch_output``
that base64-encodes the tensor in-place *before* it is serialised (exactly
the same encoding that ``DetokenizerManager._extract_routed_experts``
applies in the non-``skip_tokenizer_init`` path), so the wire format stays
consistent with both SGLang's documented behaviour and AReaL's client-side
decoder in ``areal/engine/sglang_remote.py``.

The patch is idempotent.  When R3 is disabled the patch is a no-op at
runtime because the routed-experts attribute stays ``None``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_ROUTED_EXPERTS_ATTRS = ("routed_experts", "output_routed_experts")


def _encode_routed_experts_for_wire(value):
    """Convert ``routed_experts`` to a base64 string for wire transport.

    Mirrors ``sglang.srt.managers.detokenizer_manager
    ._extract_routed_experts``: each request's tensor is encoded as
    ``pybase64.b64encode(tensor.numpy().tobytes()).decode("utf-8")``.

    Accepts tensors, numpy arrays, or already-encoded strings; other
    types are returned unchanged so the client branch can surface them.
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Already encoded by DetokenizerManager or a prior invocation.
        return value
    try:
        import numpy as np
        import pybase64
        import torch
    except Exception:  # pragma: no cover - defensive
        return value

    if isinstance(value, torch.Tensor):
        # ``to("cpu")`` is a no-op when already on CPU but protects us
        # against exotic device placements (e.g. CUDA tensors leaked by
        # a capture buffer).  ``contiguous()`` guarantees ``tobytes``
        # produces a dense layout matching ``shape`` on the decode side.
        arr = value.detach().to("cpu").contiguous().numpy()
    elif isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
    else:
        return value

    # Normalise dtype to int32 so the client's ``np.frombuffer(..., int32)``
    # matches regardless of whether the capture buffer was int64.
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)

    return pybase64.b64encode(arr.tobytes()).decode("utf-8")


def apply_sglang_r3_patch() -> bool:
    """Install the ``_handle_batch_output`` monkey patch.

    Returns ``True`` when the patch is installed (or was already
    installed).  Returns ``False`` when SGLang is unavailable in the
    current process (so the caller can gracefully skip).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        from sglang.srt.managers import tokenizer_manager as _tm
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[R3] sglang.srt.managers.tokenizer_manager not importable; "
            "skipping R3 server patch. reason=%s",
            exc,
        )
        return False

    original = _tm.TokenizerManager._handle_batch_output

    def _handle_batch_output_r3(self, recv_obj):  # type: ignore[no-redef]
        # Pre-encode the routed-experts tensor so the downstream FastAPI
        # serialisation sees a plain string (which ``jsonable_encoder``
        # passes through untouched) instead of a ``torch.Tensor`` (which
        # ``jsonable_encoder`` silently flattens to ``{}``).
        try:
            for attr_name in _ROUTED_EXPERTS_ATTRS:
                re_list = getattr(recv_obj, attr_name, None)
                if re_list is None:
                    continue
                encoded = [_encode_routed_experts_for_wire(v) for v in re_list]
                try:
                    setattr(recv_obj, attr_name, encoded)
                except Exception:
                    # Some SGLang versions freeze the dataclass; fall back
                    # to object.__setattr__ which bypasses __slots__ /
                    # frozen protection.
                    object.__setattr__(recv_obj, attr_name, encoded)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "[R3] Failed to pre-encode routed_experts on server; "
                "falling through to unpatched behaviour."
            )
        return original(self, recv_obj)

    _tm.TokenizerManager._handle_batch_output = _handle_batch_output_r3
    _PATCH_APPLIED = True
    logger.info(
        "[R3] Installed sglang TokenizerManager._handle_batch_output "
        "base64 encoder patch for routed_experts "
        "(handles both routed_experts and output_routed_experts)."
    )
    return True
