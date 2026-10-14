"""Private, per-segment filesystem mailbox for local runtime control.

This is a trusted shared-filesystem protocol, not a network API. Each launch has
an unpredictable identity; receipts are immutable and atomically published.
"""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

SCHEMA = 1
LAUNCH_ENV = "AUTOPILOT_LAUNCH_FILE"
BINDINGS = ("run_id", "segment_id", "nonce", "runtime_revision", "configuration_id")


class ProtocolError(RuntimeError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(_):
    raise ProtocolError("nonfinite JSON")


def read_json(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024:
        raise ProtocolError("control message exceeds size limit")
    value = json.loads(data, object_pairs_hook=_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ProtocolError("control message must be an object")
    return value


def publish(path: Path, value: dict) -> None:
    """Create once, with no partial JSON visible to readers; never overwrite."""
    data = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name("." + path.name + "." + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def message(launch: dict, kind: str, **payload) -> dict:
    return dict(
        schema_version=SCHEMA,
        kind=kind,
        **{key: launch[key] for key in BINDINGS},
        **({"plan_id": launch["plan_id"]} if "plan_id" in launch else {}),
        **payload,
    )


def check_message(value: dict, launch: dict, kind: str) -> dict:
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA
        or value.get("kind") != kind
        or any(value.get(key) != launch[key] for key in BINDINGS)
        or value.get("plan_id") != launch.get("plan_id")
    ):
        raise ProtocolError("receipt does not match this runtime segment")
    return value


def check_snapshot(value: dict, launch: dict, request: dict) -> dict:
    """Bind a published manifest to this exact stop request (no latest lookup).

    AReaL owns payload integrity checks and repeats them when restoring.
    """
    snapshot, context = value["snapshot"], value["context"]
    if set(snapshot) != {"path", "snapshot_id", "manifest_sha256"}:
        raise ProtocolError("invalid snapshot reference")
    if Path(snapshot["path"]).resolve() != Path(request["directory"]).resolve():
        raise ProtocolError("unexpected snapshot directory")
    if context.get("state_id") != request["state_id"] or any(
        context.get(key) != launch[key]
        for key in ("run_id", "runtime_revision", "configuration_id")
    ):
        raise ProtocolError("snapshot source context mismatch")
    step = context.get("step_info", {})
    if (
        set(step) != {"epoch", "epoch_step", "global_step", "steps_per_epoch"}
        or any(type(v) is not int or v < 0 for v in step.values())
        or step["steps_per_epoch"] <= step["epoch_step"]
        or step["global_step"]
        != step["epoch"] * step["steps_per_epoch"] + step["epoch_step"]
        or type(context.get("policy_version")) is not int
        or context["policy_version"] < 0
    ):
        raise ProtocolError("invalid saved training boundary")
    encoded = (Path(snapshot["path"]) / "manifest.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != snapshot["manifest_sha256"]:
        raise ProtocolError("snapshot manifest digest mismatch")
    manifest = json.loads(encoded, object_pairs_hook=_object)
    if (
        manifest.get("snapshot_id") != snapshot["snapshot_id"]
        or manifest.get("context") != context
    ):
        raise ProtocolError("snapshot manifest source mismatch")
    return {"snapshot": snapshot, "context": context}
