"""Data Blueprint: RTensor ``/data/*`` storage endpoints.

Provides a Flask Blueprint that handles tensor shard storage and
retrieval via HTTP.  Used by any service that needs local tensor
storage accessible over the network (e.g., the RPC server).

Routes:

- ``PUT    /data/<shard_id>``  — store a single shard
- ``GET    /data/<shard_id>``  — retrieve a single shard
- ``POST   /data/batch``       — retrieve multiple shards
- ``DELETE /data/clear``       — clear specified shards
"""

from __future__ import annotations

import traceback

import orjson
from flask import Blueprint, Response, jsonify, request

from areal.infra.rpc import rtensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging

logger = logging.getLogger("DataBP")

data_bp = Blueprint("data", __name__)


@data_bp.route("/data/<shard_id>", methods=["PUT"])
def store_batch_data(shard_id: str):
    """Store batch data shard."""
    try:
        data_bytes = request.get_data()

        # Deserialize to get tensor (already on CPU)
        serialized_data = orjson.loads(data_bytes)
        data = deserialize_value(serialized_data)

        rtensor.store(shard_id, data)

        logger.debug(f"Stored batch shard {shard_id} (size={len(data_bytes)} bytes)")
        return jsonify({"status": "ok", "shard_id": shard_id})

    except Exception as e:
        logger.error(f"Error storing batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@data_bp.route("/data/<shard_id>", methods=["GET"])
def retrieve_batch_data(shard_id: str):
    """Retrieve batch data shard."""
    logger.debug(f"Received data get request for shard {shard_id}")
    try:
        try:
            data = rtensor.fetch(shard_id)
        except KeyError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Shard {shard_id} not found",
                    }
                ),
                404,
            )

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)

        logger.debug(f"Retrieved batch shard {shard_id} (size={len(data_bytes)} bytes)")
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@data_bp.route("/data/batch", methods=["POST"])
def retrieve_batch_data_many():
    """Retrieve multiple batch data shards in one request."""
    try:
        payload = request.get_json(silent=True) or {}
        shard_ids = payload.get("shard_ids", [])
        if not isinstance(shard_ids, list) or not all(
            isinstance(shard_id, str) for shard_id in shard_ids
        ):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Expected JSON body with string list field 'shard_ids'"
                        ),
                    }
                ),
                400,
            )

        data = []
        missing_shard_ids = []
        for shard_id in shard_ids:
            try:
                data.append(rtensor.fetch(shard_id))
            except KeyError:
                missing_shard_ids.append(shard_id)

        if missing_shard_ids:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": ("One or more requested shards were not found"),
                        "missing_shard_ids": missing_shard_ids,
                    }
                ),
                400,
            )

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)
        logger.debug(
            "Retrieved %s batch shards (size=%s bytes)",
            len(shard_ids),
            len(data_bytes),
        )
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shards: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500


@data_bp.route("/data/clear", methods=["DELETE"])
def clear_batch_data():
    """Clear specified batch data shards.

    Expected JSON payload::

        {"shard_ids": ["id1", "id2", ...]}
    """
    try:
        data = request.get_json(silent=True) or {}
        shard_ids = data.get("shard_ids", [])
        if not isinstance(shard_ids, list):
            return (
                jsonify({"status": "error", "message": "'shard_ids' must be a list"}),
                400,
            )

        cleared_count = sum(rtensor.remove(sid) for sid in shard_ids)
        storage = rtensor.storage_stats()
        result = {
            "status": "ok",
            "cleared_count": cleared_count,
            **storage,
        }
        logger.info(f"Cleared {cleared_count} batch shards. Stats: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error clearing batch data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
