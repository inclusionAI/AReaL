import argparse
import json
import typing as t
from collections.abc import Iterable, Mapping
from typing import Any

import flask_jsonrpc.types.methods as tm
import flask_jsonrpc.types.params as tp
from flask import Flask
from flask_jsonrpc import JSONRPC

from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.scheduler.rpc.api import (
    CallEnginePayload,
    ConfigurePayload,
    CreateEnginePayload,
    EngineNameEnum,
    Response,
)
from areal.scheduler.rpc.serializer import Serializer
from areal.utils import logging, name_resolve
from areal.utils.seeding import set_random_seed
from areal.utils.stats_tracker import export_all

logger = logging.getLogger(__name__)


def _ensure_jsonable(value: Any) -> Any:
    """Convert Python objects into JSON-serialisable structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_ensure_jsonable(v) for v in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


class EngineRPCServer:
    engine = None
    serializer = Serializer()

    def create_engine(
        self,
        payload: CreateEnginePayload,
    ) -> Response:
        try:
            if payload.class_name == EngineNameEnum.FSDP:
                engine = FSDPEngine(payload.config)
            elif payload.class_name == EngineNameEnum.MEGATRON:
                engine = MegatronEngine(payload.config)
            elif payload.class_name == EngineNameEnum.SGLANG_REMOTE:
                engine = RemoteSGLangEngine(payload.config)
            elif payload.class_name == EngineNameEnum.VLLM_REMOTE:
                engine = RemotevLLMEngine(payload.config)
            else:
                return Response(
                    success=False, message=f"Unknown engine name: {payload.class_name}"
                )
            engine.initialize(**payload.initial_args)
            self.engine = engine
            response = Response(success=True, message="ok")
            return response
        except Exception as exc:
            response = Response(
                success=False, message=f"Failed to create engine: {exc}"
            )
            return response

    def call_engine(self, payload: CallEnginePayload) -> Response:
        try:
            if self.engine is None:
                return Response(success=False, message="Engine not found")

            method = getattr(self.engine, payload.method)
            return Response(
                success=True, message="ok", data=method(*payload.args, **payload.kwargs)
            )
        except Exception as exc:
            return Response(success=False, message=f"Failed to call engine: {exc}")

    def configure(self, payload: ConfigurePayload) -> Response:
        try:
            seed_cfg = payload.seed_cfg
            if seed_cfg:
                try:
                    base_seed = seed_cfg.get("base_seed")
                    key = seed_cfg.get("key", "default")
                    if base_seed is None:
                        raise ValueError("seed.base_seed is required")
                    set_random_seed(int(base_seed), str(key))
                except Exception as exc:
                    return Response(success=False, message=f"Failed to set seed: {exc}")

            name_resolve_cfg = payload.name_resolve
            if name_resolve_cfg:
                try:
                    name_resolve.reconfigure(name_resolve_cfg)
                    return Response(success=True, message="ok")
                except TypeError as exc:
                    return Response(
                        success=False, message=f"Invalid name_resolve payload: {exc}"
                    )
            else:
                return Response(success=True, message="ok")
        except Exception as exc:
            return Response(success=False, message=f"Failed to configure: {exc}")

    def export_stats(self, reset: bool = True) -> Response:
        try:
            stats = export_all(reset=reset)
            response = Response(
                success=True, message="ok", data=_ensure_jsonable(stats)
            )
            return response
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to export stats")
            response = Response(success=False, message=f"Failed to export stats: {exc}")
            return response

    def health(self) -> Response:
        response = Response(success=True, message="ok")
        return response


def create_app():
    app = Flask(__name__)
    engine_rpc_server = EngineRPCServer()
    jsonrpc = JSONRPC(app, "/api", enable_web_browsable_api=False)

    @jsonrpc.method(
        "areal.create_engine",
        tm.MethodAnnotated[
            tm.Summary("Create an inference or training engine instance"),
            tm.Description(
                """
                Create an inference or training engine instance, including:
                RemotevLLMEngine, RemoteSGLangEngine, FSDPEngine, MegatronEngine
                """
            ),
            tm.Tag("engine"),
            tm.Error(
                code=-32000,
                message="Create Engine Error",
                data={"message": "Create Engine Error"},
                status_code=500,
            ),
        ],
    )
    def rpc_create_engine(
        payload: t.Annotated[
            CreateEnginePayload,
            tp.Summary("Engine payload"),
            tp.Description("Engine payload"),
        ],
    ) -> t.Annotated[Response, tp.Summary("Response")]:
        return engine_rpc_server.create_engine(payload)

    @jsonrpc.method(
        "areal.call_engine",
        tm.MethodAnnotated[
            tm.Summary("Call an engine method"),
            tm.Description(
                """
                Call an engine method, including:
                RemotevLLMEngine, RemoteSGLangEngine, FSDPEngine, MegatronEngine
                """
            ),
            tm.Tag("engine"),
            tm.Error(
                code=-32000,
                message="Call Engine Error",
                data={"message": "Call Engine Error"},
                status_code=500,
            ),
        ],
    )
    def rpc_call_engine(
        payload: t.Annotated[
            CallEnginePayload,
            tp.Summary("Engine payload"),
            tp.Description("Engine payload"),
        ],
    ) -> t.Annotated[Response, tp.Summary("Response")]:
        return engine_rpc_server.call_engine(payload)

    @jsonrpc.method(
        "areal.configure",
        tm.MethodAnnotated[
            tm.Summary("Configure the engine"),
            tm.Description(
                """
                Configure the engine, including:
                RemotevLLMEngine, RemoteSGLangEngine, FSDPEngine, MegatronEngine
                """
            ),
            tm.Tag("engine"),
            tm.Error(
                code=-32000,
                message="Configure Engine Error",
                data={"message": "Configure Engine Error"},
                status_code=500,
            ),
        ],
    )
    def rpc_configure(
        payload: t.Annotated[
            ConfigurePayload,
            tp.Summary("Engine payload"),
            tp.Description("Engine payload"),
        ],
    ) -> t.Annotated[Response, tp.Summary("Response")]:
        return engine_rpc_server.configure(payload)

    @jsonrpc.method(
        "areal.health",
        tm.MethodAnnotated[
            tm.Summary("Check the health of the engine"),
            tm.Description(""),
            tm.Tag("engine"),
            tm.Error(
                code=-32000,
                message="Health Check Error",
                data={"message": "Health Check Error"},
                status_code=500,
            ),
        ],
    )
    def rpc_health() -> t.Annotated[Response, tp.Summary("Response")]:
        return engine_rpc_server.health()

    @jsonrpc.method(
        "areal.export_stats",
        tm.MethodAnnotated[
            tm.Summary("Export the stats of the engine"),
            tm.Description(""),
            tm.Tag("engine"),
            tm.Error(
                code=-32000,
                message="Export Stats Error",
                data={"message": "Export Stats Error"},
                status_code=500,
            ),
        ],
    )
    def rpc_export_stats(
        reset: t.Annotated[bool, tp.Summary("Reset stats")],
    ) -> t.Annotated[Response, tp.Summary("Response")]:
        return engine_rpc_server.export_stats(reset)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000, type=int, required=False)
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port)
