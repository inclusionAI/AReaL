import typing as t
from enum import Enum
from typing import Any

import flask_jsonrpc.types.params as tp
from pydantic import BaseModel
from typing_extensions import Self

from areal.api.cli_args import (
    InferenceEngineConfig,
    NameResolveConfig,
    TrainEngineConfig,
)


class BaseException(Exception):
    def __init__(
        self: Self, message: t.Annotated[str, tp.Summary("Exception reason")]
    ) -> None:
        super().__init__(message)


class InvalidParamsException(BaseException):
    def __init__(self: Self, params: t.Annotated[str, tp.Summary("")]) -> None:
        super().__init__(message=f"Invalid Params Received: {params}")


class EngineNameEnum(str, Enum):
    FSDP = "fsdp"
    MEGATRON = "megatron"
    SGLANG_REMOTE = "sglang_remote"
    VLLM_REMOTE = "vllm_remote"


class ConfigurePayload(BaseModel):
    seed_cfg: dict
    name_resolve: NameResolveConfig


class CreateEnginePayload(BaseModel):
    config: TrainEngineConfig | InferenceEngineConfig
    class_name: EngineNameEnum
    initial_args: dict[str, Any]


class CallEnginePayload(BaseModel):
    method: str
    args: list[Any]
    kwargs: dict[str, Any]


class Response(BaseModel):
    success: bool
    message: str
    data: Any | None = None
