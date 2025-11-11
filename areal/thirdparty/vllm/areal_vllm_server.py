import asyncio
import logging
from http import HTTPStatus

import uvloop
from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.api_server import (
    create_completion as original_create_completion,
)
from vllm.entrypoints.openai.api_server import (
    router,
    run_server,
    validate_json_request,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ErrorResponse,
    OpenAIBaseModel,
)
from vllm.entrypoints.utils import cli_env_setup, load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

logger = init_logger("areal_vllm_server")
logger.setLevel(logging.INFO)

# Global event to control generation pause/resume
_generation_pause_event = asyncio.Event()
_generation_pause_event.set()  # Initially not paused


class UpdateWeightsRequest(OpenAIBaseModel):
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: str | None = "auto"
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False


class UpdateGroupRequest(OpenAIBaseModel):
    master_address: str
    master_port: str
    rank_offset: int
    world_size: int
    backend: str
    group_name: str


class UpdateWeightsFromXcclRequest(OpenAIBaseModel):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str


def to_json_response(success, message):
    content = {"success": success, "message": message}
    if success:
        return JSONResponse(content, status_code=200)
    else:
        return JSONResponse(content, status_code=400)


def build_response(ret_list):
    success = True
    message = ""
    for rank, ret_value in enumerate(ret_list):
        if_success, msg = ret_value
        success = success if if_success else False
        if if_success:
            message += f"TP rank: {rank} success\n"
        else:
            message += f"TP rank: {rank} failed. reason: {msg}\n"
    return to_json_response(success, message)


@router.post("/areal_update_weights")
async def update_weight(request: UpdateWeightsRequest, raw_request: Request):
    logger.info(f"API server starts update_weight, {request.model_path}")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.engine_core.call_utility_async(
        "areal_injected_update_weight",
        request.model_path,
    )
    return build_response(ret_list)


@router.post("/areal_update_weights_xccl")
async def update_weight_xccl(raw_request: Request):
    logger.info("API server starts update_weight")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.engine_core.call_utility_async(
        "areal_injected_update_weight_xccl",
    )
    return build_response(ret_list)


@router.post("/areal_init_weights_update_group")
async def init_weights_update_group(request: UpdateGroupRequest, raw_request: Request):
    logger.info("API server starts init_weights_update_group")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "init_update_weight_group",
        args=(
            request.master_address,
            request.master_port,
            request.rank_offset,
            request.world_size,
            request.backend,
            request.group_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_set_update_weight_meta")
async def set_weight_meta_xccl(
    request: UpdateWeightsFromXcclRequest, raw_request: Request
):
    logger.info("API server starts upload meta")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "set_weight_meta",
        args=(
            request.names,
            request.dtypes,
            request.shapes,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_pause_generation")
async def pause_generation(raw_request: Request):
    logger.info("API server starts pause_generation and aborts all requests")
    llm = raw_request.app.state.engine_client
    # Abort all running and waiting requests
    await llm.engine_core.call_utility_async("abort_all_reqs")
    _generation_pause_event.clear()
    return to_json_response(True, "Generation paused and all requests aborted")


@router.post("/areal_continue_generation")
async def continue_generation(raw_request: Request):
    logger.info("API server starts continue_generation")
    _generation_pause_event.set()
    return to_json_response(True, "Generation continued")


async def _wait_if_paused():
    """Wait if generation is paused."""
    if not _generation_pause_event.is_set():
        await _generation_pause_event.wait()


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Wrapped completions endpoint that respects pause state."""

    await _wait_if_paused()

    # Will not use streaming response here.
    response = await original_create_completion(request, raw_request)

    return response


# engine core related hook functions
def abort_all_reqs(self):
    """Abort all running and waiting requests and clean up resources."""
    scheduler = self.scheduler
    abort_lists = list(scheduler.running) + list(scheduler.waiting)

    if not abort_lists:
        # No requests to abort
        success = scheduler.reset_prefix_cache()
        assert success, (
            f"prefix cache must be reset to prevent kv cache pollution! {success}"
        )
        return

    client_outputs = {}
    for req in abort_lists:
        engine_output = EngineCoreOutput(
            request_id=req.request_id,
            new_token_ids=[],
            finish_reason=FinishReason.ABORT,
            new_logprobs=None,
            new_prompt_logprobs_tensors=None,
            stop_reason=None,
        )
        if req.client_index not in client_outputs:
            client_outputs[req.client_index] = []
        client_outputs[req.client_index].append(engine_output)

    request_ids = [req.request_id for req in abort_lists]
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    for client_index, outputs in client_outputs.items():
        engine_core_outputs = EngineCoreOutputs(outputs=outputs)
        self.output_queue.put_nowait((client_index, engine_core_outputs))

    success = scheduler.reset_prefix_cache()
    assert success, (
        f"prefix cache must be reset to prevent kv cache pollution! {success}"
    )


def areal_injected_update_weight(self, path):
    self.abort_all_reqs()
    return self.collective_rpc("update_weights", args=(path,))


def areal_injected_update_weight_xccl(self):
    self.abort_all_reqs()
    return self.collective_rpc("update_weight_xccl")


def hook():
    setattr(EngineCore, "abort_all_reqs", abort_all_reqs)
    setattr(EngineCore, "areal_injected_update_weight", areal_injected_update_weight)
    setattr(
        EngineCore,
        "areal_injected_update_weight_xccl",
        areal_injected_update_weight_xccl,
    )


hook()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.f
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
