# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import gc
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import realhf.base.constants as constants
import realhf.base.logging as logging

logger = logging.getLogger("CUDAGraph")

CUDA_GRAPH_STORAGE: Dict[str, torch.cuda.CUDAGraph] = dict()
CUDA_GRAPH_INPUT_BUFFER: Dict[str, Dict[str, torch.Tensor]] = dict()
CUDA_GRAPH_OUTPUT_BUFFER: Dict[str, Dict[str, torch.Tensor]] = dict()
CUDA_GRAPH_FIRST_CAPTURE: Dict[str, bool] = defaultdict(lambda: True)
CUDA_GRAPH_DESTROYED: Dict[str, bool] = defaultdict(lambda: False)


@contextmanager
def capture_context(graph: torch.cuda.CUDAGraph):
    # NOTE: We use lower level API in pytorch CUDAGraph instead of `torch.cuda.graph`
    # context. This is because in `torch.cuda.graph`, `torch.cuda.empty_cache()` and
    # `gc.collect()` are unnecessarily called everytime the context is entered.
    # This will introduce large overhead.
    graph.capture_begin()
    yield
    graph.capture_end()


@contextmanager
def outer_capture_context(stream=None, no_grad=False):
    """Context wrapped outside warmup and CUDAGraph capture context:
    1. Alter stream from default.
    2. Apply torch.no_grad context if required
    """
    if stream is None:
        stream = torch.cuda.Stream()
    maybe_no_grad = nullcontext() if not no_grad else torch.no_grad()
    with torch.cuda.stream(stream), maybe_no_grad:
        yield


def reinitialize_input_buffer(cuda_graph_name, new_buf):
    global CUDA_GRAPH_INPUT_BUFFER
    assert (
        cuda_graph_name in CUDA_GRAPH_INPUT_BUFFER
    ), f"CUDAGraph {cuda_graph_name} does not exist."

    buf = CUDA_GRAPH_INPUT_BUFFER[cuda_graph_name]
    for k, v in buf.items():
        if torch.is_tensor(v):
            v.copy_(new_buf[k])
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                if torch.is_tensor(vv):
                    vv.copy_(new_buf[k][i])
        else:
            buf[k] = new_buf[k]


@torch.no_grad()
def capture_func(
    name: str,
    func: Callable,
    input_buffer: Dict[str, Any],
    force_recapture: bool = False,
    no_grad: bool = False,
) -> Tuple[Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Capture a function with cuda graph, store the graph and input/output
    buffers by name. The input/output metadata should match the inputs and
    outputs of function.

    This function uses pytorch original CUDAGraph implementation

    :param name: The identifier of the CUDAGraph to be captured/reused.
    :type name: str
    :param func: The function to be captured.
    :type func: Callable
    :param input_buffer: The input buffer of the function.
    :type input_buffer: Dict[str, Any]
    :param force_recapture: Whether to force recapture the function.
    :type force_recapture: bool
    :param no_grad: Whether to run the function in no_grad context.
    :type no_grad: bool
    """
    global CUDA_GRAPH_STORAGE
    global CUDA_GRAPH_INPUT_BUFFER
    global CUDA_GRAPH_OUTPUT_BUFFER
    global CUDA_GRAPH_FIRST_CAPTURE
    global CUDA_GRAPH_DESTROYED

    if not force_recapture and not CUDA_GRAPH_FIRST_CAPTURE[name]:
        assert name in CUDA_GRAPH_STORAGE
        assert name in CUDA_GRAPH_INPUT_BUFFER
        assert name in CUDA_GRAPH_OUTPUT_BUFFER
        reinitialize_input_buffer(name, input_buffer)
        return (
            CUDA_GRAPH_STORAGE[name],
            CUDA_GRAPH_INPUT_BUFFER[name],
            CUDA_GRAPH_OUTPUT_BUFFER[name],
        )

    stream = torch.cuda.Stream()
    st = time.monotonic()
    logger.debug(f"Rank {dist.get_rank()}: Capturing CUDA graph for {name}")
    first_capture = CUDA_GRAPH_FIRST_CAPTURE[name]

    with outer_capture_context(stream, no_grad):
        if first_capture:
            func(**input_buffer)  # warmup
            logger.debug(
                f"before clear cache before capture "
                f"mem allocated: {torch.cuda.memory_allocated()/1024/1024:.4f} MB"
                f"mem reserved: {torch.cuda.memory_reserved()/1024/1024:.4f} MB"
            )
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug(
                f"after clear cache after capture "
                f"mem allocated: {torch.cuda.memory_allocated()/1024/1024:.4f} MB"
                f"mem reserved: {torch.cuda.memory_reserved()/1024/1024:.4f} MB"
            )

        graph = torch.cuda.CUDAGraph()

        with capture_context(graph):
            output = func(**input_buffer)

    logger.debug(
        f"Rank {dist.get_rank()}: Capturing CUDA graph {name} "
        f"takes {time.monotonic() - st:.4f} seconds"
    )

    assert torch.is_tensor(output)
    output_buffer = dict(output=output)

    CUDA_GRAPH_STORAGE[name] = graph
    CUDA_GRAPH_INPUT_BUFFER[name] = input_buffer
    CUDA_GRAPH_OUTPUT_BUFFER[name] = output_buffer
    CUDA_GRAPH_FIRST_CAPTURE[name] = False
    CUDA_GRAPH_DESTROYED[name] = False
    return graph, input_buffer, output_buffer


def input_buffer_handle(graph_name: str, tensor_name: str):
    if graph_name not in CUDA_GRAPH_INPUT_BUFFER:
        return None
    if CUDA_GRAPH_DESTROYED[graph_name]:
        return None
    if tensor_name not in CUDA_GRAPH_INPUT_BUFFER[graph_name]:
        raise ValueError(
            f"Tensor {tensor_name} not found in input buffer of graph {graph_name}, "
            f"Existing keys = {CUDA_GRAPH_INPUT_BUFFER[graph_name].keys()}"
        )
    return CUDA_GRAPH_INPUT_BUFFER[graph_name][tensor_name]


def output_buffer_handle(graph_name: str, tensor_name: str):
    if graph_name not in CUDA_GRAPH_OUTPUT_BUFFER:
        return None
    if CUDA_GRAPH_DESTROYED[graph_name]:
        return None
    if tensor_name not in CUDA_GRAPH_OUTPUT_BUFFER[graph_name]:
        raise ValueError(
            f"Tensor {tensor_name} not found in output buffer of graph {graph_name}, "
            f"existing keys = {CUDA_GRAPH_OUTPUT_BUFFER[graph_name].keys()}"
        )
    return CUDA_GRAPH_OUTPUT_BUFFER[graph_name][tensor_name]


def get_graph(name: str):
    return CUDA_GRAPH_STORAGE.get(name, None)


def destroy(name):
    # NOTE: This function should only be used when the graph is not needed
    # or the graph is to be recaptured. It will free memory occupied by pinned input/output buffers
    # and destroy CUDAGraph object.

    global CUDA_GRAPH_STORAGE
    global CUDA_GRAPH_INPUT_BUFFER
    global CUDA_GRAPH_OUTPUT_BUFFER
    global CUDA_GRAPH_FIRST_CAPTURE
    global CUDA_GRAPH_DESTROYED

    assert (
        name in CUDA_GRAPH_STORAGE and not CUDA_GRAPH_FIRST_CAPTURE[name]
    ), f"CUDAGraph {name} should be created before destroy."
    assert CUDA_GRAPH_DESTROYED[name] is False, f"CUDAGraph {name} already destroyed."

    CUDA_GRAPH_STORAGE[name].reset()
    CUDA_GRAPH_STORAGE[name] = None
    CUDA_GRAPH_INPUT_BUFFER[name] = None
    CUDA_GRAPH_OUTPUT_BUFFER[name] = None
    CUDA_GRAPH_DESTROYED[name] = True


def destroy_all():
    global CUDA_GRAPH_STORAGE
    global CUDA_GRAPH_INPUT_BUFFER
    global CUDA_GRAPH_OUTPUT_BUFFER
    global CUDA_GRAPH_FIRST_CAPTURE
    global CUDA_GRAPH_DESTROYED

    for name in CUDA_GRAPH_STORAGE:
        if CUDA_GRAPH_DESTROYED[name]:
            continue
        CUDA_GRAPH_STORAGE[name].reset()
        CUDA_GRAPH_STORAGE[name] = None
        CUDA_GRAPH_INPUT_BUFFER[name] = None
        CUDA_GRAPH_OUTPUT_BUFFER[name] = None
        CUDA_GRAPH_DESTROYED[name] = True
