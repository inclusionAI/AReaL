import argparse
import gzip
import inspect
import os
import traceback
from asyncio import Future
from concurrent import futures
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, AnyStr, Dict, List

import cloudpickle
import torch
from tensordict import TensorDict

from areal.api.controller_api import DistributedBatch
from areal.api.engine_api import InferenceEngine
from areal.controller.batch import DistributedBatchMemory
from areal.utils import logging

logger = logging.getLogger("RPCServer")


def tensor_container_to_safe(
    d: Dict[str, Any] | torch.Tensor | List[torch.Tensor], *args, **kwargs
):
    """Apply `t.to(*args, **kwargs)` to all tensors in the dictionary.
    Support nested dictionaries.
    """
    new_dict = {}
    if torch.is_tensor(d):
        return d.to(*args, **kwargs)
    elif isinstance(d, list):
        return [tensor_container_to_safe(v, *args, **kwargs) for v in d]
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_dict[key] = tensor_container_to_safe(value, *args, **kwargs)
            elif torch.is_tensor(value):
                new_dict[key] = value.to(*args, **kwargs)
            else:
                new_dict[key] = value
        return new_dict
    else:
        return d


def process_input_to_distributed_batch(to_device, method, *args, **kwargs):
    """Process input arguments, converting DistributedBatch based on method signature.

    This function inspects the method signature to determine whether each parameter
    expects a dict or list format, then converts DistributedBatch instances accordingly.
    """
    # Get method signature
    try:
        sig = inspect.signature(method)
        parameters = sig.parameters
    except (ValueError, TypeError):
        # Fallback to list if signature inspection fails
        parameters = {}

    def convert_distributed_batch(obj, param_name=None):
        """Convert DistributedBatch based on expected parameter type."""
        if not isinstance(obj, DistributedBatch):
            return obj

        # Determine expected type from parameter annotation
        expected_type = None
        if param_name and param_name in parameters:
            param = parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if annotation == dict or (
                    hasattr(annotation, "__origin__") and annotation.__origin__ is dict
                ):
                    expected_type = "dict"
                elif annotation == list or (
                    hasattr(annotation, "__origin__") and annotation.__origin__ is list
                ):
                    expected_type = "list"

        # Convert based on expected type or fallback to list
        if expected_type == "list":
            return obj.to_list()
        else:
            return obj.get_data()

    # Process args
    new_args = list(args)
    for i, arg in enumerate(new_args):
        if isinstance(arg, DistributedBatch):
            # Try to get parameter name for positional arguments
            param_names = list(parameters.keys())
            param_name = param_names[i] if i < len(param_names) else None
            new_args[i] = convert_distributed_batch(arg, param_name)

    # Process kwargs
    new_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, DistributedBatch):
            new_kwargs[key] = convert_distributed_batch(value, key)
        else:
            new_kwargs[key] = value

    # Apply device transfer
    new_args = tuple(tensor_container_to_safe(new_args, to_device))
    new_kwargs = tensor_container_to_safe(new_kwargs, to_device)

    return new_args, new_kwargs


def process_output_to_distributed_batch(result):
    result = tensor_container_to_safe(result, "cpu")

    if isinstance(result, TensorDict):
        return DistributedBatchMemory.from_dict(result.to_dict())

    if isinstance(result, (Future, futures.Future)):
        return result.result()

    if isinstance(result, list) and result:
        if all(isinstance(item, dict) for item in result):
            is_list_of_dict_str_tensor = True
            for item in result:
                for key, value in item.items():
                    if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                        is_list_of_dict_str_tensor = False
                        break
                if is_list_of_dict_str_tensor:
                    DistributedBatchMemory.from_list(result)

    if isinstance(result, dict) and result:
        is_dict_of_tensor = all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in result.items()
        )
        if is_dict_of_tensor:
            return DistributedBatchMemory.from_dict(result)

    return result


class EngineRPCServer(BaseHTTPRequestHandler):
    engine = None

    def _read_body(self, timeout=120.0) -> AnyStr:
        old_timeout = None
        try:
            length = int(self.headers["Content-Length"])
            old_timeout = self.request.gettimeout()
            logger.info(f"Receive rpc call, path: {self.path}, timeout: {old_timeout}")
            # set max read timeout = 120s here, if read hang raise exception
            self.request.settimeout(timeout)
            return self.rfile.read(length)
        except Exception as e:
            raise e
        finally:
            self.request.settimeout(old_timeout)

    def do_POST(self):
        data = None
        try:
            data = self._read_body()
        except Exception as e:
            self.send_response(
                HTTPStatus.REQUEST_TIMEOUT
            )  # 408 means read request timeout
            self.end_headers()
            self.wfile.write(
                f"Exception: {e}\n{traceback.format_exc()}".encode("utf-8")
            )
            logger.error(f"Exception in do_POST: {e}\n{traceback.format_exc()}")
            return

        try:
            if self.path == "/create_engine":
                decompressed_data = gzip.decompress(data)
                engine_obj, args, kwargs = cloudpickle.loads(decompressed_data)
                EngineRPCServer.engine = engine_obj
                result = EngineRPCServer.engine.initialize(*args, **kwargs)
                logger.info(f"Engine created and initialized on RPC server: {result}")
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(cloudpickle.dumps(result))
            elif self.path == "/call":
                if EngineRPCServer.engine is None:
                    self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                    self.end_headers()
                    self.wfile.write(b"Engine is none")
                    logger.error("Call received but engine is none.")
                    return
                action, args, kwargs = cloudpickle.loads(data)
                logger.info(
                    f"Received call for action: {action} with args: {args} kwargs: {kwargs}"
                )

                method = getattr(EngineRPCServer.engine, action)

                # NOTE: DO NOT print args here, args may be a very huge tensor
                if isinstance(EngineRPCServer.engine, InferenceEngine):
                    device = "cpu"
                else:  # actor
                    device = EngineRPCServer.engine.device

                args, kwargs = process_input_to_distributed_batch(
                    device, method, *args, **kwargs
                )
                if (
                    check_attribute_type(type(EngineRPCServer.engine), action)
                    == "method"
                ):
                    result = method(*args, **kwargs)
                    result = process_output_to_distributed_batch(result)
                else:
                    result = method
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(cloudpickle.dumps(result))
            elif self.path == "/health":
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
        except Exception as e:
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.end_headers()
            self.wfile.write(
                f"Exception: {e}\n{traceback.format_exc()}".encode("utf-8")
            )
            logger.error(f"Exception in do_POST: {e}\n{traceback.format_exc()}")


def start_rpc_server(port):
    # NOTE: We must use HTTPServer rather than ThreadingHTTPServer here, since the rank and device info
    # of pytorch is thread level, if use ThreadingHTTPServer, the device set by create_engine thread
    # will not be seen by call_engine thread.
    # server = ThreadingHTTPServer(("0.0.0.0", port), EngineRPCServer)
    server = HTTPServer(("0.0.0.0", port), EngineRPCServer)
    server.serve_forever()


def check_attribute_type(cls, attr_name):
    if hasattr(cls, attr_name):
        attr = getattr(cls, attr_name)  # 从类获取
        if isinstance(attr, property):
            return "property"
        elif callable(attr):
            return "method"
        else:
            raise f"unsupported attr, type: {type(attr)}, name: {attr_name}"
    raise f"attr not found, name: {attr_name}"


def get_server_port(port: int) -> int:
    if port:
        return port
    port_list = os.environ.get("PORT_LIST", "").strip()
    ports = [p.strip() for p in port_list.split(",")]
    # use the first port as serve port
    return int(ports[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=False)

    args, unknown = parser.parse_known_args()
    port = get_server_port(args.port)

    logger.info(f"About to start RPC server on {port}")
    start_rpc_server(port)
