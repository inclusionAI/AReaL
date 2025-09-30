import argparse
import gzip
import os
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import AnyStr

import cloudpickle
from tensordict import TensorDict

from areal.api.controller_api import DistributedBatch
from areal.controller.batch import DistributedBatchMemory
from areal.utils import logging

logger = logging.getLogger("RPCServer")


def process_input_to_distributed_batch(*args, **kwargs):
    for i in range(len(args)):
        if isinstance(args[i], DistributedBatch):
            args = list(args)
            args[i] = args[i].get_data()
            args = tuple(args)

    for k in list(kwargs.keys()):
        if isinstance(kwargs[k], DistributedBatch):
            kwargs[k] = kwargs[k].get_data()

    return args, kwargs


def process_output_to_distributed_batch(result):
    if isinstance(result, dict):
        return DistributedBatchMemory.from_dict(result)
    elif isinstance(result, TensorDict):
        return DistributedBatchMemory.from_dict(result.to_dict())
    elif isinstance(result, (list, tuple)):
        return DistributedBatchMemory.from_list(list(result))
    else:
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
                engine_obj, init_args = cloudpickle.loads(decompressed_data)
                EngineRPCServer.engine = engine_obj
                result = EngineRPCServer.engine.initialize(init_args)
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
                method = getattr(EngineRPCServer.engine, action)
                # NOTE: DO NOT print args here, args may be a very huge tensor
                logger.info(f"RPC server calling engine method: {action}")
                args, kwargs = process_input_to_distributed_batch(*args, **kwargs)
                result = method(*args, **kwargs)
                result = process_output_to_distributed_batch(result)
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(cloudpickle.dumps(result))
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
    server = ThreadingHTTPServer(("0.0.0.0", port), EngineRPCServer)
    server.serve_forever()


def get_serve_port(args):
    port = args.port
    port_str = os.environ.get("PORT_LIST", "").strip()

    # Check if PORT_LIST is set
    if port_str:
        # Split by comma and strip whitespace
        ports = [p.strip() for p in port_str.split(",")]
        # Use the first valid port from the list
        if ports and ports[0]:
            try:
                return int(ports[0])
            except ValueError:
                logger.warning(
                    f"Invalid port '{ports[0]}' in PORT_LIST. Falling back to --port argument."
                )
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, required=False)

    args, unknown = parser.parse_known_args()
    port = get_serve_port(args)

    logger.info(f"About to start RPC server on {port}")

    start_rpc_server(port)
