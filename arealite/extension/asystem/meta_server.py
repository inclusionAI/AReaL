import logging
import asyncio
import threading
import pickle
import json
import time
import traceback
from typing import Any, Dict, Tuple, Union
from aiohttp import web
import socket
import requests
import struct
import os

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, force=True):
  logging.basicConfig(
    level=level,
    format="%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(process)d -- %(message)s",
    force=force,
)


def get_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def to_binary(data):
    # Serialize messages using pickle
    pickled_data = pickle.dumps(data)
    # Get the length of the pickled data
    data_len = len(pickled_data)
    # Create the binary response: length of data (4 bytes) + pickled data
    return struct.pack("!I", data_len) + pickled_data


def from_binary(binary):
    # Extract the length of the pickled data
    data_len = struct.unpack("!I", binary[:4])[0]
    # Extract and unpickle the data
    pickled_data = binary[4 : 4 + data_len]
    data = pickle.loads(pickled_data)
    return data


class MetaServer:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.storage: Dict[str, Any] = {}
        self.app = web.Application(
            middlewares=[self.error_handler],
            client_max_size=2 * 1024**3,  # 2GB max size
        )
        self.id_counter = 0
        self.runner = None
        self.site = None
        self._server_thread = None
        self._server_error = None

        # Define routes
        self.app.router.add_get("/v1/get_binary/{key}", self.get_binary_handler)
        self.app.router.add_put("/v1/put_binary/{key}", self.put_binary_handler)
        self.app.router.add_put(
            "/v1/add_object_to_set/{key}", self.add_object_to_set_handler
        )
        self.app.router.add_get("/v1/get_json/{key}", self.get_json_handler)
        self.app.router.add_put("/v1/put_json/{key}", self.put_json_handler)
        self.app.router.add_delete("/v1/delete/{key}", self.delete_handler)
        self.app.router.add_get("/v1/health", self.health_check)
        self.app.router.add_get("/v1/keys", self.list_keys)
        self.app.router.add_get("/v1/has_key/{key}", self.has_key)
        self.app.router.add_post(
            "/v1/allocate_auto_grow_id", self.allocate_auto_grow_id
        )
        logger.info(f"[{os.getpid()}] Created meta server with {self.host}:{self.port}")

    def get_binary(self, key: str) -> bytes:
        """Get binary data from storage"""
        return self.storage[key]

    def put_binary(self, key: str, binary: bytes):
        """Store binary data"""
        self.storage[key] = binary

    def get_object(self, key: str) -> Any:
        """Get pickled object from storage"""
        return pickle.loads(self.storage[key])

    def put_object(self, key: str, obj: Any):
        """Store object as pickled data"""
        self.storage[key] = pickle.dumps(obj)

    def get_json(self, key: str) -> dict:
        """Get JSON data from storage"""
        return json.loads(self.storage[key].decode("utf-8"))

    def put_json(self, key: str, json_data: dict):
        """Store JSON data"""
        self.storage[key] = json.dumps(json_data).encode("utf-8")

    def delete(self, key: str):
        """Delete data from storage"""
        if key in self.storage:
            del self.storage[key]
        else:
            logger.warning(f"Key '{key}' not found in storage")
            raise ValueError(f"Key '{key}' not found in storage")

    def get_address(self) -> str:
        """Get server address"""
        return self.host

    def get_port(self) -> int:
        """Get server port"""
        return self.port

    def get_address_and_port(self) -> Tuple[str, int]:
        """Get server address and port"""
        return self.host, self.port

    def _run_server(self):
        """Run the server in a separate thread"""
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
        logger.info(f"[{os.getpid()}] Start server with {self.host}, {self.port}")
        try:
            # If port is 0, let the OS assign a port
            if self.port == 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    actual_port = s.getsockname()[1]
                    s.close()
                self.port = actual_port

            web.run_app(self.app, host=self.host, port=self.port, handle_signals=False)
        except BaseException as e:
            logger.exception(
                f"[{os.getpid()}] Start server with {self.host}, {self.port} failed"
            )
            self._server_error = e
            raise e

    def start(self):
        """Start the server in a daemon thread"""
        self._server_thread = threading.Thread(target=self._run_server, daemon=True, name="AsystemMetaServer")
        self._server_thread.start()

        # Poll health check until server is ready
        max_attempts = 50  # 5 seconds with 0.1s intervals
        for attempt in range(max_attempts):
            # Check if server thread failed
            if not self._server_thread.is_alive():
                if self._server_error:
                    raise RuntimeError(
                        f"Server failed to start on {self.host}:{self.port}: {self._server_error}"
                    ) from self._server_error
                else:
                    raise RuntimeError(
                        f"Server thread died unexpectedly on {self.host}:{self.port}"
                    )

            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/v1/health", timeout=0.5
                )
                if response.status_code == 200:
                    logger.info(f"Meta server is ready on {self.host}:{self.port}")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)

        # Timeout reached
        raise RuntimeError(
            f"Server failed to respond within timeout on {self.host}:{self.port}"
        )

    def stop(self):
        """Stop the server"""
        if self.runner:
            asyncio.run(self.runner.cleanup())

    @web.middleware
    async def error_handler(self, request, handler):
        """A middleware error handler."""
        try:
            response = await handler(request)
            if isinstance(response, web.Response):
                return response
            else:
                return web.json_response({"success": True, "data": response})
        except web.HTTPException as e:
            return web.json_response(
                {"success": False, "error": str(e)}, status=e.status_code
            )
        except Exception as e:
            logger.error("An error occurred: %s", str(e), exc_info=True)
            tb_str = "".join(
                traceback.format_exception(type(e), value=e, tb=e.__traceback__)
            )
            return web.json_response(
                {"success": False, "error": f"An error occurred: {tb_str}."}, status=500
            )

    async def get_binary_handler(self, request):
        """Handle GET binary request"""
        key = request.match_info["key"]
        if key not in self.storage:
            return web.json_response(
                {"success": False, "error": f"Key '{key}' not found"}, status=404
            )
        data = self.storage[key]
        
        # If data is already bytes, return it directly
        if isinstance(data, bytes):
            return web.Response(body=data, content_type="application/octet-stream")
        
        # Otherwise, serialize the object to binary format
        try:
            binary_data = to_binary(data)
            return web.Response(body=binary_data, content_type="application/octet-stream")
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to serialize object: {str(e)}"},
                status=500,
            )

    async def put_binary_handler(self, request):
        """Handle PUT binary request"""
        key = request.match_info["key"]
        data = await request.read()
        self.storage[key] = data
        return web.json_response(
            {"success": True, "message": f"Binary data stored for key '{key}'"}
        )

    async def add_object_to_set_handler(self, request):
        """Handle PUT object to set request"""
        key = request.match_info["key"]
        try:
            # Read binary data and use from_binary to deserialize
            binary_data = await request.read()
            obj = from_binary(binary_data)
            # Store the pickled data
            if key not in self.storage:
                self.storage[key] = set()
            self.storage[key].add(obj)
            return web.json_response(
                {"success": True, "message": f"Object stored for key '{key}'"}
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to deserialize object: {str(e)}"},
                status=500,
            )

    async def get_json_handler(self, request):
        """Handle GET JSON request"""
        key = request.match_info["key"]
        if key not in self.storage:
            return web.json_response(
                {"success": False, "error": f"Key '{key}' not found"}, status=404
            )
        try:
            json_data = json.loads(self.storage[key].decode("utf-8"))
            return web.json_response({"success": True, "data": json_data})
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to parse JSON: {str(e)}"},
                status=500,
            )

    async def put_json_handler(self, request):
        """Handle PUT JSON request"""
        key = request.match_info["key"]
        try:
            data = await request.json()
            self.storage[key] = json.dumps(data).encode("utf-8")
            return web.json_response(
                {"success": True, "message": f"JSON data stored for key '{key}'"}
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to store JSON: {str(e)}"},
                status=500,
            )

    async def delete_handler(self, request):
        """Handle DELETE request"""
        key = request.match_info["key"]
        try:
            self.delete(key)
            return web.json_response(
                {"success": True, "message": f"Key '{key}' deleted"}
            )
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=404)
        except Exception as e:
            logger.error("An error occurred during deletion: %s", str(e), exc_info=True)
            tb_str = "".join(
                traceback.format_exception(type(e), value=e, tb=e.__traceback__)
            )
            return web.json_response(
                {
                    "success": False,
                    "error": f"An error occurred during deletion: {tb_str}.",
                },
                status=500,
            )

    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response(
            {
                "success": True,
                "status": "healthy",
                "storage_size": len(self.storage),
                "address": self.host,
                "port": self.port,
            }
        )

    async def has_key(self, request):
        """Check if a key exists on server"""
        key = request.match_info["key"]
        return web.json_response({"success": True, "has_key": key in self.storage})

    async def list_keys(self, request):
        """List all stored keys"""
        return web.json_response({"success": True, "keys": list(self.storage.keys())})

    async def allocate_auto_grow_id(self, request):
        """Allocate an auto grow id"""
        new_id = self.id_counter
        self.id_counter += 1
        return web.json_response({"success": True, "id": new_id})


def start_meta_server(host: str = "", port: int = 0) -> Tuple[str, int]:
    """
    Start a meta server on a daemon thread, and return address and port
    """
    if host == "":
        host = get_ip_address()
    if port == 0:
        # Let the OS assign a random port
        port = get_free_port()

    server = MetaServer(host, port)
    try:
        server.start()
    except Exception as e:
        logger.exception(f"Failed to start meta server on {host}:{port}")
        raise RuntimeError(f"Meta server startup failed: {e}") from e

    # Store the server instance globally so it doesn't get garbage collected
    start_meta_server._server_instance = server

    return server.get_address_and_port()
