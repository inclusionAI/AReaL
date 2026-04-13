"""
The application gateway for MCP servers.
"""
# pylint: disable=broad-exception-caught,protected-access,no-value-for-parameter
import os
import copy
import socket
import shutil
import asyncio
import subprocess
import multiprocessing
from typing import List, Optional
from contextlib import closing, AsyncExitStack, asynccontextmanager

import anyio
import click
import uvicorn
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from starlette.routing import Mount, Route
from starlette.applications import Starlette
from mcp.client.sse import sse_client
from mcp.server.sse import SseServerTransport
from mcp import stdio_client, StdioServerParameters
from mcpuniverse.common.logger import get_logger
from mcpuniverse.common.misc import AutodocABCMeta
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.mcp.config import ServerConfig


class ServerConnector(metaclass=AutodocABCMeta):
    """
    Connect to another MCP server.
    """

    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._logger = get_logger(self.__class__.__name__)
        self._read_stream = None  # Read responses from the MCP server
        self._write_stream = None  # Send requests to the MCP server

    async def cleanup(self):
        """
        Clean up client resources.
        """
        async with self._cleanup_lock:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                self._logger.error("Error during cleanup of server connector: %s", str(e))

    async def connect_to_sse_server(self, server_url: str):
        """
        Connect to an MCP server via SSE.

        Args:
            server_url (str): Server address.
        """
        try:
            sse_transport = await self._exit_stack.enter_async_context(sse_client(url=server_url))
            self._read_stream, self._write_stream = sse_transport
        except Exception as e:
            self._logger.error("Failed to initialize sse client in server connector: %s", str(e))
            await self.cleanup()
            raise e

    async def connect_to_stdio_server(self, config: ServerConfig):
        """
        Initializes a connection to an MCP server using stdio transport.

        Args:
            config (ServerConfig): Configuration object containing server settings.
        """
        command = (
            shutil.which(config.stdio.command)
            if config.stdio.command in ["npx", "docker", "python", "python3"]
            else config.stdio.command
        )
        if command is None or command == "":
            raise ValueError("The command must be a valid string")

        envs = dict(os.environ)
        envs.update(config.env)
        server_params = StdioServerParameters(
            command=command,
            args=config.stdio.args,
            env=envs
        )
        try:
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self._read_stream, self._write_stream = stdio_transport
        except Exception as e:
            self._logger.error("Failed to initialize stdio client in server connector: %s", str(e))
            await self.cleanup()
            raise e

    async def _send(self, read_stream: MemoryObjectReceiveStream):
        """
        Reads requests from read_stream and sends them to the MCP server.

        Args:
            read_stream (MemoryObjectReceiveStream): A read stream.
        """
        async with (
            read_stream,
            self._write_stream,
        ):
            async for message in read_stream:
                await self._write_stream.send(message)

    async def _receive(self, write_stream: MemoryObjectSendStream):
        """
        Reads responses from the MCP server and sends them to the output stream.

        Args:
            write_stream (MemoryObjectSendStream): A write stream.
        """
        async with (
            write_stream,
            self._read_stream,
        ):
            async for message in self._read_stream:
                await write_stream.send(message)

    async def run(
            self,
            read_stream: MemoryObjectReceiveStream,
            write_stream: MemoryObjectSendStream
    ):
        """
        Redirect requests from read_stream to the MCP server and write responses to write_stream.

        Args:
            read_stream (MemoryObjectReceiveStream): A read stream.
            write_stream (MemoryObjectSendStream): A write stream.
        """
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._send, read_stream)
            tg.start_soon(self._receive, write_stream)


class Gateway(metaclass=AutodocABCMeta):
    """
    The application gateway for available MCP servers.
    This is used when you want to deploy MCP servers in remote machines.
    """

    def __init__(self, mcp_manager: MCPManager):
        self._mcp_manager = mcp_manager
        self._server_configs = mcp_manager.get_configs()
        self._processes = {}
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._logger = get_logger(self.__class__.__name__)

    async def cleanup(self):
        """
        Clean up resources.
        """
        for name, p in self._processes.items():
            try:
                process = p.get("process", None)
                if process:
                    process.terminate()
            except Exception as e:
                self._logger.error("Error during cleanup of server %s: %s", name, str(e))
        self._processes = {}

    def _find_available_port(self, start_port=10000, end_port=65535) -> int:
        """
        Finds a free port number.

        Returns:
            int: A port number.
        """
        used_ports = set(p["port"] for _, p in self._processes.items())
        for port in range(start_port, end_port + 1):
            if port in used_ports:
                continue
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                    s.bind(("", port))
                    return port
            except Exception:
                continue
        return -1

    def init_sse_server(self, server_name: str):
        """
        Initializes an SSE server corresponding to the server name.

        Args:
            server_name (str): The server name.
        """
        assert server_name in self._server_configs, f"Unknown server: {server_name}"
        if server_name in self._processes:
            return self._processes[server_name]

        config = copy.deepcopy(self._server_configs[server_name])
        if config.sse.command == "":
            raise RuntimeError(f"Server {server_name} does not support SSE")

        port = self._find_available_port()
        if port < 0:
            raise RuntimeError("Cannot find free port")
        config.render_template(params={"PORT": port})
        if config.list_unspecified_params():
            raise RuntimeError(f"Server {server_name} has unspecified parameters: "
                               f"{config.list_unspecified_params()}")

        process = multiprocessing.Process(target=run_server, args=(config.sse.command, config.sse.args))
        self._processes[server_name] = {
            "process": process,
            "port": port,
            "url": f"http://localhost:{port}/sse"
        }
        self._processes[server_name]["routes"] = self._build_sse_routes(server_name)

    def init_stdio_server(self, server_name: str):
        """
        Initializes a Stdio server corresponding to the server name.

        Args:
            server_name (str): The server name.
        """
        assert server_name in self._server_configs, f"Unknown server: {server_name}"
        if server_name in self._processes:
            return self._processes[server_name]

        config = copy.deepcopy(self._server_configs[server_name])
        self._processes[server_name] = {
            "routes": self._build_stdio_routes(server_name, config)
        }

    def start_servers(self, join: bool = True):
        """
        Starts the initialized MCP servers.

        Args:
            join (bool): Whether to do multiprocessing join.
        """
        for name, p in self._processes.items():
            self._logger.info("Starting the MCP server %s with port %d", name, p["port"])
            p["process"].start()
        if join:
            for _, p in self._processes.items():
                p["process"].join()

    def _build_sse_routes(self, server_name: str) -> List:
        """
        Builds Starlette routes for SSE transport.

        Args:
            server_name (str): The server name.

        Returns:
            List: A list of routes.
        """
        if server_name not in self._processes:
            raise RuntimeError(f"Server {server_name} is not initialized.")
        sse = SseServerTransport(f"/{server_name}/messages/")

        async def handle_sse(request):
            connector = ServerConnector()
            async with sse.connect_sse(
                    request.scope, request.receive, request._send
            ) as streams:
                await connector.connect_to_sse_server(self._processes[server_name]["url"])
                await connector.run(streams[0], streams[1])
                await connector.cleanup()

        routes = [
            Route(f"/{server_name}/sse", endpoint=handle_sse),
            Mount(f"/{server_name}/messages/", app=sse.handle_post_message),
        ]
        return routes

    def _build_stdio_routes(self, server_name: str, config: ServerConfig) -> List:
        """
        Builds Starlette routes for Stdio transport.

        Args:
            server_name (str): The server name.

        Returns:
            List: A list of routes.
        """
        sse = SseServerTransport(f"/{server_name}/messages/")

        async def handle_sse(request):
            connector = ServerConnector()
            async with sse.connect_sse(
                    request.scope, request.receive, request._send
            ) as streams:
                await connector.connect_to_stdio_server(config)
                await connector.run(streams[0], streams[1])
                await connector.cleanup()

        routes = [
            Route(f"/{server_name}/sse", endpoint=handle_sse),
            Mount(f"/{server_name}/messages/", app=sse.handle_post_message),
        ]
        return routes

    def build_starlette_app(
            self,
            mode: str = "stdio",
            servers: Optional[List[str]] = None,
            debug: bool = True
    ) -> Starlette:
        """
        Builds a Starlette app for the gateway.

        Args:
            mode (str): Launch MCP clients via "stdio" or "sse".
            servers (List[str]): A list of selected MCP servers.
            debug (bool): Use debug mode.

        Returns:
            Starlette: A Starlette app.
        """
        assert mode in ["stdio", "sse"], "`mode` should be `stdio` or `sse`"

        # Start MCP servers
        if mode == "sse":
            for server_name, config in self._server_configs.items():
                if servers and server_name not in servers:
                    continue
                if config.sse.command != "":
                    self.init_sse_server(server_name)
            self.start_servers(join=False)
        else:
            for server_name, _ in self._server_configs.items():
                if servers and server_name not in servers:
                    continue
                self.init_stdio_server(server_name)

        # Create a Starlette app
        routes = []
        for server_name, process in self._processes.items():
            routes.extend(process["routes"])

        # Lifespan
        @asynccontextmanager
        async def lifespan(_: Starlette):
            yield
            await self.cleanup()

        return Starlette(debug=debug, routes=routes, lifespan=lifespan)


def run_server(command, args):
    """
    Runs a shell command.

    Args:
        command: A command.
        args: Command arguments.
    """
    command = (
        shutil.which(command)
        if command in ["npx", "docker", "python", "python3"]
        else command
    )
    subprocess.run([command] + args, shell=False, check=True)


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option("--config", default="", help="Server config file")
@click.option("--mode", default="stdio", help="Launch MCP clients via 'stdio' or 'sse'")
@click.option("--servers", default="", help="A list of servers to use")
def main(port: int, config: str, mode: str, servers: str):
    """Start the gateway server"""
    manager = MCPManager(config=config)
    gateway = Gateway(mcp_manager=manager)
    servers = [s.strip() for s in servers.split(",") if s.strip()]
    app = gateway.build_starlette_app(mode=mode, servers=servers)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
