"""
This module provides the MCPManager class for managing MCP (Model Control Protocol) servers.

The MCPManager class is responsible for loading and managing server configurations,
setting parameters, and building clients for MCP servers. It supports both stdio
and SSE (Server-Sent Events) transport types.
"""
import copy
# pylint: disable=broad-exception-caught
import os
import json
import asyncio
from typing import Dict, List, Union, Any, Optional
from contextlib import AsyncExitStack
from functools import lru_cache

from dotenv import load_dotenv
from mcpuniverse.common.misc import AutodocABCMeta
from mcpuniverse.common.logger import get_logger
from mcpuniverse.common.context import Context
from .config import ServerConfig
from .client import MCPClient

load_dotenv()


class MCPManager(metaclass=AutodocABCMeta):
    """
    Manages MCP (Model Control Protocol) servers.

    This class is responsible for loading server configurations, setting parameters,
    and building clients for MCP servers. It supports both stdio and SSE (Server-Sent Events)
    transport types.

    Attributes:
        _server_configs (Dict[str, ServerConfig]): A dictionary of server configurations.
        _logger: Logger instance for this class.
    """

    def __init__(
            self,
            config: Optional[Union[str, Dict]] = None,
            context: Optional[Context] = None
    ):
        """
        Initializes an MCPManager instance.

        Args:
            config (Union[str, Dict], optional): The configuration file path or a dictionary
                containing server configurations. If None, the default configuration file
                will be used.
            context (Context, optional): The context information, e.g., environment variables or metadata.
        """
        self._server_configs: Dict[str, ServerConfig] = {}
        self._logger = get_logger(self.__class__.__name__)
        self.load_configs(config)
        # Set params defined in the environment variables
        self._context = context
        params = None if context is None else context.env
        for name in self._server_configs:
            self.set_params(server_name=name, params=params)

    def load_configs(self, config: Union[str, Dict] = None):
        """
        Loads server configurations from a file or dictionary.

        Args:
            config (Union[str, Dict], optional): The configuration file path or a dictionary
                containing server configurations. If None, the default configuration file
                will be used.

        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
            ValueError: If there are duplicate server names in the configuration.
            Exception: If there's an error loading a server's configuration.

        Note:
            If a configuration fails to load for a specific server, a fatal log message
            will be recorded, but the method will continue loading other configurations.
        """
        if isinstance(config, dict):
            configs = config
        else:
            if config is None or config == "":
                folder = os.path.dirname(os.path.realpath(__file__))
                config = os.path.join(folder, "configs/server_list.json")
            assert os.path.isfile(config), f"File `{config}` does not exist"
            configs = MCPManager._open_config(config)

        self._server_configs = {}
        for name, conf in configs.items():
            try:
                self._server_configs[name] = ServerConfig.from_dict(copy.deepcopy(conf))
            except Exception as e:
                self._logger.error("Failed to load config of server `%s`: %s", name, str(e))
                raise e

    @staticmethod
    @lru_cache(maxsize=20)
    def _open_config(filepath: str) -> Any:
        """
        Open a configuration file.

        Args:
            filepath (str): The path of the configuration file.

        Returns:
            Any: The loaded configuration.
        """

        def _raise_on_duplicates(ordered_pairs):
            """Reject duplicate server names"""
            d = {}
            for k, v in ordered_pairs:
                if k in d:
                    raise ValueError(f"Duplicate server name: {k}")
                d[k] = v
            return d

        with open(filepath, "r", encoding="utf-8") as f:
            configs = json.load(f, object_pairs_hook=_raise_on_duplicates)
        return configs

    def set_params(self, server_name: str, params: Dict = None):
        """
        Sets parameters for a specific server.

        Args:
            server_name (str): The name of the server to set parameters for.
            params (Dict, optional): A dictionary of parameters to set. If None,
                only environment variables will be applied.
        """
        assert server_name in self._server_configs, f"Unknown server: {server_name}"
        self._server_configs[server_name].render_template(params=params)

    def list_unspecified_params(self, ignore_port: bool = True) -> Dict[str, List[str]]:
        """
        Lists parameters with unspecified values for all servers.

        Args:
            ignore_port (bool, optional): Whether to ignore environment variable `PORT` in the return.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are server names and values
                are lists of unspecified parameter names for each server.
        """
        ignored = ["{{ PORT }}"] if ignore_port else []
        unspecified_params = {}
        for name, config in self._server_configs.items():
            params = config.list_unspecified_params()
            params = [p for p in params if p not in ignored]
            if params:
                unspecified_params[name] = params
        return unspecified_params

    def get_configs(self) -> Dict[str, ServerConfig]:
        """
        Retrieves all server configurations.

        Returns:
            Dict[str, ServerConfig]: A dictionary of all server configurations,
                where keys are server names and values are ServerConfig objects.
        """
        return self._server_configs

    def get_config(self, name: str) -> ServerConfig:
        """
        Retrieves the configuration for a specific server.

        Args:
            name (str): The name of the server.

        Returns:
            ServerConfig: The configuration object for the specified server.
        """
        if name not in self._server_configs:
            raise RuntimeError(f"Unknown server: {name}")
        return self._server_configs[name]

    async def build_client(
            self,
            server_name: str,
            transport: str = "stdio",
            timeout: int = 30,
            mcp_gateway_address: str = "",
            permissions: Optional[List[Dict[str, str]]] = None
    ) -> MCPClient:
        """
        Builds and returns an MCP client for a specified server.

        Args:
            server_name (str): The name of the MCP server to connect to.
            transport (str, optional): The transport type, either "stdio" or "sse". Defaults to "stdio".
            timeout (int, optional): Connection timeout in seconds. Defaults to 30.
            mcp_gateway_address (str, optional): A specified MCP gateway server address.
            permissions (List[dict], optional): A list of tool permissions.

        Returns:
            MCPClient: An MCP client connected to the specified server.

        Note:
            For SSE transport, the MCP_GATEWAY_ADDRESS environment variable must be set.
        """
        assert transport in ["stdio", "sse"], "Transport type should be `stdio` or `sse`"
        assert server_name in self._server_configs, f"Unknown server: {server_name}"
        server_config = self._server_configs[server_name]
        if transport == "stdio":
            if server_config.stdio.list_unspecified_params():
                raise RuntimeError(f"Server {server_name} has unspecified parameters: "
                                   f"{server_config.list_unspecified_params()}")

        client = MCPClient(name=f"{server_name}_client", permissions=permissions)
        if transport == "stdio":
            await client.connect_to_stdio_server(server_config, timeout=timeout)
        else:
            if mcp_gateway_address:
                gateway_address = mcp_gateway_address
            else:
                gateway_address = os.environ.get("MCP_GATEWAY_ADDRESS", "")
            if gateway_address == "":
                raise ValueError("MCP_GATEWAY_ADDRESS is not set")
            await client.connect_to_sse_server(f"{gateway_address}/{server_name}/sse")
        return client

    async def execute(
            self,
            server_name: str,
            tool_name: str,
            arguments: Dict[str, Any],
            transport: str = "stdio",
            permissions: Optional[List[Dict[str, str]]] = None
    ) -> Any:
        """
        Execute a function provided by an MCP server. This method will first create an MCP client,
        then call the execute function of the MCP client.

        Args:
            server_name (str): The name of the MCP server to connect to.
            tool_name (str): The name of a tool provided by the MCP server.
            arguments (Dict): The input arguments for the tool.
            transport (str, optional): The transport type, either "stdio" or "sse". Defaults to "stdio".
            permissions (List[dict], optional): A list of tool permissions.

        Returns:
            Any: The result of the tool execution.
        """
        async with AsyncExitStack():
            client = await self.build_client(
                server_name=server_name, transport=transport, permissions=permissions)
            try:
                result = await client.execute_tool(tool_name=tool_name, arguments=arguments)
                await client.cleanup()
                return result
            except Exception as e:
                await client.cleanup()
                raise e

    async def list_tools(
            self,
            server_names: str | list[str],
            transport: str = "stdio",
    ) -> list[Any]:
        """
        Retrieves a list of available tools of a MCP server.

        Args:
            server_names (str): The names of the MCP servers to connect to.
            transport (str, optional): The transport type, either "stdio" or "sse". Defaults to "stdio".

        Returns:
            list[Any]: A list of available tools.
        """
        if isinstance(server_names, str):
            server_names = [server_names]
        async with AsyncExitStack():
            clients = [await self.build_client(server_name=name, transport=transport) for name in server_names]
            try:
                results = await asyncio.gather(*[client.list_tools() for client in clients])
                for client in clients[::-1]:
                    await client.cleanup()
                return results
            except Exception as e:
                for client in clients[::-1]:
                    await client.cleanup()
                raise e

    def add_server_config(self, server_name: str, config: Dict[str, Any]):
        """
        Dynamically add a new server configuration to the manager.

        Args:
            server_name (str): The name of the server to add.
            config (Dict[str, Any]): The server configuration dictionary containing 
                transport configurations (stdio, sse) and optional environment variables.
                
        Raises:
            ValueError: If the server name already exists.
            Exception: If there's an error loading the server configuration.
            
        Example:
            >>> manager = MCPManager()
            >>> config = {
            ...     "stdio": {
            ...         "command": "python3",
            ...         "args": ["-m", "my.dynamic.server"]
            ...     },
            ...     "env": {
            ...         "API_KEY": "{{MY_API_KEY}}"
            ...     }
            ... }
            >>> manager.add_server_config("dynamic-server", config)
        """
        if server_name in self._server_configs:
            raise ValueError(
                f"Server '{server_name}' already exists. Use update_server_config() to modify existing servers.")

        try:
            server_config = ServerConfig.from_dict(config)
            self._server_configs[server_name] = server_config
            params = None if self._context is None else self._context.env
            self.set_params(server_name=server_name, params=params)
            self._logger.info("Successfully added server configuration: %s", server_name)
        except Exception as e:
            self._logger.error("Failed to add server configuration '%s': %s", server_name, str(e))
            raise e

    def update_server_config(self, server_name: str, config: Dict[str, Any]):
        """
        Update an existing server configuration.

        Args:
            server_name (str): The name of the server to update.
            config (Dict[str, Any]): The new server configuration dictionary.
                
        Raises:
            RuntimeError: If the server name doesn't exist.
            Exception: If there's an error updating the server configuration.
        """
        if server_name not in self._server_configs:
            raise RuntimeError(f"Unknown server: {server_name}. Use add_server_config() to add new servers.")

        try:
            server_config = ServerConfig.from_dict(config)
            self._server_configs[server_name] = server_config
            params = None if self._context is None else self._context.env
            self.set_params(server_name=server_name, params=params)
            self._logger.info("Successfully updated server configuration: %s", server_name)
        except Exception as e:
            self._logger.error("Failed to update server configuration '%s': %s", server_name, str(e))
            raise e

    def remove_server_config(self, server_name: str):
        """
        Remove a server configuration from the manager.

        Args:
            server_name (str): The name of the server to remove.
                
        Raises:
            RuntimeError: If the server name doesn't exist.
        """
        if server_name not in self._server_configs:
            raise RuntimeError(f"Unknown server: {server_name}")

        del self._server_configs[server_name]
        self._logger.info("Successfully removed server configuration: %s", server_name)

    def list_server_names(self) -> List[str]:
        """
        Get a list of all configured server names.

        Returns:
            List[str]: A list of all server names currently configured.
        """
        return list(self._server_configs.keys())

    @property
    def context(self) -> Optional[Context]:
        """
        Return the context object of the MCP manager.

        Returns:
            Context: The context object.
        """
        return self._context
