"""
Configuration module for MCP servers.

This module defines the configuration classes for an MCP server, including
command configurations and server configurations.
"""
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from jinja2 import Environment, meta
from mcpuniverse.common.config import BaseConfig


@dataclass
class CommandConfig(BaseConfig):
    """
    Configuration class for a command.

    This class represents the configuration for a single command, including
    the command itself and its arguments.

    Attributes:
        command (str): The command string.
        args (List): A list of command arguments.
    """
    command: str = ""
    args: List = field(default_factory=list)

    def render_template(self, params: Dict):
        """
        Render the command arguments using the provided parameters.

        This method processes the command arguments, replacing any template
        variables with values from the provided parameters.

        Args:
            params (Dict): A dictionary of parameter names and values.
        """
        new_args = []
        for arg in self.args:
            if isinstance(arg, str):
                env = Environment(trim_blocks=True, lstrip_blocks=True)
                template = env.from_string(arg)
                undefined_vars = meta.find_undeclared_variables(env.parse(arg))
                d = {var: params.get(var, f"{{{{ {var} }}}}") for var in undefined_vars}
                new_args.append(template.render(**d))
        self.args = new_args

    def list_unspecified_params(self) -> List[str]:
        """
        List parameters in the command arguments that are not specified.

        This method identifies and returns a list of parameter names that appear
        in the command arguments but don't have specified values.

        Returns:
            List[str]: A list of unspecified parameter names.
        """
        return [arg for arg in self.args if re.findall(r"\{\{.*?\}\}", "".join(arg.strip().split()))]


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration class for an MCP server.

    This class represents the complete configuration for an MCP server,
    including standard I/O, SSE, and environment configurations.

    Attributes:
        stdio (CommandConfig): Configuration for standard I/O command.
        sse (CommandConfig): Configuration for SSE command.
        env (Dict): Dictionary of environment variables.
    """
    stdio: CommandConfig = field(default_factory=CommandConfig)
    sse: CommandConfig = field(default_factory=CommandConfig)
    env: Dict = field(default_factory=dict)

    def render_template(self, params: Optional[Dict] = None):
        """
        Render the server configuration using the provided parameters.

        This method processes the server configuration, including stdio, sse,
        and environment variables, replacing any template variables with values
        from the provided parameters and the current environment.

        Args:
            params (Optional[Dict]): A dictionary of parameter names and values.
                If None, only the current environment variables are used.
        """
        env_params = dict(os.environ)
        if params is not None:
            env_params.update(params)

        self.stdio.render_template(env_params)
        self.sse.render_template(env_params)
        for key, value in self.env.items():
            if isinstance(value, str):
                env = Environment(trim_blocks=True, lstrip_blocks=True)
                template = env.from_string(value)
                undefined_vars = meta.find_undeclared_variables(env.parse(value))
                d = {}
                for var in undefined_vars:
                    value = env_params.get(var, f"{{{{ {var} }}}}")
                    if value == "":
                        value = f"{{{{ {var} }}}}"
                    d[var] = value
                self.env[key] = template.render(**d)

    def list_unspecified_params(self) -> List[str]:
        """
        List parameters in the server configuration that are not specified.

        This method identifies and returns a list of parameter names that appear
        in the server configuration (including stdio, sse, and environment variables)
        but don't have specified values.

        Returns:
            List[str]: A list of unspecified parameter names.
        """
        env_args = [arg for arg in self.env.values()
                    if re.findall(r"\{\{.*?\}\}", "".join(arg.strip().split()))]
        return env_args + self.stdio.list_unspecified_params() + self.sse.list_unspecified_params()
