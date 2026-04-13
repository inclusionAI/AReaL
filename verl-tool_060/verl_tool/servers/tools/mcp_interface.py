import os
import json
from .base import BaseTool, register_tool
import regex as re
from typing import Tuple, Dict, Any, Optional, Union, List

from .utils.mcp_client import (
    list_tools_and_resources,
    call_mcp_tool,
    get_content_text,
    MCPClientError,
)

@register_tool
class MCPInterfaceTool(BaseTool):
    tool_type = "mcp_interface"
    def __init__(self, num_workers=1):
        super().__init__(num_workers=num_workers)
        # Per-server tool registry, server key -> {tool_name: meta}
        # server key: None (single server via MCP_SERVER_URL) or a str server label
        self._server2tools: Dict[Optional[str], Dict[str, Dict[str, Any]]] = {}
        self._tools_loaded: Dict[Optional[str], bool] = {}
        # Defer remote discovery to conduct_action or explicit refresh

    def get_usage_inst(self):
        # Collate any discovered tools across servers for preview
        discovered = []
        for _srv, tools in self._server2tools.items():
            discovered.extend(list(tools.keys()))
        if discovered:
            names = discovered
            preview = ", ".join(names[:6]) + (" ..." if len(names) > 6 else "")
            return f"You can call remote MCP tools discovered from server ({len(names)} tools). Examples: {preview}"
        return "You can call remote MCP tools using <tool_call>{\"name\":..., \"arguments\":{...}}</tool_call>."
    
    def parse_action(self, action: str) -> Tuple[Union[str, Dict[str, Any]], bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        print(f"[MCP PARSE DEBUG] Input action: {action[:200]}...")
        print(f"[MCP PARSE DEBUG] Looking for <tool_call> pattern...")
        # Ensure discovery before parsing to allow tool name existence check
        # Do not force discovery here; accept action even if discovery fails.

        if not isinstance(action, str):
            return "", False

        match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
        data = None
        if match:
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str) if json_str else {}
            except Exception:
                return "", False
        else:
            # Try plain JSON (ReAct style), accept both top-level action and root server/tool
            try:
                j = json.loads(action)
                if isinstance(j, dict):
                    if isinstance(j.get("action"), dict):
                        a = j["action"]
                        data = {
                            "server": a.get("server"),
                            "name": a.get("tool"),
                            "arguments": a.get("arguments", {}),
                        }
                    elif all(k in j for k in ("tool", "arguments")):
                        data = {
                            "server": j.get("server"),
                            "name": j.get("tool"),
                            "arguments": j.get("arguments", {}),
                        }
                    else:
                        return "", False
                else:
                    return "", False
            except Exception:
                return "", False

        if not isinstance(data, dict) or "name" not in data:
            return "", False
        # Optional server hint for legacy routing; keep if provided but do not require it
        server = data.get("server") or data.get("server_name") or data.get("server_label")
        if server is not None and not isinstance(server, str):
            server = str(server)
        data["server"] = server

        # arguments is optional; default to {}, also parse stringified JSON
        if "arguments" not in data or data["arguments"] is None:
            data["arguments"] = {}
        elif isinstance(data["arguments"], str):
            try:
                data["arguments"] = json.loads(data["arguments"]) or {}
            except Exception:
                # keep as string if not a valid JSON
                pass

        # Optional strict check: unknown tool names are invalid when we have discovery
        action_name = data.get("name", "")
        # Allow internal helper tools prefixed by 'mcp.' regardless of discovery
        tools = self._server2tools.get(server) or {}
        if server is not None:
            if tools and (action_name not in tools) and (not str(action_name).startswith("mcp.")):
                return "", False
        else:
            discovered_any = False
            for mapping in self._server2tools.values():
                if mapping:
                    discovered_any = True
                    if action_name in mapping or str(action_name).startswith("mcp."):
                        break
            else:
                if discovered_any and (not str(action_name).startswith("mcp.")):
                    return "", False

        return data, True
    
    def _format_observation(self, tool_name: str, content_text: str) -> str:
        content_text = content_text or ""
        max_len = 20000
        if len(content_text) > max_len:
            content_text = content_text[:max_len] + "\n... (truncated)"
        return f"\n<mcp_response>\n```result\n{content_text}\n```\n</mcp_response>\n"


    def _ensure_discovered(self, server: Optional[str] = None) -> None:
        print(f"[MCP DISCOVERY DEBUG] _ensure_discovered called with server: {server}")
        print(f"[MCP DISCOVERY DEBUG] Current _server2tools: {list(self._server2tools.keys())}")
        print(f"[MCP DISCOVERY DEBUG] Current _tools_loaded: {self._tools_loaded}")
        if server is None:
            return
        # Avoid repeated discovery for the same server key
        if self._tools_loaded.get(server):
            return
        try:
            # Run discovery synchronously by blocking the event loop via async helper
            import asyncio

            async def _load():
                tools, _resources = await list_tools_and_resources(server)
                mapping = {}
                for t in getattr(tools, "tools", []) or []:
                    mapping[getattr(t, "name", "")] = {
                        "description": getattr(t, "description", None),
                        "parameters": getattr(t, "inputSchema", None),
                    }
                return mapping

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Best effort: schedule background refresh, mark as loaded to avoid busy loops
                    self._tools_loaded[server] = True
                    loop.create_task(self._async_refresh(server))
                    return
                else:
                    self._server2tools[server] = loop.run_until_complete(_load())
                    self._tools_loaded[server] = True
            except RuntimeError:
                # No running loop
                self._server2tools[server] = asyncio.run(_load())
                self._tools_loaded[server] = True
        except Exception:
            # Discovery failure should not block; keep empty mapping
            self._tools_loaded[server] = True

    async def _async_refresh(self, server: Optional[str] = None) -> None:
        if server is None:
            return
        try:
            tools, _ = await list_tools_and_resources(server)
            mapping = {}
            for t in getattr(tools, "tools", []) or []:
                mapping[getattr(t, "name", "")] = {
                    "description": getattr(t, "description", None),
                    "parameters": getattr(t, "inputSchema", None),
                }
            self._server2tools[server] = mapping
        except Exception:
            pass
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute MCP tool call with given arguments.
        """
        # Debug: Log that MCP interface is being called
        print(f"[MCP DEBUG] conduct_action called for trajectory {trajectory_id}")
        print(f"[MCP DEBUG] Action: {action[:100]}...")
        # Load or initialize env for this trajectory
        # If this is the first step for the trajectory, refresh discovery synchronously (best‑effort)
        if not self.has_env(trajectory_id):
            try:
                # If action contains server hint, discover on that server
                _server_hint = None
                try:
                    _parsed, _valid = self.parse_action(action)
                    if _valid and isinstance(_parsed, dict):
                        _server_hint = _parsed.get("server")
                except Exception:
                    _server_hint = None
                self._ensure_discovered(_server_hint)
            except Exception:
                pass
        env = self.load_env(trajectory_id)

        parsed_action, valid = self.parse_action(action)
        if not valid:
            observation = ""
            done = False
            # Record invalid attempt
            execution_result = ""
            self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
            self.save_env(trajectory_id, env)
            return observation, done, False

        tool_name = parsed_action.get("name")
        tool_args = parsed_action.get("arguments", {}) or {}
        server = parsed_action.get("server") or None

        # Optional per-task server restrictions carried in extra_field (from dataset extra_info)
        try:
            allowed: List[str] = []
            use_specified = False
            if isinstance(extra_field, dict):
                use_specified = bool(extra_field.get("use_specified_server", False))
                mcp_servers = extra_field.get("mcp_servers") or []
                for s in mcp_servers:
                    if isinstance(s, dict) and s.get("name"):
                        allowed.append(str(s["name"]))
                    elif isinstance(s, str):
                        allowed.append(str(s))
            allowed_set = set(allowed)

            candidate_servers: List[str] = sorted(allowed_set)
            if not candidate_servers:
                default_server = os.getenv("MCP_DEFAULT_SERVER")
                if default_server:
                    candidate_servers = [default_server]
            if not candidate_servers:
                candidate_servers = [
                    s for s in self._server2tools.keys() if s is not None
                ]

            # Ensure discovery for the known candidates
            for srv in candidate_servers:
                try:
                    self._ensure_discovered(srv)
                except Exception:
                    continue

            # When restrictions exist, block calls to other servers
            if (use_specified or allowed_set) and (server is not None) and (server not in allowed_set):
                observation = {
                    "obs": "",
                    "error": f"Server '{server}' not allowed for this task. Allowed: {sorted(allowed_set)}",
                    "tool": tool_name,
                }
                done = True
                valid = False
                execution_result = observation["error"]
                self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
                self.save_env(trajectory_id, env)
                return observation, done, valid

            # If server not provided, try to infer it from discovered tools
            if server is None:
                matches = [
                    srv for srv in candidate_servers
                    if tool_name in (self._server2tools.get(srv) or {})
                ]
                if len(matches) == 1:
                    server = matches[0]
                elif len(matches) == 0:
                    # as a fallback, if there is exactly one candidate, use it; otherwise ask for clarification
                    if len(candidate_servers) == 1:
                        server = candidate_servers[0]
                    elif candidate_servers:
                        observation = {
                            "obs": "",
                            "error": f"Unable to locate tool '{tool_name}'. Specify one of the servers: {sorted(candidate_servers)}",
                            "tool": tool_name,
                        }
                        done = True
                        valid = False
                        execution_result = observation["error"]
                        self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
                        self.save_env(trajectory_id, env)
                        return observation, done, valid
                    else:
                        observation = {
                            "obs": "",
                            "error": "No MCP servers configured for this task.",
                            "tool": tool_name,
                        }
                        done = True
                        valid = False
                        execution_result = observation["error"]
                        self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
                        self.save_env(trajectory_id, env)
                        return observation, done, valid
            else:
                observation = {
                    "obs": "",
                    "error": f"Tool '{tool_name}' is available on multiple servers {sorted(matches)}; please specify which one to use.",
                    "tool": tool_name,
                }
                done = True
                valid = False
                execution_result = observation["error"]
                self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
                self.save_env(trajectory_id, env)
                return observation, done, valid
            parsed_action["server"] = server
        except Exception:
            # Do not block on restriction parsing errors
            pass

        # Quick gateway reachability check once server is resolved (gateway mode only)
        try:
            gw = os.getenv("MCP_GATEWAY_ADDRESS", "").rstrip("/")
            if gw and server:
                import urllib.request
                url = f"{gw}/{server}/sse"
                try:
                    with urllib.request.urlopen(url, timeout=2):
                        pass
                except Exception as conn_e:
                    raise RuntimeError(f"MCP gateway unreachable or server not active: {url}; {conn_e}")
        except Exception as e_chk:
            error = str(e_chk)
            err_payload = {
                "error": error,
                "error_type": type(e_chk).__name__,
                "server": server or "",
                "endpoint": (os.getenv("MCP_GATEWAY_ADDRESS", "").rstrip("/") + (f"/{server}/sse" if server else ""))
            }
            observation = self._format_observation(tool_name, json.dumps(err_payload))
            done = False
            valid = True
            execution_result = error
            self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
            self.save_env(trajectory_id, env)
            return observation, done, valid

        try:
            import asyncio

            async def _invoke():
                # Built-in discovery helpers
                if str(tool_name).startswith("mcp."):
                    # mcp.list_tools, mcp.list_resources
                    target_server = server or tool_args.get("server")
                    if not target_server:
                        raise RuntimeError("Server must be specified for MCP helper tools.")
                    # list tools/resources and format summary
                    try:
                        self._ensure_discovered(target_server)
                    except Exception:
                        pass
                    tools, resources = await list_tools_and_resources(target_server)
                    lines = []
                    if tool_name == "mcp.list_tools":
                        lines.append("Available tools:")
                        for t in getattr(tools, "tools", []) or []:
                            nm = getattr(t, "name", "")
                            desc = getattr(t, "description", None) or ""
                            lines.append(f"- {nm}: {desc}")
                    elif tool_name == "mcp.list_resources":
                        lines.append("Available resources:")
                        for r in getattr(resources, "resources", []) or []:
                            nm = getattr(r, "uri", "") or getattr(r, "name", "")
                            desc = getattr(r, "description", None) or ""
                            lines.append(f"- {nm}: {desc}")
                    else:
                        lines.append(f"Unknown mcp helper: {tool_name}")
                    return "\n".join(lines)

                # Ensure discovery for target server as well and forward call
                try:
                    self._ensure_discovered(server)
                except Exception:
                    pass
                result = await call_mcp_tool(tool_name, tool_args, server=server)
                return get_content_text(result)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Submit to the running loop in a threadsafe way
                    fut = asyncio.run_coroutine_threadsafe(_invoke(), loop)
                    content_text = fut.result()
                else:
                    content_text = loop.run_until_complete(_invoke())
            except RuntimeError:
                # No loop: direct asyncio.run
                content_text = asyncio.run(_invoke())

            observation = self._format_observation(tool_name, content_text)
            done = False
            valid = True
            # Record successful attempt
            execution_result = content_text or ""
            self.update_env(trajectory_id, env, parsed_action, True, extra_field, execution_result)
            self.save_env(trajectory_id, env)
            return observation, done, valid
        except Exception as e:
            error = str(e)
            # Return as a formatted observation so reward_manager logs it into mcp_eval_debug
            err_payload = {
                "error": error,
                "error_type": type(e).__name__,
                "server": server or "",
                "tool": tool_name
            }
            observation = self._format_observation(tool_name, json.dumps(err_payload))
            # Propagate tool errors back to the agent as a valid response so the model
            # consumes the <mcp_response> payload instead of tripping the invalid-action
            # branch (which prepends "Your action is not valid").
            done = False
            valid = True
            # Record failure
            execution_result = error
            self.update_env(trajectory_id, env, parsed_action, False, extra_field, execution_result)
            self.save_env(trajectory_id, env)
            return observation, done, valid
