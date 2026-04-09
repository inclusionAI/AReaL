"""
Multi-worker Tool Server Router

This module implements a load-balancing router that:
1. Spawns multiple uvicorn worker subprocesses
2. Routes incoming requests to workers using consistent hashing based on trajectory_ids
3. Falls back to round-robin when trajectory_ids is not available
"""

import os
import logging
import socket
import itertools
import json
import zlib
import time
import sys
import signal
from typing import List, Optional
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
import fire
import uvicorn
from fastapi import FastAPI, Request, Response
import httpx

logger = logging.getLogger(__name__)

# Configuration constants
MAX_CONCURRENCY_PER_UVI_WORKER = 256
ROUTER_BACKLOG = 8192      # Backlog for the router
HEALTH_CHECK_TIMEOUT = 60.0
HEALTH_CHECK_INTERVAL = 0.5
SUBPROCESS_TERMINATE_TIMEOUT = 10


# constants for environment variable keys
MAX_CONNECTIONS_PER_UVI_WORKER = 512
KEEPALIVE_EXPIRY_PER_UVI_WORKER = 300
            

def _find_free_port(host: str = "127.0.0.1") -> int:
    """
    Find a free TCP port on the given host.
    
    Args:
        host: Host address to bind
        
    Returns:
        Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 0 => ask OS to assign a free port
        s.bind((host, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _find_free_ports(host: str, count: int) -> List[int]:
    """
    Find multiple free TCP ports on the given host.
    
    Args:
        host: Host address to bind
        count: Number of free ports needed
        
    Returns:
        List of available port numbers
    """
    ports: List[int] = []
    for _ in range(count):
        port = _find_free_port(host)
        # Keep trying if we get a duplicate (unlikely but possible)
        attempts = 0
        while port in ports and attempts < 100:
            port = _find_free_port(host)
            attempts += 1
        ports.append(port)
    return ports


def _start_backend_server(
    host: str,
    port: int,
    idx: int,
    tool_type: str,
    workers_per_tool: int,
    max_concurrent_requests: int,
    request_timeout: Optional[float],
    thread_pool_size: Optional[int],
    use_tqdm: bool,
    log_level: str,
    done_if_invalid: bool,
    use_ray: bool,
    enable_hashing: bool,
    log_interval: int,
    log_file: Optional[any] = None,
) -> subprocess.Popen:
    """
    Start one backend tool server as a subprocess.
    
    Args:
        host: Host address to bind
        port: Port number to listen on
        idx: Backend index (for logging)
        tool_type: Type of tool server
        workers_per_tool: Number of workers per tool
        max_concurrent_requests: Maximum concurrent requests for this backend
        request_timeout: Request timeout in seconds
        thread_pool_size: Size of thread pool
        use_tqdm: Enable tqdm progress bars
        log_level: Logging level
        done_if_invalid: Mark task as done if invalid
        use_ray: Use Ray for distributed processing
        enable_hashing: Enable consistent hashing
        log_interval: Interval for logging statistics
        
    Returns:
        Popen object for the subprocess
    """
    cmd = [
        sys.executable,
        "-m", "verl_tool.servers.tool_server",
        "--tool_type", tool_type,
        "--host", host,
        "--port", str(port),
        "--workers_per_tool", str(workers_per_tool),
        "--max_concurrent_requests", str(max_concurrent_requests),
        "--log_level", log_level,
        "--log_interval", str(log_interval),
    ]

    if request_timeout is not None:
        cmd.extend(["--request_timeout", str(request_timeout)])
    if thread_pool_size is not None:
        cmd.extend(["--thread_pool_size", str(thread_pool_size)])
    if use_tqdm:
        cmd.extend(["--use_tqdm", "True"])
    if done_if_invalid:
        cmd.extend(["--done_if_invalid", "True"])
    if use_ray:
        cmd.extend(["--use_ray", "True"])
    if not enable_hashing:
        cmd.extend(["--enable_hashing", "False"])

    logger.info(f"[BACKEND {idx}] Starting with command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file or sys.stdout,
        stderr=log_file or sys.stderr,
    )
    
    logger.info(f"[BACKEND {idx}] Spawned PID={proc.pid} on {host}:{port}")
    return proc


def _wait_for_workers_ready(
    worker_base_urls: List[str], 
    timeout: float = HEALTH_CHECK_TIMEOUT
) -> bool:
    """
    Wait for all sub-workers to become healthy.
    
    Args:
        worker_base_urls: List of worker base URLs to check
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if all workers are healthy, False otherwise
    """
    start = time.time()
    remaining = set(worker_base_urls)
    logger.info(f"[HEALTH] Waiting for {len(remaining)} workers to become healthy...")

    with httpx.Client(timeout=3.0) as client:
        while remaining and (time.time() - start) < timeout:
            ready_workers = []
            
            for base_url in remaining:
                url = f"{base_url}/health"
                try:
                    response = client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "healthy":
                            logger.info(f"[HEALTH] Worker {base_url} is healthy")
                            ready_workers.append(base_url)
                except httpx.RequestError:
                    # Worker not ready yet, continue polling
                    pass
                except Exception as e:
                    logger.debug(f"[HEALTH] Error checking {base_url}: {e}")

            for url in ready_workers:
                remaining.discard(url)

            if remaining:
                time.sleep(HEALTH_CHECK_INTERVAL)

    if remaining:
        logger.error(
            f"[HEALTH] {len(remaining)} workers not healthy after {timeout}s: "
            f"{', '.join(remaining)}"
        )
        return False
    
    logger.info(f"[HEALTH] All {len(worker_base_urls)} workers are healthy")
    return True


import re as _re

# Regex to extract tool name from <action>{"name": "xxx", ...}</action>
_ACTION_TAG_RE = _re.compile(r'<action>(.*?)</action>', _re.DOTALL | _re.IGNORECASE)
_TOOL_NAME_RE = _re.compile(r'"name"\s*:\s*"([^"]+)"')


# =============================================================================
# Static tool_type → function_name mapping
# Mirrors enable_tools in verl_tool/servers/tools/geo_*.py
# and TOOL_CATEGORIES in geo_edit/tool_definitions/router.py
# =============================================================================
TOOL_TYPE_FUNCTIONS = {
    "geo_edit_function": [
        "image_crop", "image_label", "draw_line",
        "draw_path", "bounding_box", "image_highlight",
    ],
    "geo_paddleocr": [
        "text_ocr", "table_ocr", "formula_ocr", "chart_text_ocr",
        "text_spotting", "seal_ocr", "map_text_ocr",
    ],
    "geo_chartr1": [
        "chart_reasoning", "chart_data_extract", "chart_trend_analysis",
    ],
    "geo_sam3": [
        "auto_segment", "bbox_segment", "text_segment",
        "exemplar_segment", "concept_count", "presence_check",
    ],
    "geo_grounding_dino": ["grounding_dino"],
    "geo_multimath": ["math_latex_ocr", "math_image_describe"],
    "geo_gllava": ["gllava"],
}


def _build_function_routing_table(
    worker_base_urls: List[str],
    worker_tool_types: List[str],
) -> dict:
    """
    Build function_name → [backend_indices] mapping from static table.

    Args:
        worker_base_urls: backend URLs (for logging only)
        worker_tool_types: tool_type per backend, same order as URLs
    
    Returns:
        {"function_to_backends": {name: [indices]}, "finish_backends": [indices]}
    """
    function_to_backends = {}
    finish_backends = []

    for idx, tool_type in enumerate(worker_tool_types):
        # Every backend has a finish tool
        finish_backends.append(idx)

        func_names = TOOL_TYPE_FUNCTIONS.get(tool_type, [])
        if not func_names:
            logger.warning(
                f"[ROUTING] tool_type '{tool_type}' not in TOOL_TYPE_FUNCTIONS, "
                f"backend {idx} ({worker_base_urls[idx]}) won't receive routed requests"
            )
        for fn in func_names:
            function_to_backends.setdefault(fn, []).append(idx)

    # Sort for deterministic hashing
    for fn in function_to_backends:
        function_to_backends[fn] = sorted(set(function_to_backends[fn]))

    logger.info(f"[ROUTING] function → backends: {json.dumps(function_to_backends)}")
    logger.info(f"[ROUTING] finish backends: {finish_backends}")

    return {
        "function_to_backends": function_to_backends,
        "finish_backends": sorted(set(finish_backends)),
    }


def create_router_app(
    worker_base_urls: List[str],
    worker_tool_types: Optional[List[str]] = None,
) -> FastAPI:
    """
    Create the router FastAPI application.
    
    The router:
    - Accepts all HTTP requests
    - For /get_observation: routes by tool function name to the correct backend
    - Falls back to consistent hashing on trajectory_ids when tool name is unknown
    
    Args:
        worker_base_urls: List of backend worker URLs
        worker_tool_types: List of tool_type per backend (same order as URLs).
            If provided, enables tool-aware routing.
    
    Returns:
        FastAPI application instance
    """
    num_workers = len(worker_base_urls)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Configure connection limits for each worker client
        limits = httpx.Limits(
            max_keepalive_connections=MAX_CONNECTIONS_PER_UVI_WORKER,
            max_connections=MAX_CONNECTIONS_PER_UVI_WORKER,
            keepalive_expiry=KEEPALIVE_EXPIRY_PER_UVI_WORKER,
        )

        app.state.clients = [
            httpx.AsyncClient(
                base_url=base_url,
                timeout=None,
                limits=limits,
            )
            for base_url in worker_base_urls
        ]
        app.state.counter = itertools.count()
        
        # Build tool-aware routing table from static mapping
        if worker_tool_types:
            caps = _build_function_routing_table(worker_base_urls, worker_tool_types)
            app.state.function_to_backends = caps["function_to_backends"]
            app.state.finish_backends = caps["finish_backends"]
            logger.info(
                f"[ROUTER] Tool-aware routing enabled: "
                f"{len(app.state.function_to_backends)} function names mapped "
                f"across {num_workers} backends"
            )
        else:
            app.state.function_to_backends = {}
            app.state.finish_backends = list(range(num_workers))
            logger.info("[ROUTER] No tool types provided, using hash routing only")
        
        logger.info(
            f"[ROUTER] Started with {num_workers} workers: {', '.join(worker_base_urls)}"
        )

        try:
            yield
        finally:
            logger.info("[ROUTER] Shutting down, closing client connections...")
            for client in getattr(app.state, "clients", []):
                try:
                    await client.aclose()
                except Exception as e:
                    logger.debug(f"[ROUTER] Error closing client: {e}")

    # Attach lifespan instead of using @app.on_event
    app = FastAPI(title="Tool Server Router", lifespan=lifespan)

    def _pick_worker_index_by_hash(body_bytes: bytes) -> int:
        """
        Legacy fallback: select a worker index using trajectory_id hash or round-robin.
        """
        if not body_bytes:
            return next(app.state.counter) % num_workers

        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return next(app.state.counter) % num_workers

        if isinstance(data, dict):
            trajectory_ids = data.get("trajectory_ids")
            if isinstance(trajectory_ids, list) and len(trajectory_ids) > 0:
                tid_str = str(trajectory_ids[0])
                hash_value = zlib.crc32(tid_str.encode("utf-8")) & 0xFFFFFFFF
                return hash_value % num_workers

        return next(app.state.counter) % num_workers

    def _parse_tool_request(body_bytes: bytes):
        """
        Parse the request body to extract routing-relevant fields.
        
        Returns:
            (data_dict, trajectory_id, tool_name, is_finish) or
            (None, None, None, False) on parse failure.
        """
        if not body_bytes:
            return None, None, None, False
        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, None, None, False
        
        if not isinstance(data, dict):
            return None, None, None, False
        
        # Extract trajectory_id
        trajectory_ids = data.get("trajectory_ids")
        tid = str(trajectory_ids[0]) if isinstance(trajectory_ids, list) and trajectory_ids else None
        
        # Check finish flag
        finish_flags = data.get("finish")
        is_finish = (
            isinstance(finish_flags, list) 
            and len(finish_flags) > 0 
            and finish_flags[0]
        )
        
        # Extract tool name from action
        tool_name = None
        actions = data.get("actions")
        if isinstance(actions, list) and actions:
            action_match = _ACTION_TAG_RE.search(actions[0])
            if action_match:
                name_match = _TOOL_NAME_RE.search(action_match.group(1))
                if name_match:
                    tool_name = name_match.group(1)
        
        return data, tid, tool_name, is_finish

    def _pick_worker_for_tool(tool_name: str, trajectory_id: str) -> int:
        """
        Pick backend index by tool function name.
        
        Uses the function_to_backends mapping built at startup.
        Hashes trajectory_id within the candidate set for stickiness.
        
        Returns:
            Backend index, or -1 if no mapping found.
        """
        candidates = getattr(app.state, 'function_to_backends', {}).get(tool_name)
        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidates[0]
        # Hash trajectory_id within candidate set for sticky routing
        hash_value = zlib.crc32(trajectory_id.encode("utf-8")) & 0xFFFFFFFF
        return candidates[hash_value % len(candidates)]

    # Headers that should not be forwarded
    HOP_BY_HOP_HEADERS = frozenset({
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
        "host",
    })

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(full_path: str, request: Request):
        """
        Proxy handler that forwards requests to backend workers.
        
        For /get_observation POST requests:
        - Finish requests are fan-out to all backends with finish capability
        - Tool calls are routed by function name to the correct backend
        - Unknown tools fall back to trajectory-hash routing
        
        Other requests use trajectory-hash routing.
        """
        method = request.method
        query_params = dict(request.query_params)
        body_bytes = await request.body()
        target_path = f"/{full_path}" if full_path else "/"

        # Filter out hop-by-hop headers
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }

        # --- Tool-aware routing for /get_observation ---
        if full_path == "get_observation" and method == "POST":
            _, tid, tool_name, is_finish = _parse_tool_request(body_bytes)
            
            # Case 1: Finish — fan-out to all backends that have finish tool
            if is_finish:
                finish_indices = getattr(app.state, 'finish_backends', [])
                if finish_indices:
                    return await _fan_out_finish(
                        app, finish_indices, target_path,
                        body_bytes, headers, query_params,
                    )
                # No finish backends known, fall through to hash routing
            
            # Case 2: Known tool name — route to correct backend
            if tool_name and tid:
                worker_idx = _pick_worker_for_tool(tool_name, tid)
                if worker_idx >= 0:
                    return await _forward_to_worker(
                        app, worker_idx, target_path,
                        body_bytes, headers, query_params,
                    )
                else:
                    logger.warning(
                        f"[ROUTER] Tool name '{tool_name}' not in routing table, "
                        f"falling back to hash routing. "
                        f"Known tools: {list(getattr(app.state, 'function_to_backends', {}).keys())}"
                    )
        
        # --- Fallback: trajectory-hash routing ---
        worker_idx = _pick_worker_index_by_hash(body_bytes)
        return await _forward_to_worker(
            app, worker_idx, target_path,
            body_bytes, headers, query_params,
            method=method,
        )

    @app.get("/health")
    async def health():
        """Router health check endpoint."""
        return {
            "status": "ok",
            "workers": num_workers,
            "routing_mode": "tool-aware" if getattr(app.state, 'function_to_backends', {}) else "hash-only",
            "known_function_tools": list(getattr(app.state, 'function_to_backends', {}).keys()),
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with basic info."""
        return {
            "service": "tool-server-router",
            "workers": num_workers,
            "status": "running",
        }

    async def _forward_to_worker(
        app_instance, worker_idx: int, target_path: str,
        body_bytes: bytes, headers: dict, query_params: dict,
        method: str = "POST",
    ) -> Response:
        """Forward a request to a specific backend worker with retries."""
        client: httpx.AsyncClient = app_instance.state.clients[worker_idx]
        
        for attempt in range(3):
            try:
                response = await client.request(
                    method=method,
                    url=target_path,
                    content=body_bytes if body_bytes else None,
                    headers=headers,
                    params=query_params,
                )
                response_headers = {
                    k: v for k, v in response.headers.items()
                    if k.lower() not in HOP_BY_HOP_HEADERS
                }
                return Response(
                    status_code=response.status_code,
                    content=response.content,
                    headers=response_headers,
                    media_type=response.headers.get("content-type"),
                )
            except httpx.TimeoutException as e:
                logger.warning(f"[ROUTER] Timeout forwarding to worker {worker_idx}: {e}")
                return Response(
                    status_code=504,
                    content=b'{"detail": "upstream timeout"}',
                    media_type="application/json",
                )
            except httpx.ReadError as e:
                logger.warning(
                    f"[ROUTER] ReadError forwarding to worker {worker_idx} "
                    f"({worker_base_urls[worker_idx]}), attempt={attempt+1}: {e}"
                )
                if attempt == 2:
                    logger.error(
                        f"[ROUTER] ReadError persists after retries for worker {worker_idx}",
                        exc_info=True,
                    )
                    return Response(
                        status_code=502,
                        content=b'{"detail": "upstream read error"}',
                        media_type="application/json",
                    )
            except httpx.RequestError as e:
                logger.error(
                    f"[ROUTER] Error forwarding to worker {worker_idx} "
                    f"({worker_base_urls[worker_idx]}): {e!r}",
                    exc_info=True,
                )
                return Response(
                    status_code=502,
                    content=b'{"detail": "upstream connection error"}',
                    media_type="application/json",
                )
            except Exception as e:
                logger.exception(
                    f"[ROUTER] Unexpected error forwarding to worker {worker_idx}: {e}"
                )
                return Response(
                    status_code=502,
                    content=b'{"detail": "router internal error"}',
                    media_type="application/json",
                )
        # Should not reach here, but just in case
        return Response(
            status_code=502,
            content=b'{"detail": "all retries exhausted"}',
            media_type="application/json",
        )

    async def _fan_out_finish(
        app_instance, backend_indices: List[int], target_path: str,
        body_bytes: bytes, headers: dict, query_params: dict,
    ) -> Response:
        """
        Fan-out finish request to all backends that have the finish tool.
        
        FinishTool cleans up env_cache per trajectory; each backend that 
        may hold state for this trajectory needs to receive the finish signal.
        Returns the first successful response.
        """
        import asyncio
        
        async def _send_one(idx: int):
            client = app_instance.state.clients[idx]
            try:
                return await client.request(
                    method="POST",
                    url=target_path,
                    content=body_bytes,
                    headers=headers,
                    params=query_params,
                )
            except Exception as e:
                logger.debug(f"[ROUTER] Finish fan-out to backend {idx} failed: {e}")
                return None
        
        results = await asyncio.gather(
            *[_send_one(idx) for idx in backend_indices],
            return_exceptions=True,
        )
        
        # Return first successful response
        for r in results:
            if isinstance(r, httpx.Response) and r.status_code == 200:
                response_headers = {
                    k: v for k, v in r.headers.items()
                    if k.lower() not in HOP_BY_HOP_HEADERS
                }
                return Response(
                    status_code=r.status_code,
                    content=r.content,
                    headers=response_headers,
                    media_type=r.headers.get("content-type"),
                )
        
        # All failed — return a synthetic success (finish is best-effort cleanup)
        logger.warning(
            f"[ROUTER] Finish fan-out: none of {len(backend_indices)} backends "
            f"returned 200, returning synthetic success"
        )
        return Response(
            status_code=200,
            content=json.dumps({
                "observations": [""],
                "dones": [True],
                "valids": [True],
                "processing_time_ms": 0.0,
            }).encode(),
            media_type="application/json",
        )

    return app



class WorkerManager:
    """
    Manages the lifecycle of worker subprocesses.
    """
    
    def __init__(self, log_directory: Optional[str] = None):
        self.worker_procs: List[subprocess.Popen] = []
        self._shutdown_requested = False
        self.file_handles: List[any] = []
        self.log_directory = log_directory
        if self.log_directory:
            self.log_directory_path = Path(self.log_directory)
        else:
            self.log_directory_path = Path("tool-server-logs")
        self.log_directory_path.mkdir(parents=True, exist_ok=True)
        
    def log_worker_states(self):
        for idx, proc in enumerate(self.worker_procs):
            rc = proc.poll()
            logger.info(
                f"[WORKER_MANAGER] worker[{idx}] pid={proc.pid} returncode={rc}"
            )
        
    def start_workers(
        self,
        host: str,
        ports: List[int],
        tool_type: str,
        workers_per_tool: int,
        max_concurrent_requests: int,
        request_timeout: Optional[float],
        thread_pool_size: Optional[int],
        use_tqdm: bool,
        log_level: str,
        done_if_invalid: bool,
        use_ray: bool,
        enable_hashing: bool,
        log_interval: int,
    ) -> List[str]:
        """
        Start worker subprocesses.
        
        Returns:
            List of worker base URLs
        """
        worker_base_urls = []
        num_workers = len(ports)
        
        # Distribute concurrency across backends
        per_backend_max_concurrent = max(1, max_concurrent_requests)
        logger.info(
            f"[WORKER_MANAGER] Starting {num_workers} workers with "
            f"max_concurrent_requests={per_backend_max_concurrent} each"
        )
        
        for idx, port in enumerate(ports):
            log_file = self.log_directory_path / f"tool_server_backend_{idx}.log"
            lf = open(log_file, "ab", buffering=0)
            self.file_handles.append(lf)
            proc = _start_backend_server(
                host=host,
                port=port,
                idx=idx,
                tool_type=tool_type,
                workers_per_tool=workers_per_tool,
                max_concurrent_requests=per_backend_max_concurrent,
                request_timeout=request_timeout,
                thread_pool_size=thread_pool_size,
                use_tqdm=use_tqdm,
                log_level=log_level,
                done_if_invalid=done_if_invalid,
                use_ray=use_ray,
                enable_hashing=enable_hashing,
                log_interval=log_interval,
                log_file=lf,
            )
            self.worker_procs.append(proc)
            worker_base_urls.append(f"http://{host}:{port}")
            
        return worker_base_urls
    
    def shutdown(self):
        """Gracefully shutdown all worker processes."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        
        logger.info(f"[WORKER_MANAGER] Shutting down {len(self.worker_procs)} workers...")
        
        # First, send SIGTERM to all workers
        for proc in self.worker_procs:
            if proc.poll() is None:  # Still running
                try:
                    proc.terminate()
                except Exception as e:
                    logger.debug(f"[WORKER_MANAGER] Error terminating PID={proc.pid}: {e}")
        
        # Wait for graceful shutdown
        for proc in self.worker_procs:
            try:
                proc.wait(timeout=SUBPROCESS_TERMINATE_TIMEOUT)
                logger.info(f"[WORKER_MANAGER] Worker PID={proc.pid} terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"[WORKER_MANAGER] Worker PID={proc.pid} did not terminate, killing...")
                proc.kill()
                proc.wait()
            except Exception as e:
                logger.error(f"[WORKER_MANAGER] Error waiting for PID={proc.pid}: {e}")
                
    def check_workers_alive(self) -> bool:
        """Check if all workers are still running."""
        for proc in self.worker_procs:
            if proc.poll() is not None:
                return False
        return True

def router_factory() -> FastAPI:
    urls_str = os.environ.get("VT_WORKER_BASE_URLS")
    if not urls_str:
        raise RuntimeError("VT_WORKER_BASE_URLS not set")
    worker_base_urls = json.loads(urls_str)
    
    # Optional: tool_type per backend for tool-aware routing
    types_str = os.environ.get("VT_WORKER_TOOL_TYPES")
    worker_tool_types = json.loads(types_str) if types_str else None
    
    return create_router_app(worker_base_urls, worker_tool_types)

def main(
    tool_type: str = "base",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = 32,
    max_concurrent_requests: int = 1024,
    request_timeout: Optional[float] = None,
    thread_pool_size: Optional[int] = None,
    use_tqdm: bool = False,
    log_level: str = "info",
    done_if_invalid: bool = False,
    use_ray: bool = False,
    enable_hashing: bool = True,
    log_interval: int = 30,
    uvi_workers: int = 1,
    router_workers: int = 1,
    log_directory: Optional[str] = None,
):
    """
    Main entry point for the tool server router.
    
    This function:
    1. Spawns multiple uvicorn worker subprocesses
    2. Waits for all workers to become healthy
    3. Starts the router to distribute requests to workers
    
    Args:
        tool_type: Type of tool server to run
        host: Host address to bind the router
        port: Port number for the router
        workers_per_tool: Number of workers per tool type
        max_concurrent_requests: Maximum concurrent requests
        request_timeout: Request timeout in seconds (None for no timeout)
        thread_pool_size: Size of thread pool for blocking operations
        use_tqdm: Enable tqdm progress bars
        log_level: Logging level (debug, info, warning, error)
        done_if_invalid: Mark task as done if invalid
        use_ray: Use Ray for distributed processing
        enable_hashing: Enable consistent hashing for request routing
        log_interval: Interval for logging statistics in seconds
        uvi_workers: Number of uvicorn worker processes
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Suppress noisy debug logs from HTTP libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)

    # Calculate concurrency settings
    max_concurrency = max(max_concurrent_requests, max_concurrent_requests)
    workers_per_tool = max_concurrency if use_ray else workers_per_tool
    max_concurrent_requests = max_concurrency

    # Adjust number of workers if needed
    min_workers_needed = (max_concurrency + MAX_CONCURRENCY_PER_UVI_WORKER - 1) // MAX_CONCURRENCY_PER_UVI_WORKER
    min_routers_needed = min_workers_needed * 2  # Routers usually need more workers for handling connections
    if uvi_workers < min_workers_needed:
        uvi_workers = min_workers_needed
        logger.warning(
            f"Adjusted uvi_workers to {uvi_workers} to handle "
            f"max_concurrent_requests={max_concurrent_requests}"
        )
    if router_workers < min_routers_needed:
        router_workers = min_routers_needed
        logger.warning(
            f"Adjusted router_workers to {router_workers} to handle "
            f"max_concurrent_requests={max_concurrent_requests}"
        )

    # Normalize tool_type: fire.Fire may parse "a,b" as a tuple
    if isinstance(tool_type, (tuple, list)):
        tool_type = ",".join(tool_type)

    # Export configuration via environment variables for worker processes
    os.environ["VT_TOOL_TYPE"] = tool_type
    os.environ["VT_HOST"] = host
    os.environ["VT_PORT"] = str(port)
    os.environ["VT_WORKERS_PER_TOOL"] = str(workers_per_tool)
    os.environ["VT_MAX_CONCURRENT_REQUESTS"] = str(max_concurrent_requests)
    os.environ["VT_REQUEST_TIMEOUT"] = "None" if request_timeout is None else str(request_timeout)
    if thread_pool_size is not None:
        os.environ["VT_THREAD_POOL_SIZE"] = str(thread_pool_size)
    os.environ["VT_USE_TQDM"] = "1" if use_tqdm else "0"
    os.environ["VT_LOG_LEVEL"] = log_level
    os.environ["VT_DONE_IF_INVALID"] = "1" if done_if_invalid else "0"
    os.environ["VT_USE_RAY"] = "1" if use_ray else "0"
    os.environ["VT_ENABLE_HASHING"] = "1" if enable_hashing else "0"
    os.environ["VT_LOG_INTERVAL"] = str(log_interval)

    # Detect event loop implementation for the router
    try:
        import uvloop  # noqa: F401
        loop_impl = "uvloop"
        logger.info("[MAIN] Using uvloop event loop for router")
    except ImportError:
        loop_impl = "asyncio"
        logger.info("[MAIN] Using asyncio event loop for router (uvloop not available)")

    # Initialize Ray cluster before starting workers (if use_ray is enabled)
    # This ensures all worker subprocesses connect to the same Ray cluster
    # instead of each trying to start their own
    ray_initialized = False
    if use_ray:
        try:
            import ray
            if not ray.is_initialized():
                logger.info("[MAIN] Initializing Ray cluster...")
                ray.init(
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
                logger.info(f"[MAIN] Ray initialized: {ray.cluster_resources()}")
            else:
                logger.info("[MAIN] Ray already initialized, reusing existing cluster")
            ray_initialized = True
        except ImportError:
            logger.error("[MAIN] use_ray=True but ray is not installed!")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[MAIN] Failed to initialize Ray: {e}")
            sys.exit(1)

    # Initialize worker manager
    worker_manager = WorkerManager(log_directory=log_directory)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"[MAIN] Received signal {signum}, initiating shutdown...")
        worker_manager.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Find free ports for workers (bind to localhost only)
        worker_host = "127.0.0.1"
        worker_ports = _find_free_ports(worker_host, uvi_workers)
        
        logger.info(f"[MAIN] Starting {uvi_workers} backend worker subprocesses...")
        
        # Start worker subprocesses
        worker_base_urls = worker_manager.start_workers(
            host=worker_host,
            ports=worker_ports,
            tool_type=tool_type,
            workers_per_tool=workers_per_tool,
            max_concurrent_requests=max_concurrent_requests,
            request_timeout=request_timeout,
            thread_pool_size=thread_pool_size,
            use_tqdm=use_tqdm,
            log_level=log_level,
            done_if_invalid=done_if_invalid,
            use_ray=use_ray,
            enable_hashing=enable_hashing,
            log_interval=log_interval,
        )
        
        logger.info(f"[MAIN] Backend workers started:")
        for idx, url in enumerate(worker_base_urls):
            logger.info(f"  backend[{idx}]: {url}")

        # Wait for all workers to become healthy
        if not _wait_for_workers_ready(worker_base_urls):
            logger.error("[MAIN] Some workers failed to start, aborting...")
            worker_manager.shutdown()
            sys.exit(1)

        # Export worker base URLs for router factory
        os.environ["VT_WORKER_BASE_URLS"] = json.dumps(worker_base_urls)

        # Start the router using a factory import string so that
        # uvicorn can use multiple workers.
        logger.info(f"[MAIN] Starting router on {host}:{port} with {router_workers} workers")

        uvicorn.run(
            "verl_tool.servers.serve:router_factory",  # <--- 注意这里
            host=host,
            port=port,
            log_level=log_level,
            access_log=False,
            loop=loop_impl,
            http="httptools",
            timeout_keep_alive=30,
            backlog=ROUTER_BACKLOG,
            workers=router_workers,
            factory=True,  # <--- 关键！告诉 uvicorn 这是一个工厂函数
        )
        
    except KeyboardInterrupt:
        logger.info("[MAIN] Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"[MAIN] Fatal error: {e}")
        raise
    finally:
        worker_manager.shutdown()
        # Shutdown Ray if we initialized it
        if use_ray and ray_initialized:
            try:
                import ray
                if ray.is_initialized():
                    logger.info("[MAIN] Shutting down Ray...")
                    ray.shutdown()
            except Exception as e:
                logger.warning(f"[MAIN] Error shutting down Ray: {e}")
        logger.info("[MAIN] Shutdown complete")


if __name__ == "__main__":
    fire.Fire(main)