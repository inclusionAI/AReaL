import getpass
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Lock

import requests as http_requests
from flask import Flask, jsonify, request

from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.utils import logging
from areal.utils.network import find_free_ports, gethostip

logger = logging.getLogger("RPCGuard")

# Port tracking - allocated ports excluded from future allocations
_allocated_ports: set[int] = set()

# Forked child processes - tracked for cleanup
_forked_children: list[subprocess.Popen] = []
_forked_children_lock = Lock()
# Map (role, worker_index) to forked process for selective killing
_forked_children_map: dict[tuple[str, int], subprocess.Popen] = {}

# Server address (set at startup)
_server_host: str = "0.0.0.0"

# Server config (needed for /fork endpoint to spawn children with same config)
_experiment_name: str | None = None
_trial_name: str | None = None
_fileroot: str | None = None

# Create Flask app
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is alive."""
    return jsonify(
        {
            "status": "healthy",
            "forked_children": len(_forked_children),
        }
    )


@app.route("/alloc_ports", methods=["POST"])
def alloc_ports():
    """Allocate multiple free ports.

    Expected JSON payload:
    {
        "count": 5  # Number of ports to allocate
    }
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        count = data.get("count")
        if count is None:
            return jsonify({"error": "Missing 'count' field in request"}), 400

        if not isinstance(count, int) or count <= 0:
            return jsonify({"error": "'count' must be a positive integer"}), 400

        global _allocated_ports
        ports = find_free_ports(count, exclude_ports=_allocated_ports)
        _allocated_ports.update(ports)

        return jsonify({"status": "success", "ports": ports, "host": _server_host})

    except Exception as e:
        logger.error(f"Error in alloc_ports: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def _wait_for_worker_ready(host: str, port: int, timeout: float = 60) -> bool:
    """Wait for a worker to be ready by polling its health endpoint.

    Args:
        host: The host address of the worker.
        port: The port of the worker.
        timeout: Maximum time to wait in seconds (default: 60).

    Returns:
        True if the worker is ready, False if timeout is reached.
    """
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = http_requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except http_requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    return False


@app.route("/fork", methods=["POST"])
def fork_worker():
    """Fork a new worker process on the same node.

    Supports two modes:

    **Module-path mode** (``command`` field):
        Builds ``python -m {command} --host 0.0.0.0 --port {port} ...`` with
        scheduler args injected. Waits for health readiness before returning.

    **Raw-command mode** (``raw_cmd`` field):
        Launches the provided command list as-is. A port is allocated but NOT
        injected into the command (caller provides port in ``raw_cmd``).
        Returns immediately after spawn without readiness polling.

    Expected JSON payload (module-path mode):
    {
        "role": "ref",
        "worker_index": 0,
        "command": "areal.infra.rpc.rpc_server"
    }

    Expected JSON payload (raw-command mode):
    {
        "role": "sglang",
        "worker_index": 0,
        "raw_cmd": ["python", "-m", "sglang.launch_server", "--model", "..."]
    }

    Returns:
    {
        "status": "success",
        "host": "192.168.1.10",
        "port": 8001,
        "pid": 12345
    }
    """
    global _forked_children, _forked_children_map, _allocated_ports

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        role = data.get("role")
        worker_index = data.get("worker_index")
        command = data.get("command")  # Module-path mode
        raw_cmd = data.get("raw_cmd")  # Raw-command mode

        if role is None:
            return jsonify({"error": "Missing 'role' field in request"}), 400
        if worker_index is None:
            return jsonify({"error": "Missing 'worker_index' field in request"}), 400

        if command is None and raw_cmd is None:
            return (
                jsonify(
                    {
                        "error": "Must provide either 'command' (module path) "
                        "or 'raw_cmd' (raw command list)"
                    }
                ),
                400,
            )

        # Allocate a free port for the child process
        ports = find_free_ports(1, exclude_ports=_allocated_ports)
        child_port = ports[0]
        _allocated_ports.add(child_port)

        # Determine if this is raw-command mode or module-path mode
        is_raw_mode = raw_cmd is not None

        if is_raw_mode:
            # Raw-command mode: use command as-is, do NOT inject port or args
            cmd = list(raw_cmd)
        else:
            # Module-path mode: build command with scheduler args
            cmd = [
                sys.executable,
                "-m",
                command,
                "--host",
                "0.0.0.0",
                "--port",
                str(child_port),
                "--experiment-name",
                _experiment_name,
                "--trial-name",
                _trial_name,
                "--role",
                role,
                "--worker-index",
                str(worker_index),
            ]

        logger.info(
            f"Forking new worker process for role '{role}' index {worker_index} "
            f"on port {child_port} (raw_mode={is_raw_mode})"
        )

        # Build log paths
        log_dir = (
            Path(_fileroot or "/tmp")
            / "logs"
            / getpass.getuser()
            / (_experiment_name or "default")
            / (_trial_name or "default")
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{role}.log"
        merged_log = log_dir / "merged.log"

        logger.info(f"Forked worker logs will be written to: {log_file}")

        # Use streaming log utility for terminal, role log, and merged log output
        child_process = run_with_streaming_logs(
            cmd,
            log_file,
            merged_log,
            role,
            env=os.environ.copy(),
        )

        with _forked_children_lock:
            _forked_children.append(child_process)
            _forked_children_map[(role, worker_index)] = child_process

        child_host = _server_host

        if not is_raw_mode:
            # Module-path mode: wait for child to be ready
            if not _wait_for_worker_ready(child_host, child_port):
                # Cleanup on failure
                try:
                    kill_process_tree(child_process.pid, timeout=3, graceful=True)
                except Exception:
                    pass
                with _forked_children_lock:
                    if child_process in _forked_children:
                        _forked_children.remove(child_process)
                    _forked_children_map.pop((role, worker_index), None)
                _allocated_ports.discard(child_port)
                return jsonify(
                    {"error": "Forked worker failed to start within timeout"}
                ), 500

            logger.info(
                f"Forked worker for role '{role}' index {worker_index} ready at "
                f"{child_host}:{child_port} (pid={child_process.pid})"
            )
        else:
            # Raw-command mode: return immediately without readiness polling
            logger.info(
                f"Forked raw-command worker for role '{role}' index {worker_index} "
                f"spawned (pid={child_process.pid}), port={child_port}"
            )

        return jsonify(
            {
                "status": "success",
                "host": child_host,
                "port": child_port,
                "pid": child_process.pid,
            }
        )

    except Exception as e:
        logger.error(f"Error in fork: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/kill_forked_worker", methods=["POST"])
def kill_forked_worker():
    """Kill a specific forked worker process.

    This endpoint terminates a previously forked child process identified by
    its role and worker_index.

    Expected JSON payload:
    {
        "role": "ref",
        "worker_index": 0
    }

    Returns:
    {
        "status": "success",
        "message": "Killed forked worker ref/0 (pid=12345)"
    }
    """
    global _forked_children, _forked_children_map

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        role = data.get("role")
        worker_index = data.get("worker_index")

        if role is None:
            return jsonify({"error": "Missing 'role' field in request"}), 400
        if worker_index is None:
            return jsonify({"error": "Missing 'worker_index' field in request"}), 400

        key = (role, worker_index)

        # Remove from tracking structures first (hold lock only for dict/list ops)
        with _forked_children_lock:
            child_process = _forked_children_map.pop(key, None)
            if child_process:
                try:
                    _forked_children.remove(child_process)
                except ValueError:
                    # Defensive: process was in map but not in list
                    logger.warning(
                        f"Process for {role}/{worker_index} was in map but not in list"
                    )

        if child_process is None:
            return jsonify(
                {"error": f"Forked worker {role}/{worker_index} not found"}
            ), 404

        pid = child_process.pid

        # Kill the process tree (outside the lock to avoid blocking other operations)
        try:
            if child_process.poll() is None:  # Still running
                kill_process_tree(pid, timeout=3, graceful=True)
                logger.info(f"Killed forked worker {role}/{worker_index} (pid={pid})")
        except Exception as e:
            logger.error(
                f"Error killing forked worker {role}/{worker_index} (pid={pid}): {e}"
            )
            return jsonify(
                {
                    "error": f"Failed to kill forked worker: {str(e)}",
                    "pid": pid,
                }
            ), 500

        return jsonify(
            {
                "status": "success",
                "message": f"Killed forked worker {role}/{worker_index} (pid={pid})",
            }
        )

    except Exception as e:
        logger.error(f"Error in kill_forked_worker: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def cleanup_forked_children():
    """Clean up all forked child processes."""
    global _forked_children, _forked_children_map

    with _forked_children_lock:
        if not _forked_children:
            return

        logger.info(f"Cleaning up {len(_forked_children)} forked child processes")
        for child in _forked_children:
            try:
                if child.poll() is None:  # Still running
                    kill_process_tree(child.pid, timeout=3, graceful=True)
                    logger.info(f"Killed forked child process {child.pid}")
            except Exception as e:
                logger.error(f"Error killing forked child {child.pid}: {e}")
        _forked_children.clear()
        _forked_children_map.clear()
