"""Start geo_edit tool server on the Ray worker node."""

import os
import sys
import time

import ray

VERL_ROOT = os.environ["VERL_ROOT"]
AREAL_ROOT = os.environ["AREAL_ROOT"]
LOG_DIR = os.environ["LOG_DIR"]
PORT = os.environ.get("PORT", "30888")


def get_worker_ip() -> str:
    for node in ray.nodes():
        if node["Resources"].get("tool_agent", 0) > 0 and node["Alive"]:
            return node["NodeManagerAddress"]
    raise RuntimeError("No alive worker node with 'tool_agent' resource found")


def cleanup_worker(port: str):
    """Kill old tool server processes and free the port on the worker node."""

    # Kill old Ray actor
    for ns in ("tool", "tool_server"):
        try:
            ray.kill(ray.get_actor("tool_srv", namespace=ns))
        except Exception:
            pass

    @ray.remote(resources={"tool_agent": 0.001})
    def _cleanup(port):
        import subprocess, re, time as _time

        subprocess.run("pkill -9 -f verl_tool.servers", shell=True)
        _time.sleep(1)
        # Kill any process holding the port
        r = subprocess.run(
            f"lsof -ti tcp:{port}",
            shell=True, capture_output=True, text=True,
        )
        for pid in r.stdout.strip().splitlines():
            if pid.strip():
                subprocess.run(f"kill -9 {pid.strip()}", shell=True)
        _time.sleep(2)

    ray.get(_cleanup.remote(port))
    print("Cleanup done")


def launch_server():
    """Launch tool server as a detached Ray actor on the worker node."""

    head_ip = ray.util.get_node_ip_address()
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{VERL_ROOT}/verl:{VERL_ROOT}:{AREAL_ROOT}",
            "GEOEDIT_ENABLE_TOOLS": "general,chart",
            "RAY_ADDRESS": f"{head_ip}:6379",
        }
    }

    @ray.remote(resources={"tool_agent": 0.001}, num_cpus=1, runtime_env=runtime_env)
    class ToolServer:
        def __init__(self):
            import socket
            import subprocess
            import sys

            self.ip = socket.gethostbyname(socket.gethostname())
            os.makedirs(LOG_DIR, exist_ok=True)
            self.log_path = f"{LOG_DIR}/serve.log"
            log_file = open(self.log_path, "w")
            cmd = [
                sys.executable, "-m", "verl_tool.servers.serve",
                "--host", self.ip, "--port", PORT,
                "--tool_type", "geo_edit_tool",
                "--workers_per_tool", "1", "--uvi_workers", "1",
                "--router_workers", "1",
                "--max_concurrent_requests", "128", "--use_ray", "True",
            ]
            self.proc = subprocess.Popen(
                cmd, env=os.environ.copy(), stdout=log_file, stderr=subprocess.STDOUT
            )
            print(f"Tool server on {self.ip}:{PORT} PID={self.proc.pid}")

        def status(self):
            return {
                "ip": self.ip,
                "pid": self.proc.pid,
                "running": self.proc.poll() is None,
            }

        def tail_log(self, n=30):
            import subprocess as sp
            import glob

            # Collect serve.log + all backend logs
            logs = [self.log_path]
            logs.extend(sorted(glob.glob(f"{LOG_DIR}/tool_server_backend_*.log")))
            parts = []
            for path in logs:
                out = sp.run(
                    ["tail", "-n", str(n), path],
                    capture_output=True, text=True,
                ).stdout
                if out.strip():
                    parts.append(f"--- {path} ---\n{out}")
            return "\n".join(parts)

    srv = ToolServer.options(
        name="tool_srv", lifetime="detached", namespace="tool"
    ).remote()

    time.sleep(10)
    status = ray.get(srv.status.remote())
    print(f"Status: {status}")
    if not status["running"]:
        print("DIED! Logs:")
        print(ray.get(srv.tail_log.remote()))
        ray.kill(srv)
        print("Killed dead ToolServer actor, resources released.")
        sys.exit(1)


def wait_healthy(worker_ip: str, timeout_s: int = 180):
    """Poll /health endpoint until the server is ready."""
    import urllib.request
    import urllib.error

    url = f"http://{worker_ip}:{PORT}/health"
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if b"healthy" in resp.read():
                    print(f"HEALTHY on {worker_ip}:{PORT} ({attempt * 2}s)")
                    return
        except (urllib.error.URLError, OSError):
            pass
        if attempt % 10 == 0:
            print(f"  Waiting... ({attempt}/{timeout_s // 2})")
        time.sleep(2)
    print(f"WARNING: health check timed out after {timeout_s}s")


def verify_gpu():
    """Check that tool agents have claimed GPUs."""
    time.sleep(15)
    a = ray.available_resources()
    t = ray.cluster_resources()
    gpu_used = t["GPU"] - a.get("GPU", 0)
    tool_used = t["tool_agent"] - a.get("tool_agent", 0)
    print(f"GPU used: {gpu_used:.0f}/{t['GPU']:.0f}")
    print(f"tool_agent used: {tool_used:.0f}/{t['tool_agent']:.0f}")

    # Always print recent logs for diagnosis
    try:
        srv = ray.get_actor("tool_srv", namespace="tool")
        logs = ray.get(srv.tail_log.remote(50))
    except Exception:
        logs = "(could not retrieve logs)"

    if gpu_used >= 6:
        print("SUCCESS: All 6 tool agents loaded!")
    elif gpu_used > 0:
        print(f"PARTIAL: {gpu_used:.0f} agents loaded.")
        print(f"\n--- Tool server logs ---\n{logs}")
    else:
        print("FAIL: No agents loaded.")
        print(f"\n--- Tool server logs ---\n{logs}")


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    print("=== Step 1: Cleanup ===")
    cleanup_worker(PORT)

    print("\n=== Step 2: Launch tool server on worker ===")
    launch_server()

    print("\n=== Step 3: Health check ===")
    worker_ip = get_worker_ip()
    wait_healthy(worker_ip)

    print("\n=== Step 4: Verify GPU usage ===")
    verify_gpu()


if __name__ == "__main__":
    main()
