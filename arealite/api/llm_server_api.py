# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import json
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

from arealite.api.cli_args import LLMServiceConfig, TrainingArgs
from arealite.api.io_struct import LLMServerInfo
from realhf.base import logging, name_resolve, names

logger = logging.getLogger("LLM Server")


class LLMServiceRegistry:
    """A registry class for dynamic server discovery."""

    def __init__(self, expr_name: str, trial_name: str):
        self.expr_name = expr_name
        self.trial_name = trial_name
        self.heartbeat_timeout = 30

    def get_server_key(self, server_id: str) -> str:
        return names.gen_server(self.expr_name, self.trial_name, server_id)

    def register_server(self, server_info: LLMServerInfo):
        server_info.last_heartbeat = datetime.now().timestamp()
        key = self.get_server_key(server_info.server_id)
        name_resolve.add(
            key,
            json.dumps(asdict(server_info)),
            keepalive_ttl=self.heartbeat_timeout,
            replace=False,
        )

    def unregister_server(self, server_id: str):
        try:
            name_resolve.delete(self.get_server_key(server_id))
        except name_resolve.NameEntryNotFoundError:
            pass

    def update_heartbeat(
        self, server_id: str, status: str, load: float = 0.0, version: int = 0
    ):
        try:
            key = self.get_server_key(server_id)
            server_data = name_resolve.get(key)
            server_info = LLMServerInfo(**json.loads(server_data))
            server_info.last_heartbeat = datetime.now().timestamp()
            server_info.load = load
            server_info.status = status
            server_info.version = version
            name_resolve.add(
                key,
                json.dumps(asdict(server_info)),
                keepalive_ttl=self.heartbeat_timeout,
                replace=True,
            )
        except (name_resolve.NameEntryNotFoundError, json.JSONDecodeError):
            pass

    def get_healthy_servers(self) -> List[LLMServerInfo]:
        servers = []
        current_time = time.time()
        try:
            root = names.gen_server_root(self.expr_name, self.trial_name)
            server_infos = name_resolve.get_subtree(root)
            for server_data in server_infos:
                try:
                    server_info = LLMServerInfo(**json.loads(server_data))
                    if (
                        current_time - server_info.last_heartbeat
                        < self.heartbeat_timeout
                        and server_info.status == "healthy"
                    ):
                        servers.append(server_info)
                except (json.JSONDecodeError, TypeError):
                    continue
        except name_resolve.NameEntryNotFoundError:
            pass
        return servers


class LLMServer:
    def __init__(self, args: TrainingArgs, service_config: LLMServiceConfig):
        self.args = args
        self.server_id = str(uuid.uuid4())
        self.registry = LLMServiceRegistry(args.experiment_name, args.trial_name)
        self.running = False
        self.load = 0.0
        self.process: Optional[subprocess.Popen] = None
        self.service_config = service_config

    def launch_server(self) -> Optional[LLMServerInfo]:
        """Launch the LLM server subprocess. Returns server info or None if failed."""
        raise NotImplementedError()

    def check_health(self) -> bool:
        """Check if the server is healthy."""
        raise NotImplementedError()

    def start(self):
        """Main entry point - start server and run until exit"""
        try:
            self._startup()
            self._run()
        except Exception as e:
            logger.error(f"Server error: {e}")
            logger.error(traceback.format_exc())
            self._graceful_exit(1)

    def _startup(self):
        """Initialize and start the server"""
        self.running = True

        # Launch server process
        server_info = self.launch_server()
        if not server_info or not self.process:
            raise RuntimeError("Failed to launch server")

        logger.info(f"Server {self.server_id} starting")

        # Wait for server to be ready
        if not self._wait_for_ready():
            raise RuntimeError(
                f"Server failed to become ready in {self.service_config.startup_timeout}s"
            )

        # Register with service registry
        self.registry.register_server(server_info)

        # Start health monitoring
        health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        health_thread.start()

        logger.info(
            f"Server {self.server_id} ready and registered at http://{server_info.host}:{server_info.port}"
        )

    def _wait_for_ready(self) -> bool:
        """Wait for server to become healthy"""
        start_time = time.time()
        while time.time() - start_time < self.service_config.startup_timeout:
            if not self.running or (self.process and self.process.poll() is not None):
                return False
            if self.check_health():
                return True
            time.sleep(2)
        return False

    def _run(self):
        """Main server loop"""
        try:
            while self.running:
                # Check if subprocess died
                if self.process and self.process.poll() is not None:
                    logger.error(
                        f"Server process died (code: {self.process.returncode})"
                    )
                    self._graceful_exit(1)
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self._graceful_exit(0)

    def _health_monitor(self):
        """Monitor server health and exit if unhealthy"""
        failures = 0
        max_failures = self.service_config.max_unhealth_count

        while self.running:
            try:
                # Check process first
                if self.process and self.process.poll() is not None:
                    logger.error("Server process died")
                    self._graceful_exit(1)
                    break

                # Check health
                if self.check_health():
                    failures = 0
                    self.registry.update_heartbeat(self.server_id, "healthy", self.load)
                else:
                    failures += 1
                    logger.warning(f"Health check failed ({failures}/{max_failures})")

                    if failures >= max_failures:
                        logger.error("Too many health check failures")
                        self.registry.update_heartbeat(
                            self.server_id, "unhealthy", self.load
                        )
                        if self.service_config.graceful_shutdown_on_unhealthy:
                            self._graceful_exit(1)
                            break

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                logger.error(traceback.format_exc())
                failures += 1
                if (
                    failures >= max_failures
                    and self.service_config.graceful_shutdown_on_unhealthy
                ):
                    self._graceful_exit(1)
                    break

            time.sleep(self.service_config.health_check_interval)

    def _graceful_exit(self, exit_code: int):
        """Clean shutdown and exit"""
        if not self.running:
            return

        logger.info(f"Graceful shutdown initiated (exit code: {exit_code})")
        self.running = False

        # Cleanup registry
        try:
            self.registry.unregister_server(self.server_id)
        except Exception as e:
            logger.warning(f"Registry cleanup failed: {e}")
            logger.warning(traceback.format_exc())

        # Stop process
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("Server terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Force killing server")
                try:
                    self.process.kill()
                    self.process.wait()
                except (ProcessLookupError, OSError):
                    pass
            except Exception as e:
                logger.error(f"Process cleanup failed: {e}")
                logger.error(traceback.format_exc())

        if exit_code != 0:
            sys.exit(exit_code)


@dataclass
class LLMServerFactory:
    args: TrainingArgs

    def make_server(self, server_config: LLMServiceConfig) -> LLMServer:
        """Create an LLM server instance based on the configuration."""
        if self.args.rollout.server_backend == "sglang":
            from arealite.system.sglang_server import SGLangServer

            return SGLangServer(self.args, server_config)
        else:
            raise ValueError(
                f"Unsupported server backend: {self.args.rollout.server_backend}"
            )
