"""All types and classes for the terminal-bench server."""

import asyncio
import contextvars
import logging
import os
import threading
import warnings
from pathlib import Path
from time import time
from typing import Any

import docker
from pydantic import BaseModel
from starlette.responses import JSONResponse
from terminal_bench.handlers.trial_handler import Task
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.tmux_session import TmuxSession
from tqdm import tqdm

from .logging_config import setup_logging

# Suppress SQLAlchemy 2.0 deprecation warnings from terminal_bench
warnings.filterwarnings("ignore", category=DeprecationWarning, module="terminal_bench")

setup_logging()

logger = logging.getLogger(__name__)

# Context variable to track request start time
request_start_time = contextvars.ContextVar("request_start_time", default=None)

# --- Pydantic Models for API and Tools ---


class TaskRequest(BaseModel):
    uuid: str | None = None
    task_name: str | None = None
    container_name: str | None = None  # Optional, auto-generated from uuid + task_name


class TaskContainer(DockerComposeManager):
    def __init__(
        self,
        uuid: str,
        task_name: str,
        logs_dir: Path,
        docker_compose_path: Path,
        client_image_name: str,
        no_rebuild: bool = False,
        cleanup: bool = False,
    ):
        container_name = f"{uuid}__{task_name}"

        # Initialize parent class with required parameters
        super().__init__(
            client_container_name=container_name,
            client_image_name=client_image_name,
            docker_compose_path=docker_compose_path,
            no_rebuild=no_rebuild,
            cleanup=cleanup,
        )

        # Store additional attributes
        self.uuid = uuid
        self.task_name = task_name
        self.container_name = container_name
        self.logs_dir = logs_dir.joinpath(uuid)
        ## override env
        self.env["T_BENCH_TASK_LOGS_PATH"] = str(self.logs_dir.joinpath("client"))
        self.env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(self.logs_dir.joinpath("agent"))


class TerminalBenchServer:
    """
    Terminal Bench Server class that manages server state and operations.
    Provides thread-safe access to shared resources using asyncio locks.
    """

    _is_running = False
    preheat_image = False

    def __init__(
        self,
        tasks_dir: Path | None = None,
        tasks_log_dir: Path | None = None,
        preheat_image: bool = False,
    ):
        # In-memory registry for active tasks and their managers
        # Format: {container_name: {"uuid": str, "task_name": str, "last_seen": float, "compose_manager": TaskContainer, "container": Container}}
        self.active_tasks: dict[str, dict[str, Any]] = {}
        # Cache for TmuxSession objects
        self.tmux_sessions: dict[str, TmuxSession] = {}
        # Docker client instance
        self.docker_client = None
        # Path to the tasks directory, needs to be configured
        self.tasks_dir = tasks_dir or Path(
            os.environ.get("T_BENCH_TASKS_DIR", "/app/tasks")
        )
        # Path to the tasks logs directory
        self.tasks_log_dir = tasks_log_dir or Path(
            os.environ.get("T_BENCH_TASKS_LOG_DIR", "/var/logs/terminal-bench/")
        )
        self.preheat_image = preheat_image
        # Thread locks for thread-safe access to shared resources
        # Using threading.Lock instead of asyncio.Lock for cross-thread safety
        self.active_tasks_lock = threading.Lock()
        self.tmux_sessions_lock = threading.Lock()
        self.garbage_collector_task = None
        # Background event loop and thread for GC
        self._gc_loop = None
        self._gc_thread = None

    def init_images_sync(self):
        """Pre-build all Docker images from tasks directory."""

        print("Initializing Docker images...")
        if not self.tasks_dir.exists():
            print(f"Warning: Tasks directory {self.tasks_dir} does not exist")
            return

        # Get all task directories that contain docker-compose.yaml
        task_dirs = [d for d in self.tasks_dir.iterdir() if d.is_dir()]
        total_tasks = len(task_dirs)
        built_count = 0
        skipped_count = 0
        failed_count = 0

        print(f"Found {total_tasks} task directories")

        # Use tqdm for progress tracking
        with tqdm(total=total_tasks, desc="Building images", unit="task") as pbar:
            for task_dir in task_dirs:
                # Quit if server is shutting down
                if not self._is_running:
                    pbar.set_description("Interrupted")
                    print("\nImage initialization interrupted by shutdown")
                    break
                compose_path = task_dir / "docker-compose.yaml"
                if not compose_path.exists():
                    skipped_count += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "built": built_count,
                            "skipped": skipped_count,
                            "failed": failed_count,
                        }
                    )
                    continue

                task_name = task_dir.name
                image_name = f"tb__{task_name.replace('.', '-')}__client"

                try:
                    # Check if image already exists
                    if self._image_exists(image_name):
                        pbar.set_description(f"Skipping {task_name[:30]}")
                        skipped_count += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "built": built_count,
                                "skipped": skipped_count,
                                "failed": failed_count,
                            }
                        )
                        continue

                    pbar.set_description(f"Building {task_name[:30]}")

                    # Create a temporary TaskContainer to build the image
                    temp_manager = TaskContainer(
                        uuid="temp_build",
                        task_name=task_name,
                        logs_dir=self.tasks_log_dir,
                        docker_compose_path=compose_path,
                        client_image_name=image_name,
                        no_rebuild=False,
                    )

                    # Build the image without starting the container
                    temp_manager.build()
                    built_count += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "built": built_count,
                            "skipped": skipped_count,
                            "failed": failed_count,
                        }
                    )

                except Exception as e:
                    failed_count += 1
                    pbar.set_description(f"Failed {task_name[:30]}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "built": built_count,
                            "skipped": skipped_count,
                            "failed": failed_count,
                        }
                    )
                    print(f"\nError building {task_name}: {e}")

        print("\nImage initialization complete:")
        print(f"  Built: {built_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {total_tasks}")

    def _run_gc_loop(self):
        """Run garbage collector in a separate thread with its own event loop."""
        self._gc_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._gc_loop)
        try:
            self._gc_loop.run_until_complete(self.garbage_collector())
        except Exception as e:
            print(f"Error in GC loop: {e}")
        finally:
            self._gc_loop.close()

    def startup(self):
        """Server startup logic (synchronous)."""
        print("Server starting up...")
        try:
            self.docker_client = docker.from_env()
            self._is_running = True

            if self.preheat_image:
                self.init_images_sync()

            # Recover active tasks synchronously
            self._recover_active_tasks_sync()

            # Start garbage collector in background thread
            self._gc_thread = threading.Thread(target=self._run_gc_loop, daemon=True)
            self._gc_thread.start()

            print("Server startup complete with following configuration:")
            print(f"TASK DIR: {self.tasks_dir}")
            print(f"RECOVER TASK NUM: {len(self.active_tasks)}")
            print(f"TASK LOGS DIR: {self.tasks_log_dir}")

        except Exception as e:
            self._is_running = False
            print(f"Error during startup: {e}")

    def shutdown(self):
        """Server shutdown logic (synchronous)."""
        print("Server shutting down...")
        self._is_running = False

        # Stop garbage collector loop
        if self._gc_loop and self._gc_loop.is_running():
            self._gc_loop.call_soon_threadsafe(self._gc_loop.stop)

        # Wait for GC thread to finish
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=5.0)

        print("Server shutdown complete.")

    def _recover_active_tasks_sync(self):
        """Scan for existing containers and repopulate the active_tasks registry (synchronous)."""
        print("Recovering active tasks...")
        if not self.docker_client:
            print("Docker client not initialized, skipping recovery")
            return

        containers = self.docker_client.containers.list()

        # No need for async lock in sync context
        for container in containers:
            parts = container.name.split("__")
            if len(parts) == 2:
                uuid, task_name = parts
                print(f"Found existing container: {container.name}")
                compose_path = self.tasks_dir / task_name / "docker-compose.yaml"
                if compose_path.exists():
                    image_name = f"tb__{task_name.replace('.', '-')}__client"
                    compose_manager = TaskContainer(
                        uuid=uuid,
                        task_name=task_name,
                        client_image_name=image_name,
                        docker_compose_path=compose_path,
                        logs_dir=self.tasks_log_dir,
                        no_rebuild=True,
                    )
                    self.active_tasks[container.name] = {
                        "uuid": uuid,
                        "task_name": task_name,
                        "last_seen": time(),
                        "compose_manager": compose_manager,
                        "container": container,
                    }
                    print(f"Recovered task: {container.name}")
                else:
                    print(
                        f"Warning: Could not find compose file for task {task_name} of container {container.name}"
                    )

    async def garbage_collector(self):
        """Periodically clean up idle containers."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                print("Running garbage collector...")
                current_time = time()
                idle_timeout = 60 * 15  # 10 minutes

                with self.active_tasks_lock:
                    containers_to_cleanup = []
                    for container_name, task_info in self.active_tasks.items():
                        if current_time - task_info["last_seen"] > idle_timeout:
                            containers_to_cleanup.append(container_name)

                    for container_name in containers_to_cleanup:
                        task_info = self.active_tasks[container_name]
                        print(f"Container {container_name} is idle. Shutting down.")
                        try:
                            task_info["compose_manager"].stop()

                            # Clean up tmux session
                            with self.tmux_sessions_lock:
                                if container_name in self.tmux_sessions:
                                    del self.tmux_sessions[container_name]

                            del self.active_tasks[container_name]
                            print(f"Successfully cleaned up {container_name}.")
                        except Exception as e:
                            print(
                                f"Error during garbage collection for {container_name}: {e}"
                            )
            except asyncio.CancelledError:
                print("Garbage collector cancelled.")
                break
            except Exception as e:
                print(f"Error in garbage collector: {e}")

    async def start_task(self, req: TaskRequest) -> JSONResponse:
        """Starts a new task container or returns an existing one."""
        op_start = time()
        container_name = f"{req.uuid}__{req.task_name}"
        logger.info(f"[PERF] start_task called for {container_name}")

        # Check if container already exists
        check_start = time()
        with self.active_tasks_lock:
            if container_name in self.active_tasks:
                self.active_tasks[container_name]["last_seen"] = time()
                log_operation_time(
                    f"start_task (reused) for {container_name}", op_start
                )
                return JSONResponse(
                    {"container_name": container_name, "status": "reused"}
                )
        log_operation_time("start_task lock check", check_start)

        compose_path = self.tasks_dir / req.task_name / "docker-compose.yaml"
        if not compose_path.exists():
            return JSONResponse(
                {"error": f"Task '{req.task_name}' not found."},
                status_code=404,
            )

        logger.info(f"Creating new container for task: {req.task_name}")

        try:
            image_name = f"tb__{req.task_name.replace('.', '-')}__client"

            # Check if image exists
            check_image_start = time()
            no_rebuild = self._image_exists(image_name)
            log_operation_time(
                f"start_task check image exists for {req.task_name}", check_image_start
            )

            # Create TaskContainer
            create_manager_start = time()
            compose_manager = TaskContainer(
                uuid=req.uuid,
                task_name=req.task_name,
                client_image_name=image_name,
                docker_compose_path=compose_path,
                no_rebuild=no_rebuild,
                cleanup=False,  # set to False for future reuse
                logs_dir=self.tasks_log_dir,
            )
            log_operation_time(
                f"start_task create TaskContainer for {req.task_name}",
                create_manager_start,
            )

            # Start container (potentially slow)
            start_container_time = time()
            container = compose_manager.start()
            log_operation_time(
                f"start_task container.start() for {req.task_name}",
                start_container_time,
            )

            # Register container
            register_start = time()
            with self.active_tasks_lock:
                self.active_tasks[container_name] = {
                    "uuid": req.uuid,
                    "task_name": req.task_name,
                    "last_seen": time(),
                    "compose_manager": compose_manager,
                    "container": container,
                }
            log_operation_time(
                f"start_task register container for {container_name}", register_start
            )
            log_operation_time(f"start_task (created) for {container_name}", op_start)

            return JSONResponse({"container_name": container_name, "status": "created"})
        except Exception as e:
            logger.error(f"Error starting container: {e}, task name: {req.task_name}")
            log_operation_time(f"start_task (failed) for {container_name}", op_start)
            return JSONResponse(
                {"error": f"Failed to start container: {e}"}, status_code=500
            )

    async def list_tasks(self) -> JSONResponse:
        """Lists all active tasks."""
        with self.active_tasks_lock:
            tasks = [
                {
                    "container_name": name,
                    "uuid": info["uuid"],
                    "task_name": info["task_name"],
                    "last_seen": info["last_seen"],
                }
                for name, info in self.active_tasks.items()
            ]
        return JSONResponse(tasks)

    async def get_tmux_session(self, container_name: str) -> TmuxSession:
        """Get or create a TmuxSession for the given container."""
        with self.tmux_sessions_lock:
            if container_name not in self.tmux_sessions:
                print(f"Creating new tmux session for {container_name}")

                with self.active_tasks_lock:
                    if container_name not in self.active_tasks:
                        raise ValueError(
                            f"Container {container_name} not found in active tasks"
                        )

                    task_info = self.active_tasks[container_name]
                    container = task_info["container"]
                    user = container.attrs["Config"].get("User", "")

                    self.tmux_sessions[container_name] = TmuxSession(
                        session_name="agent",
                        container=container,
                        user=user,
                    )
                    self.tmux_sessions[container_name].start()

            return self.tmux_sessions[container_name]

    def create_validation_session(self, container_name: str):
        session = TmuxSession(
            session_name="test",
            container=self.active_tasks[container_name]["container"],
            user=self.active_tasks[container_name]["container"]
            .attrs["Config"]
            .get("User", ""),
        )
        session.start()
        return session

    def update_task_last_seen(self, container_name: str):
        """Update the last_seen timestamp for a task."""
        with self.active_tasks_lock:
            if container_name in self.active_tasks:
                self.active_tasks[container_name]["last_seen"] = time()

    def validate_container(self, container_name: str) -> bool:
        """Validate that the container exists."""
        with self.active_tasks_lock:
            return container_name in self.active_tasks

    async def validate_task(self, container_name: str) -> JSONResponse:
        """Run tests in the container and return test results."""
        op_start = time()

        if not container_name:
            return JSONResponse(
                {"error": "Container name is required"},
                status_code=400,
            )

        logger.info(f"[PERF] validate_container called for {container_name}")

        # Validate container exists
        validate_start = time()
        if not self.validate_container(container_name):
            return JSONResponse(
                {"error": f"Container '{container_name}' not found"}, status_code=404
            )
        log_operation_time(
            f"validate_container check exists for {container_name}", validate_start
        )

        try:
            # Update last seen timestamp
            self.update_task_last_seen(container_name)

            # Create a temporary tmux session
            session_start = time()
            session = self.create_validation_session(container_name)
            log_operation_time(
                f"validate_container create tmux session for {container_name}",
                session_start,
            )

            # Get task info
            with self.active_tasks_lock:
                task_info = self.active_tasks.get(container_name)
                if not task_info:
                    return JSONResponse(
                        {
                            "error": f"Task info not found for container '{container_name}'"
                        },
                        status_code=404,
                    )
                task_name = task_info["task_name"]

            # Get test timeout and parser from task config
            task_dir = self.tasks_dir / task_name
            task_config_path = task_dir / "task.yaml"

            max_test_timeout_sec = 60.0  # Default timeout
            parser_name = "pytest"  # Default parser

            if task_config_path.exists():
                try:
                    task = Task.from_yaml(task_config_path)
                    max_test_timeout_sec = task.max_test_timeout_sec
                    parser_name = task.parser_name
                except Exception as e:
                    logger.warning(f"Failed to load task config for {task_name}: {e}")

            # Copy test files to container
            run_tests_path = task_dir / "run-tests.sh"
            test_dir = task_dir / "tests"

            if not run_tests_path.exists():
                return JSONResponse(
                    {
                        "container_name": container_name,
                        "status": "error",
                        "error": f"Test script not found: {run_tests_path}",
                    },
                    status_code=404,
                )

            try:
                # Use DockerComposeManager's static method to copy files
                copy_files_start = time()
                with self.active_tasks_lock:
                    container = self.active_tasks[container_name]["container"]

                paths_to_copy = [run_tests_path]
                if test_dir.exists():
                    paths_to_copy.append(test_dir)

                DockerComposeManager.copy_to_container(
                    container=container, paths=paths_to_copy, container_dir="/tests"
                )
                log_operation_time(
                    f"validate_container copy test files for {container_name}",
                    copy_files_start,
                )
                logger.info(f"Copied test files to container {container_name}")
            except Exception as e:
                logger.error(f"Failed to copy test files: {e}")
                return JSONResponse(
                    {
                        "container_name": container_name,
                        "status": "error",
                        "error": f"Failed to copy test files: {str(e)}",
                    },
                    status_code=500,
                )

            # Run test script
            test_script_path = "/tests/run-tests.sh"

            logger.info(
                f"Running tests for container {container_name} with timeout {max_test_timeout_sec}s"
            )

            try:
                run_tests_start = time()
                session.send_keys(
                    [f"bash {test_script_path}", "Enter"],
                    block=True,
                    max_timeout_sec=max_test_timeout_sec,
                )
                log_operation_time(
                    f"validate_container run tests for {container_name}",
                    run_tests_start,
                )
            except TimeoutError:
                logger.warning(f"Test timeout for container {container_name}")
                log_operation_time(
                    f"validate_container (timeout) for {container_name}", op_start
                )
                return JSONResponse(
                    {
                        "container_name": container_name,
                        "status": "timeout",
                        "error": f"Test execution timed out after {max_test_timeout_sec} seconds",
                    }
                )

            # Capture test output
            capture_start = time()
            test_output = session.capture_pane(capture_entire=True)
            log_operation_time(
                f"validate_container capture output for {container_name}", capture_start
            )

            # Parse test results
            try:
                parser = ParserFactory.get_parser(parser_name)
                results = parser.parse(test_output)

                # Calculate weighted score
                score = _calculate_weighted_test_score(results, None)

                log_operation_time(
                    f"validate_container (success) for {container_name} with score {score}",
                    op_start,
                )

                return JSONResponse(
                    {
                        "container_name": container_name,
                        "status": "completed",
                        "score": score,
                        "raw_output": test_output,
                    },
                    status_code=200,
                )

            except Exception as e:
                logger.error(f"Error parsing test results for {task_name}: {e}")
                log_operation_time(
                    f"validate_container (parse_error) for {container_name}", op_start
                )
                return JSONResponse(
                    {
                        "container_name": container_name,
                        "status": "parse_error",
                        "error": f"Failed to parse test results: {str(e)}",
                        "raw_output": test_output,
                    },
                    status_code=500,
                )

        except Exception as e:
            logger.error(
                f"Error validating container {container_name}: {e}", exc_info=True
            )
            log_operation_time(
                f"validate_container (failed) for {container_name}", op_start
            )
            return JSONResponse(
                {"error": f"Failed to validate container: {str(e)}"}, status_code=500
            )

    async def stop_task(self, req: TaskRequest) -> JSONResponse:
        """Stops and removes a task container for resource cleanup."""
        container_name = req.container_name
        if not container_name:
            container_name = f"{req.uuid}__{req.task_name}"

        with self.active_tasks_lock:
            if container_name not in self.active_tasks:
                return JSONResponse(
                    {"error": f"Container '{container_name}' not found."},
                    status_code=404,
                )

            task_info = self.active_tasks[container_name]

        print(f"Stopping container: {container_name}")

        try:
            # Stop the compose services
            task_info["compose_manager"].stop()

            # Clean up tmux session
            with self.tmux_sessions_lock:
                if container_name in self.tmux_sessions:
                    del self.tmux_sessions[container_name]

            # Remove from active tasks
            with self.active_tasks_lock:
                del self.active_tasks[container_name]

            print(f"Successfully stopped and removed container: {container_name}")
            return JSONResponse({"container_name": container_name, "status": "stopped"})
        except Exception as e:
            print(f"Error stopping container {container_name}: {e}")
            return JSONResponse(
                {"error": f"Failed to stop container: {e}"}, status_code=500
            )

    def _image_exists(self, image_name: str) -> bool:
        """Check if a Docker image exists locally.

        Args:
            image_name: Name of the Docker image to check

        Returns:
            True if image exists, False otherwise
        """
        try:
            self.docker_client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            return False


def log_operation_time(operation_name: str, start_time: float):
    """Log the time taken for an operation."""
    duration = time() - start_time
    logger.info(
        f"[PERF] {operation_name} took {duration:.3f}s ({duration * 1000:.1f}ms)"
    )


def _calculate_weighted_test_score(
    results: dict[str, UnitTestStatus],
    test_weights: dict[str, float] | None,
) -> float:
    """
    Calculate weighted score from test results.

    Args:
        results: Test name to status mapping
        test_weights: Test name to weight mapping

    Returns:
        Weighted score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    # If no test weights provided or only placeholder, use equal weights
    # Filter out placeholder key used when test_weights.json doesn't exist
    filtered_weights = {
        k: v for k, v in (test_weights or {}).items() if not k.startswith("_")
    }

    if not filtered_weights:
        equal_weight = 1.0 / len(results)
        total_score = sum(
            equal_weight if status == UnitTestStatus.PASSED else 0.0
            for status in results.values()
        )
        return total_score

    # Calculate weighted score
    total_score = 0.0
    total_weight = 0.0

    for test_name, status in results.items():
        weight = filtered_weights.get(test_name, 0.0)
        if weight > 0:
            score = 1.0 if status == UnitTestStatus.PASSED else 0.0
            total_score += score * weight
            total_weight += weight

    # Normalize if weights don't sum to 1.0
    if total_weight > 0:
        return total_score / total_weight

    equal_weight = 1.0 / len(results)
    return sum(
        equal_weight if status == UnitTestStatus.PASSED else 0.0
        for status in results.values()
    )
