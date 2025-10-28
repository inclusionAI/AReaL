"""
Comprehensive unit tests for LocalScheduler.

This test suite covers:
1. Initialization and GPU detection
2. Worker creation with various configurations
3. GPU allocation strategies (new, colocate, round-robin)
4. Port allocation and tracking
5. Worker health checks and readiness
6. Engine creation and method calls (sync and async)
7. Error handling for all exception types
8. Resource cleanup and process termination
9. Edge cases (duplicate workers, worker not found, GPU exhaustion, port conflicts)
10. Log file handling
11. HTTP client interactions
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import psutil
import pytest

from areal.api.scheduler_api import (
    ContainerSpec,
    ScheduleStrategy,
    SchedulingConfig,
    Worker,
)
from areal.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    GPUAllocationError,
    PortAllocationError,
    RPCConnectionError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.scheduler.local_scheduler import LocalScheduler, WorkerInfo

# ============================================================================
# Fixtures and Helper Functions
# ============================================================================


@pytest.fixture
def scheduler(tmp_path):
    """Create a LocalScheduler instance with default configuration."""
    return LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))


@pytest.fixture
def multi_gpu_scheduler(tmp_path):
    """Create a LocalScheduler instance with multiple GPUs."""
    return LocalScheduler(gpu_devices=[0, 1, 2], log_dir=str(tmp_path))


def create_mock_process(pid=1234, is_alive=True, exit_code=None):
    """Create a mock subprocess.Popen process.

    Args:
        pid: Process ID
        is_alive: Whether process is still running
        exit_code: Exit code if process has terminated

    Returns:
        Mock process object
    """
    mock_proc = Mock()
    mock_proc.pid = pid
    mock_proc.poll.return_value = None if is_alive else exit_code
    if not is_alive:
        mock_proc.returncode = exit_code
    return mock_proc


def create_worker_info(
    worker_id="test/0",
    role="test",
    ip="127.0.0.1",
    ports=None,
    gpu_devices=None,
    log_file="/tmp/test.log",
    process=None,
):
    """Create a WorkerInfo instance with sensible defaults.

    Args:
        worker_id: Worker identifier
        role: Worker role name
        ip: IP address
        ports: List of port strings
        gpu_devices: List of GPU device IDs
        log_file: Path to log file
        process: Mock process object (created if not provided)

    Returns:
        WorkerInfo instance
    """
    if ports is None:
        ports = ["8000"]
    if gpu_devices is None:
        gpu_devices = [0]
    if process is None:
        process = create_mock_process()

    return WorkerInfo(
        worker=Worker(id=worker_id, ip=ip, ports=ports),
        process=process,
        role=role,
        gpu_devices=gpu_devices,
        created_at=time.time(),
        log_file=log_file,
    )


def create_mock_http_response(status_code=200, json_data=None):
    """Create a mock HTTP response.

    Args:
        status_code: HTTP status code
        json_data: Dictionary to return from response.json()

    Returns:
        Mock response object
    """
    mock_response = Mock()
    mock_response.status_code = status_code
    if json_data is not None:
        mock_response.json.return_value = json_data
    return mock_response


class TestLocalSchedulerInitialization:
    """Test LocalScheduler initialization and GPU detection."""

    def test_init_with_explicit_gpu_devices(self, tmp_path):
        """Should initialize with explicitly provided GPU devices."""
        scheduler = LocalScheduler(
            gpu_devices=[0, 1, 2],
            log_dir=str(tmp_path),
            startup_timeout=60.0,
            health_check_interval=2.0,
        )

        assert scheduler.gpu_devices == [0, 1, 2]
        assert scheduler.log_dir == tmp_path
        assert scheduler.startup_timeout == 60.0
        assert scheduler.health_check_interval == 2.0
        assert scheduler._gpu_counter == 0
        assert len(scheduler._allocated_ports) == 0
        assert len(scheduler._workers) == 0
        assert tmp_path.exists()

    def test_init_without_gpu_devices_uses_cuda_visible_devices(self, tmp_path):
        """Should detect GPUs from CUDA_VISIBLE_DEVICES environment variable."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,3"}):
            scheduler = LocalScheduler(log_dir=str(tmp_path))
            assert scheduler.gpu_devices == [0, 1, 3]

    def test_init_with_invalid_cuda_visible_devices(self, tmp_path):
        """Should fall back to default [0] when CUDA_VISIBLE_DEVICES is invalid."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "invalid,gpu,ids"}):
            scheduler = LocalScheduler(log_dir=str(tmp_path))
            assert scheduler.gpu_devices == [0]

    def test_init_without_cuda_visible_devices(self, tmp_path):
        """Should default to [0] when CUDA_VISIBLE_DEVICES is not set."""
        with patch.dict(os.environ, {}, clear=True):
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            scheduler = LocalScheduler(log_dir=str(tmp_path))
            assert scheduler.gpu_devices == [0]

    def test_init_creates_log_directory(self, tmp_path):
        """Should create log directory if it doesn't exist."""
        log_dir = tmp_path / "nested" / "log" / "dir"
        assert not log_dir.exists()

        scheduler = LocalScheduler(log_dir=str(log_dir))

        assert log_dir.exists()
        assert scheduler.log_dir == log_dir

    def test_init_creates_http_clients(self, tmp_path):
        """Should initialize both sync and async HTTP clients."""
        scheduler = LocalScheduler(log_dir=str(tmp_path))

        assert isinstance(scheduler._http_client, httpx.Client)
        assert isinstance(scheduler._async_http_client, httpx.AsyncClient)


class TestGPUAllocation:
    """Test GPU allocation strategies."""

    def test_allocate_gpus_round_robin(self, tmp_path):
        """Should allocate GPUs in round-robin fashion."""
        scheduler = LocalScheduler(gpu_devices=[0, 1, 2], log_dir=str(tmp_path))

        # First allocation
        gpus1 = scheduler._allocate_gpus(2)
        assert gpus1 == [0, 1]

        # Second allocation (wraps around)
        gpus2 = scheduler._allocate_gpus(3)
        assert gpus2 == [2, 0, 1]

        # Third allocation
        gpus3 = scheduler._allocate_gpus(1)
        assert gpus3 == [2]

    def test_allocate_gpus_exceeds_available(self, tmp_path):
        """Should raise GPUAllocationError when requesting more GPUs than available."""
        scheduler = LocalScheduler(gpu_devices=[0, 1], log_dir=str(tmp_path))

        with pytest.raises(GPUAllocationError) as exc_info:
            scheduler._allocate_gpus(3)

        assert "Requested 3 GPUs but only 2 available" in str(exc_info.value)

    def test_allocate_gpus_single_gpu_multiple_times(self, scheduler):
        """Should allow multiple workers to share a single GPU via round-robin."""
        # Multiple allocations should all get GPU 0
        for _ in range(5):
            gpus = scheduler._allocate_gpus(1)
            assert gpus == [0]

    def test_get_colocated_gpus_success(self, multi_gpu_scheduler):
        """Should return GPU devices from target worker for colocation."""
        # Create mock workers for target role
        worker1 = create_worker_info(
            worker_id="actor/0", role="actor", ports=["8000"], gpu_devices=[0, 1]
        )
        worker2 = create_worker_info(
            worker_id="actor/1", role="actor", ports=["8001"], gpu_devices=[2]
        )
        multi_gpu_scheduler._workers["actor"] = [worker1, worker2]

        # Get colocated GPUs
        gpus = multi_gpu_scheduler._get_colocated_gpus("actor", 0)
        assert gpus == [0, 1]

        gpus = multi_gpu_scheduler._get_colocated_gpus("actor", 1)
        assert gpus == [2]

    def test_get_colocated_gpus_role_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when target role doesn't exist."""
        with pytest.raises(WorkerNotFoundError) as exc_info:
            scheduler._get_colocated_gpus("nonexistent", 0)

        assert "Cannot colocate with role 'nonexistent' - role not found" in str(
            exc_info.value
        )

    def test_get_colocated_gpus_worker_index_out_of_range(self, scheduler):
        """Should raise ValueError when worker index is out of range."""
        # Create only one worker for target role
        worker = create_worker_info(worker_id="actor/0", role="actor", gpu_devices=[0])
        scheduler._workers["actor"] = [worker]

        with pytest.raises(ValueError) as exc_info:
            scheduler._get_colocated_gpus("actor", 5)

        assert "only 1 workers exist" in str(exc_info.value)


class TestPortAllocation:
    """Test port allocation and tracking."""

    def test_allocate_ports_success(self, tmp_path):
        """Should allocate requested number of free ports."""
        with patch(
            "areal.scheduler.local_scheduler.find_free_ports"
        ) as mock_find_ports:
            mock_find_ports.return_value = [8000, 8001, 8002]

            scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))
            ports = scheduler._allocate_ports(3)

            assert ports == [8000, 8001, 8002]
            assert scheduler._allocated_ports == {8000, 8001, 8002}
            mock_find_ports.assert_called_once_with(3, exclude_ports=set())

    def test_allocate_ports_excludes_already_allocated(self, tmp_path):
        """Should exclude already allocated ports from search."""
        with patch(
            "areal.scheduler.local_scheduler.find_free_ports"
        ) as mock_find_ports:
            mock_find_ports.side_effect = [
                [8000, 8001],
                [8002, 8003],
            ]

            scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

            # First allocation
            ports1 = scheduler._allocate_ports(2)
            assert ports1 == [8000, 8001]

            # Second allocation should exclude previously allocated ports
            ports2 = scheduler._allocate_ports(2)
            assert ports2 == [8002, 8003]
            assert scheduler._allocated_ports == {8000, 8001, 8002, 8003}

            # Verify excluded ports were passed
            calls = mock_find_ports.call_args_list
            assert calls[0] == call(2, exclude_ports=set())
            assert calls[1] == call(2, exclude_ports={8000, 8001})

    def test_allocate_ports_failure(self, tmp_path):
        """Should raise PortAllocationError when port allocation fails."""
        with patch(
            "areal.scheduler.local_scheduler.find_free_ports"
        ) as mock_find_ports:
            mock_find_ports.side_effect = ValueError("No free ports available")

            scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

            with pytest.raises(PortAllocationError) as exc_info:
                scheduler._allocate_ports(5)

            assert "No free ports available" in str(exc_info.value)


class TestWorkerCreation:
    """Test worker creation with various configurations."""

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_with_default_spec(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should create workers with default spec (1 GPU, 2 ports) when no specs provided."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000, 8001], [8002, 8003]]

        # Mock process
        mock_process1 = Mock()
        mock_process1.pid = 1234
        mock_process1.poll.return_value = None
        mock_process2 = Mock()
        mock_process2.pid = 1235
        mock_process2.poll.return_value = None
        mock_popen.side_effect = [mock_process1, mock_process2]

        scheduler = LocalScheduler(gpu_devices=[0, 1], log_dir=str(tmp_path))

        config = SchedulingConfig(replicas=2, role="rollout")
        worker_ids = scheduler.create_workers("rollout", config)

        assert worker_ids == ["rollout/0", "rollout/1"]
        assert "rollout" in scheduler._workers
        assert len(scheduler._workers["rollout"]) == 2

        # Verify default spec was used
        assert mock_popen.call_count == 2

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_with_single_spec_for_all(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should use single spec for all workers when specs length is 1."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000, 8001, 8002]] * 3

        # Mock processes
        mock_processes = []
        for i in range(3):
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = LocalScheduler(gpu_devices=[0, 1, 2], log_dir=str(tmp_path))

        config = SchedulingConfig(
            replicas=3,
            role="actor",
            specs=[ContainerSpec(gpu=2, port_count=3)],
        )
        worker_ids = scheduler.create_workers("actor", config)

        assert len(worker_ids) == 3
        assert mock_popen.call_count == 3

        # All workers should use the same spec
        for worker_info in scheduler._workers["actor"]:
            assert len(worker_info.worker.ports) == 3

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_with_per_worker_specs(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should use individual specs when specs length equals replicas."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [[8000], [8001, 8002]]

        # Mock processes
        mock_proc1 = Mock()
        mock_proc1.pid = 1000
        mock_proc1.poll.return_value = None
        mock_proc2 = Mock()
        mock_proc2.pid = 1001
        mock_proc2.poll.return_value = None
        mock_popen.side_effect = [mock_proc1, mock_proc2]

        scheduler = LocalScheduler(gpu_devices=[0, 1], log_dir=str(tmp_path))

        config = SchedulingConfig(
            replicas=2,
            role="critic",
            specs=[
                ContainerSpec(gpu=1, port_count=1),
                ContainerSpec(gpu=1, port_count=2),
            ],
        )
        worker_ids = scheduler.create_workers("critic", config)

        assert len(worker_ids) == 2
        assert len(scheduler._workers["critic"][0].worker.ports) == 1
        assert len(scheduler._workers["critic"][1].worker.ports) == 2

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_with_custom_command(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should use custom command from spec when provided."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(
            replicas=1,
            role="custom",
            specs=[
                ContainerSpec(
                    gpu=1, port_count=2, cmd="python my_custom_server.py --port 8000"
                )
            ],
        )
        worker_ids = scheduler.create_workers("custom", config)

        assert len(worker_ids) == 1

        # Verify custom command was used
        popen_call = mock_popen.call_args
        cmd_args = popen_call[0][0]
        assert cmd_args == ["python", "my_custom_server.py", "--port", "8000"]

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_with_environment_variables(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should merge environment variables from spec into worker environment."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(
            replicas=1,
            role="envtest",
            specs=[
                ContainerSpec(
                    gpu=1,
                    port_count=2,
                    env_vars={"CUSTOM_VAR": "custom_value", "ANOTHER_VAR": "123"},
                )
            ],
        )
        worker_ids = scheduler.create_workers("envtest", config)

        assert len(worker_ids) == 1

        # Verify environment variables were passed
        popen_call = mock_popen.call_args
        env = popen_call[1]["env"]
        assert env["CUSTOM_VAR"] == "custom_value"
        assert env["ANOTHER_VAR"] == "123"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["WORKER_ID"] == "envtest/0"

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_with_colocate_strategy(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should colocate workers on same GPUs as target role when colocate strategy is used."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_processes = []
        for i in range(4):
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = LocalScheduler(gpu_devices=[0, 1, 2, 3], log_dir=str(tmp_path))

        # Create target workers (actors)
        actor_config = SchedulingConfig(
            replicas=2, role="actor", specs=[ContainerSpec(gpu=2, port_count=2)]
        )
        scheduler.create_workers("actor", actor_config)

        # Get GPU allocations for actors
        actor_gpus_0 = scheduler._workers["actor"][0].gpu_devices
        actor_gpus_1 = scheduler._workers["actor"][1].gpu_devices

        # Reset mock
        mock_find_ports.reset_mock()
        mock_find_ports.return_value = [8010, 8011]

        # Create colocated workers (critics)
        critic_config = SchedulingConfig(
            replicas=2,
            role="critic",
            specs=[ContainerSpec(gpu=2, port_count=2)],
            schedule_strategy=ScheduleStrategy(type="colocate", uid="actor"),
        )
        critic_ids = scheduler.create_workers("critic", critic_config)

        assert len(critic_ids) == 2

        # Verify critics are colocated with actors
        critic_gpus_0 = scheduler._workers["critic"][0].gpu_devices
        critic_gpus_1 = scheduler._workers["critic"][1].gpu_devices

        assert critic_gpus_0 == actor_gpus_0
        assert critic_gpus_1 == actor_gpus_1

    def test_create_workers_duplicate_role_error(self, tmp_path):
        """Should raise WorkerCreationError when attempting to create workers for existing role."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        with (
            patch("areal.scheduler.local_scheduler.subprocess.Popen") as mock_popen,
            patch("areal.scheduler.local_scheduler.find_free_ports") as mock_find_ports,
            patch("areal.scheduler.local_scheduler.gethostip") as mock_gethostip,
        ):
            mock_gethostip.return_value = "127.0.0.1"
            mock_find_ports.return_value = [8000, 8001]
            mock_proc = Mock()
            mock_proc.pid = 1234
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            config = SchedulingConfig(replicas=1, role="test")
            scheduler.create_workers("test", config)

            # Try to create again
            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers("test", config)

            assert "Worker group already exists" in str(exc_info.value)
            assert exc_info.value.worker_key == "test"

    def test_create_workers_zero_replicas_error(self, tmp_path):
        """Should raise WorkerCreationError when replicas is 0."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(replicas=0, role="test")

        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers("test", config)

        assert "replicas must be greater than 0" in str(exc_info.value)

    def test_create_workers_invalid_specs_length(self, tmp_path):
        """Should raise WorkerCreationError when specs length is invalid."""
        scheduler = LocalScheduler(gpu_devices=[0, 1], log_dir=str(tmp_path))

        config = SchedulingConfig(
            replicas=3,
            role="test",
            specs=[
                ContainerSpec(gpu=1, port_count=2),
                ContainerSpec(gpu=1, port_count=2),
            ],  # 2 specs for 3 replicas
        )

        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers("test", config)

        assert "specs length (2) must be 1 or equal to replicas (3)" in str(
            exc_info.value
        )

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_subprocess_fails_immediately(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should raise WorkerCreationError when subprocess exits immediately."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        # Mock process that exits immediately
        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = 1  # Exit code 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        # Create log file with error message
        log_file = tmp_path / "test_0.log"
        log_file.write_text("Error: Failed to start server\n")

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(replicas=1, role="test")

        with patch.object(
            scheduler, "_read_log_tail", return_value="Error: Failed to start server"
        ):
            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers("test", config)

            assert "exited immediately with code 1" in str(exc_info.value)

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_create_workers_cleanup_on_partial_failure(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should clean up successfully created workers when a later worker fails."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [
            [8000, 8001],  # First worker succeeds
            ValueError("No free ports"),  # Second worker fails
        ]

        # First process succeeds
        mock_proc1 = Mock()
        mock_proc1.pid = 1234
        mock_proc1.poll.return_value = None
        mock_popen.return_value = mock_proc1

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(replicas=2, role="test")

        with patch.object(scheduler, "_cleanup_workers") as mock_cleanup:
            with pytest.raises(WorkerCreationError) as exc_info:
                scheduler.create_workers("test", config)

            # Verify cleanup was called
            assert mock_cleanup.called
            assert "Resource allocation failed" in str(exc_info.value)

    def test_create_workers_colocate_strategy_missing_uid(self, tmp_path):
        """Should raise WorkerCreationError when colocate strategy is missing target role uid."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(
            replicas=1,
            role="test",
            specs=[ContainerSpec(gpu=1, port_count=2)],
            schedule_strategy=ScheduleStrategy(type="colocate", uid=""),  # Missing uid
        )

        with pytest.raises(WorkerCreationError) as exc_info:
            scheduler.create_workers("test", config)

        assert "Colocate strategy requires uid" in str(exc_info.value)


class TestGetWorkers:
    """Test getting workers and waiting for readiness."""

    def test_get_workers_role_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when role doesn't exist."""
        with pytest.raises(WorkerNotFoundError) as exc_info:
            scheduler.get_workers("nonexistent")

        assert exc_info.value.worker_id == "nonexistent"

    @patch("areal.scheduler.local_scheduler.time.sleep")
    def test_get_workers_success(self, mock_sleep, scheduler, tmp_path):
        """Should return workers when all are ready."""
        # Create mock workers
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test_0.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test_1.log")
        )

        scheduler._workers["test"] = [worker1, worker2]

        with patch.object(scheduler, "_is_worker_ready", return_value=True):
            workers = scheduler.get_workers("test", timeout=10.0)

            assert len(workers) == 2
            assert workers[0].id == "test/0"
            assert workers[1].id == "test/1"

    @patch("areal.scheduler.local_scheduler.time.time")
    @patch("areal.scheduler.local_scheduler.time.sleep")
    def test_get_workers_timeout(self, mock_sleep, mock_time, scheduler, tmp_path):
        """Should raise WorkerTimeoutError when timeout is exceeded."""
        # Mock time progression - provide enough values
        mock_time.side_effect = [0.0] + [i for i in range(1, 20)]

        worker = create_worker_info(log_file=str(tmp_path / "test_0.log"))
        worker.created_at = 0.0

        scheduler._workers["test"] = [worker]

        # Worker never becomes ready
        with patch.object(scheduler, "_is_worker_ready", return_value=False):
            with pytest.raises(WorkerTimeoutError) as exc_info:
                scheduler.get_workers("test", timeout=5.0)

            assert exc_info.value.worker_key == "test"
            assert exc_info.value.timeout == 5.0

    def test_get_workers_process_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when worker process dies during readiness check."""
        log_file = tmp_path / "test_0.log"
        log_file.write_text("Error: Connection refused\n")

        # Process dies after first check
        mock_proc = create_mock_process()
        mock_proc.poll.side_effect = [None, 1]  # None (alive), then 1 (dead)
        mock_proc.returncode = 1

        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with patch.object(scheduler, "_is_worker_ready", return_value=False):
            with pytest.raises(WorkerFailedError) as exc_info:
                scheduler.get_workers("test", timeout=10.0)

            assert exc_info.value.worker_id == "test/0"
            assert exc_info.value.exit_code == 1

    @patch("areal.scheduler.local_scheduler.time.sleep")
    def test_get_workers_gradual_readiness(self, mock_sleep, scheduler, tmp_path):
        """Should wait for all workers to become ready gradually."""
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test_0.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test_1.log")
        )

        scheduler._workers["test"] = [worker1, worker2]

        # Worker 1 ready immediately, worker 2 ready on second check
        ready_calls = [True, False, True, True]
        with patch.object(scheduler, "_is_worker_ready", side_effect=ready_calls):
            workers = scheduler.get_workers("test", timeout=10.0)

            assert len(workers) == 2


class TestWorkerHealthCheck:
    """Test worker health checking functionality."""

    @pytest.mark.parametrize(
        "status_code,expected",
        [
            (200, True),  # Success
            (503, False),  # Service unavailable
            (500, False),  # Internal server error
        ],
    )
    def test_is_worker_ready_http_status(
        self, scheduler, tmp_path, status_code, expected
    ):
        """Should return appropriate result based on HTTP status code."""
        worker_info = create_worker_info(log_file=str(tmp_path / "test.log"))
        mock_response = create_mock_http_response(status_code=status_code)

        with patch.object(scheduler._http_client, "get", return_value=mock_response):
            assert scheduler._is_worker_ready(worker_info) is expected

    def test_is_worker_ready_connection_error(self, scheduler, tmp_path):
        """Should return False when connection to worker fails."""
        worker_info = create_worker_info(log_file=str(tmp_path / "test.log"))

        with patch.object(
            scheduler._http_client,
            "get",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            assert scheduler._is_worker_ready(worker_info) is False

    def test_check_worker_health_all_healthy(self, scheduler, tmp_path):
        """Should pass when all workers are healthy."""
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test_0.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test_1.log")
        )

        scheduler._workers["test"] = [worker1, worker2]

        # Should not raise
        scheduler._check_worker_health("test")

    def test_check_worker_health_worker_failed(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when a worker has failed."""
        log_file = tmp_path / "test_0.log"
        log_file.write_text("Killed by signal\n")

        mock_proc = create_mock_process(is_alive=False, exit_code=137)
        worker = create_worker_info(process=mock_proc, log_file=str(log_file))

        scheduler._workers["test"] = [worker]

        with pytest.raises(WorkerFailedError) as exc_info:
            scheduler._check_worker_health("test")

        assert exc_info.value.worker_id == "test/0"
        assert exc_info.value.exit_code == 137

    def test_check_worker_health_nonexistent_role(self, scheduler):
        """Should silently pass when role doesn't exist."""
        # Should not raise
        scheduler._check_worker_health("nonexistent")


class TestDeleteWorkers:
    """Test worker deletion and cleanup."""

    def test_delete_workers_specific_role(self, scheduler, tmp_path):
        """Should delete workers for specific role."""
        # Create mock workers for multiple roles
        worker1 = create_worker_info(
            worker_id="role1/0",
            role="role1",
            ports=["8000"],
            log_file=str(tmp_path / "role1_0.log"),
        )
        worker2 = create_worker_info(
            worker_id="role2/0",
            role="role2",
            ports=["8001"],
            log_file=str(tmp_path / "role2_0.log"),
        )

        scheduler._workers["role1"] = [worker1]
        scheduler._workers["role2"] = [worker2]
        scheduler._allocated_ports = {8000, 8001}

        with patch.object(scheduler, "_terminate_process_tree"):
            scheduler.delete_workers("role1")

        # role1 should be deleted, role2 should remain
        assert "role1" not in scheduler._workers
        assert "role2" in scheduler._workers
        assert 8000 not in scheduler._allocated_ports
        assert 8001 in scheduler._allocated_ports

    def test_delete_workers_all_roles(self, scheduler, tmp_path):
        """Should delete all workers when role is None."""
        worker1 = create_worker_info(
            worker_id="role1/0",
            role="role1",
            ports=["8000"],
            log_file=str(tmp_path / "role1_0.log"),
        )
        worker2 = create_worker_info(
            worker_id="role2/0",
            role="role2",
            ports=["8001"],
            log_file=str(tmp_path / "role2_0.log"),
        )

        scheduler._workers["role1"] = [worker1]
        scheduler._workers["role2"] = [worker2]
        scheduler._allocated_ports = {8000, 8001}

        with patch.object(scheduler, "_terminate_process_tree"):
            scheduler.delete_workers(None)

        # All workers should be deleted
        assert len(scheduler._workers) == 0
        assert len(scheduler._allocated_ports) == 0

    def test_delete_workers_nonexistent_role(self, scheduler):
        """Should log warning and return when role doesn't exist."""
        # Should not raise
        scheduler.delete_workers("nonexistent")

    def test_cleanup_workers_releases_ports(self, scheduler, tmp_path):
        """Should release allocated ports when cleaning up workers."""
        worker = create_worker_info(
            ports=["8000", "8001"], log_file=str(tmp_path / "test.log")
        )
        scheduler._allocated_ports = {8000, 8001, 8002}

        with patch.object(scheduler, "_terminate_process_tree"):
            scheduler._cleanup_workers([worker])

        # Ports 8000 and 8001 should be released
        assert scheduler._allocated_ports == {8002}

    def test_cleanup_workers_handles_errors(self, scheduler, tmp_path):
        """Should continue cleanup even if terminating a process fails."""
        worker1 = create_worker_info(
            worker_id="test/0", ports=["8000"], log_file=str(tmp_path / "test_0.log")
        )
        worker2 = create_worker_info(
            worker_id="test/1", ports=["8001"], log_file=str(tmp_path / "test_1.log")
        )

        # First termination fails, second succeeds
        with patch.object(
            scheduler,
            "_terminate_process_tree",
            side_effect=[Exception("Failed to terminate"), None],
        ):
            # Should not raise, just log error
            scheduler._cleanup_workers([worker1, worker2])


class TestProcessTermination:
    """Test process termination functionality."""

    @patch("areal.scheduler.local_scheduler.psutil.Process")
    @patch("areal.scheduler.local_scheduler.psutil.wait_procs")
    def test_terminate_process_tree_graceful(
        self, mock_wait_procs, mock_process_class, tmp_path
    ):
        """Should gracefully terminate process tree."""
        # Mock parent process
        mock_parent = Mock()
        mock_child1 = Mock()
        mock_child2 = Mock()

        mock_parent.children.return_value = [mock_child1, mock_child2]
        mock_process_class.return_value = mock_parent

        # All processes terminate gracefully
        mock_wait_procs.return_value = ([], [])  # (gone, alive)

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))
        scheduler._terminate_process_tree(1234)

        # Verify termination sequence
        mock_child1.terminate.assert_called_once()
        mock_child2.terminate.assert_called_once()
        mock_parent.terminate.assert_called_once()

        # Should not call kill since all terminated gracefully
        mock_child1.kill.assert_not_called()
        mock_child2.kill.assert_not_called()
        mock_parent.kill.assert_not_called()

    @patch("areal.scheduler.local_scheduler.psutil.Process")
    @patch("areal.scheduler.local_scheduler.psutil.wait_procs")
    def test_terminate_process_tree_force_kill(
        self, mock_wait_procs, mock_process_class, tmp_path
    ):
        """Should force kill processes that don't terminate gracefully."""
        mock_parent = Mock()
        mock_child = Mock()

        mock_parent.children.return_value = [mock_child]

        # Return mock_parent only when called with pid=1234, otherwise raise NoSuchProcess
        # This prevents interference from __del__ cleanup of previous test's schedulers
        def process_side_effect(pid):
            if pid == 1234:
                return mock_parent
            raise psutil.NoSuchProcess(pid)

        mock_process_class.side_effect = process_side_effect

        # Child doesn't terminate gracefully
        mock_wait_procs.return_value = ([], [mock_child])  # (gone, alive)

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))
        scheduler._terminate_process_tree(1234)

        # Verify force kill was called
        mock_child.terminate.assert_called_once()
        mock_child.kill.assert_called_once()

    @patch("areal.scheduler.local_scheduler.psutil.Process")
    def test_terminate_process_tree_no_such_process(self, mock_process_class, tmp_path):
        """Should handle gracefully when process doesn't exist."""
        mock_process_class.side_effect = psutil.NoSuchProcess(1234)

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        # Should not raise
        scheduler._terminate_process_tree(1234)

    @patch("areal.scheduler.local_scheduler.psutil.Process")
    def test_terminate_process_tree_handles_child_no_such_process(
        self, mock_process_class, tmp_path
    ):
        """Should handle when child process disappears during termination."""
        mock_parent = Mock()
        mock_child = Mock()
        mock_child.terminate.side_effect = psutil.NoSuchProcess(1235)

        mock_parent.children.return_value = [mock_child]
        mock_process_class.return_value = mock_parent

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        # Should not raise
        scheduler._terminate_process_tree(1234)


class TestLogFileHandling:
    """Test log file reading and handling."""

    def test_read_log_tail_success(self, tmp_path):
        """Should read last N lines from log file."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        log_file = tmp_path / "test.log"
        log_lines = [f"Line {i}\n" for i in range(100)]
        log_file.write_text("".join(log_lines))

        tail = scheduler._read_log_tail(str(log_file), lines=10)

        # Should contain last 10 lines
        assert "Line 90" in tail
        assert "Line 99" in tail
        assert "Line 89" not in tail

    def test_read_log_tail_file_not_found(self, tmp_path):
        """Should return error message when log file doesn't exist."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        tail = scheduler._read_log_tail("/nonexistent/file.log")

        assert "Could not read log file" in tail

    def test_read_log_tail_fewer_lines_than_requested(self, tmp_path):
        """Should return all lines when file has fewer lines than requested."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        log_file = tmp_path / "test.log"
        log_file.write_text("Line 1\nLine 2\nLine 3\n")

        tail = scheduler._read_log_tail(str(log_file), lines=50)

        assert "Line 1" in tail
        assert "Line 2" in tail
        assert "Line 3" in tail


class TestEngineCreation:
    """Test engine creation on workers."""

    def test_create_engine_success(self, scheduler, tmp_path):
        """Should successfully create engine on worker."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=200,
            json_data={"result": {"status": "initialized", "name": "TestEngine"}},
        )

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            result = asyncio.run(
                scheduler.create_engine(
                    "test/0", "test_engines.DummyEngine", name="TestEngine", param=123
                )
            )

            assert result == {"status": "initialized", "name": "TestEngine"}

    def test_create_engine_worker_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when worker doesn't exist."""
        with pytest.raises(WorkerNotFoundError) as exc_info:
            asyncio.run(
                scheduler.create_engine("nonexistent/0", "test_engines.DummyEngine")
            )

        assert exc_info.value.worker_id == "nonexistent/0"

    def test_create_engine_worker_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when worker process has died."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Worker crashed\n")

        mock_proc = create_mock_process(is_alive=False, exit_code=1)
        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with pytest.raises(WorkerFailedError) as exc_info:
            asyncio.run(scheduler.create_engine("test/0", "test_engines.DummyEngine"))

        assert exc_info.value.worker_id == "test/0"
        assert exc_info.value.exit_code == 1

    def test_create_engine_invalid_engine_type(self, scheduler, tmp_path):
        """Should raise EngineCreationError when engine is not a string."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        with pytest.raises(EngineCreationError) as exc_info:
            asyncio.run(scheduler.create_engine("test/0", 123))  # Invalid type

        assert "Engine must be a string import path" in str(exc_info.value)

    def test_create_engine_import_error(self, scheduler, tmp_path):
        """Should raise EngineImportError when engine import fails."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=400,
            json_data={"detail": "Failed to import 'nonexistent.Engine'"},
        )

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            with pytest.raises(EngineImportError) as exc_info:
                asyncio.run(scheduler.create_engine("test/0", "nonexistent.Engine"))

            assert "nonexistent.Engine" in str(exc_info.value)

    def test_create_engine_initialization_error(self, scheduler, tmp_path):
        """Should raise EngineCreationError when engine initialization fails."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=500,
            json_data={"detail": "Engine initialization failed: out of memory"},
        )

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            with pytest.raises(EngineCreationError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert "out of memory" in str(exc_info.value)
            assert exc_info.value.status_code == 500

    def test_create_engine_connection_error_worker_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when connection fails and worker is dead."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Worker crashed during engine creation\n")

        # First call returns None (alive), second call returns exit code (dead)
        mock_proc = create_mock_process()
        mock_proc.poll.side_effect = [None, 1]
        mock_proc.returncode = 1

        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with patch.object(
            scheduler._http_client,
            "post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(WorkerFailedError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert exc_info.value.worker_id == "test/0"

    def test_create_engine_connection_error_worker_alive(self, scheduler, tmp_path):
        """Should raise RPCConnectionError when connection fails but worker is alive."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        with patch.object(
            scheduler._http_client,
            "post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(RPCConnectionError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert exc_info.value.worker_id == "test/0"
            assert exc_info.value.host == "127.0.0.1"
            assert exc_info.value.port == 8000

    def test_create_engine_timeout(self, scheduler, tmp_path):
        """Should raise EngineCreationError when request times out."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        with patch.object(
            scheduler._http_client,
            "post",
            side_effect=httpx.TimeoutException("Request timeout"),
        ):
            with pytest.raises(EngineCreationError) as exc_info:
                asyncio.run(
                    scheduler.create_engine("test/0", "test_engines.DummyEngine")
                )

            assert "Request timed out" in str(exc_info.value)


class TestEngineMethodCalls:
    """Test calling methods on engines (sync and async)."""

    def test_call_engine_success(self, scheduler, tmp_path):
        """Should successfully call engine method synchronously."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=200, json_data={"result": 42}
        )

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            result = scheduler.call_engine("test/0", "compute", arg1=10, arg2=20)

            assert result == 42

    def test_call_engine_worker_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when worker doesn't exist."""
        with pytest.raises(WorkerNotFoundError):
            scheduler.call_engine("nonexistent/0", "method")

    def test_call_engine_worker_died(self, scheduler, tmp_path):
        """Should raise WorkerFailedError when worker dies before call."""
        log_file = tmp_path / "test.log"
        log_file.write_text("Worker crashed\n")

        mock_proc = create_mock_process(is_alive=False, exit_code=1)
        worker = create_worker_info(process=mock_proc, log_file=str(log_file))
        scheduler._workers["test"] = [worker]

        with pytest.raises(WorkerFailedError):
            scheduler.call_engine("test/0", "method")

    def test_call_engine_method_error(self, scheduler, tmp_path):
        """Should raise EngineCallError when method call returns 400/500."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(
            status_code=400, json_data={"detail": "Method 'nonexistent' not found"}
        )

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            with pytest.raises(EngineCallError) as exc_info:
                scheduler.call_engine("test/0", "nonexistent")

            assert "Method 'nonexistent' not found" in str(exc_info.value)

    @patch("areal.scheduler.local_scheduler.time.sleep")
    def test_call_engine_retry_on_503(self, mock_sleep, scheduler, tmp_path):
        """Should retry on 503 Service Unavailable."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # First call returns 503, second call succeeds
        mock_response_503 = create_mock_http_response(status_code=503)
        mock_response_200 = create_mock_http_response(
            status_code=200, json_data={"result": "success"}
        )

        with patch.object(
            scheduler._http_client,
            "post",
            side_effect=[mock_response_503, mock_response_200],
        ):
            result = scheduler.call_engine("test/0", "method", max_retries=3)

            assert result == "success"
            assert mock_sleep.called

    @patch("areal.scheduler.local_scheduler.time.sleep")
    def test_call_engine_max_retries_exhausted(self, mock_sleep, scheduler, tmp_path):
        """Should raise EngineCallError after max retries."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(status_code=503)

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            with pytest.raises(EngineCallError) as exc_info:
                scheduler.call_engine("test/0", "method", max_retries=3)

            assert "Max retries exceeded" in str(
                exc_info.value
            ) or "Service unavailable" in str(exc_info.value)
            assert exc_info.value.attempt == 3

    @patch("areal.scheduler.local_scheduler.time.sleep")
    def test_call_engine_exponential_backoff(self, mock_sleep, scheduler, tmp_path):
        """Should use exponential backoff for retries."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        mock_response = create_mock_http_response(status_code=503)

        with patch.object(scheduler._http_client, "post", return_value=mock_response):
            try:
                scheduler.call_engine(
                    "test/0", "method", max_retries=3, retry_delay=1.0
                )
            except EngineCallError:
                pass

        # Verify exponential backoff: 1.0, 2.0
        sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        assert sleep_calls[0] == 1.0  # First retry
        assert sleep_calls[1] == 2.0  # Second retry

    def test_async_call_engine_success(self, scheduler, tmp_path):
        """Should successfully call engine method asynchronously."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # Use Mock instead of AsyncMock for the response object
        mock_response = create_mock_http_response(
            status_code=200, json_data={"result": 42}
        )

        # But AsyncMock for post() since it's an async method
        async_mock_post = AsyncMock(return_value=mock_response)
        with patch.object(scheduler._async_http_client, "post", async_mock_post):
            result = asyncio.run(
                scheduler.async_call_engine("test/0", "compute", arg1=10, arg2=20)
            )

            assert result == 42

    def test_async_call_engine_worker_not_found(self, scheduler):
        """Should raise WorkerNotFoundError when worker doesn't exist (async)."""
        with pytest.raises(WorkerNotFoundError):
            asyncio.run(scheduler.async_call_engine("nonexistent/0", "method"))

    def test_async_call_engine_retry_with_backoff(self, scheduler, tmp_path):
        """Should retry with exponential backoff in async mode."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        # First call returns 503, second call succeeds
        # Use Mock (not AsyncMock) for response objects since response.json() is synchronous
        mock_response_503 = create_mock_http_response(status_code=503)
        mock_response_200 = create_mock_http_response(
            status_code=200, json_data={"result": "success"}
        )

        # AsyncMock for post() since it's an async method
        async_mock_post = AsyncMock(side_effect=[mock_response_503, mock_response_200])
        with patch.object(scheduler._async_http_client, "post", async_mock_post):
            with patch("asyncio.sleep") as mock_async_sleep:
                result = asyncio.run(
                    scheduler.async_call_engine("test/0", "method", max_retries=3)
                )

                assert result == "success"
                assert mock_async_sleep.called


class TestFindWorkerById:
    """Test finding workers by ID."""

    def test_find_worker_by_id_success(self, scheduler, tmp_path):
        """Should find worker by ID."""
        worker1 = create_worker_info(
            worker_id="role1/0",
            role="role1",
            ports=["8000"],
            log_file=str(tmp_path / "role1_0.log"),
        )
        worker2 = create_worker_info(
            worker_id="role2/0",
            role="role2",
            ports=["8001"],
            log_file=str(tmp_path / "role2_0.log"),
        )

        scheduler._workers["role1"] = [worker1]
        scheduler._workers["role2"] = [worker2]

        found = scheduler._find_worker_by_id("role2/0")

        assert found is worker2
        assert found.worker.id == "role2/0"

    def test_find_worker_by_id_not_found(self, scheduler, tmp_path):
        """Should return None when worker ID is not found."""
        worker = create_worker_info(
            worker_id="role1/0", role="role1", log_file=str(tmp_path / "role1_0.log")
        )
        scheduler._workers["role1"] = [worker]

        found = scheduler._find_worker_by_id("nonexistent/99")

        assert found is None


class TestSchedulerCleanup:
    """Test scheduler cleanup and destructor."""

    def test_destructor_deletes_all_workers(self, scheduler, tmp_path):
        """Should delete all workers when scheduler is destroyed."""
        worker = create_worker_info(log_file=str(tmp_path / "test.log"))
        scheduler._workers["test"] = [worker]

        with patch.object(scheduler, "delete_workers") as mock_delete:
            scheduler.__del__()

            mock_delete.assert_called_once()

    def test_destructor_closes_http_clients(self, scheduler):
        """Should close HTTP clients when scheduler is destroyed."""
        with patch.object(scheduler._http_client, "close") as mock_close:
            scheduler.__del__()

            mock_close.assert_called_once()

    def test_destructor_handles_errors_gracefully(self, scheduler):
        """Should handle errors gracefully in destructor."""
        with patch.object(scheduler, "delete_workers", side_effect=Exception("Error")):
            # Should not raise
            scheduler.__del__()


class TestEdgeCases:
    """Test various edge cases and corner scenarios."""

    def test_gpu_counter_wraps_correctly(self, tmp_path):
        """Should correctly wrap GPU counter for round-robin allocation."""
        scheduler = LocalScheduler(gpu_devices=[0, 1], log_dir=str(tmp_path))

        # Allocate many times to ensure wrapping
        for i in range(10):
            gpus = scheduler._allocate_gpus(1)
            expected_gpu = i % 2
            assert gpus == [expected_gpu]

    def test_port_allocation_accumulates_correctly(self, tmp_path):
        """Should correctly accumulate allocated ports over multiple allocations."""
        with patch(
            "areal.scheduler.local_scheduler.find_free_ports"
        ) as mock_find_ports:
            mock_find_ports.side_effect = [
                [8000, 8001],
                [8002, 8003],
                [8004, 8005, 8006],
            ]

            scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

            scheduler._allocate_ports(2)
            scheduler._allocate_ports(2)
            scheduler._allocate_ports(3)

            assert scheduler._allocated_ports == {
                8000,
                8001,
                8002,
                8003,
                8004,
                8005,
                8006,
            }

    @patch("areal.scheduler.local_scheduler.gethostip")
    @patch("areal.scheduler.local_scheduler.subprocess.Popen")
    @patch("areal.scheduler.local_scheduler.find_free_ports")
    def test_worker_id_format(
        self, mock_find_ports, mock_popen, mock_gethostip, tmp_path
    ):
        """Should create worker IDs in correct format (role/index)."""
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = [8000, 8001]

        mock_processes = []
        for i in range(5):
            mock_proc = Mock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = None
            mock_processes.append(mock_proc)
        mock_popen.side_effect = mock_processes

        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        config = SchedulingConfig(replicas=5, role="worker")
        worker_ids = scheduler.create_workers("worker", config)

        assert worker_ids == [
            "worker/0",
            "worker/1",
            "worker/2",
            "worker/3",
            "worker/4",
        ]

    def test_empty_workers_dict_operations(self, tmp_path):
        """Should handle operations on empty workers dictionary gracefully."""
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(tmp_path))

        # Delete all workers when none exist
        scheduler.delete_workers(None)

        # Check health of non-existent role
        scheduler._check_worker_health("nonexistent")

        # Find worker by ID when no workers exist
        assert scheduler._find_worker_by_id("any/0") is None

    def test_concurrent_gpu_allocations(self, tmp_path):
        """Should handle concurrent GPU allocations correctly."""
        scheduler = LocalScheduler(gpu_devices=[0, 1, 2], log_dir=str(tmp_path))

        # Simulate multiple workers requesting GPUs simultaneously
        results = []
        for _ in range(6):
            gpus = scheduler._allocate_gpus(1)
            results.append(gpus[0])

        # Should cycle through GPUs in order
        assert results == [0, 1, 2, 0, 1, 2]

    def test_log_directory_with_special_characters(self, tmp_path):
        """Should handle log directory paths with special characters."""
        log_dir = tmp_path / "logs with spaces" / "special-chars_123"
        scheduler = LocalScheduler(gpu_devices=[0], log_dir=str(log_dir))

        assert log_dir.exists()
        assert scheduler.log_dir == log_dir
