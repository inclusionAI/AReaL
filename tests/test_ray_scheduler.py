# SPDX-License-Identifier: Apache-2.0

import gc
import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest
import ray

import areal.infra.scheduler.ray as ray_scheduler
from areal.api import Job, Worker
from areal.api.cli_args import (
    NameResolveConfig,
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
    vLLMConfig,
)
from areal.infra.scheduler.exceptions import (
    WorkerCreationError,
    WorkerNotFoundError,
)
from areal.infra.scheduler.ray import RayScheduler, RayWorkerInfo


def _scheduler(tmp_path, n_gpus_per_node: int = 8) -> RayScheduler:
    scheduler = object.__new__(RayScheduler)
    scheduler._n_gpus_per_node = n_gpus_per_node
    scheduler.experiment_name = "test_ray_scheduler"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)
    scheduler.enable_tms_offload = False
    scheduler.ray_device_resource = "GPU"
    scheduler.device_control_env_var = "CUDA_VISIBLE_DEVICES"
    scheduler.name_resolve_config = NameResolveConfig(
        type="nfs",
        nfs_record_root=str(tmp_path / "name_resolve"),
        etcd3_addr="localhost:2379",
    )
    scheduler.startup_timeout = 30.0
    scheduler.health_check_interval = 0.1
    scheduler.exp_config = None
    scheduler._workers = {}
    scheduler._launchers = {}
    scheduler._placement_groups = {}
    scheduler._multi_node_rollout = None
    scheduler._colocated_roles = {}
    return scheduler


def _worker_info(worker_id: str, role: str = "actor") -> RayWorkerInfo:
    return RayWorkerInfo(
        worker=Worker(
            id=worker_id,
            ip="127.0.0.1",
            worker_ports=["10000"],
            engine_ports=[],
        ),
        role=role,
        task_index=int(worker_id.rsplit("/", 1)[1]),
        launchers=[],
        spec=SchedulingSpec(cpu=1, gpu=1, mem=1),
    )


def _sleep_cmd(seconds: int = 60) -> str:
    return f'{sys.executable} -c "import time; time.sleep({seconds})"'


@pytest.fixture(autouse=True)
def _disable_scheduler_destructor(monkeypatch):
    monkeypatch.setattr(RayScheduler, "__del__", lambda self: None)
    yield
    gc.collect()


@pytest.fixture(scope="module")
def local_ray_cluster():
    ray.shutdown()
    try:
        ray.init(
            address="local",
            num_cpus=16,
            num_gpus=8,
            include_dashboard=False,
            ignore_reinit_error=True,
        )
    except Exception as e:
        pytest.skip(f"Unable to start local Ray cluster: {e}")
    yield
    ray.shutdown()


def test_ray_scheduler_rejects_cpu_only_cluster(monkeypatch):
    """Test that RayScheduler fails clearly on CPU-only clusters."""
    monkeypatch.setattr(ray_scheduler.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_scheduler, "ray_resource_type", lambda: "CPU")

    with pytest.raises(RuntimeError, match="does not support CPU-only clusters"):
        RayScheduler(experiment_name="test-exp", trial_name="test-trial")


def test_prepare_worker_specs_expands_single_spec(tmp_path):
    """Test that a single scheduling spec expands to all replicas."""
    scheduler = _scheduler(tmp_path)
    spec = SchedulingSpec(cpu=1, gpu=1, mem=1)

    schedulings = scheduler._prepare_worker_specs("actor", 2, [spec])

    assert schedulings == [spec, spec]


def test_empty_scheduling_spec_fails(tmp_path):
    """Test that job with no scheduling specs fails."""
    scheduler = _scheduler(tmp_path)

    with pytest.raises(ValueError, match="No scheduling specs"):
        scheduler.create_workers(Job(role="empty_spec", replicas=2, tasks=[]))


def test_mismatched_spec_count_fails(tmp_path):
    """Test that mismatched spec count fails."""
    scheduler = _scheduler(tmp_path)
    job = Job(
        role="mismatched",
        replicas=3,
        tasks=[
            SchedulingSpec(cpu=1, gpu=1, mem=1),
            SchedulingSpec(cpu=2, gpu=1, mem=2),
        ],
    )

    with pytest.raises(ValueError, match="must be 1 or match"):
        scheduler.create_workers(job)


def test_zero_replicas_fails(tmp_path):
    """Test that zero replicas fails."""
    scheduler = _scheduler(tmp_path)
    job = Job(role="zero_replicas", replicas=0, tasks=[SchedulingSpec()])

    with pytest.raises(WorkerCreationError, match="replicas must be greater than 0"):
        scheduler.create_workers(job)


def test_additional_bash_cmds_fails(tmp_path):
    """Test that Ray rejects Slurm-style additional bash commands."""
    scheduler = _scheduler(tmp_path)
    job = Job(
        role="bash_cmd",
        replicas=1,
        tasks=[SchedulingSpec(additional_bash_cmds=["echo should-not-run"])],
    )

    with pytest.raises(ValueError, match="additional_bash_cmds"):
        scheduler.create_workers(job)


def test_build_node_plan_large_training_keeps_contiguous_node_groups(tmp_path):
    """Test full-node training creates contiguous node-sized bundles."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=8)
    spec = SchedulingSpec(cpu=1, gpu=1, mem=1)

    bundles, plan, nodes_per_worker = scheduler._build_node_plan(replicas=32, spec=spec)

    assert nodes_per_worker == 1
    assert [int(bundle["GPU"]) for bundle in bundles] == [8, 8, 8, 8]
    assert [item["workers"] for item in plan] == [8, 8, 8, 8]


def test_build_node_plan_partial_single_node_role(tmp_path):
    """Test partial-node Ray allocation builds one partial bundle."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=8)
    spec = SchedulingSpec(cpu=2, gpu=1, mem=3)

    bundles, plan, nodes_per_worker = scheduler._build_node_plan(replicas=6, spec=spec)

    assert nodes_per_worker == 1
    assert len(bundles) == 1
    assert int(bundles[0]["GPU"]) == 6
    assert bundles[0]["CPU"] == 12
    assert plan == [
        {"bundle_index": 0, "node_rank": 0, "workers": 6, "gpus_on_node": 6}
    ]


def test_build_node_plan_multi_node_instance_uses_head_and_worker_nodes(tmp_path):
    """Test multi-node workers reserve all nodes but start only the head worker."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=8)
    spec = SchedulingSpec(cpu=16, gpu=16, mem=32)

    bundles, plan, nodes_per_worker = scheduler._build_node_plan(replicas=2, spec=spec)

    assert nodes_per_worker == 2
    assert [int(bundle["GPU"]) for bundle in bundles] == [8, 8, 8, 8]
    assert [
        (item["worker_idx"], item["node_rank"], item["workers"]) for item in plan
    ] == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]


def test_build_node_plan_multi_node_requires_even_cpu_mem_split(tmp_path):
    """Test multi-node workers require CPU and memory to split across nodes."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=8)
    spec = SchedulingSpec(cpu=15, gpu=16, mem=32)

    with pytest.raises(ValueError, match="evenly split CPU and memory"):
        scheduler._build_node_plan(replicas=1, spec=spec)


def test_launcher_worker_spec_defaults_to_rpc_server_and_auto_port():
    """Test launcher worker command defaults to rpc_server with port zero."""
    launcher_cls = (
        ray_scheduler.RayWorkerProcessLauncher.__ray_metadata__.modified_class
    )
    launcher = object.__new__(launcher_cls)
    launcher.env_vars = {"EXTRA_ENV": "1"}
    launcher.device_control_env_var = "CUDA_VISIBLE_DEVICES"

    worker = launcher._build_worker_spec(
        role="actor",
        worker_index=0,
        gpu_devices=["2", "3"],
        cmd=None,
        experiment_name="exp",
        trial_name="trial",
        name_resolve_type="nfs",
        nfs_record_root="/tmp/name_resolve",
        etcd3_addr="localhost:2379",
        fileroot=None,
    )

    assert worker["worker_id"] == "actor/0"
    assert worker["cmd"][:3] == [
        sys.executable,
        "-m",
        "areal.infra.rpc.rpc_server",
    ]
    assert "--port" in worker["cmd"]
    assert worker["cmd"][worker["cmd"].index("--port") + 1] == "0"
    assert worker["env"]["EXTRA_ENV"] == "1"
    assert worker["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"


@pytest.mark.parametrize(
    "cmd",
    [
        "python -m areal.infra.rpc.rpc_server --port 1234",
        "python -m areal.infra.rpc.rpc_server --port=1234",
    ],
)
def test_launcher_worker_spec_rejects_custom_port_argument(cmd):
    """Test that custom commands cannot override scheduler-managed ports."""
    launcher_cls = (
        ray_scheduler.RayWorkerProcessLauncher.__ray_metadata__.modified_class
    )
    launcher = object.__new__(launcher_cls)
    launcher.env_vars = {}
    launcher.device_control_env_var = "CUDA_VISIBLE_DEVICES"

    with pytest.raises(RuntimeError, match="should not include --port"):
        launcher._build_worker_spec(
            role="actor",
            worker_index=0,
            gpu_devices=["0"],
            cmd=cmd,
            experiment_name="exp",
            trial_name="trial",
            name_resolve_type="nfs",
            nfs_record_root="/tmp/name_resolve",
            etcd3_addr="localhost:2379",
            fileroot=None,
        )


def test_launcher_actor_starts_and_stops_worker_processes(tmp_path, local_ray_cluster):
    """Test RayWorkerProcessLauncher as a real Ray actor."""
    launcher = ray_scheduler.RayWorkerProcessLauncher.options(
        num_cpus=1,
        num_gpus=2,
    ).remote(
        "actor",
        str(tmp_path / "actor.log"),
        str(tmp_path / "merged.log"),
        "GPU",
        "CUDA_VISIBLE_DEVICES",
        {},
    )
    node_info = ray.get(launcher.get_node_info.remote(), timeout=30)
    assert len(node_info["visible_devices"]) == 2

    ray.get(
        launcher.start_workers.remote(
            [
                dict(
                    role="actor",
                    worker_index=0,
                    gpu_devices=node_info["visible_devices"][:1],
                    cmd=_sleep_cmd(),
                    experiment_name="test_ray_scheduler",
                    trial_name="trial",
                    name_resolve_type="nfs",
                    nfs_record_root=str(tmp_path / "name_resolve"),
                    etcd3_addr="localhost:2379",
                    fileroot=str(tmp_path),
                ),
                dict(
                    role="actor",
                    worker_index=1,
                    gpu_devices=node_info["visible_devices"][1:],
                    cmd=_sleep_cmd(),
                    experiment_name="test_ray_scheduler",
                    trial_name="trial",
                    name_resolve_type="nfs",
                    nfs_record_root=str(tmp_path / "name_resolve"),
                    etcd3_addr="localhost:2379",
                    fileroot=str(tmp_path),
                ),
            ]
        ),
        timeout=30,
    )
    statuses = ray.get(
        launcher.worker_statuses.remote(["actor/0", "actor/1"]), timeout=30
    )
    assert statuses == {
        "actor/0": {"exists": True, "returncode": None},
        "actor/1": {"exists": True, "returncode": None},
    }

    ray.get(launcher.stop_all_processes.remote(), timeout=30)
    stopped = ray.get(
        launcher.worker_statuses.remote(["actor/0", "actor/1"]), timeout=30
    )
    assert stopped == {
        "actor/0": {"exists": False, "returncode": None},
        "actor/1": {"exists": False, "returncode": None},
    }
    ray.kill(launcher, no_restart=True)


def test_backend_worker_launch_env_overrides_launcher_env(monkeypatch, tmp_path):
    """Test backend launch uses role env while retaining node-local devices."""
    captured = {}

    class FakeVLLMBackend:
        @staticmethod
        def build_server_env(base_env):
            return base_env.copy()

    fake_vllm_remote = types.ModuleType("areal.engine.vllm_remote")
    fake_vllm_remote.VLLMBackend = FakeVLLMBackend
    monkeypatch.setitem(sys.modules, "areal.engine.vllm_remote", fake_vllm_remote)
    monkeypatch.setattr(
        vLLMConfig,
        "build_cmd_from_args",
        staticmethod(lambda _: ["fake-vllm-server"]),
    )

    class FakeProcess:
        pid = 1234
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_run_with_streaming_logs(cmd, log_file, merged_log, role, *, env):
        captured.update(cmd=cmd, env=env)
        return FakeProcess()

    monkeypatch.setattr(
        ray_scheduler, "run_with_streaming_logs", fake_run_with_streaming_logs
    )
    monkeypatch.setattr(ray_scheduler.time, "sleep", lambda _: None)

    launcher_env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ACTOR_ONLY": "1",
    }
    manager = ray_scheduler.RayBackendWorkerProcessManager(
        role="actor",
        log_file=str(tmp_path / "actor.log"),
        merged_log=str(tmp_path / "merged.log"),
        device_control_env_var="ASCEND_RT_VISIBLE_DEVICES",
        env_vars=launcher_env,
        visible_devices=["4", "5"],
        host="worker-node",
    )

    manager.start_backend_worker(
        group_key=("rollout/0", "rollout/0"),
        backend="vllm",
        server_args={"node_rank": 1},
        env_overrides={
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            "ROLLOUT_ONLY": "1",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        },
    )

    assert captured["cmd"] == ["fake-vllm-server"]
    assert captured["env"]["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert captured["env"]["ACTOR_ONLY"] == "1"
    assert captured["env"]["ROLLOUT_ONLY"] == "1"
    assert captured["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert launcher_env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.asyncio
async def test_prepare_launch_server_forwards_rollout_env(monkeypatch):
    """Test multi-node backend workers receive the rollout scheduling env."""

    class FakeRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)

            async def complete():
                return {}

            return complete()

    class FakeLauncher:
        def __init__(self):
            self.start_backend_worker = FakeRemoteMethod()

    secondary_launcher = FakeLauncher()
    rollout_env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
        "TORCH_COMPILE_DISABLE": "0",
    }
    worker_info = RayWorkerInfo(
        worker=Worker(
            id="rollout/0",
            ip="127.0.0.1",
            worker_ports=["10000"],
            engine_ports=[],
        ),
        role="rollout",
        task_index=0,
        launchers=[object(), secondary_launcher],
        spec=SchedulingSpec(cpu=1, gpu=16, mem=1, env_vars=rollout_env),
    )
    coordinator = object.__new__(ray_scheduler.RayMultiNodeRolloutCoordinator)
    coordinator.startup_timeout = 1.0
    coordinator._rollout_inference_backend = "vllm"

    async def fake_build_multi_node_server_args(worker_info, backend, server_args):
        return ({**server_args, "node_rank": 0}, [{"node_rank": 1}])

    monkeypatch.setattr(
        coordinator,
        "_build_multi_node_server_args",
        fake_build_multi_node_server_args,
    )

    result = await coordinator.prepare_launch_server(
        worker_info,
        engine_name="rollout/0",
        kwargs={"server_args": {"model": "test-model"}},
    )

    assert result["server_args"] == {"model": "test-model", "node_rank": 0}
    assert secondary_launcher.start_backend_worker.calls == [
        (
            ("rollout/0", "rollout/0"),
            "vllm",
            {"node_rank": 1},
            rollout_env,
        )
    ]
    forwarded_env = secondary_launcher.start_backend_worker.calls[0][3]
    assert forwarded_env is not rollout_env


@pytest.fixture
def build_multi_node_args(monkeypatch):
    response = AsyncMock()
    response.raise_for_status = Mock()
    response.json = AsyncMock(return_value={"host": "10.0.0.1", "ports": [12345]})
    response.__aenter__.return_value = response
    response.__aexit__.return_value = None

    session = AsyncMock()
    session.post = Mock(return_value=response)
    session.__aenter__.return_value = session
    session.__aexit__.return_value = None
    monkeypatch.setattr(
        ray_scheduler.aiohttp, "ClientSession", Mock(return_value=session)
    )
    monkeypatch.setattr(ray_scheduler, "get_default_connector", lambda: None)

    coordinator = object.__new__(ray_scheduler.RayMultiNodeRolloutCoordinator)
    coordinator.startup_timeout = 1.0

    async def build(backend, n_nodes, allocated_devices, **overrides):
        worker_info = _worker_info("rollout/0", role="rollout")
        worker_info.launchers = [object() for _ in range(n_nodes)]
        worker_info.gpu_devices = [str(i) for i in range(allocated_devices)]
        server_args = {
            "model": "test-model",
            "host": "127.0.0.1",
            "port": 8000,
            **overrides,
        }
        original_args = server_args.copy()
        head_args, worker_args = await coordinator._build_multi_node_server_args(
            worker_info, backend, server_args
        )
        assert server_args == original_args
        response.json.assert_awaited_once()
        return original_args, head_args, worker_args

    return build


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("n_nodes", "dp_size", "tp_size", "pp_size", "devices"),
    [(4, 32, 2, 1, 64), (2, 4, 2, 2, 16), (2, 2, 8, 1, 16)],
    ids=["multiple-dp-per-node", "node-local-tp-pp", "one-dp-per-node"],
)
async def test_build_vllm_multi_node_args_assigns_node_local_dp(
    build_multi_node_args, n_nodes, dp_size, tp_size, pp_size, devices
):
    """Each node owns disjoint DP ranks and only the head serves HTTP."""
    original, head, workers = await build_multi_node_args(
        "vllm",
        n_nodes,
        devices,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    local_dp = dp_size // n_nodes
    assert len(workers) == n_nodes - 1
    for node_rank, args in enumerate([head, *workers]):
        expected = {
            **original,
            "data_parallel_size_local": local_dp,
            "data_parallel_start_rank": node_rank * local_dp,
            "data_parallel_address": "10.0.0.1",
            "data_parallel_rpc_port": 12345,
        }
        if node_rank:
            expected.pop("host")
            expected.pop("port")
            expected["headless"] = True
        assert args == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("n_nodes", "dp_size", "tp_size", "pp_size", "devices"),
    [(2, 1, 16, 1, 16), (4, 2, 16, 1, 32), (4, 2, 1, 16, 32), (2, 4, 2, 1, 16)],
    ids=["single-dp", "cross-node-tp", "cross-node-pp", "allocation-mismatch"],
)
async def test_build_vllm_multi_node_args_preserves_cross_node_launch(
    build_multi_node_args, n_nodes, dp_size, tp_size, pp_size, devices
):
    """Do not use node-local DP unless replicas fit the allocated node layout."""
    original, head, workers = await build_multi_node_args(
        "vllm",
        n_nodes,
        devices,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    assert len(workers) == n_nodes - 1
    for node_rank, args in enumerate([head, *workers]):
        expected = {
            **original,
            "nnodes": n_nodes,
            "node_rank": node_rank,
            "master_addr": "10.0.0.1",
            "master_port": "12345",
        }
        if node_rank:
            expected.pop("host")
            expected.pop("port")
            expected["headless"] = True
        assert args == expected


@pytest.mark.asyncio
async def test_build_sglang_multi_node_args_preserves_rendezvous(build_multi_node_args):
    """The vLLM-specific DP change must not alter SGLang's launch arguments."""
    original, head, workers = await build_multi_node_args("sglang", 2, 16, tp_size=16)
    common = {"nnodes": 2, "dist_init_addr": "10.0.0.1:12345"}
    assert head == {**original, **common, "node_rank": 0}
    original.pop("host")
    original.pop("port")
    assert workers == [{**original, **common, "node_rank": 1}]


@pytest.mark.asyncio
async def test_prepare_single_node_server_bypasses_multi_node_coordinator(monkeypatch):
    """Single-node launches must retain their original arguments."""
    worker_info = _worker_info("rollout/0", role="rollout")
    worker_info.launchers = [object()]
    coordinator = object.__new__(ray_scheduler.RayMultiNodeRolloutCoordinator)
    build = AsyncMock()
    monkeypatch.setattr(coordinator, "_build_multi_node_server_args", build)
    kwargs = {"server_args": {"model": "test-model"}}
    assert (
        await coordinator.prepare_launch_server(worker_info, "rollout/0", kwargs)
        is kwargs
    )
    build.assert_not_awaited()


@pytest.fixture
def vllm_engine_config():
    pytest.importorskip("vllm")
    from vllm.engine.arg_utils import EngineArgs

    def create(args):
        engine_args = {
            k: v for k, v in args.items() if k not in {"host", "port", "headless"}
        }
        return EngineArgs(
            **engine_args,
            skip_tokenizer_init=True,
            dtype="float16",
            enforce_eager=True,
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            distributed_executor_backend="mp",
        ).create_engine_config(headless=args.get("headless", False))

    return create


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dp_size", "tp_size", "pp_size", "n_nodes"), [(32, 2, 1, 4), (4, 2, 2, 2)]
)
async def test_node_local_dp_remains_single_node_after_dense_engine_reset(
    build_multi_node_args,
    vllm_engine_config,
    tmp_path,
    dp_size,
    tp_size,
    pp_size,
    n_nodes,
):
    """Dense EngineCores must not reinterpret global nodes as TP/PP nodes."""
    from transformers import LlamaConfig

    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=32,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)
    _, head, workers = await build_multi_node_args(
        "vllm",
        n_nodes,
        dp_size * tp_size * pp_size,
        model=str(tmp_path),
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    for args in (head, workers[0]):
        parallel = vllm_engine_config(args).parallel_config
        assert parallel.data_parallel_rank == args["data_parallel_start_rank"]
        assert parallel.data_parallel_size_local == dp_size // n_nodes
        assert parallel.data_parallel_hybrid_lb is False

        # Match the three assignments in vLLM's dense EngineCore startup.
        parallel.data_parallel_size = 1
        parallel.data_parallel_size_local = 1
        parallel.data_parallel_rank = 0
        assert parallel.nnodes_within_dp == 1
        assert parallel.local_world_size == tp_size * pp_size


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_node_local_moe_ep_preserves_attention_dp(
    build_multi_node_args, vllm_engine_config, tmp_path
):
    """Wide EP retains attention DP and internal load balancing on every node."""
    from transformers import MixtralConfig

    MixtralConfig(
        architectures=["MixtralForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_local_experts=64,
        num_experts_per_tok=2,
        vocab_size=32,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)
    _, head, workers = await build_multi_node_args(
        "vllm",
        4,
        64,
        model=str(tmp_path),
        data_parallel_size=32,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
    )
    for node_rank, args in enumerate([head, *workers]):
        config = vllm_engine_config(args)
        assert config.model_config.is_moe
        assert config.model_config.get_num_experts() == 64
        parallel = config.parallel_config
        assert parallel.enable_expert_parallel is True
        assert parallel.data_parallel_size == 32
        assert parallel.data_parallel_size_local == 8
        assert parallel.data_parallel_rank == node_rank * 8
        assert parallel.data_parallel_index == node_rank * 8
        assert parallel.tensor_parallel_size == 2
        assert parallel.pipeline_parallel_size == 1
        assert parallel.nnodes_within_dp == 1
        assert parallel.local_world_size == 2
        assert parallel.data_parallel_hybrid_lb is False
        assert parallel.data_parallel_external_lb is False


def test_create_workers_uses_real_ray_launchers_and_tracks_state(
    tmp_path, local_ray_cluster
):
    """Test worker creation records real Ray launchers and worker metadata."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=8)
    scheduler._destroy_engines_on_workers = lambda workers: None

    job = Job(
        role="actor",
        replicas=2,
        tasks=[SchedulingSpec(cpu=2, gpu=1, mem=1, cmd=_sleep_cmd())],
    )

    try:
        worker_ids = scheduler.create_workers(job)
        assert worker_ids == ["actor/0", "actor/1"]
        assert len(scheduler._launchers["actor"]) == 1
        assert [
            worker_info.worker.id for worker_info in scheduler._workers["actor"]
        ] == [
            "actor/0",
            "actor/1",
        ]

        launcher = scheduler._launchers["actor"][0]
        statuses = ray.get(launcher.worker_statuses.remote(worker_ids), timeout=30)
        assert statuses == {
            "actor/0": {"exists": True, "returncode": None},
            "actor/1": {"exists": True, "returncode": None},
        }
    finally:
        if "actor" in scheduler._workers:
            scheduler.delete_workers("actor")

    assert "actor" not in scheduler._workers
    assert "actor" not in scheduler._launchers
    assert "actor" not in scheduler._placement_groups


def test_create_workers_multi_node_only_starts_head_launcher(
    tmp_path, local_ray_cluster
):
    """Test multi-node worker creation starts only the head launcher process."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=4)
    scheduler._destroy_engines_on_workers = lambda workers: None

    job = Job(
        role="rollout",
        replicas=1,
        tasks=[SchedulingSpec(cpu=8, gpu=8, mem=2, cmd=_sleep_cmd())],
    )

    try:
        worker_ids = scheduler.create_workers(job)
        assert worker_ids == ["rollout/0"]
        worker_info = scheduler._workers["rollout"][0]
        assert len(worker_info.launchers) == 2

        head_status = ray.get(
            worker_info.launchers[0].worker_statuses.remote(["rollout/0"]),
            timeout=30,
        )
        non_head_status = ray.get(
            worker_info.launchers[1].worker_statuses.remote(["rollout/0"]),
            timeout=30,
        )
        assert head_status == {"rollout/0": {"exists": True, "returncode": None}}
        assert non_head_status == {"rollout/0": {"exists": False, "returncode": None}}
    finally:
        if "rollout" in scheduler._workers:
            scheduler.delete_workers("rollout")

    assert "rollout" not in scheduler._workers
    assert "rollout" not in scheduler._launchers
    assert "rollout" not in scheduler._placement_groups


def test_delete_all_workers_removes_multiple_roles(tmp_path, local_ray_cluster):
    """Test deleting all Ray workers clears every owned role."""
    scheduler = _scheduler(tmp_path, n_gpus_per_node=8)
    scheduler._destroy_engines_on_workers = lambda workers: None

    scheduler.create_workers(
        Job(
            role="role1",
            replicas=1,
            tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1, cmd=_sleep_cmd())],
        )
    )
    scheduler.create_workers(
        Job(
            role="role2",
            replicas=1,
            tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1, cmd=_sleep_cmd())],
        )
    )

    scheduler.delete_workers()

    assert scheduler._workers == {}
    assert scheduler._launchers == {}
    assert scheduler._placement_groups == {}


def test_create_workers_with_non_fork_colocation_reuses_target_workers(tmp_path):
    """Test non-fork colocation reuses existing target workers."""
    scheduler = _scheduler(tmp_path)
    scheduler._workers["actor"] = [
        _worker_info("actor/0", role="actor"),
        _worker_info("actor/1", role="actor"),
    ]
    job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor", fork=False
        ),
    )

    worker_ids = scheduler.create_workers(job)

    assert worker_ids == ["actor/0", "actor/1"]
    assert "ref" not in scheduler._workers
    assert scheduler._colocated_roles["ref"] == "actor"


def test_create_workers_with_fork_colocation_delegates_to_fork_workers(
    tmp_path, monkeypatch
):
    """Test fork colocation delegates worker creation to fork_workers."""
    scheduler = _scheduler(tmp_path)
    scheduler._workers["actor"] = [
        _worker_info("actor/0", role="actor"),
        _worker_info("actor/1", role="actor"),
    ]
    called = []

    def fake_fork_workers(job: Job):
        called.append((job.role, job.scheduling_strategy.target))
        return ["ref/0", "ref/1"]

    monkeypatch.setattr(scheduler, "fork_workers", fake_fork_workers)
    job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor", fork=True
        ),
    )

    worker_ids = scheduler.create_workers(job)

    assert worker_ids == ["ref/0", "ref/1"]
    assert called == [("ref", "actor")]


def test_colocation_replica_mismatch_raises_error(tmp_path):
    """Test that colocation fails if replica count does not match target."""
    scheduler = _scheduler(tmp_path)
    scheduler._workers["actor"] = [_worker_info("actor/0", role="actor")]
    job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor", fork=False
        ),
    )

    with pytest.raises(WorkerCreationError, match="Replica count mismatch"):
        scheduler.create_workers(job)


def test_colocation_target_not_found_raises_error(tmp_path):
    """Test that colocation fails if target role does not exist."""
    scheduler = _scheduler(tmp_path)
    job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)],
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="missing"
        ),
    )

    with pytest.raises(WorkerNotFoundError):
        scheduler.create_workers(job)
