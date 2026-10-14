"""Child-side bridge for the pinned AReaL snapshot/restore interfaces.

Only the request listener runs in a thread. Training, restore acknowledgement,
trainer cleanup and the stopped receipt execute in order on the main thread.
"""

import hashlib
import os
import shlex
import threading
from dataclasses import asdict
from pathlib import Path

from autopilot_harness.validator.checker import configuration_id

from .protocol import (
    LAUNCH_ENV,
    ProtocolError,
    check_message,
    message,
    publish,
    read_json,
)


def source_digest(package: Path) -> str:
    """Identify actual Python source, including uncommitted runtime changes."""
    digest = hashlib.sha256()
    files = sorted(package.rglob("*.py"))
    if not files:
        raise ValueError("runtime source files are unavailable")
    for path in files:
        digest.update(str(path.relative_to(package)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def load_managed_configuration(configuration: dict) -> dict:
    """Select the frozen full target before constructing a trainer/datasets.

    Custom entrypoints call this before consuming tunables. run_areal handles
    it automatically. The original entrypoint config is also fingerprinted so
    changing its file/arguments cannot silently change immutable task settings.
    """
    launch = read_json(Path(os.environ[LAUNCH_ENV]))
    target = launch.get("configuration_file")
    if target is not None:
        if configuration_id(configuration) != launch["entry_configuration_id"]:
            raise ProtocolError("entrypoint configuration changed across restart")
        configuration = read_json(Path(target))
    if configuration_id(configuration) != launch["configuration_id"]:
        raise ProtocolError("managed target configuration digest mismatch")
    return configuration


def run_training(
    trainer_factory,
    *,
    configuration: dict,
    runtime_source_id: str,
    train_kwargs: dict | None = None,
    snapshot_api=None,
    supports_config_apply: bool = False,
):
    """Run one managed trainer instance and service cooperative stop requests.

    trainer_factory(restore=request_or_none) must return a fresh context-managed
    PPOTrainer. snapshot_api is injectable for CPU protocol tests; normal use
    imports the installed AReaL implementation. Arguments/settings are trusted
    application input, not Planner-generated executable code.
    """
    path = Path(os.environ[LAUNCH_ENV]).resolve(strict=True)
    launch = read_json(path)
    directory = path.parent
    if (
        launch.get("schema_version") != 1
        or directory.name != launch.get("segment_id")
        or not launch.get("nonce")
    ):
        raise ProtocolError("invalid launch identity")
    errors = []
    stopping = threading.Event()
    listener = None
    request = []
    stage = "initialization"
    source = launch.get("restore")

    def emit(kind, **payload):
        publish(directory / (kind + ".json"), message(launch, kind, **payload))

    def failure(error, where):
        if not (directory / "failed.json").exists():
            emit("failed", stage=where, error_type=type(error).__name__)

    def restored_source(trainer):
        if source is None:
            return None
        result = trainer.snapshot_restore_result
        restored = {
            "snapshot": asdict(result.snapshot),
            "context": asdict(result.context),
        }
        if restored != source:
            raise ProtocolError("runtime restore acknowledgement mismatch")
        if (
            transition is not None
            and trainer._snapshot_restore.transition != restore.transition
        ):
            raise ProtocolError("runtime did not apply the planned state mapping")
        return restored

    try:
        if configuration_id(configuration) != launch["configuration_id"]:
            raise ProtocolError(
                "resolved configuration differs from managed configuration"
            )
        if not isinstance(runtime_source_id, str) or not runtime_source_id:
            raise ProtocolError("missing runtime source identity")
        if launch.get("runtime_source_id") not in (None, runtime_source_id):
            raise ProtocolError("runtime source changed across restart")
        if snapshot_api is None:
            from areal.infra import reconfiguration as snapshot_api
        api = snapshot_api
        adapter = (
            getattr(api, "CONFIGURATION_APPLY_ADAPTER", None)
            if supports_config_apply
            else None
        )
        restore = None
        transition = launch.get("transition")
        if transition is not None and (
            source is None or adapter != transition.get("adapter_id")
        ):
            raise ProtocolError(
                "configuration transition requires the implemented restore adapter"
            )
        if source is not None:
            mapping = None
            if transition is not None:
                if (
                    transition["plan_id"] != launch.get("plan_id")
                    or transition["source_configuration_id"]
                    != source["context"]["configuration_id"]
                    or transition["target_configuration_id"]
                    != launch["configuration_id"]
                ):
                    raise ProtocolError("transition configuration binding mismatch")
                mapping = api.ConfigurationTransition(**transition)
            elif source["context"]["configuration_id"] != launch["configuration_id"]:
                raise ProtocolError(
                    "changed configuration requires an explicit transition"
                )
            restore = api.SnapshotRestoreRequest(
                api.SnapshotRef(**source["snapshot"]),
                api.SnapshotContext(**source["context"]),
                **({"transition": mapping} if mapping is not None else {}),
            )
        control = api.TrainerReconfiguration(
            run_id=launch["run_id"],
            runtime_revision=launch["runtime_revision"],
            configuration_id=launch["configuration_id"],
        )
        kwargs = dict(train_kwargs or {})
        if "reconfiguration" in kwargs:
            raise ValueError("reconfiguration channel is owned by the bridge")

        def listen():
            try:
                while not stopping.is_set():
                    pending = directory / "stop_request.json"
                    if pending.exists():
                        value = check_message(
                            read_json(pending), launch, "stop_request"
                        )
                        expected = (
                            directory.parents[1] / "snapshots" / value["request_id"]
                        )
                        if (
                            value.get("state_id") != value.get("request_id")
                            or len(value["request_id"]) != 32
                            or any(
                                c not in "0123456789abcdef" for c in value["request_id"]
                            )
                            or Path(value["directory"]).resolve() != expected
                        ):
                            raise ProtocolError("invalid stop request binding")
                        future = control.request_snapshot(
                            value["directory"],
                            state_id=value["state_id"],
                            stop_after_save=True,
                        )
                        request.append((value, future))
                        return
                    stopping.wait(0.02)
            except BaseException as error:
                errors.append(error)
                failure(error, "request")

        with trainer_factory(restore=restore) as trainer:
            if source is not None:
                # This is the only private AReaL coupling, tested against the
                # pinned implementation. A synchronous hook cannot miss a very
                # short resumed run or acknowledge before admission opens.
                controller = trainer._snapshot_restore
                original_resume = controller.resume

                def resume(runtime):
                    original_resume(runtime)
                    if not controller.released:
                        raise ProtocolError(
                            "runtime has not released restored rollouts"
                        )
                    emit(
                        "ready",
                        status="resumed",
                        restored=restored_source(runtime),
                        runtime_source_id=runtime_source_id,
                        apply_adapter=adapter,
                    )

                controller.resume = resume
            else:
                emit(
                    "ready",
                    status="initialized",
                    runtime_source_id=runtime_source_id,
                    apply_adapter=adapter,
                )
            listener = threading.Thread(
                target=listen, name="AutopilotRequests", daemon=True
            )
            listener.start()
            stage = "training"
            result = trainer.train(reconfiguration=control, **kwargs)
            # Stop accepting requests before cleanup. Late/unlatched requests
            # cannot be acknowledged as a successful snapshot.
            stopping.set()
            listener.join()
            if errors:
                raise errors[0]
            stage = "cleanup"
        # __exit__/close has returned. Parent still verifies leader AND group exit.
        if isinstance(result, api.SnapshotStopResult):
            if not request or request[0][1].result() != result.snapshot:
                raise ProtocolError("training stopped without the requested snapshot")
            emit(
                "stopped",
                request_id=request[0][0]["request_id"],
                cleanup_completed=True,
                snapshot=asdict(result.snapshot),
                context=asdict(result.context),
            )
        else:
            if request:
                raise ProtocolError(
                    "training finished before the requested snapshot boundary"
                )
            emit(
                "completed",
                status="completed",
                cleanup_completed=True,
                restored=restored_source(trainer),
                runtime_source_id=runtime_source_id,
                apply_adapter=adapter,
            )
    except BaseException as error:
        failure(error, stage)
        raise
    finally:
        stopping.set()
        if listener is not None:
            listener.join()


def _check_local_workers(config, *, single_controller, remote_dataset):
    """Check the containment assumptions of this process-group backend."""
    if config.scheduler.type != "local" or not single_controller:
        raise NotImplementedError(
            "managed process-group restart requires v1 LocalScheduler"
        )

    def check_command(cmd, module):
        command = shlex.split(cmd)
        if (
            len(command) != 3
            or not Path(command[0]).name.startswith("python")
            or command[1:] != ["-m", module]
        ):
            raise NotImplementedError(
                "process-group backend requires standard local worker commands"
            )

    roles = [
        getattr(config, role, None) for role in ("actor", "critic", "ref", "rollout")
    ]
    if config.teacher is not None:
        if config.teacher.engine_type not in {"train", "rollout"}:
            raise ValueError("unknown teacher engine type")
        teacher = getattr(config.teacher, config.teacher.engine_type)
        if teacher is None:
            raise ValueError("selected teacher engine configuration is missing")
        roles.append(teacher)
    for settings in roles:
        if settings is None:
            continue
        if settings._version != "v1":
            raise NotImplementedError("all managed roles must use v1")
        for spec in settings.scheduling_spec:
            check_command(spec.cmd, "areal.infra.rpc.rpc_server")
    if remote_dataset:
        check_command(config.train_dataset.scheduling_spec.cmd, "areal.infra.rpc.guard")
    if getattr(config.rollout.agent, "mode", None) == "online":
        raise NotImplementedError(
            "external online sessions need a coordinated admission/exit backend"
        )


def run_areal(config, *, train_dataset=None, valid_dataset=None, **train_kwargs):
    """Drop-in managed training call for standard v1 LocalScheduler entrypoints.

    The caller constructs the resolved config/datasets as usual. This adapter
    rejects remote schedulers/v2: their workers require a different exit backend.
    Custom commands that detach or contact external runtimes are not supported.
    """
    from omegaconf import OmegaConf

    import areal
    from areal import PPOTrainer
    from areal.api.cli_args import to_structured_cfg
    from areal.infra.data_service.rdataset import RDataset
    from areal.utils.environ import is_single_controller

    resolved = load_managed_configuration(asdict(config))
    config = OmegaConf.to_object(
        to_structured_cfg(OmegaConf.create(resolved), type(config))
    )
    if configuration_id(asdict(config)) != configuration_id(resolved):
        raise ProtocolError("native config conversion changed the validated target")

    _check_local_workers(
        config,
        single_controller=is_single_controller(),
        remote_dataset=isinstance(train_dataset, RDataset),
    )
    return run_training(
        lambda restore: PPOTrainer(
            config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            restore=restore,
        ),
        configuration=asdict(config),
        runtime_source_id=source_digest(Path(areal.__file__).parent),
        train_kwargs=train_kwargs,
        supports_config_apply=True,
    )
