# SPDX-License-Identifier: Apache-2.0

"""Run fresh SWE episodes with data-triggered training and idle GPU release."""

import argparse
import asyncio
import json
import os
import threading
import time
from pathlib import Path


async def run(args):
    from examples.autopilot.elastic.fixtures.groups import create_ppo_group
    from examples.autopilot.elastic.fixtures.inference_service import InferenceService
    from examples.autopilot.elastic.fixtures.weight_publication import WeightPublisher
    from examples.autopilot.elastic.swe.rollout_to_queue import collect_rollout
    from examples.autopilot.elastic.validation.probes import probe_inference

    from areal.autopilot.elastic.loop import ElasticLoop
    from areal.autopilot.elastic.policy import IdleTimeoutPolicy
    from areal.autopilot.elastic.runtime.history import RuntimeHistory
    from areal.autopilot.elastic.runtime.policy import PolicyApplication, PolicyRequest
    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
    from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer

    args.output.mkdir(parents=True, exist_ok=False)
    queue = BatchQueue(args.state_root / "queue")
    group = create_ppo_group(
        root=args.state_root / "training",
        model=args.model,
        revision=args.model_revision,
        devices=[args.inference_device, *args.devices]
        if args.policy == "joint"
        else args.devices,
        elastic=args.policy in {"basic", "joint"},
    )
    if args.policy == "joint":
        from examples.autopilot.elastic.basic_policy import measured_joint_policy

        policy, initial = measured_joint_policy(
            args.profiles,
            compatibility=group.compatibility,
            horizon=args.horizon,
            idle_seconds=args.idle_seconds,
        )
        if (
            args.initial_train_gpus is not None
            and args.initial_train_gpus != initial.actor.world_size
        ):
            raise ValueError(
                "Joint initial Actor comes from the complete profile allowlist"
            )
        group.configure(initial.actor)
    elif args.policy == "basic":
        from examples.autopilot.elastic.basic_policy import measured_policy

        from areal.autopilot.contracts import TrainingConfig

        group.configure(TrainingConfig(args.initial_train_gpus or len(args.devices)))
        policy = measured_policy(
            args.profiles,
            horizon=args.horizon,
            cost_scope=args.cost_scope,
            idle_seconds=args.idle_seconds,
            inference_gpus=1,
            compatibility=group.compatibility,
            revision_root=getattr(args, "revision_root", None),
        )
    else:
        policy = IdleTimeoutPolicy(idle_seconds=args.idle_seconds)
    trainer = QueuedTrainer(group, queue)
    config_application = None
    if args.policy == "joint":
        from areal.autopilot.elastic.runtime.config import (
            ConfigApplication,
            JointConfigExecutor,
        )
        from areal.autopilot.elastic.runtime.rollout import RolloutGroup

        service = RolloutGroup(
            config=initial.rollout,
            model=args.model,
            model_name=args.model_name,
            model_revision=args.model_revision,
            root=args.state_root / "inference",
            lock_root=args.state_root.parent / "gpu-locks",
            service_factory=InferenceService,
            context_length=16384,
        )
        publisher = WeightPublisher(service, state=args.state_root / "published.json")
        config_application = ConfigApplication(
            JointConfigExecutor(trainer, service, publisher),
            args.state_root / "config-application",
        )
    else:
        service = InferenceService(
            model=args.model,
            model_name=args.model_name,
            device=args.inference_device,
            root=args.output / "inference",
            lock_root=args.state_root.parent / "gpu-locks",
        )
        publisher = WeightPublisher(service, state=args.state_root / "published.json")
    application, history, revision = None, None, None
    initial_id = "constructor-policy"
    policy_root = args.state_root / "policy-application"
    if args.policy == "joint":
        from areal.autopilot.revision.contracts import digest

        initial_id = "constructor-joint:" + digest(
            {
                "profiles": json.loads(args.profiles.read_text()),
                "horizon": args.horizon,
                "idle_seconds": args.idle_seconds,
            }
        )
        history = RuntimeHistory(
            args.state_root / "runtime-history.json",
            context={
                "queue_id": queue.queue_id,
                "checkpoint_root": str(group.store.root.resolve()),
                "compatibility": group.compatibility,
                "joint_executor": config_application.executor.context_id,
                "initial_policy": initial_id,
            },
        )
    if (
        getattr(args, "revision_command", None)
        or getattr(args, "policy_bundle", None)
        or (policy_root / "active.json").exists()
    ):
        from dataclasses import asdict

        from areal.autopilot.revision.contracts import digest

        if (
            args.policy not in {"basic", "joint"}
            or getattr(args, "revision_root", None) is not None
        ):
            raise ValueError(
                "Runtime revisions require an explicit basic policy and phase profiles"
            )
        profile = json.loads(args.profiles.read_text())
        settings = asdict(policy.optimizer)
        if args.policy == "joint":
            from areal.autopilot.elastic.runtime.serial_policy import (
                serial_policy_identity,
            )

            execution = {
                "queue_id": queue.queue_id,
                "checkpoint_root": str(group.store.root.resolve()),
                "scope": "serial-joint-timing-v1",
                "training_pool": group.pool_uuids,
                "max_total_gpus": len(group.pool_devices),
                "joint_executor": config_application.executor.context_id,
                "joint_policy_identity": serial_policy_identity(
                    profile,
                    settings,
                    horizon=args.horizon,
                    idle_seconds=args.idle_seconds,
                ),
            }
        else:
            initial_id = "constructor-basic:" + digest(
                {
                    "profile": profile,
                    "settings": settings,
                    "idle_seconds": args.idle_seconds,
                }
            )
            execution = {
                "queue_id": queue.queue_id,
                "checkpoint_root": str(group.store.root.resolve()),
                "inference_gpus": 1,
                "training_pool": group.pool_uuids,
                "inference_gpu": service.gpu_uuid,
            }
        application = PolicyApplication(
            policy_root,
            compatibility=group.compatibility,
            execution_context=execution,
            initial_bundle_id=initial_id,
        )
        if args.policy != "joint":
            history = RuntimeHistory(
                args.state_root / "runtime-history.json",
                context={**execution, "compatibility": group.compatibility},
            )
        if getattr(args, "revision_command", None):
            from areal.autopilot.revision.agent import CommandRevisionAgent
            from areal.autopilot.revision.online import RuntimeRevisionLoop

            agent = CommandRevisionAgent(tuple(args.revision_command))
            revision = RuntimeRevisionLoop(
                args.state_root / "runtime-revisions",
                profile=profile,
                settings=settings,
                model_agent=agent,
                optimizer_agent=agent,
                every_updates=args.revision_every_updates,
                max_windows=args.max_revision_windows,
                max_model_rounds=args.model_revision_rounds,
                max_optimizer_rounds=args.optimizer_revision_rounds,
                idle_seconds=args.idle_seconds,
                scope="serial-joint-timing-v1"
                if args.policy == "joint"
                else "lifecycle",
                horizon=args.horizon,
            )
    loop = ElasticLoop(
        trainer,
        publisher,
        policy=policy,
        journal=args.output / "elastic-loop.json",
        policy_application=application,
        policy_bundle_id=initial_id,
        runtime_history=history,
        config_application=config_application,
    )
    stop = threading.Event()
    errors = []
    thread = None
    report = {"status": "running", "rounds": [], "idle_seconds": args.idle_seconds}

    def serve():
        try:
            loop.run(stop, poll_seconds=0.5)
        except BaseException as error:
            errors.append(error)

    async def wait_until(predicate, timeout=900):
        deadline = time.monotonic() + timeout
        if revision is not None:
            await asyncio.to_thread(revision.tick, loop)
        while not predicate():
            if errors:
                raise RuntimeError("Elastic loop failed") from errors[0]
            if time.monotonic() >= deadline:
                raise TimeoutError("Elastic loop did not reach the expected state")
            if revision is not None:
                await asyncio.to_thread(revision.tick, loop)
            await asyncio.sleep(0.5)
        if errors:
            raise RuntimeError("Elastic loop failed") from errors[0]

    try:
        if args.policy != "joint":
            await asyncio.to_thread(service.start)
        await asyncio.to_thread(loop.start)
        if getattr(args, "policy_bundle", None) is not None:
            prepared = await asyncio.to_thread(loop.prepare_policy, args.policy_bundle)
            request = PolicyRequest(
                "startup-" + prepared.bundle_id[:16],
                prepared.bundle_id,
                application.state()["generation"],
                trainer.version,
            )
            await asyncio.to_thread(loop.apply_policy, request, prepared)
        if trainer.has_pending_batch():
            raise RuntimeError(
                "Validation requires a recovered queue without pending data"
            )
        initial_step = trainer.version
        report["initial_version"] = initial_step
        thread = threading.Thread(target=serve, name="elastic-loop", daemon=False)
        thread.start()
        await wait_until(lambda: publisher.ready and loop.state == "SUSPENDED")
        await asyncio.sleep(2)
        if trainer.is_active or any(e["event"] == "WAKING" for e in loop.events):
            raise RuntimeError("An empty queue allocated training GPUs")
        report["empty_queue_kept_training_off"] = True
        for index in range(args.rounds):
            output = args.output / f"rollout-{index + 1:03d}"
            output.mkdir()
            episode_args = argparse.Namespace(
                **{**vars(args), "queue": queue.root, "output": output}
            )
            with loop.episode() as version:
                started_while_active = trainer.is_active
                episode_task = asyncio.create_task(
                    collect_rollout(episode_args, service, version)
                )
                try:
                    if index and args.policy == "idle":
                        # The next real agent holds admission while idle release
                        # proceeds independently; inference weights stay fixed.
                        await wait_until(
                            lambda: loop.state == "SUSPENDED" or episode_task.done()
                        )
                        if loop.state != "SUSPENDED":
                            raise RuntimeError(
                                "Validation episode ended before idle release; lower the timeout"
                            )
                        report["rounds"][-1].update(
                            inference_after_release=await probe_inference(
                                service, version
                            ),
                            release_overlapped_next_episode=started_while_active
                            and not episode_task.done(),
                        )
                    episode = await episode_task
                finally:
                    if not episode_task.done():
                        episode_task.cancel()
                        try:
                            await episode_task
                        except asyncio.CancelledError:
                            pass
            target = initial_step + index + 1
            await wait_until(lambda: loop.published_version == target)
            counts = queue.status()["counts"]
            if counts != dict(pending=0, leased=0, applied=0, committed=target):
                raise RuntimeError(
                    "Automatic checkpoint did not commit exactly one episode"
                )
            report["rounds"].append(
                {
                    "episode": episode,
                    "published_version": target,
                }
            )
        if args.policy == "idle":
            await wait_until(lambda: loop.state == "SUSPENDED")
        report["rounds"][-1]["inference_after_release"] = await probe_inference(
            service, trainer.version
        )
        if (
            args.policy == "idle"
            and args.rounds > 1
            and not any(
                item.get("release_overlapped_next_episode") for item in report["rounds"]
            )
        ):
            raise RuntimeError("No training release overlapped a live SWE episode")
        wakes = [e for e in loop.events if e["event"] == "WAKING"]
        releases = [e for e in loop.events if e["event"] == "RELEASED"]
        if args.policy == "idle" and (
            len(wakes) != args.rounds or len(releases) != args.rounds
        ):
            raise RuntimeError("Expected one automatic wake/release per episode")
        # Idle observation must not produce another checkpoint or GPU generation.
        before = (len(group.events), trainer.latest_checkpoint)
        await asyncio.sleep(args.idle_seconds + 1)
        if args.policy == "idle" and before != (
            len(group.events),
            trainer.latest_checkpoint,
        ):
            raise RuntimeError("A clean idle group repeated lifecycle work")
        fronts = [event["frontend"] for event in group.events if event.get("frontend")]
        for key in ("services", "processes", "api_key_sha256"):
            if not fronts or any(front[key] != fronts[0][key] for front in fronts):
                raise RuntimeError("Training frontend changed across automatic cycles")
        report.update(
            status="passed",
            final_version=trainer.version,
            queue=queue.status(),
            frontend_stable=True,
            policy=args.policy,
            committed_samples=trainer.committed_samples,
            effective_tokens=trainer.committed_tokens,
            config_state=None
            if config_application is None
            else config_application.state(),
            quality_status="not_evaluated",
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            # to_thread cancellation does not stop its work. Join the actual lifecycle
            # thread before touching workers, frontend, or inference from this caller.
            stop.set()
            if thread is not None and thread.ident is not None:
                await asyncio.to_thread(thread.join)
            if revision is not None:
                await asyncio.to_thread(revision.close, loop)
                report["revision_outcomes"] = [
                    json.loads(p.read_text())
                    for p in sorted(
                        (args.state_root / "runtime-revisions").glob(
                            "window-*/outcome.json"
                        )
                    )
                ]
            report["policy_bundle_id"] = loop.policy_bundle_id
            report["policy_generation"] = loop.policy_generation
        except BaseException as error:
            report.update(status="failed", revision_cleanup_error=str(error))
            raise
        finally:
            try:
                try:
                    try:
                        await asyncio.to_thread(loop.close)
                    finally:
                        await asyncio.to_thread(group.close)
                finally:
                    report["inference_cleanup"] = await asyncio.to_thread(
                        service.stop if args.policy == "joint" else service.close
                    )
            except BaseException as error:
                report.update(status="failed", cleanup_error=str(error))
                raise
            finally:
                report["elastic_loop"] = loop.events
                report["lifecycle"] = group.events
                if errors:
                    report.update(
                        status="failed",
                        loop_error=f"{type(errors[0]).__name__}: {errors[0]}",
                    )
                atomic_json(args.output / "report.json", report)
        if errors:
            raise RuntimeError("Elastic loop failed") from errors[0]


def main():
    import uvloop

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "state-root",
        "output",
        "fixture",
        "agent-python",
        "sandbox-python",
        "kubectl",
        "kubeconfig",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-name", default="Qwen3-0.6B")
    parser.add_argument("--environment", required=True)
    parser.add_argument("--sandbox-namespace", default="aenv-sandbox")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path(__file__).parent / "swe/agent-config.yaml",
    )
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--inference-device", type=int, default=0)
    parser.add_argument("--policy", choices=["idle", "basic", "joint"], default="idle")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--revision-root", type=Path)
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--revision-every-updates", type=int, default=4)
    parser.add_argument("--max-revision-windows", type=int, default=3)
    parser.add_argument("--model-revision-rounds", type=int, default=2)
    parser.add_argument("--optimizer-revision-rounds", type=int, default=2)
    parser.add_argument("--revision-command", nargs=argparse.REMAINDER)
    parser.add_argument("--initial-train-gpus", type=int)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--cost-scope", choices=["total", "training"], default="total")
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.inference_device in args.devices or args.rounds < 1:
        parser.error("Use disjoint GPU pools and at least one round")
    if args.revision_root is not None and args.policy != "basic":
        parser.error("--revision-root requires --policy basic")
    if (args.revision_command or args.policy_bundle) and (
        args.policy not in {"basic", "joint"} or args.revision_root is not None
    ):
        parser.error(
            "Runtime revision/bundle application requires --policy basic/joint and excludes --revision-root"
        )
    if args.policy in {"basic", "joint"} and args.profiles is None:
        parser.error("Basic/joint policy requires --profiles from matching calibration")
    if args.policy == "joint" and args.horizon < 2:
        parser.error("Joint policy needs --horizon >= 2 for a future episode")
    if args.initial_train_gpus is not None and not 1 <= args.initial_train_gpus <= len(
        args.devices
    ):
        parser.error("Initial training count must fit the GPU pool")
    uvloop.run(run(args))


if __name__ == "__main__":
    main()
