# SPDX-License-Identifier: Apache-2.0

"""Calibrate all pool sizes and run basic policy on real FSDP/PPO workers.

Synthetic on-policy-shaped data checks scheduling and checkpoint recovery. Joint
application mode also tests real SGLang generation. This is not a quality benchmark.
"""

import argparse
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path


class CheckpointPublisher:
    """Diagnostic publication receipt; no inference service in this experiment."""

    def __init__(self):
        self.lock = threading.RLock()
        self.ready = False
        self.version = 0

    def initialize(self):
        self.ready = True

    @contextmanager
    def episode(self):
        with self.lock:
            if not self.ready:
                raise RuntimeError("Publication is not ready")
            yield self.version

    def publish(self, checkpoint):
        self.version = json.loads((checkpoint / "manifest.json").read_text())[
            "global_step"
        ]
        self.ready = True
        return {"version": self.version, "scope": "diagnostic-receipt"}


def payload(version, revision):
    from examples.autopilot.elastic.swe.trajectory import trajectory_payload

    length = 64
    trajectory = {
        "input_ids": [[10 + (index + version) % 100 for index in range(length)]],
        "attention_mask": [[1] * length],
        "loss_mask": [[0] * 32 + [1] * 32],
        "logprobs": [[-5.5] * length],
        "versions": [[version] * length],
        "rewards": [1.0],
    }
    return trajectory_payload(
        trajectory,
        task_id="synthetic",
        rollout_id=f"synthetic-{version}",
        model_revision=revision,
        policy_version=version,
    )


def require_target(trainer, world_size):
    from areal.autopilot.contracts import TrainingConfig

    if (
        trainer.last_reconfiguration_error is not None
        or trainer.training_config != TrainingConfig(world_size)
    ):
        raise RuntimeError(
            f"Requested {world_size} GPUs were not successfully activated"
        )


def verify_config_application(loop, report, verify_restore, source, expected, before):
    """Exercise real workers, duplicate requests and interrupted receipt recovery."""
    import areal.autopilot.elastic.runtime.config as module
    from areal.autopilot.contracts import TrainingConfig
    from areal.autopilot.elastic.runtime.config import ConfigRequest

    application = loop.config_application
    executor = application.executor
    initial = executor.observe()
    receipts = []

    def request(identity, target):
        return ConfigRequest(
            identity,
            target,
            application.state()["generation"],
            loop.trainer.version,
            "measured-basic-validation",
        )

    for index, count in enumerate((1, 2, 1, len(loop.trainer.group.pool_devices))):
        target = replace(executor.observe().config, actor=TrainingConfig(count))
        change = request(f"validation-{index}", target)
        receipt = loop.apply_config(change)
        if receipt["status"] != "applied":
            raise AssertionError("Configuration target was not activated")
        if executor.observe() != replace(initial, config=target):
            raise AssertionError("Application changed checkpoint/cursor/progress")
        verify_restore(source, expected, before)
        events = len(loop.trainer.group.events)
        if (
            loop.apply_config(change) != receipt
            or len(loop.trainer.group.events) != events
        ):
            raise AssertionError("Duplicate request replayed a transition")
        receipts.append(receipt)

    current = executor.observe()
    unsupported = replace(
        current.config, rollout=replace(current.config.rollout, replicas=2)
    )
    try:
        loop.apply_config(request("unsupported-rollout", unsupported))
    except ValueError:
        if not loop.publisher.ready or executor.observe() != current:
            raise AssertionError("Unsupported target changed admission or execution")
    else:
        raise AssertionError("Actor adapter accepted a Rollout change")

    change = request(
        "interrupted-receipt", replace(current.config, actor=TrainingConfig(2))
    )
    write = module.atomic_json
    failed = False

    def interrupt(path, state):
        nonlocal failed
        if (
            not failed
            and state["pending"] is None
            and change.request_id in state["receipts"]
        ):
            failed = True
            raise OSError("Injected receipt write failure after activation")
        write(path, state)

    module.atomic_json = interrupt
    try:
        try:
            loop.apply_config(change)
        except OSError:
            pass
        else:
            raise AssertionError("Receipt failure was not injected")
    finally:
        module.atomic_json = write
    if not failed or loop.state != "FAILED" or loop.publisher.ready:
        raise AssertionError("Uncertain application did not close admission")
    if application.state()["pending"] is None:
        raise AssertionError("Interrupted intent was lost")
    recovered = loop.recover_config()
    if recovered["status"] != "rolled_back" or executor.observe() != current:
        raise AssertionError("Recovery did not restore the source configuration")
    verify_restore(source, expected, before)
    report["config_application"] = {
        "receipts": receipts,
        "duplicate_without_restart": True,
        "unsupported_rollout_rejected": True,
        "interrupted_receipt": recovered,
        "same_checkpoint_cursor_and_progress": True,
    }


def run(args):
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    from examples.autopilot.elastic.basic_policy import measured_policy
    from examples.autopilot.elastic.fixtures.groups import create_ppo_group
    from examples.autopilot.elastic.validation.train_lifecycle import compare_states

    from areal.autopilot.contracts import TrainingConfig
    from areal.autopilot.elastic.loop import ElasticLoop
    from areal.autopilot.elastic.policy.basic.cost_model.rollout import RolloutConfig
    from areal.autopilot.elastic.policy.basic.simulator import Measurement
    from areal.autopilot.elastic.runtime.config import (
        ActorConfigExecutor,
        ConfigApplication,
    )
    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.state.checkpoint import (
        artifact_manifest,
        atomic_json,
    )
    from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer

    args.output.mkdir(parents=True, exist_ok=False)
    model = args.model
    revision = args.model_revision
    if args.tiny:
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        model = args.output / "tiny-model"
        torch.manual_seed(19)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=512,
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        Qwen3ForCausalLM(config).save_pretrained(model)
        AutoTokenizer.from_pretrained(
            args.model, local_files_only=True
        ).save_pretrained(model)
        revision = "tiny-qwen3-random-seed19"
    group = create_ppo_group(
        root=args.output / "training",
        model=model,
        revision=revision,
        devices=args.devices,
        elastic=True,
    )
    group.configure(TrainingConfig(1))
    queue = BatchQueue(args.output / "queue")
    trainer = QueuedTrainer(group, queue)
    measurements = []
    report = {
        "status": "running",
        "scope": "synthetic PPO, diagnostic CPU publication",
        "quality_status": "not_evaluated",
        "calibration": [],
        "restore_checks": [],
    }
    loop = None

    def measure(phase, world_size, started, tokens=0):
        measurements.append(
            Measurement(phase, world_size, time.monotonic() - started, tokens)
        )

    def verify_restore(source, expected, before):
        directory = args.output / "verification"
        directory.mkdir(exist_ok=True)
        from areal.api import SaveLoadMeta

        group.actor.save(
            SaveLoadMeta(
                path=str(directory / "dcp"), weight_format="dcp", with_optim=True
            )
        )
        dcp_to_torch_save(directory / "dcp", directory / "actual.pt")
        actual = torch.load(
            directory / "actual.pt", map_location="cpu", weights_only=False
        )
        count = compare_states(actual, expected, rtol=0, atol=0)
        if artifact_manifest(source) != before:
            raise AssertionError("Restoration modified its source checkpoint")
        report["restore_checks"].append(
            {
                "source": source.name,
                "target_world_size": group.training_config.world_size,
                "tensor_elements_compared": count,
                "exact": True,
                "source_immutable": True,
            }
        )
        shutil.rmtree(directory)

    try:
        trainer.prepare()
        expected = source = None
        for count in range(1, len(args.devices) + 1):
            if count > 1:
                before = artifact_manifest(source)
                records = trainer.configure(TrainingConfig(count))
                require_target(trainer, count)
                measurements.extend(
                    Measurement(phase, config.world_size, seconds)
                    for phase, config, seconds in records
                )
                verify_restore(source, expected, before)
            else:
                started = time.monotonic()
                trainer.activate()
                measure("start", count, started)
            batch = payload(trainer.version, revision)
            queue.enqueue(batch["rollout_id"], batch, policy_version=trainer.version)
            started = time.monotonic()
            trainer.train_next()
            measure("train", count, started, group.workload(batch).sequence_tokens)
            started = time.monotonic()
            source = trainer.checkpoint()
            measure("checkpoint", count, started)
            measurements.append(Measurement("publish", count, 0))
            dcp_to_torch_save(source / "dcp", args.output / "expected.pt")
            expected = torch.load(
                args.output / "expected.pt", map_location="cpu", weights_only=False
            )
            # Verify that a nontrivial Adam state, rather than an empty template,
            # is what every subsequent topology restores.
            optimizer = expected["dcp"]["optim"]
            adam = list(optimizer["state"].values())
            if not adam or not any(
                torch.count_nonzero(state["exp_avg"]).item() for state in adam
            ):
                raise AssertionError("Source Adam moments are empty or all zero")
            if any(float(state["step"]) != trainer.version for state in adam):
                raise AssertionError("Source Adam step differs from committed version")
            report["calibration"].append(
                {
                    "world_size": count,
                    "version": trainer.version,
                    "samples": trainer.committed_samples,
                    "effective_tokens": trainer.committed_tokens,
                }
            )
            started = time.monotonic()
            trainer.suspend()
            measure("release", count, started)
            if count == 1:
                before = artifact_manifest(source)
                started = time.monotonic()
                trainer.activate()
                measure("restore", count, started)
                verify_restore(source, expected, before)
                started = time.monotonic()
                trainer.suspend()
                measure("release", count, started)
            atomic_json(args.output / "report.json", report)
        if args.all_pairs:
            # Euler tour covers every directed change, all at the same committed
            # step/cursor. Saving after restore changes only the checkpoint layout.
            adjacency = {
                n: [m for m in range(1, len(args.devices) + 1) if m != n]
                for n in range(1, len(args.devices) + 1)
            }
            stack, route = [len(args.devices)], []
            while stack:
                if adjacency[stack[-1]]:
                    stack.append(adjacency[stack[-1]].pop())
                else:
                    route.append(stack.pop())
            for target in list(reversed(route))[1:]:
                before = artifact_manifest(source)
                records = trainer.configure(TrainingConfig(target))
                require_target(trainer, target)
                measurements.extend(
                    Measurement(phase, config.world_size, seconds)
                    for phase, config, seconds in records
                )
                verify_restore(source, expected, before)
                source = trainer.checkpoint()
                started = time.monotonic()
                trainer.suspend()
                measure("release", target, started)
                atomic_json(args.output / "report.json", report)
        profile = args.output / "profiles.json"
        atomic_json(
            profile,
            {
                "format_version": 1,
                "inference_gpus": 0,
                "compatibility": group.compatibility,
                "measurements": [asdict(x) for x in measurements],
            },
        )
        # Enter the actual lifecycle with calibrated profiles and a committed head.
        trainer.stop()
        trainer = QueuedTrainer(group, queue)
        policy = measured_policy(
            profile, horizon=2, idle_seconds=1, compatibility=group.compatibility
        )
        publisher = CheckpointPublisher()
        application = None
        if args.config_application:
            # Declared identity only: this validation has no inference service.
            rollout = RolloutConfig(
                revision,
                "cpu",
                "diagnostic-receipt",
                "synthetic",
                1,
                1,
                1,
                64,
                64,
                1,
                backend="diagnostic",
            )
            application = ConfigApplication(
                ActorConfigExecutor(trainer, rollout), args.output / "applications"
            )
        loop = ElasticLoop(
            trainer,
            publisher,
            policy=policy,
            journal=args.output / "loop.json",
            config_application=application,
            policy_bundle_id="measured-basic-validation",
        )
        loop.start()
        if application is not None:
            verify_config_application(
                loop,
                report,
                verify_restore,
                source,
                expected,
                artifact_manifest(source),
            )
        for _ in range(2):
            with loop.episode() as version:
                batch = payload(version, revision)
                queue.enqueue(batch["rollout_id"], batch, policy_version=version)
            loop.tick()
        report.update(
            status="passed",
            final_version=trainer.version,
            samples=trainer.committed_samples,
            effective_tokens=trainer.committed_tokens,
            queue=queue.status(),
            optimized_world_sizes=[
                x["config"] for x in loop.events if x["event"] == "DECIDED"
            ],
            history_workloads=len(policy.history.snapshot().workloads),
        )
        if application is not None:
            report["config_application"]["final_state"] = application.state()
        if trainer.committed_samples != len(args.devices) + 2:
            raise AssertionError("Logical sample count differs from committed updates")
        fronts = [event["frontend"] for event in group.events if event.get("frontend")]
        if not fronts or any(
            any(
                front[key] != fronts[0][key]
                for key in ("services", "processes", "api_key_sha256")
            )
            for front in fronts
        ):
            raise AssertionError(
                "Persistent training frontend changed across topologies"
            )
        report["frontend_stable"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if loop is not None:
                loop.close()
            else:
                trainer.stop()
        finally:
            try:
                group.close()
            finally:
                report["lifecycle"] = group.events
                atomic_json(args.output / "report.json", report)


def run_joint(args):
    """Real SGLang generation and FSDP state checks on a common local GPU pool."""
    import asyncio

    import httpx
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from examples.autopilot.elastic.fixtures.groups import create_ppo_group
    from examples.autopilot.elastic.fixtures.inference_service import InferenceService
    from examples.autopilot.elastic.fixtures.weight_publication import WeightPublisher
    from examples.autopilot.elastic.validation.probes import probe_inference
    from examples.autopilot.elastic.validation.train_lifecycle import compare_states

    import areal.autopilot.elastic.runtime.config as module
    from areal.api import SaveLoadMeta
    from areal.autopilot.contracts import TrainingConfig
    from areal.autopilot.elastic.loop import ElasticLoop
    from areal.autopilot.elastic.policy.basic.cost_model.rollout import RolloutConfig
    from areal.autopilot.elastic.policy.basic.joint import JointConfig
    from areal.autopilot.elastic.runtime.config import (
        ConfigApplication,
        ConfigRequest,
        JointConfigExecutor,
    )
    from areal.autopilot.elastic.runtime.rollout import RolloutGroup
    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.state.checkpoint import (
        artifact_manifest,
        atomic_json,
    )
    from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer

    args.output.mkdir(parents=True, exist_ok=False)
    model, revision = args.model, args.model_revision
    if args.tiny:
        model, revision = args.output / "tiny-model", "tiny-qwen3-joint-seed19"
        torch.manual_seed(19)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=512,
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        Qwen3ForCausalLM(config).save_pretrained(model)
        vocab = {
            word: index
            for index, word in enumerate(
                ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + [f"w{i}" for i in range(4, 256)]
            )
        }
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            model_max_length=512,
        )
        tokenizer.chat_template = "{% for message in messages %}{{message['role']}}: {{message['content']}} {% endfor %}assistant:"
        tokenizer.save_pretrained(model)
    group = create_ppo_group(
        root=args.output / "training",
        model=model,
        revision=revision,
        devices=args.devices,
        elastic=True,
    )
    group.configure(TrainingConfig(1))
    trainer = QueuedTrainer(group, BatchQueue(args.output / "queue"))
    initial_rollout = RolloutConfig(
        revision, "H20", "local-sglang-v2", "fixed-synthetic", 1, 1, 2, 512, 1024, 1
    )
    rollout = RolloutGroup(
        config=initial_rollout,
        model=model,
        model_name="joint-validation",
        model_revision=revision,
        root=args.output / "rollout",
        lock_root=args.output / "gpu-locks",
        service_factory=InferenceService,
        context_length=512,
    )
    publisher = WeightPublisher(rollout, state=args.output / "published.json")
    executor = JointConfigExecutor(trainer, rollout, publisher)
    application = ConfigApplication(executor, args.output / "applications")
    loop = ElasticLoop(
        trainer,
        publisher,
        config_application=application,
        journal=args.output / "loop.json",
    )
    report = {
        "status": "running",
        "scope": "synthetic PPO with real SGLang replica/TP changes",
        "quality_status": "not_evaluated",
        "receipts": [],
        "probes": [],
        "restore_checks": [],
    }

    def save():
        atomic_json(args.output / "report.json", report)

    def probe():
        with loop.episode() as version:
            record = asyncio.run(probe_inference(rollout, version))
            with httpx.Client(timeout=120, trust_env=False) as client:
                tokens = []
                for backend in rollout.controller._inf_addrs:
                    response = client.post(
                        f"{backend}/generate",
                        json={
                            "input_ids": [10, 11, 12],
                            "sampling_params": {
                                "max_new_tokens": 4,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                    )
                    response.raise_for_status()
                    count = response.json()["meta_info"]["completion_tokens"]
                    if count != 4:
                        raise AssertionError(
                            "Replica did not generate the requested tokens"
                        )
                    tokens.append(count)
            record.update(
                config=asdict(executor.observe().config),
                per_replica_tokens=tokens,
                backend_settings=rollout.service.backend_settings,
                actor_active=trainer.is_active,
                actor_search_capacity=loop.observe().max_train_gpus,
                actor_devices=list(group.devices) if trainer.is_active else [],
                actor_planned_devices=list(
                    executor.placement(executor.observe().config)[0]
                ),
                rollout_devices=list(rollout.devices),
            )
            if loop.observe().max_train_gpus != (
                len(args.devices) - rollout.config.world_size
            ):
                raise AssertionError("Actor search capacity ignores Rollout GPUs")
            report["probes"].append(record)

    def train():
        with loop.episode() as version:
            batch = payload(version, revision)
            trainer.queue.enqueue(batch["rollout_id"], batch, policy_version=version)
        if loop.tick() != "ACTIVE":
            raise AssertionError("Training did not complete after joint application")

    def verify(source, expected, before):
        directory = args.output / "verify"
        directory.mkdir()
        group.actor.save(
            SaveLoadMeta(
                path=str(directory / "dcp"), weight_format="dcp", with_optim=True
            )
        )
        dcp_to_torch_save(directory / "dcp", directory / "actual.pt")
        actual = torch.load(
            directory / "actual.pt", map_location="cpu", weights_only=False
        )
        count = compare_states(actual, expected, rtol=0, atol=0)
        if artifact_manifest(source) != before:
            raise AssertionError("Joint application modified its checkpoint")
        report["restore_checks"].append(
            {"exact": True, "source_immutable": True, "tensor_elements_compared": count}
        )
        shutil.rmtree(directory)

    def request(identity, target):
        return ConfigRequest(
            identity,
            target,
            application.state()["generation"],
            trainer.version,
            "joint-validation",
        )

    try:
        loop.start()
        probe()
        train()
        source = trainer.latest_checkpoint
        before = artifact_manifest(source)
        dcp_to_torch_save(source / "dcp", args.output / "expected.pt")
        expected = torch.load(
            args.output / "expected.pt", map_location="cpu", weights_only=False
        )
        boundary = executor.observe()
        targets = [
            JointConfig(TrainingConfig(2), initial_rollout),
            JointConfig(
                TrainingConfig(1),
                replace(
                    initial_rollout,
                    replicas=2,
                    max_batch_sequences=3,
                    max_batch_tokens=1024,
                    kv_capacity_tokens=2048,
                ),
            ),
            JointConfig(
                TrainingConfig(1),
                replace(
                    initial_rollout,
                    tp=2,
                    max_batch_sequences=3,
                    max_batch_tokens=1024,
                    kv_capacity_tokens=2048,
                ),
            ),
            JointConfig(
                TrainingConfig(1), replace(initial_rollout, max_batch_sequences=1)
            ),
        ]
        for index, target in enumerate(targets):
            change = request(f"joint-{index}", target)
            receipt = loop.apply_config(change)
            if receipt["status"] != "applied" or executor.observe() != replace(
                boundary, config=target
            ):
                raise AssertionError(
                    "Joint target was not activated at the original boundary"
                )
            verify(source, expected, before)
            owner, events = rollout.owner, len(group.events)
            if (
                loop.apply_config(change) != receipt
                or rollout.owner != owner
                or len(group.events) != events
            ):
                raise AssertionError("Duplicate request restarted a role")
            probe()
            report["receipts"].append(receipt)
            save()
        current = executor.observe()
        change = request(
            "joint-interrupted",
            replace(
                current.config, rollout=replace(current.config.rollout, replicas=2)
            ),
        )
        write = module.atomic_json
        fired = False

        def interrupt(path, state):
            nonlocal fired
            if (
                not fired
                and state["pending"] is None
                and change.request_id in state["receipts"]
            ):
                if publisher.ready:
                    raise AssertionError("Admission opened before joint receipt commit")
                fired = True
                raise OSError("Injected joint receipt failure")
            write(path, state)

        module.atomic_json = interrupt
        try:
            try:
                loop.apply_config(change)
            except OSError:
                pass
            else:
                raise AssertionError("Joint interruption was not injected")
        finally:
            module.atomic_json = write
        if not fired or publisher.ready or loop.state != "FAILED":
            raise AssertionError(
                "Interrupted joint application did not fence admission"
            )
        recovery = loop.recover_config()
        if recovery["status"] != "rolled_back" or executor.observe() != current:
            raise AssertionError("Joint source was not restored")
        verify(source, expected, before)
        probe()
        if trainer.version != 1 or trainer.committed_samples != 1:
            raise AssertionError("Application replayed training")
        train()
        train()
        probe()
        if (trainer.version, trainer.committed_samples, trainer.committed_tokens) != (
            3,
            3,
            96,
        ):
            raise AssertionError(
                "Post-application logical progress differs from three updates"
            )
        report.update(
            status="passed",
            recovery=recovery,
            final_weight_version=trainer.version,
            final_config_state=application.state(),
            samples=trainer.committed_samples,
            effective_tokens=trainer.committed_tokens,
            role_events=executor.role_events,
            queue=trainer.queue.status(),
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            loop.close()
        finally:
            try:
                group.close()
            finally:
                report["lifecycle"] = group.events
                save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Verify every directed resize at the same committed step",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a small random Qwen3 for infrastructure checks",
    )
    parser.add_argument(
        "--config-application",
        action="store_true",
        help="Verify durable Actor config requests and interrupted receipt recovery",
    )
    parser.add_argument(
        "--joint-config-application",
        action="store_true",
        help="Verify real SGLang and Actor joint changes on a shared pool",
    )
    args = parser.parse_args()
    if args.joint_config_application and (
        len(args.devices) < 4 or args.config_application
    ):
        parser.error(
            "Joint validation requires at least four GPUs and excludes --config-application"
        )
    if args.config_application and len(args.devices) < 3:
        parser.error("--config-application requires a pool of at least three GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    (run_joint if args.joint_config_application else run)(args)


if __name__ == "__main__":
    main()
