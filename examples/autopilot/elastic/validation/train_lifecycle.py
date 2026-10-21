# SPDX-License-Identifier: Apache-2.0

"""Compare continuous FSDP updates with two complete worker-group restarts.

Run with ordinary Python, not torchrun. This is a fixed LM-loss lifecycle
experiment, not a SWE rollout or PPO quality evaluation.
"""

import argparse
import fcntl
import getpass
import json
import math
import os
import secrets
import time
from pathlib import Path


def compare_states(actual, expected, *, rtol, atol, path="root"):
    import torch

    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=path)
        return actual.numel()
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"State keys differ at {path}")
        return sum(
            compare_states(
                value, expected[key], rtol=rtol, atol=atol, path=f"{path}.{key}"
            )
            for key, value in actual.items()
        )
    if isinstance(actual, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"State lengths differ at {path}")
        return sum(
            compare_states(a, b, rtol=rtol, atol=atol, path=f"{path}[{index}]")
            for index, (a, b) in enumerate(zip(actual, expected))
        )
    if actual != expected:
        raise AssertionError(f"State differs at {path}: {actual!r} != {expected!r}")
    return 0


def run_queued(group, *, root, steps, segment_steps, restart, inject_faults):
    """Six distinct durable batches, including two deliberately lost commits."""
    import torch

    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer

    queue = BatchQueue(root / "queue")
    generator = torch.Generator().manual_seed(47)
    for index in range(steps):
        payload = {"source": "synthetic-lm", "samples": []}
        for _ in range(2):
            payload["samples"].append(
                {
                    "input_ids": torch.randint(
                        1000, 4000, (64,), generator=generator
                    ).tolist(),
                    "attention_mask": [1] * 64,
                    "loss_mask": [0] + [1] * 63,
                }
            )
        batch_id = f"batch-{index + 1:08d}"
        sequence = queue.enqueue(batch_id, payload, policy_version=0)
        if queue.enqueue(batch_id, payload, policy_version=0) != sequence:
            raise AssertionError("Producer retry duplicated a batch")
    # Discard the producer object to exercise loading durable data from disk.
    queue = BatchQueue(queue.root)
    trainer = QueuedTrainer(group, queue)
    records, attempts, faults = {}, [], []
    injected_update = False
    try:
        trainer.start()
        while group.step < steps:
            record = trainer.train_next(
                lambda batch: group.train_fixed_batch(batch.payload["samples"])
            )
            if record is None:
                raise AssertionError("Queue ran out before all optimizer updates")
            records[group.step] = record
            attempts.append(record)
            if restart and inject_faults and group.step == 3 and not injected_update:
                injected_update = True
                # The updated weights are deliberately discarded without saving.
                trainer.stop()
                trainer = QueuedTrainer(group, BatchQueue(queue.root))
                trainer.start()
                if group.step != 2:
                    raise AssertionError("Uncommitted update was not rolled back")
                records.pop(3)
                faults.append(
                    {
                        "window": "after_update_before_checkpoint",
                        "restored_step": group.step,
                    }
                )
            elif restart and group.step < steps and group.step % segment_steps == 0:
                lost_ack = inject_faults and group.step == 4
                if lost_ack:
                    # Simulate process loss between the durable latest pointer and
                    # queue acknowledgement, without adding production fault hooks.
                    group.checkpoint(data_cursor=trainer.consumer.cursor)
                    if queue.status()["cursor"]["applied_batches"] != 2:
                        raise AssertionError("Queue acknowledged before checkpoint")
                else:
                    trainer.checkpoint()
                expected_step = group.step
                trainer.stop()
                trainer = QueuedTrainer(group, BatchQueue(queue.root))
                trainer.start()
                if group.step != expected_step:
                    raise AssertionError("Published update was lost during recovery")
                if lost_ack:
                    if queue.status()["counts"]["committed"] != 4:
                        raise AssertionError("Lost acknowledgement was not repaired")
                    faults.append(
                        {
                            "window": "after_checkpoint_before_ack",
                            "restored_step": group.step,
                        }
                    )
        checkpoint = trainer.checkpoint()
        status = queue.status()
        if status["counts"] != {
            "pending": 0,
            "leased": 0,
            "applied": 0,
            "committed": steps,
        }:
            raise AssertionError("Queue did not commit each batch exactly once")
        if len({record["batch_digest"] for record in records.values()}) != steps:
            raise AssertionError("Queue fixture did not use distinct batch contents")
        if restart and inject_faults:
            third = [record for record in attempts if record["step"] == 3]
            keys = ("step", "batch_id", "batch_digest", "rng_probe")
            if len(third) != 2 or any(third[0][key] != third[1][key] for key in keys):
                raise AssertionError(
                    "Uncommitted batch replay differs from first attempt"
                )
        return (
            checkpoint,
            list(records.values()),
            {
                **status,
                "attempts": attempts,
                "faults": faults,
                "scope": "GPU generations and consumer sessions restart; CPU coordinator stays alive",
            },
        )
    finally:
        trainer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--controller", choices=["v1", "v2"], default="v2")
    parser.add_argument("--frontend-lifecycle", choices=["persistent", "rebuild"])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--segment-steps", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--batch-source", choices=["fixed", "queue"], default="fixed")
    parser.add_argument("--queue-faults", action="store_true")
    args = parser.parse_args()
    if args.queue_faults and (
        args.batch_source != "queue" or args.steps != 6 or args.segment_steps != 2
    ):
        parser.error(
            "Queue fault fixture requires --batch-source queue --steps 6 --segment-steps 2"
        )
    if args.frontend_lifecycle is None:
        args.frontend_lifecycle = "persistent" if args.controller == "v2" else "rebuild"
    if args.controller == "v1" and args.frontend_lifecycle == "persistent":
        parser.error("Persistent frontend requires --controller v2")
    if len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("Training GPU IDs must be nonnegative and unique")
    if (
        args.segment_steps < 1
        or args.steps < 3
        or args.steps % args.segment_steps
        or args.steps // args.segment_steps < 3
    ):
        parser.error("Use at least three equal segments to exercise two restarts")
    if args.output.exists():
        parser.error(
            "Output directory must be new; previous experiments are never overwritten"
        )
    args.output.mkdir(parents=True)
    # Hide CUDA before importing AReaL/Torch. Each worker gets its own explicit
    # GPU assignment below, so the coordinator never owns a CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cache_root = args.output.parent / "cache"
    for key, directory in (
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
    ):
        path = cache_root / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    for key in (
        "AREAL_SPMD_MODE",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(key, None)

    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    from examples.autopilot.elastic.fixtures.groups import LMGroup

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        MicroBatchSpec,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.autopilot.elastic.runtime.processes import (
        gpu_processes,
        resolve_gpu_uuids,
    )
    from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json

    started = time.monotonic()
    report = {
        "status": "running",
        "model": str(args.model),
        "devices": args.devices,
        "steps": args.steps,
        "batch_source": args.batch_source,
        "controller_version": args.controller,
        "frontend_lifecycle": args.frontend_lifecycle,
        "restart_scope": (
            "workers_and_data_proxy"
            if args.frontend_lifecycle == "persistent"
            else "training_service"
            if args.controller == "v2"
            else "rpc_workers"
        ),
        "runs": {},
    }
    atomic_json(args.output / "report.json", report)
    # A project-local lease prevents two copies of this experiment sharing a GPU.
    lease_root = args.output.parent / "gpu-leases"
    lease_root.mkdir(exist_ok=True)
    leases = []
    for device in sorted(args.devices):
        stream = (lease_root / f"gpu-{device}.lock").open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(stream)

    generator = torch.Generator().manual_seed(47)
    batch = []
    for _ in range(2):
        ids = torch.randint(1000, 4000, (64,), generator=generator).tolist()
        batch.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * 64,
                "loss_mask": [0] + [1] * 63,
            }
        )
    global_batch = len(batch) * len(args.devices)
    ft_spec = FinetuneSpec(1, args.steps * global_batch, global_batch)
    compatibility = {
        "model": str(args.model.resolve()),
        "backend": f"fsdp:d{len(args.devices)}",
        "total_steps": args.steps,
        "seed": 17,
        "optimizer_dtype": "float32",
        "scheduler": "linear",
        "lr": 1e-5,
        "batch_seed": 47,
        "batch_source": args.batch_source,
        "samples_per_rank": len(batch),
        "sequence_length": 64,
        "controller_version": args.controller,
    }
    outputs = {}
    try:
        for mode in ("continuous", "restarted"):
            root = args.output / mode
            root.mkdir()
            config = TrainEngineConfig(
                experiment_name="autopilot-lifecycle",
                trial_name=mode,
                path=str(args.model.resolve()),
                backend=compatibility["backend"],
                dtype="bfloat16",
                optimizer_dtype="float32",
                _version=args.controller,
                admin_api_key=secrets.token_urlsafe(32),
                setup_timeout=120,
                workers_ready_timeout=120,
                request_timeout=120,
                disable_dropout=True,
                gradient_checkpointing=False,
                attn_impl="flash_attention_2",
                mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
                optimizer=OptimizerConfig(
                    lr=1e-5, lr_scheduler_type="linear", warmup_steps_proportion=0.0
                ),
                # Guards stay on CPU; the v2 adapter or v1 bootstrap selects GPUs.
                scheduling_spec=(
                    SchedulingSpec(
                        cpu=4,
                        gpu=1,
                        cmd="python -m examples.autopilot.elastic.fixtures.worker_entrypoint",
                        env_vars={
                            "LOGNAME": getpass.getuser(),
                            "CUDA_VISIBLE_DEVICES": "",
                            "AREAL_ELASTIC_GPU_POOL": ":".join(map(str, args.devices)),
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            # The H20 runtime hangs during first P2P/CUMEM
                            # connection setup; use NCCL's legacy allocation path.
                            "NCCL_CUMEM_ENABLE": "0",
                            "AREAL_ELASTIC_DIAGNOSTICS_DIR": str(root / "diagnostics"),
                        },
                    ),
                ),
            )
            group = LMGroup(
                devices=args.devices,
                config=config,
                engine_class="examples.autopilot.elastic.fixtures.lm_engine.LifecycleEngine",
                ft_spec=ft_spec,
                root=root,
                compatibility=compatibility,
                keep_frontend=args.frontend_lifecycle == "persistent",
            )
            records = []
            report["runs"][mode] = records
            try:
                if args.batch_source == "queue":
                    checkpoint, records, evidence = run_queued(
                        group,
                        root=root,
                        steps=args.steps,
                        segment_steps=args.segment_steps,
                        restart=mode == "restarted",
                        inject_faults=args.queue_faults,
                    )
                    outputs[mode] = checkpoint
                    report["runs"][mode] = records
                    report.setdefault("queues", {})[mode] = evidence
                else:
                    group.start()
                    while group.step < args.steps:
                        records.append(group.train_fixed_batch(batch))
                        atomic_json(args.output / "report.json", report)
                        if (
                            mode == "restarted"
                            and group.step < args.steps
                            and group.step % args.segment_steps == 0
                        ):
                            group.checkpoint()
                            group.stop()
                            group.start(restore=True)
                    outputs[mode] = group.checkpoint()
            finally:
                try:
                    group.close()
                finally:
                    report.setdefault("lifecycle", {})[mode] = group.events
                    atomic_json(args.output / "report.json", report)

        if args.frontend_lifecycle == "persistent":
            for mode, events in report["lifecycle"].items():
                snapshots = [e["frontend"] for e in events if "frontend" in e]
                if not snapshots:
                    raise AssertionError(f"Missing frontend evidence for {mode}")
                first = snapshots[0]
                for snapshot in snapshots:
                    for key in ("services", "processes", "api_key_sha256"):
                        if snapshot[key] != first[key]:
                            raise AssertionError(f"Frontend {key} changed for {mode}")
                for event in events:
                    if "frontend" not in event:
                        continue
                    expected = 200 if event["state"] == "ACTIVE" else 503
                    if event["frontend"]["training_status"] != expected:
                        raise AssertionError(
                            "Training was exposed before recovery completed"
                        )
                closed = [e for e in events if e["state"] == "FRONTEND_CLOSED"]
                if len(closed) != 1 or not closed[0]["process_exit_verified"]:
                    raise AssertionError("Final frontend shutdown was not verified")
            report["frontend_lifecycle_verified"] = True

        for step_a, step_b in zip(
            report["runs"]["continuous"], report["runs"]["restarted"]
        ):
            if step_a["rng_probe"] != step_b["rng_probe"]:
                raise AssertionError(
                    f"RNG continuation differs at step {step_a['step']}"
                )
            if args.batch_source == "queue":
                for key in ("batch_id", "batch_digest"):
                    if step_a[key] != step_b[key]:
                        raise AssertionError(f"Queue consumption differs: {key}")
        if args.batch_source == "queue":
            cursors = [
                report["queues"][mode]["cursor"] for mode in ("continuous", "restarted")
            ]
            for key in ("through_sequence", "applied_batches", "chain_sha256"):
                if cursors[0][key] != cursors[1][key]:
                    raise AssertionError(f"Final queue cursor differs: {key}")
            report["queue_consumption_equal"] = True
        norms = [
            step["rank_stats"][rank]["grad_norm"]
            for records in report["runs"].values()
            for step in records
            for rank in range(len(args.devices))
        ]
        if not norms or not all(math.isfinite(norm) and norm > 0 for norm in norms):
            raise AssertionError("Training did not produce finite nonzero gradients")
        for rank in range(len(args.devices)):
            states = [
                json.loads((outputs[mode] / f"runtime-{rank}.json").read_text())
                for mode in ("continuous", "restarted")
            ]
            if states[0] != states[1]:
                raise AssertionError(f"Final runtime state differs for rank {rank}")

        compare_dir = args.output / "comparison"
        compare_dir.mkdir()
        for mode in outputs:
            dcp_to_torch_save(outputs[mode] / "dcp", compare_dir / f"{mode}.pt")
        continuous = torch.load(
            compare_dir / "continuous.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        restarted = torch.load(
            compare_dir / "restarted.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        compared = compare_states(restarted, continuous, rtol=args.rtol, atol=args.atol)
        optimizer_steps = [
            value["step"].item()
            for value in restarted["dcp"]["optim"]["state"].values()
        ]
        if not optimizer_steps or any(step != args.steps for step in optimizer_steps):
            raise AssertionError(
                "Final Adam step counter does not equal completed updates"
            )
        processes = gpu_processes()
        if any(processes.get(uuid) for uuid in resolve_gpu_uuids(args.devices)):
            raise AssertionError("GPU processes remain after experiment")
        report.update(
            status="passed",
            compared_tensor_elements=compared,
            rng_continuation_equal=True,
            runtime_state_equal=True,
            gpu_pool_free=True,
            elapsed_seconds=time.monotonic() - started,
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "report.json", report)
        for stream in leases:
            stream.close()


if __name__ == "__main__":
    main()
