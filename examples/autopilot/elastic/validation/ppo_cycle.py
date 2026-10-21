# SPDX-License-Identifier: Apache-2.0

"""Validate real SWE PPO consumption, serving publication and worker restoration."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from examples.autopilot.elastic.validation.probes import probe_inference


async def run(args):
    from examples.autopilot.elastic.fixtures.groups import create_ppo_group
    from examples.autopilot.elastic.fixtures.inference_service import InferenceService
    from examples.autopilot.elastic.fixtures.weight_publication import WeightPublisher

    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
    from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer

    args.output.mkdir(parents=True, exist_ok=False)
    args.state_root.mkdir(parents=True, exist_ok=True)
    queue = BatchQueue(args.state_root / "queue")
    payload = json.loads(args.trajectory.read_text())
    queue.enqueue(
        "swe:" + payload["rollout_id"],
        payload,
        policy_version=payload["policy_version"],
    )
    group = create_ppo_group(
        root=args.state_root / "training",
        model=args.model,
        revision=args.model_revision,
        devices=args.devices,
    )
    trainer = QueuedTrainer(group, queue)
    service = InferenceService(
        model=args.model,
        model_name="Qwen3-0.6B",
        device=args.inference_device,
        root=args.output / "inference",
        lock_root=args.state_root.parent / "gpu-locks",
    )
    publisher = WeightPublisher(service, state=args.state_root / "published.json")
    report = {"status": "running", "algorithm": group.compatibility, "updates": []}
    try:
        await asyncio.to_thread(service.start)
        publisher.initialize()
        await asyncio.to_thread(trainer.start)
        update = await asyncio.to_thread(
            trainer.train_next, lambda batch: group.train_trajectory(batch.payload)
        )
        if update is None:
            raise RuntimeError("Validation requires an unconsumed initial trajectory")
        report["updates"].append(update)
        checkpoint = await asyncio.to_thread(trainer.checkpoint)
        report["publication"] = await asyncio.to_thread(publisher.publish, checkpoint)
        report["inference_probe"] = await probe_inference(service, 1)
        await asyncio.to_thread(trainer.stop)
        await asyncio.to_thread(trainer.start)
        if (
            group.step != 1
            or await asyncio.to_thread(
                trainer.train_next, lambda batch: group.train_trajectory(batch.payload)
            )
            is not None
        ):
            raise RuntimeError("Committed trajectory was not recovered exactly once")
        report["retry_publication"] = await asyncio.to_thread(
            publisher.publish, checkpoint
        )
        await asyncio.to_thread(trainer.stop)
        report["inference_after_training_release"] = await probe_inference(service, 1)
        report.update(
            status="passed",
            restored_step=group.step,
            checkpoint=str(checkpoint),
            queue=queue.status(),
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            try:
                try:
                    await asyncio.to_thread(trainer.stop)
                finally:
                    await asyncio.to_thread(group.close)
            finally:
                report["inference_cleanup"] = await asyncio.to_thread(service.close)
        except BaseException as error:
            report.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            report["lifecycle"] = group.events
            atomic_json(args.output / "report.json", report)


def main():
    import uvloop

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "state-root", "output", "trajectory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--inference-device", type=int, default=0)
    args = parser.parse_args()
    if args.inference_device in args.devices:
        parser.error("Training and inference pools must be disjoint")
    uvloop.run(run(args))


if __name__ == "__main__":
    main()
