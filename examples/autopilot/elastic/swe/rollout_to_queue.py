# SPDX-License-Identifier: Apache-2.0

"""Run a real, bounded SWE rollout through v2 and persist materialized training data."""

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path


async def run_agent(args: argparse.Namespace, gateway: str, key: str) -> dict:
    from areal.autopilot.elastic.runtime.processes import (
        OWNER_ENV,
        stop_owned_processes,
    )

    report_path = args.output / "agent-report.json"
    owner = uuid.uuid4().hex
    with (args.output / "agent.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            str(args.agent_python),
            "-u",
            str(Path(__file__).with_name("agent_worker.py")),
            "--fixture",
            str(args.fixture),
            "--config",
            str(args.agent_config),
            "--report",
            str(report_path),
            "--sandbox-python",
            str(args.sandbox_python),
            "--environment",
            args.environment,
            env={**os.environ, "OPENAI_MODEL": args.model_name, OWNER_ENV: owner},
            stdin=asyncio.subprocess.PIPE,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(
                process.communicate(
                    json.dumps({"base_url": gateway, "api_key": key}).encode() + b"\n"
                ),
                timeout=3000,
            )
        finally:
            await asyncio.to_thread(stop_owned_processes, owner)
            await asyncio.wait_for(process.wait(), timeout=10)
    report = json.loads(report_path.read_text())
    if process.returncode or report["status"] != "passed":
        raise RuntimeError(
            "SWE agent or grading failed; see agent-report.json and agent.log"
        )
    return report


async def verify_sandbox_cleanup(args: argparse.Namespace) -> bool:
    from examples.autopilot.elastic.swe.cleanup import wait_for_pod_removal

    path = args.output / "agent-report.json"
    ownership = path.with_suffix(".sandbox.json")
    instance_id = None
    for record in (ownership, path):
        if record.exists():
            instance_id = (
                json.loads(record.read_text()).get("instance_id") or instance_id
            )
    if not instance_id:
        return False
    # The SDK may have been terminated before its own finally block. Delete only
    # this episode's journaled pod, then independently query for its absence.
    process = await asyncio.create_subprocess_exec(
        str(args.kubectl),
        "--kubeconfig",
        str(args.kubeconfig),
        "--request-timeout=10s",
        "delete",
        f"pod/{instance_id}",
        "-n",
        args.sandbox_namespace,
        "--ignore-not-found=true",
        "--wait=true",
        "--timeout=60s",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=70)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f"Could not delete the owned sandbox: {stderr.decode()}")
    await wait_for_pod_removal(args, instance_id)
    return True


async def collect_rollout(
    args: argparse.Namespace, service, policy_version: int
) -> dict:
    """Use an existing service; caller owns the episode gate and new output folder."""
    import httpx

    from examples.autopilot.elastic.swe.trajectory import trajectory_payload

    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
    from areal.infra.rpc.rtensor import RTensor
    from areal.infra.rpc.serialization import deserialize_value

    summary = {
        "status": "running",
        "model_revision": args.model_revision,
        "policy_version": policy_version,
        "run_id": uuid.uuid4().hex,
    }
    try:
        gateway = service.controller._gateway_addr
        if service.controller.get_version() != policy_version:
            raise RuntimeError("Rollout admission version differs from loaded policy")
        summary["v2_gateway"] = gateway
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            admin = {"Authorization": f"Bearer {service.admin_key}"}
            task_id = json.loads(args.fixture.read_text())["instance_id"]
            response = await client.post(
                f"{gateway}/rl/start_session",
                headers=admin,
                # Gateway counters restart with inference. Namespace sessions by
                # this producer attempt so repeated tasks cannot collide in SQLite.
                json={"task_id": f"{task_id}-{summary['run_id']}", "group_size": 1},
            )
            response.raise_for_status()
            session = response.json()
            credentials = session["sessions"][0]
            session_id = credentials["session_id"]
            agent_report = await run_agent(
                args, gateway, credentials["session_api_key"]
            )
            response = await client.post(
                f"{gateway}/rl/set_reward",
                headers={"Authorization": f"Bearer {credentials['session_api_key']}"},
                json={"interaction_id": None, "reward": agent_report["reward"]},
            )
            response.raise_for_status()
            # Upstream export is destructive: do not blindly retry this request.
            response = await client.post(
                f"{gateway}/export_trajectories",
                headers=admin,
                json={
                    "session_ids": [session_id],
                    "group_id": session.get("group_id"),
                    "trajectory_id": response.json().get("trajectory_id"),
                    "discount": 1.0,
                    "style": "individual",
                    "remove_session": True,
                    "drop_retry_orphans": False,
                    "reward_normalization": False,
                },
            )
            response.raise_for_status()
            tensors = await asyncio.to_thread(
                RTensor.localize, deserialize_value(response.json()["traj"])
            )
        payload = trajectory_payload(
            tensors,
            task_id=task_id,
            rollout_id=session_id,
            model_revision=args.model_revision,
            policy_version=policy_version,
        )
        if payload["trajectory"]["rewards"][-1] != agent_report["reward"]:
            raise RuntimeError(
                "Exported terminal reward does not match sandbox grading"
            )
        # Fsync a recovery copy before enqueue; the earlier export-to-copy crash gap remains.
        atomic_json(args.output / "trajectory.json", payload)
        queue = BatchQueue(args.queue)
        batch_id = f"swe:{task_id}:{session_id}"
        sequence = queue.enqueue(batch_id, payload, policy_version=policy_version)
        reopened = BatchQueue(args.queue)
        if (
            reopened.enqueue(batch_id, payload, policy_version=policy_version)
            != sequence
        ):
            raise RuntimeError("Idempotent queue reopen failed")
        summary.update(
            status="passed",
            batch_id=batch_id,
            sequence=sequence,
            generated_tokens=payload["generated_tokens"],
            interactions=len(payload["trajectory"]["input_ids"]),
            reward=agent_report["reward"],
            queue=reopened.status(),
        )
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            summary["sandbox_removed"] = await verify_sandbox_cleanup(args)
            if summary["status"] == "passed" and not summary["sandbox_removed"]:
                raise RuntimeError("Sandbox cleanup was not verified")
        except BaseException as error:
            summary.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            atomic_json(args.output / "report.json", summary)
    return summary


async def run(args: argparse.Namespace) -> None:
    from examples.autopilot.elastic.fixtures.inference_service import InferenceService

    from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json

    args.output.mkdir(parents=True, exist_ok=False)
    service = InferenceService(
        model=args.model,
        model_name=args.model_name,
        device=args.device,
        root=args.output / "inference",
        lock_root=args.queue.parent / "gpu-locks",
    )
    summary = {"status": "running"}
    try:
        await asyncio.to_thread(service.start)
        summary = await collect_rollout(args, service, 0)
    except BaseException as error:
        path = args.output / "report.json"
        if path.exists():
            summary = json.loads(path.read_text())
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            summary["inference_cleanup"] = await asyncio.to_thread(service.close)
        except BaseException as error:
            summary.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            atomic_json(args.output / "report.json", summary)


def main() -> None:
    import uvloop

    # Only the independently spawned SGLang backend receives a GPU UUID.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "fixture",
        "agent-python",
        "sandbox-python",
        "queue",
        "output",
        "kubectl",
        "kubeconfig",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--sandbox-namespace", default="aenv-sandbox")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path(__file__).with_name("agent-config.yaml"),
    )
    # Uvicorn's callback thread changes the global loop policy during startup.
    # Start with uvloop explicitly so later subprocess creation remains supported.
    uvloop.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
