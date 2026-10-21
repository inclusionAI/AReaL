# SPDX-License-Identifier: Apache-2.0

"""Independent verification that an owned SWE sandbox has been deleted."""

import argparse
import asyncio
import json


async def wait_for_pod_removal(args: argparse.Namespace, instance_id: str) -> None:
    """Verify deletion independently because the upstream SDK swallows errors."""
    deadline = asyncio.get_running_loop().time() + 60
    while True:
        process = await asyncio.create_subprocess_exec(
            str(args.kubectl),
            "--kubeconfig",
            str(args.kubeconfig),
            "--request-timeout=5s",
            "get",
            "pods",
            "-n",
            args.sandbox_namespace,
            "--field-selector",
            f"metadata.name={instance_id}",
            "-o",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise RuntimeError(
                "Timed out querying Kubernetes for sandbox deletion"
            ) from None
        if process.returncode:
            raise RuntimeError(f"Could not verify sandbox deletion: {stderr.decode()}")
        if not json.loads(stdout)["items"]:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(f"Sandbox {instance_id} still exists after release")
        await asyncio.sleep(1)
