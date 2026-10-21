# SPDX-License-Identifier: Apache-2.0

"""Inference probes shared by lifecycle validation runners."""

import asyncio
import uuid


async def probe_inference(service, expected_version: int) -> dict:
    import httpx

    from areal.infra.rpc.rtensor import RTensor
    from areal.infra.rpc.serialization import deserialize_value

    gateway = service.controller._gateway_addr
    admin = {"Authorization": f"Bearer {service.admin_key}"}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(
            f"{gateway}/rl/start_session",
            headers=admin,
            json={"task_id": "publish-check-" + uuid.uuid4().hex},
        )
        response.raise_for_status()
        session = response.json()
        member = session["sessions"][0]
        headers = {"Authorization": f"Bearer {member['session_api_key']}"}
        response = await client.post(
            f"{gateway}/chat/completions",
            headers=headers,
            json={
                "model": service.model_name,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_completion_tokens": 32,
                "temperature": 0.6,
                "stream": False,
            },
        )
        response.raise_for_status()
        response = await client.post(
            f"{gateway}/rl/set_reward",
            headers=headers,
            json={"interaction_id": None, "reward": 0.0},
        )
        response.raise_for_status()
        response = await client.post(
            f"{gateway}/export_trajectories",
            headers=admin,
            json={
                "session_ids": [member["session_id"]],
                "group_id": session.get("group_id"),
                "trajectory_id": response.json().get("trajectory_id"),
                "discount": 1.0,
                "style": "individual",
                "remove_session": True,
            },
        )
        response.raise_for_status()
        tensors = await asyncio.to_thread(
            RTensor.localize, deserialize_value(response.json()["traj"])
        )
    versions = tensors["versions"][tensors["loss_mask"].bool()].tolist()
    if not versions or set(versions) != {expected_version}:
        raise RuntimeError("Post-publication tokens carry the wrong weight version")
    return {"generated_tokens": len(versions), "version": expected_version}
