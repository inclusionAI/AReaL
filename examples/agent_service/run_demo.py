"""One-click demo: Agent Service + Tau2 (PydanticAI).

Usage::

    python examples/agent_service/run_demo.py                       # single task
    python examples/agent_service/run_demo.py --domain telecom      # different domain
    python examples/agent_service/run_demo.py --full                # all tasks
    python examples/agent_service/run_demo.py --config my.yaml      # custom config

Requires::

    pip install pydantic-ai
    pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import uvicorn
import yaml

from areal.experimental.agent_service import (
    OpenResponsesBridge,
    create_data_proxy_app,
    create_gateway_app,
    create_router_app,
    create_worker_app,
    mount_bridge,
)

ROUTER_PORT = 18081
WORKER_PORT = 19000
PROXY_PORT = 19100
GATEWAY_PORT = 18080

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _start_in_thread(app, port: int, name: str) -> threading.Thread:
    def run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    t = threading.Thread(target=run, daemon=True, name=name)
    t.start()
    return t


async def _wait_healthy(url: str, timeout: float = 10.0) -> None:
    async with httpx.AsyncClient() as client:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError(f"Service at {url} did not become healthy")


async def run_task(gateway_addr: str, task, domain: str) -> float:
    """Run a single tau2 task. Returns the reward."""
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    session_key = f"tau2-{domain}-{task.id}"
    print(f"\n  Task: {task.id}")
    print(f"  Scenario: {str(task.user_scenario)[:120]}...")

    scripted_messages = [
        str(task.user_scenario),
        "Yes, please go ahead and help me with that.",
        "Can you check the status of my request?",
        "Thank you, that's all I need.",
    ]

    tau2_messages = []
    error_occurred = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, msg in enumerate(scripted_messages, 1):
            resp = await client.post(
                f"{gateway_addr}/v1/responses",
                json={
                    "input": [{"type": "message", "content": msg}],
                    "model": "tau2-agent",
                    "user": session_key,
                },
            )
            data = resp.json()

            tau2_messages.append(
                UserMessage(role="user", content=msg, turn_idx=len(tau2_messages))
            )

            if data.get("status") == "completed":
                agent_text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                agent_text += block["text"]
                                print(f"    [Turn {i}] Agent: {block['text'][:150]}")
                    elif item.get("type") == "function_call":
                        print(f"    [Turn {i}] [tool] {item.get('name', '')}")

                tau2_messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=agent_text or "(no response)",
                        turn_idx=len(tau2_messages),
                    )
                )
            elif data.get("error"):
                err = data["error"].get("message", "")[:100]
                print(f"    [Turn {i}] Error: {err}")
                tau2_messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=f"Error: {err}",
                        turn_idx=len(tau2_messages),
                    )
                )
                error_occurred = True
                break

    reward = 0.0
    if not error_occurred:
        try:
            simulation = SimulationRun(
                id=f"demo-{task.id}",
                task_id=task.id,
                messages=tau2_messages,
                start_time="",
                end_time="",
                duration=0.0,
                termination_reason=TerminationReason.USER_STOP,
            )
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
            )
            reward = reward_info.reward
        except Exception as e:
            print(f"    Eval error: {e}")

    print(f"    Reward: {reward:.3f}")
    return reward


async def run_demo(gateway_addr: str, domain: str, full: bool) -> None:
    from tau2.registry import registry

    print(f"\n{'=' * 60}")
    print(f"  Tau2 Agent Service Demo — domain: {domain}")
    print(f"{'=' * 60}")

    tasks = registry.get_tasks_loader(domain)(None)
    total = len(tasks)

    if not full:
        tasks = tasks[:1]
        print(f"  Running 1 task (use --full for all {total} tasks)")
    else:
        print(f"  Running all {total} tasks")

    rewards = []
    for task in tasks:
        reward = await run_task(gateway_addr, task, domain)
        rewards.append((task.id, reward))

    print(f"\n{'=' * 60}")
    print(f"  Results — {len(rewards)} task(s)")
    print(f"{'=' * 60}")
    for task_id, reward in rewards:
        print(f"  Task {task_id}: reward = {reward:.3f}")
    if rewards:
        avg = sum(r for _, r in rewards) / len(rewards)
        print(f"\n  Average reward: {avg:.3f}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tau2 Agent Service Demo")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Config YAML path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--domain",
        choices=["airline", "retail", "telecom"],
        help="Override tau2.domain from config",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tasks (default: single task)",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    tau2_cfg = config.setdefault("tau2", {})

    domain = args.domain or tau2_cfg.get("domain", "airline")
    tau2_cfg["domain"] = domain

    data_dir = tau2_cfg.get("data_dir") or os.environ.get("TAU2_DATA_DIR")
    if data_dir:
        os.environ["TAU2_DATA_DIR"] = data_dir

    admin_key = config.get("admin_key", "areal-agent-admin")

    router_addr = f"http://127.0.0.1:{ROUTER_PORT}"
    worker_addr = f"http://127.0.0.1:{WORKER_PORT}"
    proxy_addr = f"http://127.0.0.1:{PROXY_PORT}"
    gateway_addr = f"http://127.0.0.1:{GATEWAY_PORT}"

    # 1. Router
    _start_in_thread(create_router_app(admin_key=admin_key), ROUTER_PORT, "router")

    # 2. Worker (Tau2Agent with PydanticAI + tau2 tools)
    def _make_agent_cls():
        from examples.agent_service.agent import Tau2Agent

        class _Configured(Tau2Agent):
            def __init__(self, **kw: Any):
                super().__init__(config=config, **kw)

        return _Configured

    with patch(
        "areal.experimental.agent_service.worker.app.import_from_string",
        return_value=_make_agent_cls(),
    ):
        worker_app = create_worker_app("examples.agent_service.agent.Tau2Agent")
    _start_in_thread(worker_app, WORKER_PORT, "worker")

    # 3. DataProxy
    _start_in_thread(
        create_data_proxy_app(worker_addr=worker_addr), PROXY_PORT, "proxy"
    )

    # 4. Gateway + Bridge
    gw_app = create_gateway_app(router_addr=router_addr, admin_key=admin_key)
    mount_bridge(
        gw_app,
        OpenResponsesBridge(router_addr=router_addr, admin_key=admin_key),
    )
    _start_in_thread(gw_app, GATEWAY_PORT, "gateway")

    # 5. Wait + register
    async def setup():
        await _wait_healthy(f"{router_addr}/health")
        await _wait_healthy(f"{worker_addr}/health")
        await _wait_healthy(f"{proxy_addr}/health")
        await _wait_healthy(f"{gateway_addr}/health")
        from areal.experimental.agent_service.auth import admin_headers

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{router_addr}/register",
                json={"addr": proxy_addr},
                headers=admin_headers(admin_key),
            )

    asyncio.run(setup())
    print("All services started.")

    # 6. Run demo
    asyncio.run(run_demo(gateway_addr, domain=domain, full=args.full))


if __name__ == "__main__":
    main()
