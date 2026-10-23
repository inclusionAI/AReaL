# SPDX-License-Identifier: Apache-2.0

"""Drive benchmark episodes through any task-capable v2 Agent Service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from examples.task_workflows.types import Benchmark, TaskResult, get_benchmark


class InboxAudit:
    """Read-only observation of the existing coordinator inbox, never a new queue."""

    def __init__(self, state_root: Path) -> None:
        self.database = state_root / "inbox/queue.sqlite3"

    def _read(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        with sqlite3.connect(
            self.database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5
        ) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(query, params)]

    def episode(self, session_id: str, episode: dict) -> dict[str, Any] | None:
        rows = self._read(
            "SELECT episode_id,source_id,session_id,trajectory_id,status,digest,"
            "source_acked,min_version,max_version,batch_id,rejection "
            "FROM online_episodes WHERE session_id=? AND trajectory_id=?",
            (session_id, episode["trajectory_id"]),
        )
        rows = [
            row
            for row in rows
            if row["source_id"].endswith(":" + episode["incarnation_id"])
        ]
        if len(rows) > 1:
            raise RuntimeError("Ambiguous durable episode identity")
        return rows[0] if rows else None

    def capacity(self, starting: list[dict]) -> bool:
        rows = self._read(
            "SELECT session_id,trajectory_id,source_id,status FROM online_episodes"
        )
        progress = self._read("SELECT * FROM online_progress")[0]
        seen = {
            (
                row["session_id"],
                row["trajectory_id"],
                row["source_id"].rsplit(":", 1)[-1],
            )
            for row in rows
        }
        unseen = sum(
            (
                item["session_id"],
                item["areal_episode"]["trajectory_id"],
                item["areal_episode"]["incarnation_id"],
            )
            not in seen
            for item in starting
        )
        active = sum(row["status"] not in ("published", "rejected") for row in rows)
        return (
            bool(progress["admission_open"])
            and active + unseen
            < (progress["max_staleness"] + 1) * progress["batch_size"]
        )


@dataclass(frozen=True)
class ClientSettings:
    agent_url: str
    inference_url: str
    agent_key: str
    inference_key: str
    model: str
    mode: str
    output: Path
    state_root: Path | None = None
    concurrency: int = 4
    request_timeout: float = 900.0
    handoff_timeout: float = 180.0
    benchmark: str = "dapo_math"

    def __post_init__(self) -> None:
        if self.mode not in ("on_demand", "resident", "eval"):
            raise ValueError("Unsupported client mode")
        if self.mode == "on_demand" and (
            self.state_root is None or not self.state_root.is_absolute()
        ):
            raise ValueError(
                "on_demand requires the trainer's absolute state root on this host"
            )
        if min(self.concurrency, self.request_timeout, self.handoff_timeout) <= 0:
            raise ValueError("Concurrency and timeouts must be positive")
        if not self.agent_key or not self.inference_key:
            raise ValueError("Agent and inference API keys are required")


class EpisodeClient:
    def __init__(
        self,
        settings: ClientSettings,
        http: httpx.AsyncClient,
        benchmark: Benchmark | None = None,
    ) -> None:
        self.settings, self.http = settings, http
        self.benchmark = benchmark or get_benchmark(settings.benchmark)
        self._write_lock = asyncio.Lock()
        self.audit = (
            InboxAudit(settings.state_root) if settings.mode == "on_demand" else None
        )
        self._admission_lock = asyncio.Lock()
        self._starting: dict[str, dict] = {}

    async def _post(
        self,
        base: str,
        path: str,
        key: str,
        body: dict,
        *,
        session_key: str | None = None,
    ) -> dict:
        headers = {"Authorization": "Bearer " + key}
        if session_key:
            headers["X-AReaL-Session-Key"] = session_key
        response = await self.http.post(
            base.rstrip("/") + path, headers=headers, json=body
        )
        response.raise_for_status()
        return response.json()

    async def _admit(self, credentials: dict) -> None:
        if self.audit is None:
            return
        deadline = time.monotonic() + self.settings.handoff_timeout
        while time.monotonic() < deadline:
            async with self._admission_lock:
                available = await asyncio.to_thread(
                    self.audit.capacity, list(self._starting.values())
                )
                if available:
                    self._starting[credentials["session_id"]] = credentials
                    return
            await asyncio.sleep(0.2)
        raise TimeoutError(
            "Waiting for episode admission; check trainer progress and batch size"
        )

    async def _handoff(self, credentials: dict) -> dict:
        if self.audit is None:
            return {"verified": False, "reason": "resident_mode_has_no_durable_inbox"}
        deadline = time.monotonic() + self.settings.handoff_timeout
        while time.monotonic() < deadline:
            row = await asyncio.to_thread(
                self.audit.episode,
                credentials["session_id"],
                credentials["areal_episode"],
            )
            if row is not None and row["digest"] and row["source_acked"]:
                return {"verified": True, **row}
            if row is not None and row["status"] == "rejected":
                raise RuntimeError(
                    f"Episode rejected before handoff: {row['rejection']}"
                )
            await asyncio.sleep(0.2)
        raise TimeoutError(
            "Reward was submitted but durable export/source ACK was not confirmed"
        )

    def _task_result(self, response: dict) -> TaskResult:
        return TaskResult.from_response(response)

    async def run_task(self, task: Any) -> dict[str, Any]:
        started = time.time()
        clock = time.perf_counter()
        request_id = self.benchmark.request_prefix + "-" + uuid.uuid4().hex
        credentials = None
        rewarded = False
        agent_finished = False
        result: dict[str, Any] = {
            "task_id": task.task_id,
            "request_id": request_id,
            "started": started,
        }
        try:
            settings = self.settings
            if settings.mode != "eval":
                response = await self._post(
                    settings.inference_url,
                    "/rl/start_session",
                    settings.inference_key,
                    {"task_id": request_id},
                )
                if len(response["sessions"]) != 1:
                    raise RuntimeError("Independent task requests require one session")
                credentials = response["sessions"][0]
                if settings.mode == "on_demand" and not credentials.get(
                    "areal_episode"
                ):
                    raise RuntimeError(
                        "Inference service did not return a durable episode identity"
                    )
                result.update(
                    session_id=credentials["session_id"],
                    episode=credentials.get("areal_episode"),
                )
                await self._admit(credentials)
            task_input = self.benchmark.prepare(task)
            result["submitted"] = time.time()
            payload = {
                "input": [{"type": "message", "content": task_input.message}],
                "model": "task-agent",
                "metadata": {
                    "task_id": task_input.task_id,
                    "task_environment": asdict(task_input.environment),
                },
                "user": request_id,
                "inf_base_url": settings.inference_url.rstrip("/") + "/v1",
                "inf_model": settings.model,
                # Eval uses the admin-authenticated standalone inference path:
                # no RL session, reward callback, or training admission credit.
                "session_api_key": credentials["session_api_key"]
                if credentials
                else settings.inference_key,
            }
            if credentials and credentials.get("areal_episode"):
                payload["inf_episode"] = credentials["areal_episode"]
            response = await self._post(
                settings.agent_url,
                "/v1/responses",
                settings.agent_key,
                payload,
                session_key=request_id,
            )
            agent_finished = True
            task_result = self._task_result(response)
            metadata = response.get("metadata", {})
            # Normalized environment metadata is shared by every benchmark/harness.
            metadata = {**metadata, "environment": task_result.environment}
            result["metadata"] = metadata
            result.update(answer=task_result.answer, answered=time.time())
            result.update(self.benchmark.result_metadata(task))
            score = await asyncio.to_thread(
                self.benchmark.grade, task, task_result, settings.output / request_id
            )
            result["score"] = asdict(score)
            result["scored"] = time.time()
            if credentials:
                reward_body = {"reward": score.reward}
                if credentials.get("areal_episode"):
                    reward_body["areal_episode"] = credentials["areal_episode"]
                receipt = await self._post(
                    settings.inference_url,
                    "/rl/set_reward",
                    credentials["session_api_key"],
                    reward_body,
                )
                rewarded = True
                if receipt.get("trajectory_id") is None or not receipt.get(
                    "trajectory_ready"
                ):
                    raise RuntimeError(
                        "Reward did not seal the expected trajectory; use zero finish timeout"
                    )
                if (
                    credentials.get("areal_episode")
                    and receipt["trajectory_id"]
                    != credentials["areal_episode"]["trajectory_id"]
                ):
                    raise RuntimeError("Reward receipt targets a different trajectory")
                result["reward_receipt"] = receipt
                result["handoff"] = await self._handoff(credentials)
                result["handed_off"] = time.time()
            result["status"] = "completed"
            return result
        except Exception as exc:
            result.update(
                status="infrastructure_error", error=f"{type(exc).__name__}: {exc}"
            )
            # No generation retries. A known-finished unscored episode may be
            # explicitly aborted; uncertain in-flight work is left for inspection.
            if (
                credentials
                and credentials.get("areal_episode")
                and agent_finished
                and not rewarded
            ):
                try:
                    result["abort_receipt"] = await self._post(
                        self.settings.inference_url,
                        "/rl/abort_episode",
                        credentials["session_api_key"],
                        {
                            **credentials["areal_episode"],
                            "reason": "task_client_infrastructure_error",
                        },
                    )
                except Exception as abort_error:
                    result["abort_error"] = type(abort_error).__name__
            return result
        finally:
            if credentials:
                self._starting.pop(credentials["session_id"], None)
            if agent_finished:
                try:
                    await self._post(
                        self.settings.agent_url,
                        "/sessions/close",
                        self.settings.agent_key,
                        {"session_key": request_id},
                    )
                except Exception as close_error:
                    result["agent_cleanup_error"] = type(close_error).__name__
            result["finished"] = time.time()
            metadata = result.get("metadata", {})
            inference = metadata.get("inference_calls")
            tool_metadata = metadata.get("environment", {})
            tools = tool_metadata.get("timing_calls", tool_metadata.get("calls"))
            result["timing"] = {
                "total_seconds": time.perf_counter() - clock,
                "inference_seconds": sum(call["seconds"] for call in inference)
                if inference is not None
                and all(call["seconds"] is not None for call in inference)
                else None,
                "tool_seconds": sum(
                    call["finished"] - call["started"] for call in tools
                )
                if tools is not None
                and all(call["finished"] is not None for call in tools)
                else None,
            }

    async def run(self, tasks: list[Any]) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(self.settings.concurrency)
        failed = asyncio.Event()
        progress = tqdm(total=len(tasks), desc="Task episodes", unit="episode")

        async def one(task: Any) -> dict:
            async with semaphore:
                if failed.is_set():
                    row = {"task_id": task.task_id, "status": "not_submitted"}
                else:
                    row = await self.run_task(task)
                    if row["status"] != "completed":
                        failed.set()
                async with self._write_lock:
                    await asyncio.to_thread(self._append_episode, row)
                progress.update(1)
                return row

        try:
            rows = await asyncio.gather(*(one(task) for task in tasks))
        finally:
            progress.close()
        completed = [row for row in rows if row["status"] == "completed"]
        return {
            "mode": self.settings.mode,
            "tasks": len(rows),
            "completed": len(completed),
            "correct": sum(row["score"]["reward"] == 1 for row in completed),
            "infrastructure_errors": sum(
                row["status"] == "infrastructure_error" for row in rows
            ),
            "not_submitted": sum(row["status"] == "not_submitted" for row in rows),
            "durable_handoffs": sum(
                row.get("handoff", {}).get("verified", False) for row in completed
            ),
            "model_calls": sum(row["metadata"]["api_calls"] for row in completed),
            "tool_executions": sum(
                len(row["metadata"]["environment"]["calls"]) for row in completed
            ),
        }

    def _append_episode(self, row: dict[str, Any]) -> None:
        with (self.settings.output / "episodes.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _main(args: argparse.Namespace) -> int:
    benchmark = get_benchmark(getattr(args, "benchmark", "dapo_math"))
    tasks, manifest = await asyncio.to_thread(benchmark.load_tasks, args.tasks)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(tasks):
            raise ValueError("Limit must fit the frozen task file")
        tasks = tasks[: args.limit]
    settings = ClientSettings(
        agent_url=args.agent_url.rstrip("/"),
        inference_url=args.inference_url.rstrip("/"),
        agent_key=os.environ["AREAL_AGENT_ADMIN_KEY"],
        inference_key=os.environ["AREAL_INFERENCE_ADMIN_KEY"],
        model=args.model,
        mode=args.mode,
        output=args.output,
        state_root=args.state_root,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        handoff_timeout=args.handoff_timeout,
        benchmark=benchmark.name,
    )
    settings.output.mkdir(parents=True, exist_ok=False)
    (settings.output / "tasks-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (settings.output / "run.json").write_text(
        json.dumps(
            {
                "mode": settings.mode,
                "benchmark": benchmark.name,
                "tasks": str(args.tasks.resolve()),
                "limit": args.limit,
                "model": settings.model,
                "concurrency": settings.concurrency,
                "task_ids": [task.task_id for task in tasks],
            },
            indent=2,
        )
        + "\n"
    )
    async with httpx.AsyncClient(timeout=settings.request_timeout) as http:
        summary = await EpisodeClient(settings, http, benchmark).run(tasks)
    (settings.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return int(bool(summary["infrastructure_errors"] or summary["not_submitted"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=("dapo_math", "swe_rebench"), default="dapo_math"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--agent-url", required=True, help="Agent gateway origin, without /v1"
    )
    parser.add_argument(
        "--inference-url", required=True, help="Inference gateway origin, without /v1"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode", choices=("on_demand", "resident", "eval"), default="resident"
    )
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--handoff-timeout", type=float, default=180)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
