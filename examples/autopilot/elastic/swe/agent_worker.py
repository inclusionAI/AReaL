# SPDX-License-Identifier: Apache-2.0

"""Run pinned upstream SWEAgent with an isolated sandbox SDK process."""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_grading(reward: float, output: str, failures: int) -> None:
    """A caught SDK/transport failure is not a legitimate negative reward."""
    if failures or reward not in (0.0, 1.0):
        raise RuntimeError("Sandbox infrastructure failed or reward is invalid")
    expected = "AUTOPILOT_SWE_RESOLVED=" + str(bool(reward)).lower()
    other = "AUTOPILOT_SWE_RESOLVED=" + str(not bool(reward)).lower()
    if expected not in output or other in output:
        raise RuntimeError("Reward lacks an unambiguous completed grading report")


async def run(args: argparse.Namespace, credentials: dict) -> None:
    from aweagent.agents.swe import SWEAgent
    from process_env import SandboxProcess

    data = json.loads(args.fixture.read_text())
    data.pop("patch", None)
    report = {"status": "running", "task_id": data["instance_id"]}
    agent = None
    with args.report.with_suffix(".sandbox.log").open("w") as diagnostics:
        env = SandboxProcess(
            str(args.sandbox_python),
            stderr=diagnostics,
            ownership_file=args.report.with_suffix(".sandbox.json"),
        )
        try:
            await env.start(data, args.environment)
            report["instance_id"] = env.instance_id
            write_report(args.report, report)
            agent = SWEAgent(
                env,
                config_name=str(args.config.resolve().with_suffix("")),
                base_url=credentials["base_url"],
                api_key=credentials["api_key"],
            )
            result = await asyncio.wait_for(
                agent.run(data["problem_statement"]),
                timeout=args.agent_timeout,
            )
            report["agent_result"] = result
            if not result["total_model_calls"] or not any(
                call["method"] in ("execute_bash", "search_replace")
                for call in env.calls
            ):
                raise RuntimeError(
                    "Agent did not execute a model-selected repository tool"
                )
            final = result["final_answer"]
            if final is None:
                # An explicitly bounded rollout still gets its actual patch graded.
                final = await env.submit_final_solution()
            if final["returncode"]:
                raise RuntimeError("Could not extract the agent's patch")
            reward, output = await env.request(
                "get_reward",
                patch=final["output"],
                return_output=True,
            )
            validate_grading(reward, output, env.tool_call_failed_count)
            report.update(
                status="passed",
                reward=reward,
                grading_output=output,
                patch=final["output"],
            )
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report.update(
                instance_id=env.instance_id,
                tool_call_failed_count=env.tool_call_failed_count,
                calls=env.calls,
                messages=agent.messages if agent else [],
                tools=agent.tools_schema if agent else [],
                token_usage=dict(agent.token_usage) if agent else {},
            )
            try:
                write_report(args.report, report)
            finally:
                try:
                    await env.close()
                    report["sdk_process_exited"] = True
                except BaseException as error:
                    report.update(status="failed", cleanup_error=str(error))
                    raise
                finally:
                    write_report(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sandbox-python", required=True, type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--agent-timeout", type=float, default=600)
    args = parser.parse_args()
    # Session credentials never appear in process arguments or report artifacts.
    credentials = json.loads(sys.stdin.readline())
    try:
        asyncio.run(run(args, credentials))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
