# SPDX-License-Identifier: Apache-2.0

"""Check real SWE sandbox state, command failures, and empty/gold patch rewards."""

import argparse
import asyncio
import hashlib
import json
import shlex
from pathlib import Path

from examples.autopilot.elastic.swe.cleanup import wait_for_pod_removal


async def run(args: argparse.Namespace) -> None:
    from aweagent.envs.swe import AenvSWE

    fixture = json.loads(args.fixture.read_text())
    records = []
    try:
        for mode, expected in (("empty", 0.0), ("gold", 1.0)):
            env = AenvSWE(
                data_id=fixture["instance_id"],
                data=fixture,
                faas_image=fixture["instance_id"],
                aenv_version=args.environment,
            )
            record = {"mode": mode, "expected_reward": expected}
            records.append(record)
            try:
                if not await env.check_env():
                    raise RuntimeError(f"Sandbox initialization failed for {mode}")
                record["instance_id"] = env.env_info["instance_id"]
                setup = await env.execute_bash(
                    f"cd {shlex.quote(env.config.repo_dir)}; "
                    "export AUTOPILOT_SHELL_PROBE=state_preserved"
                )
                check = await env.execute_bash(
                    'test "$AUTOPILOT_SHELL_PROBE" = state_preserved && '
                    f'test "$PWD" = {shlex.quote(env.config.repo_dir)}'
                )
                failure = await env.execute_bash("false")
                if setup["returncode"] or check["returncode"]:
                    raise RuntimeError("Shell did not preserve its working state")
                if failure["returncode"] == 0:
                    raise RuntimeError("A failing command was reported as successful")
                record["shell_state_preserved"] = True
                record["command_failure_preserved"] = True

                patch = fixture["patch"] if mode == "gold" else ""
                if patch:
                    marker = "PATCH_" + hashlib.sha256(patch.encode()).hexdigest()
                    if marker in patch:
                        raise ValueError("Patch collides with its heredoc delimiter")
                    applicability = await env.execute_bash(
                        f"git apply --check - <<'{marker}'\n{patch}\n{marker}"
                    )
                    if applicability["returncode"]:
                        raise RuntimeError(
                            f"Gold patch is not applicable: {applicability}"
                        )
                reward, output = await env.get_reward(patch=patch, return_output=True)
                record["reward"] = reward
                record["output"] = output
                marker = "AUTOPILOT_SWE_RESOLVED=" + str(bool(expected)).lower()
                if marker not in output:
                    raise RuntimeError(
                        "Reward lacks a matching completed grading report"
                    )
                if reward != expected:
                    raise RuntimeError(
                        f"{mode} reward was {reward}, expected {expected}; "
                        f"grading output: {output[-4000:]}"
                    )
            finally:
                await env.release()
                if record.get("instance_id"):
                    await wait_for_pod_removal(args, record["instance_id"])
                    record["sandbox_removed"] = True
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"instance_id": fixture["instance_id"], "checks": records}, indent=2
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--kubectl", required=True, type=Path)
    parser.add_argument("--kubeconfig", required=True, type=Path)
    parser.add_argument("--sandbox-namespace", default="aenv-sandbox")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
