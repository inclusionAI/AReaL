# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


def _run_provisioning(
    tmp_path: Path, failed_token_request: int = 0
) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    workflow = yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows/test-areal.yml").read_text()
    )
    script = next(
        step["run"]
        for step in workflow["jobs"]["provision-runner"]["steps"]
        if step.get("id") == "create-instance"
    )
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    mock = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json
        import os
        import sys
        from pathlib import Path

        log = Path(os.environ["MOCK_CALLS"])
        events = [json.loads(line) for line in log.read_text().splitlines()]
        if Path(sys.argv[0]).name == "curl":
            request = 1 + sum(event["kind"] == "token" for event in events)
            event = {"kind": "token", "request": request}
            failed = request == int(os.environ["MOCK_FAILED_TOKEN_REQUEST"])
            result = {"token": f"registration-token-{request}"}
        elif sys.argv[1:4] == ["compute", "zones", "list"]:
            print("zone-a\\nzone-b")
            sys.exit(0)
        else:
            assert sys.argv[1:4] == ["compute", "instances", "create"]
            zone = sys.argv[sys.argv.index("--zone") + 1]
            metadata = sys.argv[sys.argv.index("--metadata-from-file") + 1]
            token_file = dict(item.split("=", 1) for item in metadata.split(","))[
                "runner_token"
            ]
            event = {"kind": "create", "zone": zone,
                     "token": Path(token_file).read_text().strip()}
            failed = zone == "zone-a"
            result = None
        with log.open("a") as output:
            output.write(json.dumps(event) + "\\n")
        if failed:
            sys.exit(1)
        if result is not None:
            print(json.dumps(result))
        """
    )
    for command in ("curl", "gcloud"):
        path = mock_bin / command
        path.write_text(mock)
        path.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    calls.write_text("")
    (tmp_path / "runner-token.txt").write_text("stale-registration-token")
    env = os.environ | {
        "PATH": f"{mock_bin}{os.pathsep}{os.environ['PATH']}",
        "GH_PAT": "test-pat-must-not-reach-vm",
        "GITHUB_API_URL": "https://api.invalid",
        "GITHUB_REPOSITORY": "test-owner/test-repo",
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        "GCP_PROJECT_ID": "test-project",
        "GCP_OS_IMAGE": "test-image",
        "INSTANCE_NAME": "test-instance",
        "RUNNER_LABELS": "test-labels",
        "MOCK_CALLS": str(calls),
        "MOCK_FAILED_TOKEN_REQUEST": str(failed_token_request),
    }
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, [json.loads(line) for line in calls.read_text().splitlines()]


def test_runner_provisioning_zone_retry_refreshes_registration_token(
    tmp_path: Path,
) -> None:
    """A later GPU allocation attempt must receive its own fresh token."""
    result, events = _run_provisioning(tmp_path)

    assert result.returncode == 0, result.stderr
    assert events == [
        {"kind": "token", "request": 1},
        {"kind": "create", "zone": "zone-a", "token": "registration-token-1"},
        {"kind": "token", "request": 2},
        {"kind": "create", "zone": "zone-b", "token": "registration-token-2"},
    ]
    assert (tmp_path / "github-output").read_text() == "zone=zone-b\n"
    for secret in (
        "test-pat-must-not-reach-vm",
        "registration-token-1",
        "registration-token-2",
    ):
        assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("failed_request", [1, 2])
def test_runner_provisioning_registration_failure_prevents_instance_creation(
    tmp_path: Path, failed_request: int
) -> None:
    """An API failure must abort rather than allocate with a stale token."""
    result, events = _run_provisioning(tmp_path, failed_request)

    assert result.returncode != 0
    assert events[-1] == {"kind": "token", "request": failed_request}
    creates = [event for event in events if event["kind"] == "create"]
    assert len(creates) == failed_request - 1
    assert not (tmp_path / "github-output").exists()
    assert "test-pat-must-not-reach-vm" not in result.stdout + result.stderr
