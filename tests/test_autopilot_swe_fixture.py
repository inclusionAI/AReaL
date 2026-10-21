# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import os
import subprocess
import sys

import pytest

from examples.autopilot.elastic.swe.cleanup import wait_for_pod_removal
from examples.autopilot.elastic.swe.prepare_fixture import build_fixture


@pytest.fixture
def dataset_row():
    return {
        "instance_id": "example__project-1",
        "repo": "example/project",
        "version": "1.0",
        "base_commit": "abc123",
        "FAIL_TO_PASS": '["test_bug"]',
        "PASS_TO_PASS": '["test_existing"]',
        "patch": "a gold patch",
        "problem_statement": "Fix the bug",
    }


@pytest.mark.parametrize("key", ["base_commit", "repo", "version", "FAIL_TO_PASS"])
def test_fixture_mismatched_metadata_rejected(dataset_row, key):
    """Do not grade a patch against a different checkout or test set."""
    details = {**dataset_row, "script": "true", key: '["other"]'}

    with pytest.raises(ValueError, match=key):
        build_fixture(dataset_row, details)


def test_fixture_empty_failing_tests_rejected(dataset_row):
    """A smoke test must distinguish a broken checkout from a fixed one."""
    dataset_row["FAIL_TO_PASS"] = "[]"
    details = {**dataset_row, "script": "true"}

    with pytest.raises(ValueError, match="at least one failing test"):
        build_fixture(dataset_row, details)


@pytest.mark.parametrize("resolved,raw_exit", [(False, 0), (True, 1)])
def test_reward_uses_completed_grading_report_not_shell_exit(
    dataset_row, tmp_path, monkeypatch, resolved, raw_exit
):
    """The grading report decides reward even when shell status disagrees."""
    # Supply the external grader's interface without downloading its repository.
    (tmp_path / "swe_model.py").write_text(
        "class SweRun:\n"
        "    def __init__(self, instance, *args):\n"
        "        self.instance_id = instance\n"
    )
    (tmp_path / "grading.py").write_text(
        "def get_logs_eval(spec, path):\n"
        "    return {test: 'PASSED' for test in "
        "spec.FAIL_TO_PASS + spec.PASS_TO_PASS}, True\n"
    )
    (tmp_path / "aenv_evaluation.py").write_text(
        "import json\n"
        "def get_eval_report(spec, path, include_tests_status):\n"
        "    assert spec.FAIL_TO_PASS == ['test_bug']\n"
        "    assert spec.PASS_TO_PASS == ['test_existing']\n"
        "    assert include_tests_status\n"
        "    with open(path) as stream:\n"
        "        resolved = json.load(stream)['resolved']\n"
        "    return {spec.instance_id: {'resolved': resolved, 'tests_status': {}}}\n"
    )
    monkeypatch.setenv("SWE_EVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("SWE_GRADER_PYTHON", sys.executable)
    script = f"echo '{{\"resolved\": {str(resolved).lower()}}}'\nexit {raw_exit}"
    fixture = build_fixture(dataset_row, {**dataset_row, "script": script})
    path = tmp_path / "eval.sh"
    path.write_text(fixture["eval_script"])

    result = subprocess.run(
        ["bash", str(path)], capture_output=True, text=True, env=os.environ, timeout=10
    )

    assert result.returncode == (0 if resolved else 1), result.stderr
    assert f"AUTOPILOT_SWE_RESOLVED={str(resolved).lower()}" in result.stdout


def test_reward_missing_grading_report_fails(dataset_row, tmp_path, monkeypatch):
    """A crashed grader must not produce a successful reward marker."""
    monkeypatch.setenv("SWE_EVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("SWE_GRADER_PYTHON", sys.executable)
    fixture = build_fixture(dataset_row, {**dataset_row, "script": "true"})
    path = tmp_path / "eval.sh"
    path.write_text(fixture["eval_script"])

    result = subprocess.run(
        ["bash", str(path)], capture_output=True, text=True, timeout=10
    )

    assert result.returncode != 0
    assert "AUTOPILOT_SWE_RESOLVED=" not in result.stdout


@pytest.mark.parametrize("found,missing_test", [(False, False), (True, True)])
def test_reward_incomplete_test_coverage_fails(
    dataset_row, tmp_path, monkeypatch, found, missing_test
):
    """Missing test markers or reference tests cannot count as an empty-patch pass."""
    (tmp_path / "swe_model.py").write_text(
        "class SweRun:\n"
        "    def __init__(self, instance, *args):\n"
        "        self.instance_id = instance\n"
    )
    (tmp_path / "aenv_evaluation.py").write_text(
        "def get_eval_report(*args, **kwargs):\n"
        "    raise AssertionError('Incomplete logs must be rejected before grading')\n"
    )
    observed = {"test_existing": "PASSED"}
    if not missing_test:
        observed["test_bug"] = "FAILED"
    (tmp_path / "grading.py").write_text(
        f"def get_logs_eval(*args):\n    return {observed!r}, {found!r}\n"
    )
    monkeypatch.setenv("SWE_EVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("SWE_GRADER_PYTHON", sys.executable)
    fixture = build_fixture(dataset_row, {**dataset_row, "script": "true"})
    path = tmp_path / "eval.sh"
    path.write_text(fixture["eval_script"])

    result = subprocess.run(
        ["bash", str(path)], capture_output=True, text=True, timeout=10
    )

    assert result.returncode != 0
    assert "AUTOPILOT_SWE_RESOLVED=" not in result.stdout
    assert "did not report every reference test" in result.stderr


@pytest.mark.parametrize("exit_code", [0, 1])
def test_sandbox_removal_query_failure_is_not_success(tmp_path, exit_code):
    """API authorization/network errors must not masquerade as a removed pod."""
    command = tmp_path / "kubectl"
    command.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('invalid response')\n"
        "print('query failed', file=sys.stderr)\n"
        f"sys.exit({exit_code})\n"
    )
    command.chmod(0o755)
    args = argparse.Namespace(
        kubectl=command, kubeconfig=tmp_path / "config", sandbox_namespace="sandbox"
    )

    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(wait_for_pod_removal(args, "example-pod"))
