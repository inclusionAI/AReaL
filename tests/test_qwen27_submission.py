"""CPU-only tests for shared source staging and simulated Slurm submission."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.swe.rl.qwen38_27b.source_snapshot import PROFILE, stage, verify

REPO = Path(__file__).resolve().parents[1]
RECIPE = Path("examples/swe/rl/qwen38_27b")


@pytest.fixture
def checkout(tmp_path):
    """Use existing Git objects; no commits or changes to the user's repository."""
    source = tmp_path / "local"
    recipe = source / RECIPE
    recipe.mkdir(parents=True)
    for filename in (
        "submit.sh",
        "runtime.sh",
        "train.sbatch",
        "train.yaml",
        "source_snapshot.py",
    ):
        shutil.copy2(REPO / RECIPE / filename, recipe / filename)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    objects = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "objects",
        ],
        text=True,
    ).strip()
    alternates = source / ".git/objects/info/alternates"
    alternates.parent.mkdir(exist_ok=True)
    alternates.write_text(objects + "\n")
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(["git", "-C", str(source), "update-ref", "HEAD", head], check=True)
    subprocess.run(["git", "-C", str(source), "add", "examples"], check=True)
    (source / ".git/info/exclude").write_text("site.env\nignored.py\n")
    (source / "examples/new_runtime.py").write_text("answer = 42\n")
    (source / "scratch.zip").write_bytes(b"do not copy root scratch")
    (source / "examples/ignored.py").write_text("ignored\n")
    (source / "examples/unrelated.env").write_text("SECRET=not-for-this-run\n")
    profile = source / PROFILE
    profile.write_text("QWEN_REPO=/a/local/path\nTEST_CREDENTIAL=dummy\n")
    return source, profile, tmp_path / "shared" / "trial" / "source"


def test_snapshot_dirty_source_and_new_files_are_copied(checkout):
    """Tracked edits and new runtime files survive without .git or extra secrets."""
    source, profile, destination = checkout
    revision = stage(source, profile, destination)
    verify(destination, revision)
    assert (destination / "code/examples/new_runtime.py").read_text() == "answer = 42\n"
    assert (destination / "code" / PROFILE).read_bytes() == profile.read_bytes()
    assert not (destination / "code/.git").exists()
    assert not (destination / "code/scratch.zip").exists()
    assert not (destination / "code/examples/unrelated.env").exists()
    assert not (destination / "code/examples/ignored.py").exists()
    assert "dummy" not in (destination / "manifest.json").read_text()


def test_snapshot_repeated_submission_reuses_but_never_overwrites(checkout):
    """A new local edit cannot overwrite a snapshot used by an earlier submission."""
    source, profile, destination = checkout
    original = stage(source, profile, destination)
    assert stage(source, profile, destination) == original
    (source / "examples/new_runtime.py").write_text("answer = 43\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        stage(source, profile, destination)
    verify(destination, original)


@pytest.mark.parametrize("mutation", ["source", "profile", "extra"])
def test_snapshot_remote_mutation_fails_verification(checkout, mutation):
    """The controller detects modified source, changed credentials, or added code."""
    source, profile, destination = checkout
    revision = stage(source, profile, destination)
    target = {
        "source": Path("examples/new_runtime.py"),
        "profile": PROFILE,
        "extra": Path("extra.py"),
    }[mutation]
    (destination / "code" / target).write_text("changed\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify(destination, revision)


def test_snapshot_excludes_generated_bytecode(checkout):
    """Normal interpreter caches do not invalidate otherwise identical snapshots."""
    source, profile, destination = checkout
    revision = stage(source, profile, destination)
    cache = destination / "code/examples/__pycache__"
    cache.mkdir()
    (cache / "test.pyc").write_bytes(b"cache")
    verify(destination, revision)


def test_snapshot_rejects_destination_inside_checkout(checkout):
    source, profile, _ = checkout
    with pytest.raises(ValueError, match="overlap"):
        stage(source, profile, source / "output")


@pytest.mark.parametrize("sync_only", [False, True])
def test_submit_stages_profile_and_pins_shared_paths(checkout, tmp_path, sync_only):
    """Use fake sudo/sbatch: no real privileges or jobs, including --sync-only."""
    source, profile, snapshot = checkout
    dependencies = tmp_path / "dependencies"
    for suffix in (
        "model/config.json",
        "awex/awex/models/qwen3_5.py",
        "dense/dependencies/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py",
    ):
        target = dependencies / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    run = tmp_path / "runs" / "trial"
    output = tmp_path / "outputs"
    profile.write_text(
        "\n".join(
            [
                f"QWEN_REPO={source}",
                f"QWEN_RUN_DIR={run}",
                f"QWEN_OUTPUT_ROOT={output}",
                f"QWEN_SYNC_DIR={snapshot}",
                "QWEN_EXPERIMENT_NAME=test-exp",
                "QWEN_ARENA_TRIAL=test-trial",
                "QWEN_CONTROLLER_NODE=test-node",
                "QWEN_PARTITION=test",
                "QWEN_RESERVATION=test-reservation",
                "QWEN_NODELIST=",
                f"QWEN_MODEL={dependencies}/model",
                f"AWEX_RUNTIME_ROOT={dependencies}/awex",
                f"QWEN_DENSE_RUNTIME={dependencies}/dense",
                *[
                    f"{key}=dummy"
                    for key in (
                        "QWEN_ACTOR_IMAGE",
                        "QWEN_ROLLOUT_IMAGE",
                        "QWEN_MOUNTS",
                        "QWEN_CONTROLLER_MOUNTS",
                        "QWEN_ACTOR_EXTRA_PYTHONPATH",
                        "QWEN_ROLLOUT_RUNTIME_DEPS",
                        "QWEN_PROXY_DEPS",
                        "MEGATRON_ROOT",
                        "ARENA_OPENAPI_BASE",
                        "ARENA_OPENAPI_TOKEN",
                        "ARENA_LLM_API_KEY",
                        "WANDB_BASE_URL",
                        "WANDB_API_KEY",
                    )
                ],
                "",
            ]
        )
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sudo = bindir / "sudo"
    sudo.write_text('#!/bin/bash\nexec "$@"\n')
    sudo.chmod(0o755)
    sbatch = bindir / "sbatch"
    receipt = tmp_path / "submission.json"
    # Literal paths are safe here: pytest temp paths contain no shell metacharacters.
    sbatch.write_text(
        "#!/usr/bin/env python3\nimport os,json,sys\n"
        "keys=['QWEN_REPO','QWEN_ENV_FILE','USER','LOGNAME','QWEN_RUN_DIR','QWEN_RUNTIME_SCRIPT']\n"
        f"open({str(receipt)!r},'w').write(json.dumps({{'args':sys.argv[1:],'env':{{k:os.environ[k] for k in keys}}}}))\n"
    )
    sbatch.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("QWEN_", "SLURM_"))
    }
    env.update(
        PATH=f"{bindir}:{os.environ['PATH']}",
        QWEN_RUN_USER="admin",
        BASH_ENV="/dev/null",
        ENV="/dev/null",
        TMPDIR=str(tmp_path),
    )
    command = ["bash", str(source / RECIPE / "submit.sh")]
    if sync_only:
        command.append("--sync-only")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (snapshot / "code" / PROFILE).is_file()
    assert "TEST_CREDENTIAL" not in result.stdout
    if sync_only:
        assert not receipt.exists()
        assert not (run / "train.yaml").exists()
        return
    submitted = json.loads(receipt.read_text())
    shared_repo = str(snapshot / "code")
    assert submitted["env"]["QWEN_REPO"] == shared_repo
    assert submitted["env"]["QWEN_ENV_FILE"] == str(snapshot / "code" / PROFILE)
    assert submitted["env"]["USER"] == submitted["env"]["LOGNAME"] == "admin"
    assert submitted["env"]["QWEN_RUN_DIR"] == str(run)
    assert submitted["env"]["QWEN_RUNTIME_SCRIPT"].startswith(shared_repo)
    assert f"--chdir={shared_repo}" in submitted["args"]
    assert any("/logs/admin/test-exp/test-trial/" in arg for arg in submitted["args"])
    assert submitted["args"][-1] == str(snapshot / "code" / RECIPE / "train.sbatch")
