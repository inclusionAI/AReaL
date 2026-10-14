# SPDX-License-Identifier: Apache-2.0
"""Save-then-exit tests using the production trainer loop with CPU participants."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from unittest.mock import Mock

import pytest

from tests.test_reconfiguration import (
    REPO_ROOT,
    STEP,
    M,
    _bind_channel,
    _source_function,
    _trainer_fixture,
)


def _control():
    return M.TrainerReconfiguration(
        run_id="stop-run",
        runtime_revision="test-revision",
        configuration_id="test-config",
    )


@pytest.fixture
def control():
    channel = _control()
    yield channel
    channel.close()


def _with_lifecycle(trainer, events):
    """Use actual trainer context/cleanup code with fake external resources."""
    namespace = {
        "logger": Mock(),
        "perf_tracer": types.SimpleNamespace(
            save=lambda **kwargs: events.append("trace.close")
        ),
    }
    for name in ("__enter__", "__exit__", "close"):
        setattr(
            type(trainer),
            name,
            _source_function(
                "areal/trainer/rl_trainer.py", "PPOTrainer", name, namespace
            ),
        )
    trainer.saver.finalize = lambda: events.append("saver.close")
    trainer.stats_logger.close = lambda: events.append("stats.close")
    trainer.rollout.destroy = lambda: events.append("rollout.destroy")
    trainer.actor.destroy = lambda: events.append("actor.destroy")
    if trainer.critic is not None:
        trainer.critic.destroy = lambda: events.append("critic.destroy")
    trainer._save_perf_tracer = lambda **kwargs: events.append("trace.step")
    return trainer


def _several_steps(trainer):
    trainer.config.total_train_epochs = 2
    trainer.config.total_train_steps = 3


@pytest.mark.parametrize("awex", [False, True])
@pytest.mark.parametrize("critic", [False, True])
def test_stop_saves_completed_boundary_then_exits_without_resuming(
    monkeypatch, control, tmp_path, awex, critic
):
    """Both native checkpoint placements exit through the usual context cleanup."""
    trainer, events = _trainer_fixture(monkeypatch, awex=awex, critic=critic)
    _several_steps(trainer)
    _with_lifecycle(trainer, events)
    saved = control.request_snapshot(
        tmp_path / "snapshot", state_id="boundary", stop_after_save=True
    )
    saved.add_done_callback(lambda _: events.append("saved"))
    with trainer:
        result = trainer.train(workflow=object(), reconfiguration=control)
        assert isinstance(result, M.SnapshotStopResult)
        assert trainer.rollout.frozen
        assert "actor.destroy" not in events
        events.append("train.returned")
    assert result.snapshot == saved.result()
    assert result.context.step_info == STEP
    assert result.context.policy_version == 1
    manifest = M.inspect_snapshot(result.snapshot)
    assert manifest["context"]["state_id"] == "boundary"
    assert events.count("actor.update") == 1
    if critic:
        assert events.count("critic.update") == 1
        assert events.index("critic.destroy") < events.index("actor.destroy")
    assert "rollout.release" not in events
    assert "rollout.resume" not in events
    assert events.index("stats") < events.index("saved") < events.index("trace.step")
    assert events.index("train.returned") < events.index("saver.close")
    assert events.index("rollout.destroy") < events.index("actor.destroy")
    assert events[-1] == "trace.close"


def test_ordinary_snapshot_keeps_training_by_default(monkeypatch, control, tmp_path):
    trainer, events = _trainer_fixture(monkeypatch)
    _several_steps(trainer)
    saved = control.request_snapshot(tmp_path / "snapshot", state_id="continue")
    assert trainer.train(workflow=object(), reconfiguration=control) is None
    assert saved.result().path == str(tmp_path / "snapshot")
    assert events.count("actor.update") == events.count("rollout.resume") == 3
    assert events.count("rollout.release") == 1


def test_stop_callback_cannot_submit_or_reenter_training(
    monkeypatch, control, tmp_path
):
    """Closing the channel before Future callbacks prevents an unserviceable save."""
    trainer, events = _trainer_fixture(monkeypatch)
    _several_steps(trainer)
    saved = control.request_snapshot(
        tmp_path / "snapshot", state_id="stop", stop_after_save=True
    )
    observed = []

    def completed(future):
        assert future.result().path == str(tmp_path / "snapshot")
        assert trainer.rollout.frozen
        with pytest.raises(M.SnapshotError, match="closed"):
            control.request_snapshot(tmp_path / "after", state_id="after")
        with pytest.raises(RuntimeError, match="already running"):
            trainer.train(workflow=object())
        observed.append("terminal")

    saved.add_done_callback(completed)
    assert isinstance(
        trainer.train(workflow=object(), reconfiguration=control), M.SnapshotStopResult
    )
    assert observed == ["terminal"]
    assert events.count("actor.update") == 1
    assert not (tmp_path / "after").exists()


def test_stopped_trainer_rejects_reuse_and_settles_new_request(
    monkeypatch, control, tmp_path
):
    trainer, events = _trainer_fixture(monkeypatch)
    control.request_snapshot(
        tmp_path / "snapshot", state_id="stop", stop_after_save=True
    )
    trainer.train(workflow=object(), reconfiguration=control)
    new_channel = _control()
    future = new_channel.request_snapshot(tmp_path / "reuse", state_id="reuse")
    with pytest.raises(RuntimeError, match="create a new trainer") as error:
        trainer.train(workflow=object(), reconfiguration=new_channel)
    assert future.exception() is error.value
    assert events.count("actor.update") == 1
    assert not (tmp_path / "reuse").exists()
    assert not trainer._train_running


def test_cancelled_stop_request_does_not_stop_training(monkeypatch, control, tmp_path):
    trainer, events = _trainer_fixture(monkeypatch)
    _several_steps(trainer)
    future = control.request_snapshot(
        tmp_path / "cancelled", state_id="cancelled", stop_after_save=True
    )
    assert future.cancel()
    assert trainer.train(workflow=object(), reconfiguration=control) is None
    assert events.count("actor.update") == events.count("rollout.resume") == 3
    assert "rollout.drain" not in events
    assert not (tmp_path / "cancelled").exists()


@pytest.mark.parametrize("last_step", [False, True])
def test_late_stop_waits_for_next_boundary_or_reports_end(
    monkeypatch, control, tmp_path, last_step
):
    trainer, events = _trainer_fixture(monkeypatch)
    if not last_step:
        _several_steps(trainer)
    requests = []
    update = trainer.actor.ppo_update

    def request_during_training(batch):
        if not requests:
            requests.append(
                control.request_snapshot(
                    tmp_path / "late", state_id="late", stop_after_save=True
                )
            )
        update(batch)

    trainer.actor.ppo_update = request_during_training
    result = trainer.train(workflow=object(), reconfiguration=control)
    if last_step:
        assert result is None
        with pytest.raises(M.SnapshotError, match="Training ended"):
            requests[0].result()
        assert not (tmp_path / "late").exists()
    else:
        assert isinstance(result, M.SnapshotStopResult)
        assert result.context.step_info["global_step"] == 1
        assert result.snapshot == requests[0].result()
        assert events.count("actor.update") == 2
    assert events.count("rollout.resume") == 1


@pytest.mark.parametrize("phase", ["drain", "save", "wait", "state", "publish"])
def test_stop_save_failure_never_reports_success_and_still_leaves_context(
    monkeypatch, control, tmp_path, phase
):
    trainer, events = _trainer_fixture(monkeypatch)
    _with_lifecycle(trainer, events)
    root = tmp_path / "failed"
    saved = control.request_snapshot(root, state_id="fail", stop_after_save=True)
    failure = RuntimeError("injected failure")
    if phase == "drain":
        trainer.rollout.freeze_and_drain = Mock(side_effect=failure)
    elif phase == "save":
        trainer.actor.fail_save = True
    elif phase == "wait":
        trainer.actor.fail_wait = True
    elif phase == "state":
        trainer.actor.runtime_state_dict = Mock(side_effect=failure)
    else:
        monkeypatch.setattr(M.os, "replace", Mock(side_effect=OSError("disk failed")))
    with pytest.raises((RuntimeError, M.SnapshotError)):
        with trainer:
            trainer.train(workflow=object(), reconfiguration=control)
    assert saved.done() and saved.exception() is not None
    assert not (root / "manifest.json").exists()
    assert "rollout.resume" not in events
    assert "actor.destroy" in events
    assert not getattr(trainer, "_reconfiguration_stopped", False)


@pytest.mark.parametrize("phase", ["trace", "cleanup"])
def test_post_save_failure_is_not_misreported_as_process_exit(
    monkeypatch, control, tmp_path, phase
):
    """A durable saved Future remains valid even if later tracing/cleanup fails."""
    trainer, events = _trainer_fixture(monkeypatch)
    _with_lifecycle(trainer, events)
    saved = control.request_snapshot(
        tmp_path / "snapshot", state_id="stop", stop_after_save=True
    )
    failure = RuntimeError(f"{phase} failed")
    if phase == "trace":
        trainer._save_perf_tracer = Mock(side_effect=failure)
    else:
        trainer.saver.finalize = Mock(side_effect=failure)
    with pytest.raises(RuntimeError) as error:
        with trainer:
            trainer.train(workflow=object(), reconfiguration=control)
    assert error.value is failure
    assert M.inspect_snapshot(saved.result())["context"]["state_id"] == "stop"
    assert trainer.rollout.frozen
    assert "rollout.resume" not in events
    assert trainer._reconfiguration_stopped


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_stop_flag_requires_explicit_boolean(control, tmp_path, invalid):
    with pytest.raises(TypeError, match="must be a bool"):
        control.request_snapshot(
            tmp_path / "invalid", state_id="stop", stop_after_save=invalid
        )
    assert not control.has_active_snapshot
    assert not list(tmp_path.iterdir())


def test_standalone_stop_result_is_published_and_terminal(control, tmp_path):
    _bind_channel(control)
    future = control.request_snapshot(
        tmp_path / "snapshot", state_id="stop", stop_after_save=True
    )
    control.begin_step_snapshot(step_info=STEP)
    assert not future.cancel()
    assert not future.done()
    result = control.finish_step_snapshot(policy_version=1)
    assert isinstance(result, M.SnapshotStopResult)
    assert result.snapshot == future.result()
    assert not control.has_active_snapshot
    with pytest.raises(M.SnapshotError, match="closed"):
        control.request_snapshot(tmp_path / "again", state_id="again")


def test_save_then_exit_process_closes_cpu_workers(tmp_path):
    """A normal entry process returns, cleans up real CPU children and exits 0.

    Worker destruction is supplied by the test; this verifies trainer control
    flow and process exit, not Local/Ray/Slurm termination or GPU checkpoints.
    """
    script = textwrap.dedent("""
        import json
        import subprocess
        import sys
        from pathlib import Path
        import pytest
        from tests.test_reconfiguration import _trainer_fixture
        from tests.test_reconfiguration_stop import _control, _several_steps, _with_lifecycle

        children = []
        try:
            with pytest.MonkeyPatch.context() as patcher:
                trainer, events = _trainer_fixture(patcher)
                _several_steps(trainer)
                _with_lifecycle(trainer, events)
                for role in ('actor', 'rollout'):
                    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                    children.append(child)
                    def destroy(child=child, role=role):
                        child.terminate()
                        child.wait(timeout=5)
                        events.append(role + '.destroy')
                    getattr(trainer, role).destroy = destroy
                channel = _control()
                saved = channel.request_snapshot(Path(sys.argv[1]) / 'snapshot', state_id='stop', stop_after_save=True)
                with trainer:
                    result = trainer.train(workflow=object(), reconfiguration=channel)
                assert result.snapshot == saved.result()
                assert all(child.poll() is not None for child in children)
                print(json.dumps({'updates': events.count('actor.update'), 'resumes': events.count('rollout.resume'), 'destroyed': sum(event.endswith('.destroy') for event in events)}))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"updates": 1, "resumes": 0, "destroyed": 2}
    assert (tmp_path / "snapshot" / "manifest.json").is_file()
