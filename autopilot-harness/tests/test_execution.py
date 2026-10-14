import errno
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from autopilot_harness.execution import (
    LaunchSpec,
    RestartCoordinator,
    RestartError,
    Timeouts,
)
from autopilot_harness.execution.process import LocalProcess
from autopilot_harness.execution.protocol import (
    ProtocolError,
    check_message,
    message,
    publish,
)
from autopilot_harness.execution.runtime import _check_local_workers
from autopilot_harness.monitor.outputs import HistoryStore
from autopilot_harness.validator import configuration_id

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = {"steps": 8, "delay": 0.08}


@unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, fault="", **timeouts):
        spec = LaunchSpec(
            (sys.executable, "examples/execution/cpu_runtime.py"),
            PROJECT,
            "test-run",
            "fixture-runtime",
            configuration_id(CONFIG),
            {
                "PYTHONPATH": str(PROJECT / "src"),
                "AUTOPILOT_FIXTURE_FAULT": fault,
                "EXAMPLE_API_KEY": "must-not-enter-history",
            },
        )
        options = dict(startup=8, snapshot=8, restore=8, exit=3, poll=0.01)
        options.update(timeouts)
        runner = RestartCoordinator(
            spec, self.root / "run", timeouts=Timeouts(**options)
        )
        self.addCleanup(runner.close)
        return runner

    def events(self, runner):
        return HistoryStore(runner.directory / "events.jsonl").read(run_id="test-run")

    def test_restart_orders_exit_before_launch_and_preserves_progress(self):
        runner = self.runner()
        old = runner.start()
        result = runner.restart()
        new = runner.current
        self.assertEqual(result["status"], "resumed")
        self.assertNotEqual(old.process.pgid, new.process.pgid)
        self.assertTrue(old.process.group_gone())
        self.assertEqual(old.process.poll(), 0)
        complete = runner.wait(timeout=8)
        self.assertEqual(complete["restored"]["snapshot"], result["snapshot"])
        final = json.loads((new.directory / "fixture-final.json").read_text())
        self.assertEqual(final, {"count": 8, "restored": True})
        self.assertTrue(new.process.group_gone())
        events = self.events(runner)
        exited = next(i for i, e in enumerate(events) if e["kind"] == "runtime.exited")
        launched = [i for i, e in enumerate(events) if e["kind"] == "runtime.launched"]
        self.assertLess(exited, launched[1])
        self.assertEqual(sum(e["kind"] == "restart.confirmed" for e in events), 1)
        self.assertNotIn(
            "must-not-enter-history", (runner.directory / "events.jsonl").read_text()
        )
        self.assertNotIn(
            "must-not-enter-history", (new.directory / "launch.json").read_text()
        )

    def test_final_boundary_restore_completes_without_waiting_for_admission(self):
        runner = self.runner("last_step")
        runner.start()
        result = runner.restart()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["context"]["step_info"]["global_step"], 7)
        self.assertTrue(runner.current.process.group_gone())
        runner.wait(timeout=2)

    def test_repeated_restarts_use_new_segments_and_snapshot_ids(self):
        runner = self.runner()
        runner.start()
        first = runner.restart()
        second = runner.restart()
        self.assertNotEqual(
            first["snapshot"]["snapshot_id"], second["snapshot"]["snapshot_id"]
        )
        self.assertGreater(
            second["context"]["step_info"]["global_step"],
            first["context"]["step_info"]["global_step"],
        )
        runner.wait(timeout=8)
        self.assertEqual(len(list((runner.directory / "segments").iterdir())), 3)

    def test_save_timeout_preserves_live_source_and_never_launches_replacement(self):
        runner = self.runner("no_snapshot", snapshot=0.15)
        with self.assertRaises(RestartError) as error:
            with runner:
                source = runner.start()
                runner.restart()
        self.assertEqual(error.exception.stage, "save_and_cleanup")
        self.assertIsNone(source.process.poll())
        self.assertIs(runner.retained_source, source)
        self.assertTrue(
            any(e["kind"] == "runtime.retained" for e in self.events(runner))
        )
        self.assertEqual(len(list((runner.directory / "segments").iterdir())), 1)
        with self.assertRaises(RuntimeError):
            runner.restart()

    def test_zero_leader_exit_does_not_substitute_for_group_exit(self):
        runner = self.runner(exit=0.15)
        source = runner.start()
        with patch.object(source.process, "group_gone", return_value=False):
            with self.assertRaises(RestartError) as error:
                runner.restart()
        self.assertEqual(error.exception.stage, "old_group_exit")
        self.assertEqual(source.process.poll(), 0)
        self.assertEqual(len(list((runner.directory / "segments").iterdir())), 1)

    def test_exit_race_requires_observed_zero_status(self):
        runner = self.runner()
        handle = Mock()
        handle.process.poll.side_effect = [None, 1]
        handle.process.group_gone.return_value = True
        with self.assertRaisesRegex(RuntimeError, "unsuccessfully"):
            runner._wait_exit(handle)
        handle.process.group_gone.assert_not_called()

    def test_cleanup_failure_and_corrupt_manifest_prevent_replacement(self):
        for fault in ("cleanup", "corrupt"):
            with self.subTest(fault=fault):
                # Each coordinator owns a fresh directory.
                self.root = Path(self.temp.name) / fault
                runner = self.runner(fault)
                runner.start()
                with self.assertRaises(RestartError):
                    runner.restart()
                self.assertEqual(
                    len(list((runner.directory / "segments").iterdir())), 1
                )
                self.assertFalse(
                    any(e["kind"] == "restart.confirmed" for e in self.events(runner))
                )
                runner.close()

    def test_restore_failure_and_code_drift_retain_snapshot_without_success(self):
        for fault in ("restore", "source_drift"):
            with self.subTest(fault=fault):
                self.root = Path(self.temp.name) / fault
                runner = self.runner(fault)
                old = runner.start()
                with self.assertRaises(RestartError) as error:
                    runner.restart()
                self.assertEqual(error.exception.stage, "restore")
                self.assertTrue(old.process.group_gone())
                self.assertTrue(runner.current.process.group_gone())
                self.assertEqual(
                    len(list((runner.directory / "snapshots").glob("*/manifest.json"))),
                    1,
                )
                self.assertFalse(
                    any(e["kind"] == "restart.confirmed" for e in self.events(runner))
                )
                runner.close()

    def test_stale_nonce_receipt_is_rejected(self):
        runner = self.runner()
        source = runner.start()
        stale = message(
            source.launch, "stopped", request_id="old", cleanup_completed=True
        )
        stale["nonce"] = "previous-runtime"
        publish(source.directory / "stopped.json", stale)
        with self.assertRaises(RestartError):
            runner.restart()
        self.assertEqual(len(list((runner.directory / "segments").iterdir())), 1)

    def test_concurrent_operation_is_rejected_before_request(self):
        runner = self.runner()
        source = runner.start()
        runner._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "in progress"):
                runner.restart()
            self.assertFalse((source.directory / "stop_request.json").exists())
        finally:
            runner._lock.release()

    def test_cli_runs_and_waits_for_resumed_job(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "autopilot_harness",
                "restart",
                "--launch",
                "examples/execution/launch.json",
                "--output",
                str(self.root / "cli"),
                "--completion-timeout",
                "8",
            ],
            cwd=PROJECT,
            env=dict(os.environ, PYTHONPATH=str(PROJECT / "src")),
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Training completed", result.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("pytest"),
        "native AReaL fixture requires test-only pytest",
    )
    def test_native_trainer_hooks_restore_actor_critic_and_complete(self):
        spec = LaunchSpec(
            (sys.executable, "tests/areal_restart_runtime.py"),
            PROJECT,
            "native-hooks",
            "pinned-areal",
            configuration_id({"fixture": "native-ppo-hooks"}),
            {"PYTHONPATH": str(PROJECT / "src")},
        )
        with RestartCoordinator(
            spec,
            self.root / "native",
            timeouts=Timeouts(startup=15, snapshot=15, restore=15, exit=5, poll=0.01),
        ) as runner:
            source = runner.start()
            result = runner.restart()
            runner.wait(timeout=10)
            final = json.loads(
                (runner.current.directory / "native-fixture.json").read_text()
            )
            self.assertTrue(final["restored"])
            self.assertEqual(final["actor_loads"], 1)
            self.assertEqual(final["critic_loads"], 1)
            self.assertEqual(final["schedule_step"], 8)
            self.assertLess(result["context"]["step_info"]["global_step"], 7)
            self.assertIn("rollout.resume", final["events"])
            self.assertLess(
                final["events"].index("actor.load"), final["events"].index("batch")
            )
            self.assertIn("actor.destroy", final["events"])
            self.assertTrue(source.process.group_gone())

    def test_incomplete_completion_receipts_are_rejected(self):
        runner = self.runner()
        source = runner.start()
        valid = message(
            source.launch,
            "completed",
            status="completed",
            cleanup_completed=True,
            restored=None,
            runtime_source_id=source.ready["runtime_source_id"],
        )
        for key, value in (
            ("cleanup_completed", False),
            ("status", "resumed"),
            ("runtime_source_id", "different"),
            ("restored", {}),
            ("kind", "ready"),
        ):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                runner._check_completed(
                    source,
                    dict(valid, **{key: value}),
                    source.ready["runtime_source_id"],
                )


class AdapterTests(unittest.TestCase):
    def config(self):
        def engine():
            return SimpleNamespace(
                _version="v1",
                agent=None,
                scheduling_spec=[
                    SimpleNamespace(cmd="python3 -m areal.infra.rpc.rpc_server")
                ],
            )

        return SimpleNamespace(
            scheduler=SimpleNamespace(type="local"),
            actor=engine(),
            rollout=engine(),
            critic=None,
            ref=None,
            teacher=SimpleNamespace(
                engine_type="rollout", rollout=engine(), train=engine()
            ),
            train_dataset=SimpleNamespace(
                scheduling_spec=SimpleNamespace(cmd="python3 -m areal.infra.rpc.guard")
            ),
        )

    def test_selected_teacher_engine_and_data_service_are_checked(self):
        config = self.config()
        for role in ("train", "rollout"):
            config.teacher.engine_type = role
            _check_local_workers(config, single_controller=True, remote_dataset=True)
            settings = getattr(config.teacher, role)
            settings._version = "v2"
            with self.assertRaises(NotImplementedError):
                _check_local_workers(
                    config, single_controller=True, remote_dataset=True
                )
            settings._version = "v1"
        config.train_dataset.scheduling_spec.cmd = (
            "python3 custom_detached_data_service.py"
        )
        with self.assertRaises(NotImplementedError):
            _check_local_workers(config, single_controller=True, remote_dataset=True)

    def test_unsupported_process_containment_is_rejected(self):
        for backend in ("ray", "slurm"):
            config = self.config()
            config.scheduler.type = backend
            with self.assertRaises(NotImplementedError):
                _check_local_workers(
                    config, single_controller=True, remote_dataset=False
                )
        config = self.config()
        config.actor.scheduling_spec[0].cmd = "python3 detached_worker.py"
        with self.assertRaises(NotImplementedError):
            _check_local_workers(config, single_controller=True, remote_dataset=False)


class ProtocolTests(unittest.TestCase):
    def test_permission_denial_is_not_exit(self):
        process = LocalProcess.__new__(LocalProcess)
        process.process = Mock()
        process.process.poll.return_value = 0
        process.pgid = 123
        process._gone = False
        with patch("os.killpg", side_effect=PermissionError(errno.EPERM, "denied")):
            with self.assertRaises(PermissionError):
                process.group_gone()
        self.assertFalse(process._gone)

    def test_messages_cannot_overwrite_prior_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            publish(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                publish(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_nan_and_invalid_timeouts_are_rejected(self):
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Timeouts(snapshot=value)

    def test_config_and_segment_bindings_are_checked(self):
        launch = dict(
            run_id="r",
            segment_id="s",
            nonce="n",
            runtime_revision="v",
            configuration_id="c",
        )
        receipt = message(launch, "ready", status="initialized")
        for field in launch:
            with self.subTest(field=field):
                bad = dict(receipt, **{field: "stale"})
                with self.assertRaises(ProtocolError):
                    check_message(bad, launch, "ready")


if __name__ == "__main__":
    unittest.main()
