import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

from autopilot_harness.cli import main
from autopilot_harness.monitor import sources
from autopilot_harness.monitor._io import SourceError, read_json, redacted, timestamp
from autopilot_harness.monitor.collector import collect, save
from autopilot_harness.monitor.outputs import (
    HistoryStore,
    build_configuration,
    make_event,
)

PROJECT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT / "examples" / "monitor"


def wb_options():
    return {"run": "team/project/run", "metrics": {"duration": "s", "reward": None}}


def wb_factory(rows, configuration=None):
    run = SimpleNamespace(
        config=configuration or {"actor": {"batch_size": 2}},
        scan_history=Mock(return_value=iter(rows)),
    )
    factory = Mock(return_value=SimpleNamespace(run=Mock(return_value=run)))
    return factory, run


def prom_options():
    return {
        "base_url": "http://prom.example/prefix",
        "start": 100,
        "end": 130,
        "step_seconds": 15,
        "queries": [
            {
                "name": "requests",
                "expression": 'running{trial="a&b"}',
                "unit": "requests",
            }
        ],
    }


def prom_matrix(values=None, labels=None):
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": labels or {"instance": "worker-0"},
                    "values": values
                    if values is not None
                    else [[100, "2"], [115, "3"], [130, "4"]],
                }
            ],
        },
    }


class WandbTests(unittest.TestCase):
    def test_sparse_rows_keep_distinct_timestamps_and_steps(self):
        factory, run = wb_factory(
            [
                {"_step": 9, "_timestamp": 100, "duration": 1.5},
                {"_step": 9, "_timestamp": 101, "reward": 0.2},
            ]
        )
        result = sources.collect_wandb(wb_options(), EXAMPLE, api_factory=factory)
        run.scan_history.assert_called_once_with(page_size=1000)
        self.assertEqual(result.report.status, "ok")
        self.assertEqual([item.observed_at for item in result.observations], [100, 101])
        self.assertEqual([item.logging_step for item in result.observations], [9, 9])
        self.assertTrue(all(item.train_step is None for item in result.observations))
        self.assertNotEqual(
            result.observations[0].source_ref, result.observations[1].source_ref
        )

    def test_train_step_requires_explicit_key(self):
        factory, _ = wb_factory(
            [{"_step": 9, "training_step": 3, "duration": 1, "reward": 0.2}]
        )
        result = sources.collect_wandb(
            wb_options() | {"train_step_key": "training_step"},
            EXAMPLE,
            api_factory=factory,
        )
        self.assertEqual(result.observations[0].train_step, 3)

    def test_limits_are_reported_and_not_silent(self):
        factory, _ = wb_factory([{"duration": 1, "reward": 0.2}, {"duration": 2}])
        result = sources.collect_wandb(
            wb_options() | {"max_rows": 1}, EXAMPLE, api_factory=factory
        )
        self.assertEqual(result.report.status, "partial")
        self.assertIn("history truncated by max_rows", result.report.errors)
        self.assertEqual(result.report.details["rows_read"], 1)

    def test_partial_scan_retains_samples_and_hides_sdk_secrets(self):
        def broken_rows():
            yield {"duration": 1}
            raise RuntimeError("Authorization: secret-should-not-appear")

        factory, _ = wb_factory(broken_rows())
        result = sources.collect_wandb(wb_options(), EXAMPLE, api_factory=factory)
        self.assertEqual(result.report.status, "partial")
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.report.details["missing_metrics"], ["reward"])
        self.assertNotIn("secret-should-not-appear", repr(result))

    def test_nonfinite_null_and_non_numeric_values(self):
        for value, reason in [
            (float("nan"), "non_finite"),
            (None, "source_null"),
            (False, "not_numeric"),
            ({}, "not_numeric"),
        ]:
            with self.subTest(value=value):
                factory, _ = wb_factory([{"duration": value}])
                result = sources.collect_wandb(
                    wb_options(), EXAMPLE, api_factory=factory
                )
                self.assertIsNone(result.observations[0].value)
                self.assertEqual(result.observations[0].missing_reason, reason)
                self.assertEqual(result.report.status, "partial")

    def test_url_parsing_and_missing_key(self):
        self.assertEqual(
            sources.wandb_run_path("https://wandb.ai/team/project/runs/abc?x=1"),
            "team/project/abc",
        )
        with self.assertRaises(ValueError):
            sources.wandb_run_path("https://user:pass@example/team/project/runs/abc")
        with patch.dict(os.environ, {}, clear=True):
            result = sources.collect_wandb(wb_options(), EXAMPLE)
        self.assertEqual(result.report.status, "error")
        self.assertIn("key missing", result.report.errors[0])


class PrometheusTests(unittest.TestCase):
    def test_labels_query_timestamps_and_coverage_are_preserved(self):
        data = prom_matrix()
        second = copy.deepcopy(data["data"]["result"][0])
        second["metric"] = {"instance": "worker-1", "gpu": "0"}
        second["values"] = [[115, "9"]]
        data["data"]["result"].append(second)
        result = sources.collect_prometheus(
            prom_options(), EXAMPLE, transport=lambda *_: data
        )
        self.assertEqual(result.report.status, "partial")
        self.assertEqual(len(result.observations), 4)
        last = result.observations[-1]
        self.assertEqual(last.entity, {"instance": "worker-1", "gpu": "0"})
        self.assertEqual(last.coverage["expected"], 3)
        self.assertEqual(last.coverage["present"], 1)
        self.assertEqual(last.timestamp_basis, "prometheus_query_evaluation")
        self.assertIsNone(last.train_step)

    def test_nonfinite_values_and_warnings_remain_visible(self):
        payload = prom_matrix([[100, "NaN"], [115, "+Inf"], [130, "-Inf"]])
        payload["warnings"] = ["partial backend response"]
        result = sources.collect_prometheus(
            prom_options(), EXAMPLE, transport=lambda *_: payload
        )
        self.assertEqual(result.report.status, "partial")
        self.assertTrue(all(item.value is None for item in result.observations))
        self.assertTrue(
            all(item.missing_reason == "non_finite" for item in result.observations)
        )
        self.assertEqual(
            result.report.details["queries"][0]["warnings"], payload["warnings"]
        )

    def test_empty_result_is_not_zero(self):
        payload = prom_matrix()
        payload["data"]["result"] = []
        result = sources.collect_prometheus(
            prom_options(), EXAMPLE, transport=lambda *_: payload
        )
        self.assertEqual(result.report.status, "empty")
        self.assertEqual(result.observations, [])

    def test_one_failed_query_does_not_discard_another(self):
        options = prom_options()
        options["queries"].append({"name": "failed", "expression": "bad"})

        def query(_options, expression):
            if expression == "bad":
                raise OSError("secret-should-not-appear")
            return prom_matrix()

        result = sources.collect_prometheus(options, EXAMPLE, transport=query)
        self.assertEqual(result.report.status, "partial")
        self.assertEqual(len(result.observations), 3)
        self.assertNotIn("secret-should-not-appear", repr(result))

    def test_api_error_wrong_type_and_out_of_window_are_errors(self):
        for payload in [
            {"status": "error", "error": "server details"},
            {"status": "success", "data": {"resultType": "vector"}},
            prom_matrix([[200, "1"]]),
        ]:
            with self.subTest(payload=payload):
                result = sources.collect_prometheus(
                    prom_options(), EXAMPLE, transport=lambda *_: payload
                )
                self.assertEqual(result.report.status, "error")
                self.assertEqual(result.observations, [])

    def test_real_http_request_encoding_and_auth_with_mocked_connection(self):
        response = io.BytesIO(json.dumps(prom_matrix()).encode())
        with (
            patch.dict(os.environ, {"PROMETHEUS_BEARER_TOKEN": "test-token"}),
            patch.object(sources, "urlopen", return_value=response) as opening,
        ):
            payload = sources.query_prometheus_range(
                prom_options(), 'metric{trial="a&b"}'
            )
        request = opening.call_args.args[0]
        self.assertEqual(
            request.full_url, "http://prom.example/prefix/api/v1/query_range"
        )
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            parse_qs(request.data.decode())["query"], ['metric{trial="a&b"}']
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(payload["status"], "success")

    def test_response_size_limit(self):
        with patch.object(sources, "urlopen", return_value=io.BytesIO(b"x" * 20)):
            with self.assertRaises(SourceError):
                sources.query_prometheus_range(
                    prom_options() | {"max_response_bytes": 10}, "metric"
                )

    def test_credentials_in_environment_url_do_not_leak(self):
        options = prom_options()
        del options["base_url"]
        with patch.dict(
            os.environ, {"PROMETHEUS_URL": "https://user:secret-in-url@example.com"}
        ):
            result = sources.collect_prometheus(options, EXAMPLE)
        self.assertEqual(result.report.status, "error")
        self.assertNotIn("secret-in-url", repr(result))


class ContextTests(unittest.TestCase):
    def test_four_outputs_and_safe_serialization(self):
        config = read_json(EXAMPLE / "config.json")
        snapshot = collect(config, root=EXAMPLE)
        result = snapshot.to_dict()
        for key in (
            "training_specification",
            "current_configuration",
            "runtime_observations",
            "execution_history",
        ):
            self.assertIn(key, result)
        self.assertEqual(len(result["runtime_observations"]["samples"]), 10)
        self.assertEqual(snapshot.training_specification.missing_fields, [])
        self.assertEqual(snapshot.current_configuration.status, "recorded")
        self.assertTrue(
            all(
                item["available_at"] == snapshot.collected_at
                for item in result["runtime_observations"]["samples"]
            )
        )
        self.assertTrue(
            all(
                item["basis"] == "assumed"
                for item in result["runtime_observations"]["samples"]
            )
        )
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("synthetic-secret-for-redaction-test", serialized)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            path = save(snapshot, config, output)
            self.assertEqual(read_json(path), result)
            self.assertEqual(
                read_json(output / "manifest.json")["snapshot_id"], snapshot.snapshot_id
            )
            self.assertEqual(len(HistoryStore(output / "events.jsonl").read()), 1)
            with self.assertRaises(FileExistsError):
                save(snapshot, config, output)
            self.assertEqual(read_json(path), result)

    def test_source_outage_preserves_other_source(self):
        factory = Mock(side_effect=ConnectionError("secret-should-not-appear"))
        snapshot = collect(
            {
                "run_id": "r",
                "segment_id": "s",
                "wandb": wb_options(),
                "prometheus": prom_options(),
            },
            wandb_api_factory=factory,
            prometheus_transport=lambda *_: prom_matrix(),
        )
        self.assertEqual(len(snapshot.runtime_observations["samples"]), 3)
        self.assertEqual(
            [item["status"] for item in snapshot.runtime_observations["sources"]],
            ["error", "ok"],
        )
        self.assertEqual(snapshot.current_configuration.status, "missing")
        self.assertIn(
            "runtime_code.version", snapshot.training_specification.missing_fields
        )

    def test_recorded_config_does_not_become_confirmed(self):
        config = build_configuration(
            None, {"batch": 2, "wandb_api_key": "secret"}, "wandb:run", "s", 100
        )
        self.assertEqual(config.status, "recorded")
        self.assertIsNone(config.confirmation)
        self.assertEqual(config.values["wandb_api_key"], "[REDACTED]")
        self.assertTrue(config.configuration_id.startswith("redacted-sha256:"))

    def test_confirmed_configuration_requires_segment_and_application_evidence(self):
        value = {"values": {"batch": 2}, "status": "confirmed"}
        with self.assertRaises(ValueError):
            build_configuration(value, None, None, "s", 100)
        value["confirmation"] = {
            "event_id": "event-1",
            "segment_id": "s",
            "applied_at": 90,
        }
        self.assertEqual(
            build_configuration(value, None, None, "s", 100).status, "confirmed"
        )
        with self.assertRaises(ValueError):
            build_configuration(value, None, None, "other-segment", 100)
        with self.assertRaises(ValueError):
            build_configuration(value, None, None, "s", 80)

    def test_history_is_append_only_and_filters_other_runs_and_future_events(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HistoryStore(Path(directory) / "events.jsonl")
            first = make_event(
                "r", "old-segment", "configuration.applied", 50, {"batch": 1}
            )
            store.append(first)
            store.append(make_event("other-run", "s", "planner.decision", 60, {}))
            store.append(make_event("r", "s", "planner.revision", 99999999999, {}))
            self.assertEqual(store.read(run_id="r", as_of=100), [first])
            with self.assertRaises(ValueError):
                store.append(first)
            self.assertEqual(len(store.read()), 3)
            config = read_json(EXAMPLE / "config.json") | {
                "run_id": "r",
                "history_path": str(store.path),
            }
            snapshot = collect(config, root=EXAMPLE)
            self.assertEqual(snapshot.execution_history[0], first)
            self.assertEqual(snapshot.execution_history[1]["kind"], "monitor.captured")
            self.assertEqual(len(store.read()), 3)

    def test_invalid_local_config_is_rejected_before_source_access(self):
        invalids = [
            {"run_id": "r", "segment_id": "s"},
            {
                "run_id": "r",
                "segment_id": "s",
                "wandb": wb_options(),
                "training_specification": [1],
            },
            {
                "run_id": "r",
                "segment_id": "s",
                "wandb": wb_options(),
                "current_configuration": [1],
            },
            {
                "run_id": "r",
                "segment_id": "s",
                "wandb": wb_options() | {"max_rows": -1},
            },
            {
                "run_id": "r",
                "segment_id": "s",
                "prometheus": prom_options() | {"start": 200},
            },
            {
                "run_id": "r",
                "segment_id": "s",
                "prometheus": prom_options() | {"step_seconds": 0},
            },
        ]
        factory = Mock()
        for config in invalids:
            with self.subTest(config=config), self.assertRaises(ValueError):
                collect(config, wandb_api_factory=factory)
        factory.assert_not_called()

    def test_cli_success_and_overwrite_rejection(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            args = [
                "monitor",
                "--config",
                str(EXAMPLE / "config.json"),
                "--output",
                str(Path(directory) / "capture"),
            ]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 2)

    def test_cli_source_failure_still_saves_snapshot(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            patch.dict(os.environ, {}, clear=True),
        ):
            path = Path(directory)
            (path / "config.json").write_text(
                json.dumps({"run_id": "r", "segment_id": "s", "wandb": wb_options()})
            )
            self.assertEqual(
                main(
                    [
                        "monitor",
                        "--config",
                        str(path / "config.json"),
                        "--output",
                        str(path / "out"),
                    ]
                ),
                1,
            )
            self.assertTrue((path / "out" / "snapshots" / "monitor.json").is_file())

    def test_timestamps_require_timezones_and_finite_values(self):
        self.assertEqual(timestamp("2026-10-01T00:00:00Z"), 1790812800)
        for value in ["2026-10-01T00:00:00", "NaN", float("inf"), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                timestamp(value)

    def test_redaction_keeps_token_metrics_and_environment_variable_names(self):
        config = {
            "wandb_api_key": "secret",
            "api_key_env": "WANDB_API_KEY",
            "token_usage": 0.3,
            "nested": {"authorization": "secret", "max_total_num_tokens": 42},
        }
        result = redacted(config)
        self.assertNotIn("secret", repr(result))
        self.assertEqual(result["api_key_env"], "WANDB_API_KEY")
        self.assertEqual(result["token_usage"], 0.3)


if __name__ == "__main__":
    unittest.main()
