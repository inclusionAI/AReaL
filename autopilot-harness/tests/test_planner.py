import copy
import json
import unittest
from pathlib import Path

from autopilot_harness.contracts.planner import PlannerInput
from autopilot_harness.monitor.collector import collect
from autopilot_harness.planner import plan
from autopilot_harness.validator import load_policy, validate_configuration
from validator_fixtures import context_for

PROJECT = Path(__file__).resolve().parents[1]
MONITOR_EXAMPLE = PROJECT / "examples/monitor"


class PlannerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(PROJECT / "configs/validator.json")
        self.config = json.loads((MONITOR_EXAMPLE / "config.json").read_text())
        self.baseline = json.loads(
            (PROJECT / "examples/validator/baseline.json").read_text()
        )
        self.config["training_specification"]["runtime_code"]["version"] = (
            self.policy.runtime_revision
        )
        self.config["current_configuration"] = {
            "values": self.baseline,
            "status": "recorded",
            "source_ref": "fixture:validator/baseline.json",
        }

    def test_monitor_planner_validator_contracts_connect(self):
        snapshot = collect(self.config, root=MONITOR_EXAMPLE)
        before = copy.deepcopy(snapshot.to_dict())
        inputs = PlannerInput.from_snapshot(snapshot)
        result = plan(inputs)
        validation = validate_configuration(
            snapshot.current_configuration.values,
            result.config,
            self.policy,
            runtime_revision=inputs.training_specification.values["runtime_code"][
                "version"
            ],
            context=context_for(self.baseline, result.config),
        )
        self.assertTrue(validation.passed)
        self.assertEqual(validation.changes, ())
        self.assertEqual(result.config, self.baseline)
        self.assertFalse(result.changed)
        self.assertEqual(result.search_status, "not_run")
        self.assertEqual(snapshot.to_dict(), before)
        self.assertEqual(snapshot.current_configuration.status, "recorded")
        self.assertEqual(inputs.runtime_observations, snapshot.runtime_observations)
        self.assertEqual(inputs.execution_history, snapshot.execution_history)
        self.assertEqual(
            json.loads(json.dumps(result.to_dict()))["config"], self.baseline
        )

    def test_downstream_edits_cannot_mutate_monitor_or_other_results(self):
        snapshot = collect(self.config, root=MONITOR_EXAMPLE)
        inputs = PlannerInput.from_snapshot(snapshot)
        before = copy.deepcopy(snapshot.to_dict())
        first, second = plan(inputs), plan(inputs)
        first.config["actor"]["mb_spec"]["n_mbs"] = 8
        first.config["actor"]["scheduling_spec"][0]["gpu"] = 8
        first.config["actor"]["scheduling_spec"].append({"gpu": 1})
        del first.config["rollout"]
        self.assertEqual(snapshot.to_dict(), before)
        self.assertEqual(second.config, self.baseline)

    def test_passthrough_does_not_repair_or_approve_invalid_config(self):
        self.baseline["actor"]["mb_spec"]["n_mbs"] = 0
        snapshot = collect(self.config, root=MONITOR_EXAMPLE)
        result = plan(PlannerInput.from_snapshot(snapshot))
        validation = validate_configuration(
            snapshot.current_configuration.values,
            result.config,
            self.policy,
            runtime_revision=self.policy.runtime_revision,
        )
        self.assertEqual(result.config["actor"]["mb_spec"]["n_mbs"], 0)
        self.assertFalse(result.changed)
        self.assertEqual(result.search_status, "not_run")
        self.assertEqual(validation.status, "rejected")
        self.assertIn("below_minimum", [issue.code for issue in validation.issues])


if __name__ == "__main__":
    unittest.main()
