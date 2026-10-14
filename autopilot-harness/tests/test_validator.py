import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from autopilot_harness.cli import main
from autopilot_harness.validator.checker import (
    load_policy,
    parse_policy,
    read_document,
    validate_configuration,
)
from validator_fixtures import context_for

PROJECT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT / "configs/validator.json"
EXAMPLES = PROJECT / "examples/validator"


def set_path(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(POLICY_PATH)
        self.baseline = read_document(EXAMPLES / "baseline.json")

    def validate(self, candidate, **kwargs):
        return validate_configuration(
            self.baseline,
            candidate,
            self.policy,
            runtime_revision=kwargs.get("revision", self.policy.runtime_revision),
            context=kwargs.get("context", context_for(self.baseline, candidate)),
        )

    def test_example_passes_without_mutating_inputs(self):
        candidate = read_document(EXAMPLES / "candidate.json")
        before = copy.deepcopy((self.baseline, candidate))
        result = self.validate(candidate)
        self.assertTrue(result.passed)
        self.assertEqual(result.scope, "configuration_and_transition")
        self.assertEqual(result.runtime_applicability, "supported")
        self.assertIsNotNone(result.context_id)
        self.assertEqual(
            {change.path for change in result.changes},
            {
                ("actor", "mb_spec", "n_mbs"),
                ("actor", "mb_spec", "max_tokens_per_mb"),
                ("rollout", "max_concurrent_rollouts"),
            },
        )
        self.assertEqual((self.baseline, candidate), before)
        self.assertEqual(result.policy_id, self.policy.policy_id)
        self.assertNotEqual(result.baseline_id, result.candidate_id)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_unchanged_configuration_passes(self):
        result = self.validate(copy.deepcopy(self.baseline))
        self.assertTrue(result.passed)
        self.assertEqual(result.changes, ())
        self.assertEqual(result.baseline_id, result.candidate_id)

    def test_fixed_training_fields_cannot_change(self):
        cases = [
            (("seed",), 2),
            (("train_dataset", "batch_size"), 64),
            (("gconfig", "n_samples"), 2),
            (("gconfig", "max_new_tokens"), 2048),
            (("gconfig", "temperature"), 0.8),
            (("actor", "optimizer", "lr"), 0.01),
            (("actor", "ppo_n_minibatches"), 2),
            (("actor", "mb_spec", "granularity"), 2),
            (("actor", "mb_spec", "packing_algorithm"), "kk"),
            (("rollout", "max_head_offpolicyness"), 4),
            (("rollout", "consumer_batch_size"), 64),
        ]
        for path, value in cases:
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.baseline)
                set_path(candidate, path, value)
                result = self.validate(candidate)
                self.assertEqual(result.status, "rejected")
                self.assertTrue(
                    any(
                        issue.path == path and issue.code == "unauthorized_change"
                        for issue in result.issues
                    )
                )

    def test_extra_fields_and_embedded_policy_are_rejected(self):
        for key, value in [
            ("new_flag", True),
            ("parameters", {}),
            ("actor.mb_spec.n_mbs", 2),
            ("actor/mb_spec/n_mbs", 2),
        ]:
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.baseline)
                candidate[key] = value
                result = self.validate(candidate)
                self.assertEqual(result.status, "rejected")
                self.assertIn((key,), [issue.path for issue in result.issues])

    def test_whitelisted_field_presence_must_be_preserved(self):
        candidate = copy.deepcopy(self.baseline)
        del candidate["actor"]["mb_spec"]["n_mbs"]
        result = self.validate(candidate)
        self.assertIn("field_presence", [issue.code for issue in result.issues])
        self.assertEqual(result.changes[0].operation, "remove")
        del self.baseline["actor"]["mb_spec"]["n_mbs"]
        candidate["actor"]["mb_spec"]["n_mbs"] = 2
        result = self.validate(candidate)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.changes[0].operation, "add")

    def test_containers_and_arrays_do_not_grant_nested_permissions(self):
        candidates = []
        candidate = copy.deepcopy(self.baseline)
        del candidate["actor"]
        candidates.append(candidate)
        candidate = copy.deepcopy(self.baseline)
        candidate["actor"]["mb_spec"] = 2
        candidates.append(candidate)
        candidate = copy.deepcopy(self.baseline)
        candidate["actor"]["scheduling_spec"][0]["gpu"] = 2
        candidates.append(candidate)
        candidate = copy.deepcopy(self.baseline)
        candidate["actor"]["mb_spec"]["n_mbs"] = {"value": 2}
        candidates.append(candidate)
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(self.validate(candidate).status, "rejected")

    def test_integer_type_is_not_coerced(self):
        for value in [True, 2.0, "2", [], {}, 0, -1]:
            with self.subTest(value=value):
                candidate = copy.deepcopy(self.baseline)
                candidate["actor"]["mb_spec"]["n_mbs"] = value
                result = self.validate(candidate)
                self.assertEqual(result.status, "rejected")

    def test_boolean_integer_equality_does_not_hide_fixed_changes(self):
        for path in [("seed",), ("actor", "scheduling_spec", 0, "gpu")]:
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.baseline)
                set_path(candidate, path, True)
                result = self.validate(candidate)
                self.assertEqual(result.status, "rejected")
                self.assertIn(
                    "unauthorized_change", [issue.code for issue in result.issues]
                )

    def test_nullable_and_nonnullable_fields(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["actor"]["mb_spec"]["n_mbs"] = None
        candidate["actor"]["mb_spec"]["max_tokens_per_mb"] = None
        candidate["rollout"]["max_concurrent_rollouts"] = None
        self.assertTrue(self.validate(candidate).passed)
        candidate["actor"]["pad_to_maximum"] = None
        self.assertIn(
            "null_not_allowed",
            [issue.code for issue in self.validate(candidate).issues],
        )

    def test_bounds_and_nonfinite_input(self):
        for value in [0, -0.1, 1.1, True, "0.8", float("nan"), float("inf")]:
            with self.subTest(value=value):
                candidate = copy.deepcopy(self.baseline)
                candidate["sglang"]["mem_fraction_static"] = value
                self.assertEqual(self.validate(candidate).status, "rejected")
        for value in [0.4, 1.0, None]:
            candidate = copy.deepcopy(self.baseline)
            candidate["sglang"]["mem_fraction_static"] = value
            self.assertTrue(self.validate(candidate).passed)

    def test_candidate_values_are_checked_even_when_unchanged(self):
        self.baseline["actor"]["mb_spec"]["n_mbs"] = 0
        result = self.validate(copy.deepcopy(self.baseline))
        self.assertEqual(result.status, "rejected")
        self.assertIn("below_minimum", [issue.code for issue in result.issues])

    def test_padding_combination_requires_explicit_token_limit(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["actor"]["pad_to_maximum"] = True
        self.assertTrue(self.validate(candidate).passed)
        for value in [None, 10**12]:
            candidate["actor"]["mb_spec"]["max_tokens_per_mb"] = value
            result = self.validate(candidate)
            self.assertIn(
                "padding_requires_limit", [issue.code for issue in result.issues]
            )

    def test_megatron_requires_nonnull_microbatch_count(self):
        self.baseline["actor"]["backend"] = "megatron:d4p1t1"
        candidate = copy.deepcopy(self.baseline)
        candidate["actor"]["mb_spec"]["n_mbs"] = None
        result = self.validate(candidate)
        self.assertIn(
            "megatron_requires_n_mbs", [issue.code for issue in result.issues]
        )

    def test_backend_substitution_is_rejected(self):
        for role, value in [("actor", "megatron:d4p1t1"), ("rollout", "vllm:d4p1t1")]:
            with self.subTest(role=role):
                candidate = copy.deepcopy(self.baseline)
                candidate[role]["backend"] = value
                result = self.validate(candidate)
                self.assertEqual(result.status, "rejected")
                self.assertIn(
                    "backend_substitution", [issue.code for issue in result.issues]
                )

    def test_layout_changes_are_not_silently_approved(self):
        for path, value in [
            (("actor", "backend"), "fsdp:d2p1t2"),
            (("rollout", "backend"), "sglang:d2p1t2"),
            (("sglang", "dp_size"), 2),
            (("sglang", "enable_ep_moe"), True),
        ]:
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.baseline)
                set_path(candidate, path, value)
                result = self.validate(candidate, context=None)
                self.assertEqual(result.status, "undetermined")
                self.assertFalse(result.passed)
                self.assertIn(
                    "validation_context_missing",
                    [issue.code for issue in result.issues],
                )

    def test_inactive_backend_fields_cannot_change(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["vllm"]["max_num_seqs"] = 128
        self.assertIn(
            "inactive_backend",
            [issue.code for issue in self.validate(candidate).issues],
        )
        self.baseline["rollout"]["backend"] = "vllm:d4p1t1"
        candidate = copy.deepcopy(self.baseline)
        candidate["vllm"]["max_num_seqs"] = 128
        self.assertTrue(self.validate(candidate).passed)
        candidate["sglang"]["max_prefill_tokens"] = 4096
        self.assertIn(
            "inactive_backend",
            [issue.code for issue in self.validate(candidate).issues],
        )

    def test_missing_context_and_revision_are_undetermined(self):
        for revision in [None, "another-revision"]:
            result = self.validate(copy.deepcopy(self.baseline), revision=revision)
            self.assertEqual(result.status, "undetermined")
        del self.baseline["actor"]["mb_spec"]["n_mbs"]
        result = self.validate(copy.deepcopy(self.baseline))
        self.assertEqual(result.status, "undetermined")
        self.assertIn("missing_parameter", [issue.code for issue in result.issues])

    def test_unresolved_inputs_are_rejected(self):
        for value in ["???", "${train_dataset.batch_size}"]:
            candidate = copy.deepcopy(self.baseline)
            candidate["rollout"]["consumer_batch_size"] = value
            result = self.validate(candidate)
            self.assertEqual(result.status, "rejected")
            self.assertIn("unresolved_input", [issue.code for issue in result.issues])
            self.assertNotIn("parameter_values", result.checked_rules)

    def test_results_do_not_contain_sensitive_config_values(self):
        self.baseline["credentials"] = {"password": "original-secret"}
        candidate = copy.deepcopy(self.baseline)
        candidate["credentials"]["password"] = "candidate-secret"
        result = self.validate(candidate)
        serialized = json.dumps(result.to_dict())
        self.assertNotIn("original-secret", serialized)
        self.assertNotIn("candidate-secret", serialized)
        self.assertIn("password", serialized)


class PolicyTests(unittest.TestCase):
    def test_invalid_policy_is_not_silently_ignored(self):
        original = read_document(POLICY_PATH)
        invalids = []
        for key, value in [
            ("type", "object"),
            ("minimum", True),
            ("choices", [True]),
            ("nullable", "yes"),
            ("maximum", 0),
            ("minimun", 1),
        ]:
            policy = copy.deepcopy(original)
            policy["parameters"]["actor.mb_spec.n_mbs"][key] = value
            invalids.append(policy)
        policy = copy.deepcopy(original)
        policy["checks"].append("arbitrary_python")
        invalids.append(policy)
        policy = copy.deepcopy(original)
        policy["parameters"]["actor.*"] = {"type": "integer"}
        invalids.append(policy)
        policy = copy.deepcopy(original)
        policy["parameters"]["actor"] = {"type": "string"}
        invalids.append(policy)
        policy = copy.deepcopy(original)
        policy["runtime_revision"] = "latest"
        invalids.append(policy)
        for policy in invalids:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                parse_policy(policy)

    def test_choices_and_upper_bounds_can_be_narrowed_by_harness(self):
        raw = read_document(POLICY_PATH)
        raw["parameters"]["actor.mb_spec.n_mbs"].update(choices=[1, 2, 4], maximum=4)
        policy = parse_policy(raw)
        baseline = read_document(EXAMPLES / "baseline.json")
        candidate = copy.deepcopy(baseline)
        candidate["actor"]["mb_spec"]["n_mbs"] = 3
        result = validate_configuration(
            baseline, candidate, policy, runtime_revision=policy.runtime_revision
        )
        self.assertIn("invalid_choice", [issue.code for issue in result.issues])
        self.assertNotEqual(policy.policy_id, load_policy(POLICY_PATH).policy_id)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text('{"actor": {}, "actor": {"mb_spec": {"n_mbs": 2}}}')
            with self.assertRaises(ValueError):
                read_document(path)


class ValidatorCliTests(unittest.TestCase):
    def args(self):
        return [
            "validate",
            "--baseline",
            str(EXAMPLES / "baseline.json"),
            "--candidate",
            str(EXAMPLES / "candidate.json"),
            "--policy",
            str(POLICY_PATH),
            "--context",
            str(EXAMPLES / "context.json"),
        ]

    def test_cli_pass_and_result_file_cannot_be_overwritten(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            path = Path(directory) / "result.json"
            args = self.args() + [
                "--runtime-revision",
                load_policy(POLICY_PATH).runtime_revision,
                "--output",
                str(path),
            ]
            self.assertEqual(main(args), 0)
            result = json.loads(path.read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["runtime_applicability"], "supported")
            self.assertEqual(main(args), 2)
            self.assertEqual(json.loads(path.read_text()), result)

    def test_cli_missing_revision_saves_undetermined(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(self.args()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "undetermined")

    def test_cli_rejection_returns_failure_with_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = read_document(EXAMPLES / "candidate.json")
            candidate["seed"] = 99
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(candidate))
            args = self.args()
            args[args.index("--candidate") + 1] = str(path)
            args += ["--runtime-revision", load_policy(POLICY_PATH).runtime_revision]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(args), 1)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "rejected")
            self.assertIn(
                "unauthorized_change", [issue["code"] for issue in result["issues"]]
            )


if __name__ == "__main__":
    unittest.main()
