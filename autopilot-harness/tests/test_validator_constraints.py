import copy
import json
import unittest

from autopilot_harness.validator import (
    configuration_id,
    load_policy,
    validate_configuration,
)
from autopilot_harness.validator.checker import read_document
from autopilot_harness.validator.constraints import parse_layout
from validator_fixtures import PROJECT, context_for


class NativeLayoutTests(unittest.TestCase):
    def test_world_size_and_moe_folding(self):
        cases = [
            ("fsdp:d2t2c2", 8, 1, 1),
            ("megatron[actor]:d2p2t2e4", 8, 4, 1),
            ("megatron:(attn:d2p2t2|ffn:p2e4)", 8, 4, 1),
            ("megatron:(attn:d2p2t2|ffn:d1p2t2e2)", 8, 2, 2),
            ("sglang[rollout]:d2p2t4", 16, 1, 1),
            ("vllm:d2t4", 8, 1, 1),
            ("archon:d4", 4, 1, 1),
        ]
        for value, world, ep, etp in cases:
            with self.subTest(value=value):
                layout = parse_layout(value)
                self.assertEqual(
                    (layout.world_size, layout.ep, layout.etp), (world, ep, etp)
                )

    def test_malformed_and_unsupported_layouts(self):
        cases = [
            "fsdp:d0",
            "fsdp:d2d4",
            "fsdp:p2",
            "fsdp:e2",
            "sglang:d1e2",
            "megatron:d2e3",
            "fsdp:d4+sglang:d4",
            "other:d4",
            "d4",
            "fsdp:d2junk",
            "megatron:(attn:d2p2t2|ffn:p1e4)",
            "megatron:(attn:d2p2t2|ffn:d2p2e4)",
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_layout(value)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(PROJECT / "configs/validator.json")
        self.baseline = read_document(PROJECT / "examples/validator/baseline.json")
        self.candidate = read_document(PROJECT / "examples/validator/candidate.json")
        self.ctx = context_for(self.baseline, self.candidate)

    def refresh(self):
        self.ctx["baseline_id"] = configuration_id(self.baseline)
        self.ctx["candidate_id"] = configuration_id(self.candidate)
        self.ctx["transition"]["changes"] = context_for(self.baseline, self.candidate)[
            "transition"
        ]["changes"]

    def validate(self):
        return validate_configuration(
            self.baseline,
            self.candidate,
            self.policy,
            runtime_revision=self.policy.runtime_revision,
            context=self.ctx,
        )

    def assert_issue(self, code, status="rejected"):
        result = self.validate()
        self.assertEqual(result.status, status, result.to_dict())
        self.assertIn(code, [i.code for i in result.issues], result.to_dict())
        return result

    def actor_layout_change(self):
        self.candidate["actor"]["backend"] = "fsdp:d2t2"
        self.ctx["transition"]["state_mapping"]["parameters"]["method"] = "reshard"
        self.ctx["transition"]["state_mapping"]["optimizer"]["method"] = "reshard"
        self.ctx["transition"]["state_mapping"]["rng"]["method"] = "remap"
        self.refresh()

    def test_full_context_is_pure_and_serializable(self):
        before = copy.deepcopy((self.baseline, self.candidate, self.ctx))
        result = self.validate()
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual((self.baseline, self.candidate, self.ctx), before)
        self.assertEqual(result.context_id, configuration_id(self.ctx))
        self.assertEqual(result.state_id, self.ctx["state_id"])
        self.assertIn("memory_capacity", result.checked_rules)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_all_active_sglang_tunables_can_be_validated_together(self):
        self.actor_layout_change()
        self.candidate["actor"]["pad_to_maximum"] = True
        self.candidate["rollout"]["backend"] = "sglang:d1t4"
        self.candidate["sglang"].update(
            enable_dp_attention=True,
            dp_size=2,
            enable_ep_moe=True,
            ep_size=2,
            max_running_requests=64,
            max_prefill_tokens=8192,
            mem_fraction_static=0.7,
        )
        self.ctx["model"]["num_experts"] = 8
        self.ctx["memory"]["inference_memory_fraction"] = 0.7
        self.refresh()
        result = self.validate()
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(len(result.changes), 13)  # vLLM's field is inactive.

    def test_missing_context_never_passes_static_only(self):
        self.ctx = None
        self.assert_issue("validation_context_missing", "undetermined")

    def test_missing_sections_are_unknown_not_zero_or_unlimited(self):
        original = copy.deepcopy(self.ctx)
        for key in (
            "model",
            "resources",
            "backends",
            "memory",
            "dispatch",
            "transition",
            "state_id",
        ):
            with self.subTest(key=key):
                self.ctx = copy.deepcopy(original)
                del self.ctx[key]
                self.assert_issue("evidence_missing", "undetermined")

    def test_evidence_must_match_entire_configuration_pair_and_revision(self):
        original = copy.deepcopy(self.ctx)
        for key in ("baseline_id", "candidate_id", "runtime_revision"):
            with self.subTest(key=key):
                self.ctx = copy.deepcopy(original)
                self.ctx[key] = "wrong-scope"
                self.assert_issue("evidence_scope_mismatch", "undetermined")

    def test_unknown_or_malformed_context_fields_do_not_crash_or_pass(self):
        original = copy.deepcopy(self.ctx)
        for value in (
            [],
            True,
            42,
            {**original, "resouces": {}},
            {**original, "schema_version": True},
        ):
            with self.subTest(value=type(value)):
                self.ctx = value
                self.assert_issue("invalid_context")

    def test_gpu_reservation_and_world_size(self):
        self.candidate["actor"]["backend"] = "fsdp:d8"
        self.refresh()
        self.assert_issue("gpu_world_size")
        self.ctx["resources"]["role_pools"]["actor"].append("not-reserved")
        self.assert_issue("gpu_reservation")

    def test_separated_role_pools_must_be_disjoint(self):
        self.ctx["resources"]["role_pools"]["rollout"].append("gpu0")
        self.assert_issue("gpu_overlap")

    def test_colocation_requires_explicit_group_and_matching_devices(self):
        r = self.ctx["resources"]
        r["role_pools"]["rollout"] = r["role_pools"]["actor"][:]
        r["rank_devices"]["rollout"] = r["rank_devices"]["actor"][:]
        self.assert_issue("gpu_overlap")
        r["colocation_groups"] = [["actor", "rollout"]]
        for i in range(4):
            m = self.ctx["memory"]["devices"]
            m[f"gpu{i}"]["roles"]["rollout"] = m[f"gpu{i + 4}"]["roles"].pop("rollout")
        self.assertTrue(self.validate().passed)
        r["rank_devices"]["rollout"] = r["rank_devices"]["rollout"][:-1]
        self.assert_issue("gpu_world_size")

    def test_topology_profile_rejects_unsupported_node_span(self):
        self.candidate["rollout"]["backend"] = "sglang:d1t4"
        self.ctx["resources"]["gpus"]["gpu7"]["node"] = "node2"
        self.refresh()
        self.assert_issue("unsupported_topology")

    def test_supported_layout_change_with_state_mapping_passes(self):
        self.actor_layout_change()
        result = self.validate()
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.runtime_applicability, "supported")

    def test_backend_model_support_profile_is_required(self):
        self.actor_layout_change()
        self.ctx["backends"]["actor"]["supported_layouts"] = ["fsdp:d4"]
        self.assert_issue("unsupported_layout")

    def test_attention_and_kv_divisibility_are_distinct(self):
        self.actor_layout_change()
        self.ctx["model"]["num_attention_heads"] = 31
        self.assert_issue("attention_divisibility")
        self.ctx["model"]["num_attention_heads"] = 32
        self.ctx["model"]["num_key_value_heads"] = 1
        # FSDP's own implementation requires KV divisibility even if a profile
        # mistakenly claims replication support.
        self.ctx["backends"]["actor"]["kv_head_replication"] = True
        self.assert_issue("kv_head_divisibility")

    def test_inference_kv_replication_requires_backend_evidence(self):
        self.candidate["rollout"]["backend"] = "sglang:d1t4"
        self.ctx["model"]["num_key_value_heads"] = 2
        self.refresh()
        self.assertTrue(self.validate().passed)
        self.ctx["backends"]["rollout"]["kv_head_replication"] = False
        self.assert_issue("kv_head_divisibility")

    def test_pipeline_layer_partition(self):
        self.baseline["actor"]["backend"] = "megatron:d4"
        self.candidate["actor"]["backend"] = "megatron:d1p2t2"
        self.ctx["model"]["num_hidden_layers"] = 25
        for name in ("parameters", "optimizer"):
            self.ctx["transition"]["state_mapping"][name]["method"] = "reshard"
        self.ctx["transition"]["state_mapping"]["rng"]["method"] = "remap"
        self.refresh()
        self.assert_issue("pipeline_divisibility")
        self.ctx["backends"]["actor"]["layer_partition"] = [12, 13]
        self.assertTrue(self.validate().passed)
        self.ctx["backends"]["actor"]["layer_partition"] = [12, 12]
        self.assert_issue("pipeline_partition")

    def test_moe_experts_fold_without_extra_gpus(self):
        self.baseline["actor"]["backend"] = "megatron:d4"
        self.candidate["actor"]["backend"] = "megatron:(attn:d2t2|ffn:e4)"
        self.ctx["backends"]["actor"]["supported_layouts"].append(
            self.candidate["actor"]["backend"]
        )
        self.ctx["model"]["num_experts"] = 8
        for name in ("parameters", "optimizer"):
            self.ctx["transition"]["state_mapping"][name]["method"] = "reshard"
        self.ctx["transition"]["state_mapping"]["rng"]["method"] = "remap"
        self.refresh()
        self.assertTrue(self.validate().passed)
        self.ctx["model"]["num_experts"] = 7
        self.assert_issue("expert_divisibility")

    def test_sglang_internal_dp_does_not_multiply_reserved_gpu_count(self):
        self.candidate["rollout"]["backend"] = "sglang:d1t4"
        self.candidate["sglang"].update(enable_dp_attention=True, dp_size=2)
        self.refresh()
        self.assertTrue(self.validate().passed)
        self.candidate["sglang"]["dp_size"] = 3
        self.refresh()
        self.assert_issue("sglang_dp_layout")

    def test_sglang_dp_ep_flags_and_model_constraints(self):
        self.candidate["rollout"]["backend"] = "sglang:d1t4"
        self.candidate["sglang"].update(enable_ep_moe=True, ep_size=2)
        self.refresh()
        self.assert_issue("expert_model_required")
        self.ctx["model"]["num_experts"] = 8
        self.assertTrue(self.validate().passed)
        self.candidate["sglang"]["enable_ep_moe"] = False
        self.refresh()
        self.assert_issue("sglang_ep_layout")

    def test_batch_dispatch_groups_not_naive_batch_mod_dp(self):
        self.ctx["dispatch"].update(train_batch_size=10, group_size=2)
        # Five groups can be greedily distributed across four DP ranks.
        self.assertTrue(self.validate().passed)
        self.ctx["dispatch"]["group_size"] = 3
        self.assert_issue("batch_dispatch")
        self.ctx["dispatch"].update(train_batch_size=2, group_size=1)
        self.assert_issue("batch_dispatch")
        self.ctx["dispatch"].update(
            train_batch_size=10, group_size=2, allow_uneven_dp=False
        )
        self.assert_issue("batch_dispatch")

    def test_memory_steady_state_and_transition_peak(self):
        m = self.ctx["memory"]["devices"]["gpu0"]
        m["transition_extra"] = 50 * 1024**3
        self.assert_issue("memory_capacity")
        m["transition_extra"] = 0
        m["roles"]["actor"]["activations"] = 100 * 1024**3
        self.assert_issue("memory_capacity")

    def test_memory_fraction_and_colocated_roles_are_accounted_for(self):
        self.candidate["sglang"]["mem_fraction_static"] = 0.1
        self.refresh()
        self.ctx["memory"]["inference_memory_fraction"] = 0.1
        self.assert_issue("inference_memory_budget")
        self.candidate["sglang"]["mem_fraction_static"] = 0.8
        self.refresh()
        self.ctx["memory"]["inference_memory_fraction"] = 0.8
        r, m = self.ctx["resources"], self.ctx["memory"]["devices"]
        r["role_pools"]["rollout"] = r["role_pools"]["actor"][:]
        r["rank_devices"]["rollout"] = r["rank_devices"]["actor"][:]
        r["colocation_groups"] = [["actor", "rollout"]]
        self.assert_issue("memory_role_coverage", "undetermined")
        for i in range(4):
            m[f"gpu{i}"]["roles"]["rollout"] = m[f"gpu{i + 4}"]["roles"].pop("rollout")
        m["gpu0"]["roles"]["rollout"]["buffers"] = 20 * 1024**3
        self.assert_issue("memory_capacity")

    def test_memory_missing_unknown_measured_and_invalid_bounds(self):
        original = copy.deepcopy(self.ctx)
        del self.ctx["memory"]["devices"]["gpu0"]["roles"]["actor"]["activations"]
        self.assert_issue("evidence_missing", "undetermined")
        self.ctx = copy.deepcopy(original)
        self.ctx["memory"]["basis"] = "measured_peak"
        self.assert_issue("memory_bound_required", "undetermined")
        self.ctx = copy.deepcopy(original)
        self.ctx["memory"]["devices"]["gpu0"]["roles"]["actor"]["activations"] = -1
        self.assert_issue("invalid_context")
        self.ctx["memory"]["devices"]["gpu0"]["roles"]["actor"]["activations"] = float(
            "nan"
        )
        self.assert_issue("invalid_input")

    def test_memory_covers_idle_reserved_devices_and_host_disk(self):
        self.ctx["resources"]["gpus"]["idle"] = {
            "node": "node3",
            "memory_bytes": 1024,
            "unavailable_bytes": 0,
        }
        self.assert_issue("memory_coverage", "undetermined")
        del self.ctx["resources"]["gpus"]["idle"]
        self.ctx["memory"]["host_bytes"] = (
            self.ctx["resources"]["host_memory_bytes"] + 1
        )
        self.assert_issue("auxiliary_capacity")
        self.ctx["memory"]["host_bytes"] = 0
        self.ctx["memory"]["disk_bytes"] = self.ctx["resources"]["disk_bytes"] + 1
        self.assert_issue("auxiliary_capacity")

    def test_all_state_and_recovery_evidence_required(self):
        del self.ctx["transition"]["state_mapping"]["optimizer"]
        self.assert_issue("evidence_missing", "undetermined")
        self.ctx = context_for(self.baseline, self.candidate)
        del self.ctx["transition"]["recovery_ref"]
        self.assert_issue("evidence_missing", "undetermined")

    def test_layout_change_cannot_keep_old_optimizer_shards(self):
        self.actor_layout_change()
        self.ctx["transition"]["state_mapping"]["optimizer"]["method"] = "preserve"
        self.assert_issue("state_mapping")

    def test_combined_change_requires_combined_transition_evidence(self):
        self.ctx["transition"]["changes"].pop()
        self.assert_issue("transition_coverage", "undetermined")
        self.ctx["transition"]["supported"] = False
        self.assert_issue("unsupported_transition")

    def test_dropping_requests_or_resetting_training_is_rejected(self):
        original = copy.deepcopy(self.ctx)
        for field, method in (
            ("data_cursor", "reset"),
            ("in_flight_rollouts", "drop"),
            ("policy_version", "reset"),
            ("optimizer", "discard"),
        ):
            with self.subTest(field=field):
                self.ctx = copy.deepcopy(original)
                self.ctx["transition"]["state_mapping"][field]["method"] = method
                self.assert_issue("state_mapping")

    def test_noop_needs_no_transition_evidence_but_still_checks_capacity(self):
        self.candidate = copy.deepcopy(self.baseline)
        self.refresh()
        del self.ctx["transition"]
        result = self.validate()
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.runtime_applicability, "not_required")
        self.ctx["memory"]["devices"]["gpu0"]["roles"]["actor"]["parameters"] = (
            100 * 1024**3
        )
        self.assert_issue("memory_capacity")


if __name__ == "__main__":
    unittest.main()
