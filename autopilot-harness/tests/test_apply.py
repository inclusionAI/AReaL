"""Pre-pause validation and cross-process config application, without GPUs."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autopilot_harness.contracts.execution import APPLY_ADAPTER, ApplyPlan
from autopilot_harness.execution import (
    LaunchSpec,
    RestartCoordinator,
    RestartError,
    Timeouts,
    build_apply_plan,
)
from autopilot_harness.execution.plan import check_apply_plan
from autopilot_harness.execution.protocol import ProtocolError, check_message, message
from autopilot_harness.monitor.outputs import HistoryStore
from autopilot_harness.validator import (
    configuration_id,
    load_policy,
    validate_configuration,
)
from autopilot_harness.validator.checker import read_document
from validator_fixtures import PROJECT, context_for


def baseline_config():
    value = read_document(PROJECT / "examples/validator/baseline.json")
    value["scheduler"] = {"type": "local"}
    value["actor"].update(_version="v1", weight_update_mode="disk")
    value["rollout"]["_version"] = "v1"
    return value


class ApplyFixtures:
    def setUp(self):
        self.baseline = baseline_config()
        self.candidate = deepcopy(self.baseline)
        self.candidate["actor"]["mb_spec"]["n_mbs"] = 2
        self.candidate["rollout"]["max_concurrent_rollouts"] = 64
        self.policy = load_policy(PROJECT / "configs/validator.json")
        self.execution = dict(
            run_id="apply-test",
            segment_id="source",
            nonce="nonce",
            runtime_revision=self.policy.runtime_revision,
            configuration_id=configuration_id(self.baseline),
            runtime_source_id="fixture:source-digest",
            apply_adapter=APPLY_ADAPTER,
        )

    def validate(self, execution=True, context=None):
        self.execution["configuration_id"] = configuration_id(self.baseline)
        return validate_configuration(
            self.baseline,
            self.candidate,
            self.policy,
            runtime_revision=self.policy.runtime_revision,
            context=context or context_for(self.baseline, self.candidate),
            execution=self.execution if execution else None,
        )

    def plan(self):
        result = self.validate()
        self.assertTrue(result.passed, result.to_dict())
        return build_apply_plan(
            self.baseline, self.candidate, result, execution=self.execution
        )


class ApplyValidationTests(ApplyFixtures, unittest.TestCase):
    def teacher_context(self, backend="sglang", layout="d2p1t2"):
        self.baseline["rollout"]["backend"] = f"{backend}:d2p1t2"
        self.baseline["teacher"] = {
            "engine_type": "rollout",
            "path": "fixture:teacher-model",
            "rollout": {"backend": f"{backend}:{layout}", "_version": "v1"},
        }
        self.candidate = deepcopy(self.baseline)
        if backend == "sglang":
            self.candidate["sglang"].update(enable_dp_attention=True, dp_size=2)
        else:
            self.candidate["vllm"]["max_num_seqs"] = 512
        ctx = context_for(self.baseline, self.candidate)
        devices = [f"gpu{i}" for i in range(8, 12)]
        ctx["resources"]["role_pools"]["teacher"] = devices
        ctx["resources"]["rank_devices"]["teacher"] = devices
        for gpu in devices:
            ctx["resources"]["gpus"][gpu] = deepcopy(ctx["resources"]["gpus"]["gpu4"])
            ctx["memory"]["devices"][gpu] = {
                "roles": {
                    "teacher": deepcopy(
                        ctx["memory"]["devices"]["gpu4"]["roles"]["rollout"]
                    )
                },
                "transition_extra": 0,
            }
        ctx["auxiliary"] = {
            "teacher": {
                "model": dict(ctx["model"], source_ref="fixture:teacher-model-profile"),
                "backends": {"teacher": deepcopy(ctx["backends"]["rollout"])},
                "inference_memory_fraction": 0.8,
            }
        }
        return ctx

    def test_shared_sglang_dp_checks_teacher_layout(self):
        ctx = self.teacher_context(layout="d4p1t1")
        result = self.validate(context=ctx)
        self.assertIn("sglang_dp_layout", {i.code for i in result.issues})
        ctx = self.teacher_context()
        result = self.validate(context=ctx)
        self.assertTrue(result.passed, result.to_dict())
        self.assertIn("fixture:teacher-model-profile", result.evidence_refs)

    def test_shared_backend_checks_require_teacher_evidence(self):
        for backend in ("sglang", "vllm"):
            with self.subTest(backend=backend):
                ctx = self.teacher_context(backend=backend)
                result = self.validate(context=ctx)
                self.assertTrue(result.passed, result.to_dict())
                del ctx["auxiliary"]
                result = self.validate(context=ctx)
                self.assertEqual(result.status, "undetermined")
                self.assertIn("evidence_missing", {i.code for i in result.issues})

    def test_shared_ep_requires_teacher_moe_model_and_memory_budget(self):
        ctx = self.teacher_context()
        self.candidate["sglang"].update(enable_ep_moe=True, ep_size=2)
        ctx["candidate_id"] = configuration_id(self.candidate)
        ctx["model"]["num_experts"] = 8
        result = self.validate(context=ctx)
        self.assertIn("expert_model_required", {i.code for i in result.issues})
        ctx["auxiliary"]["teacher"]["model"]["num_experts"] = 8
        self.assertTrue(self.validate(context=ctx).passed)
        # It can fit total GPU memory yet exceed the teacher server's static budget.
        ctx["memory"]["devices"]["gpu8"]["roles"]["teacher"]["kv_cache"] = 60 * 2**30
        result = self.validate(context=ctx)
        self.assertIn("inference_memory_budget", {i.code for i in result.issues})

    def test_passed_plan_freezes_inputs_and_covers_concurrency_mapping(self):
        plan = self.plan()
        value = check_apply_plan(plan, self.execution)
        self.assertEqual(value["transition"]["source_limits"], [256, 128, 2])
        self.assertEqual(value["transition"]["target_limits"], [64, 128, 2])
        self.candidate["actor"]["mb_spec"]["n_mbs"] = 16
        value["candidate"]["actor"]["mb_spec"]["n_mbs"] = 32
        self.assertEqual(plan.to_dict()["candidate"]["actor"]["mb_spec"]["n_mbs"], 2)

    def test_offline_supported_evidence_cannot_authorize_apply(self):
        result = self.validate(execution=False)
        self.assertTrue(result.passed)
        with self.assertRaises(ValueError):
            build_apply_plan(
                self.baseline, self.candidate, result, execution=self.execution
            )

    def test_changed_pair_result_or_segment_rejected(self):
        plan = self.plan()
        for key in ("segment_id", "nonce", "runtime_source_id", "configuration_id"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_apply_plan(plan, dict(self.execution, **{key: "changed"}))
        document = plan.to_dict()
        document["candidate"]["actor"]["mb_spec"]["n_mbs"] = 16
        with self.assertRaises(ValueError):
            check_apply_plan(ApplyPlan(json.dumps(document)), self.execution)
        with self.assertRaises(ValueError):
            build_apply_plan(
                self.baseline,
                self.candidate,
                replace(self.validate(), status="rejected"),
                execution=self.execution,
            )

    def test_fsdp_training_layout_records_native_reshard_and_new_rng_policy(self):
        self.candidate["actor"]["backend"] = "fsdp:d2p1t2"
        result = self.validate()
        self.assertTrue(result.passed, result.to_dict())
        mapping = self.plan().to_dict()["transition"]["training_layout"]
        self.assertEqual(mapping["source"], [4, 1, 1, 1, 1, 1])
        self.assertEqual(mapping["target"], [2, 2, 1, 1, 1, 1])
        self.assertEqual(mapping["rng_policy"], "fsdp-mesh-reseed-v1")

    def megatron_context(self, target="megatron:d2p1t2"):
        self.baseline["actor"]["backend"] = "megatron:d4p1t1"
        self.candidate["actor"]["backend"] = target
        ctx = context_for(self.baseline, self.candidate)
        if target not in ctx["backends"]["actor"]["supported_layouts"]:
            ctx["backends"]["actor"]["supported_layouts"].append(target)
        ctx["training_checkpoint"] = {
            "source_ref": "fixture:canonical-checkpoint-schema",
            "format": "areal-megatron-fully-reshardable-v1",
            "model_schema": {
                "source": "fixture:model-schema",
                "target": "fixture:model-schema",
            },
            "optimizer_schema": {
                "source": "fixture:adam-schema",
                "target": "fixture:adam-schema",
            },
        }
        return ctx

    def test_megatron_layout_records_portable_mapping(self):
        for target in ("megatron:d2p1t2", "megatron:d1p2t2", "megatron:d2p1t2e2"):
            with self.subTest(target=target):
                ctx = self.megatron_context(target)
                ctx["model"]["num_experts"] = 8
                result = self.validate(context=ctx)
                self.assertTrue(result.passed, result.to_dict())
                plan = build_apply_plan(
                    self.baseline, self.candidate, result, execution=self.execution
                )
                mapping = plan.to_dict()["transition"]["training_layout"]
                self.assertEqual(mapping["backend"], "megatron")
                self.assertEqual(
                    mapping["rng_policy"], "megatron-model-parallel-reseed-v1"
                )

    def test_megatron_layout_requires_schema_evidence_before_pause(self):
        ctx = self.megatron_context()
        del ctx["training_checkpoint"]
        result = self.validate(context=ctx)
        self.assertEqual(result.status, "undetermined")
        self.assertIn("evidence_missing", {i.code for i in result.issues})
        ctx = self.megatron_context()
        ctx["training_checkpoint"]["format"] = "areal-megatron-dcp"
        result = self.validate(context=ctx)
        self.assertIn("training_checkpoint_format", {i.code for i in result.issues})
        ctx = self.megatron_context()
        ctx["training_checkpoint"]["model_schema"]["target"] = "different-aliases"
        result = self.validate(context=ctx)
        self.assertIn("training_checkpoint_schema", {i.code for i in result.issues})

    def test_megatron_fallback_pipeline_and_custom_fsdp_rejected(self):
        for config in (self.baseline, self.candidate):
            config["actor"]["megatron"] = {"bridge_type": "none"}
        ctx = self.megatron_context("megatron:d1p2t2")
        result = self.validate(context=ctx)
        self.assertIn("training_checkpoint_pipeline", {i.code for i in result.issues})
        for config in (self.baseline, self.candidate):
            config["actor"]["megatron"] = {"use_custom_fsdp": True}
        result = self.validate(context=self.megatron_context())
        self.assertIn("training_checkpoint_optimizer", {i.code for i in result.issues})

    def test_runtime_adapter_cannot_be_enabled_by_supported_flag(self):
        self.execution["apply_adapter"] = "unimplemented"
        result = self.validate()
        self.assertEqual(result.status, "rejected")
        self.assertIn("unsupported_apply_adapter", {i.code for i in result.issues})

    def test_megatron_lora_is_rejected_before_snapshot_request(self):
        for config in (self.baseline, self.candidate):
            config["actor"].update(backend="megatron:d4p1t1", use_lora=True)
        result = self.validate()
        self.assertIn("unsupported_snapshot_lora", {i.code for i in result.issues})

    def test_rollout_layout_with_fixed_actor_uses_native_queue_preservation(self):
        self.candidate["rollout"]["backend"] = "sglang:d2p1t2"
        self.assertTrue(self.validate().passed, self.validate().to_dict())

    def test_native_xccl_pipeline_constraints_are_checked_before_apply(self):
        for actor, inference in (("fsdp", "vllm"), ("megatron", "sglang")):
            with self.subTest(actor=actor, inference=inference):
                self.baseline = baseline_config()
                self.baseline["actor"].update(
                    backend=f"{actor}:d4p1t1", weight_update_mode="xccl"
                )
                self.baseline["rollout"]["backend"] = f"{inference}:d4p1t1"
                self.candidate = deepcopy(self.baseline)
                self.candidate["rollout"]["backend"] = f"{inference}:d1p2t2"
                ctx = context_for(self.baseline, self.candidate)
                ctx["backends"]["rollout"]["supported_layouts"].append(
                    self.candidate["rollout"]["backend"]
                )
                result = self.validate(context=ctx)
                self.assertIn("weight_sync_layout", {i.code for i in result.issues})

    def test_each_tunable_is_classified_by_adapter(self):
        cases = {
            "actor.backend": "fsdp:d2p1t2",
            "actor.mb_spec.n_mbs": 2,
            "actor.mb_spec.max_tokens_per_mb": 20480,
            "actor.pad_to_maximum": True,
            "rollout.backend": "sglang:d2p1t2",
            "rollout.max_concurrent_rollouts": None,
            "sglang.enable_dp_attention": True,
            "sglang.dp_size": 2,
            "sglang.enable_ep_moe": True,
            "sglang.ep_size": 2,
            "sglang.max_running_requests": 128,
            "sglang.max_prefill_tokens": 16384,
            "sglang.mem_fraction_static": 0.7,
            "vllm.max_num_seqs": 512,
        }
        for path, value in cases.items():
            with self.subTest(path=path):
                self.baseline = baseline_config()
                if path.startswith("sglang."):
                    self.baseline["rollout"]["backend"] = "sglang:d2p1t2"
                if path == "vllm.max_num_seqs":
                    self.baseline["rollout"]["backend"] = "vllm:d4p1t1"
                self.candidate = deepcopy(self.baseline)
                owner = self.candidate
                parts = path.split(".")
                for part in parts[:-1]:
                    owner = owner[part]
                owner[parts[-1]] = value
                if path == "sglang.dp_size":
                    self.candidate["sglang"]["enable_dp_attention"] = True
                if path == "sglang.ep_size":
                    self.candidate["sglang"]["enable_ep_moe"] = True
                ctx = context_for(self.baseline, self.candidate)
                if "ep_" in path or "ep_size" in path:
                    ctx["model"]["num_experts"] = 8
                result = self.validate(context=ctx)
                self.assertEqual(
                    result.status,
                    "passed",
                    result.to_dict(),
                )

    def test_plan_receipts_cannot_be_replayed_from_another_plan(self):
        launch = dict(self.execution, plan_id="plan-one")
        receipt = message(launch, "ready")
        with self.assertRaises(ProtocolError):
            check_message(receipt, dict(launch, plan_id="plan-two"), "ready")

    def test_awex_gpu_sharing_does_not_imply_scheduler_worker_sharing(self):
        self.baseline["actor"].update(
            backend="megatron:d4p1t1", weight_update_mode="awex"
        )
        self.candidate = deepcopy(self.baseline)
        self.candidate["rollout"]["backend"] = "sglang:d2p1t2"
        ctx = context_for(self.baseline, self.candidate)
        for field in ("role_pools", "rank_devices"):
            ctx["resources"][field]["rollout"] = list(ctx["resources"][field]["actor"])
        ctx["resources"]["colocation_groups"] = [["actor", "rollout"]]
        for i in range(4):
            devices = ctx["memory"]["devices"]
            devices[f"gpu{i}"]["roles"]["rollout"] = devices[f"gpu{i + 4}"][
                "roles"
            ].pop("rollout")
        result = self.validate(context=ctx)
        self.assertTrue(result.passed, result.to_dict())
        for config in (self.baseline, self.candidate):
            config["rollout"]["scheduling_strategy"] = {
                "type": "colocation",
                "target": "actor",
            }
        ctx["baseline_id"] = configuration_id(self.baseline)
        ctx["candidate_id"] = configuration_id(self.candidate)
        result = self.validate(context=ctx)
        self.assertIn("colocation_replicas", {i.code for i in result.issues})


@unittest.skipUnless(
    os.name == "posix" and importlib.util.find_spec("pytest"),
    "native CPU fixture requires POSIX and pytest",
)
class ApplyProcessTests(ApplyFixtures, unittest.TestCase):
    # Keep subprocess scenarios separate from the pure contract suite.
    def runner(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        baseline = root / "baseline.json"
        baseline.write_text(json.dumps(self.baseline))
        spec = LaunchSpec(
            (sys.executable, "tests/areal_restart_runtime.py"),
            PROJECT,
            "apply-test",
            self.policy.runtime_revision,
            configuration_id(self.baseline),
            {
                "PYTHONPATH": str(PROJECT / "src"),
                "AUTOPILOT_APPLY_FIXTURE": str(baseline),
            },
        )
        runner = RestartCoordinator(
            spec,
            root / "run",
            timeouts=Timeouts(startup=8, snapshot=8, exit=4, restore=8, poll=0.01),
        )
        self.addCleanup(runner.close)
        runner.start()
        self.execution = runner.execution_context()
        return runner

    def test_apply_then_restart_keeps_target_and_progress(self):
        runner = self.runner()
        source = runner.current
        plan = self.plan()
        result = runner.apply(plan)
        self.assertEqual(result.status, "applied")
        self.assertTrue(source.process.group_gone())
        self.assertEqual(
            runner.current.launch["configuration_id"], configuration_id(self.candidate)
        )
        # A subsequent same-config restart must use the newly applied config.
        runner.restart()
        runner.wait(timeout=8)
        final = read_document(runner.current.directory / "native-fixture.json")
        self.assertEqual(final["configuration"], self.candidate)
        self.assertEqual(set(final["observed_n_mbs"]), {2})
        self.assertEqual(final["schedule_step"], 40)
        self.assertEqual(final["staleness"]["limits"], [64, 128, 2])
        self.assertEqual(final["actor_loads"], 1)
        events = HistoryStore(runner.directory / "events.jsonl").read(
            run_id="apply-test"
        )
        kinds = [e["kind"] for e in events]
        self.assertLess(
            kinds.index("apply.validated"), kinds.index("snapshot.requested")
        )
        self.assertLess(kinds.index("runtime.exited"), kinds.index("apply.confirmed"))
        self.assertNotIn("apply.failed", kinds)

    def test_unchanged_and_rejected_plans_never_pause(self):
        runner = self.runner()
        source = runner.current
        self.candidate = deepcopy(self.baseline)
        result = runner.apply(self.plan())
        self.assertEqual(result.status, "unchanged")
        self.assertIs(source, runner.current)
        self.assertFalse((source.directory / "stop_request.json").exists())
        self.candidate["actor"]["optimizer"]["lr"] = 0.1
        validation = self.validate()
        with self.assertRaises(ValueError):
            build_apply_plan(
                self.baseline, self.candidate, validation, execution=self.execution
            )
        self.assertFalse((source.directory / "stop_request.json").exists())
        self.assertIsNone(source.process.poll())

    def test_layout_mapping_then_same_layout_restart_keeps_training_progress(self):
        runner = self.runner()
        self.candidate["actor"]["backend"] = "fsdp:d2p1t2"
        result = runner.apply(self.plan())
        self.assertEqual(result.status, "applied")
        mapped = runner.current
        runner.restart()
        mapped_fixture = read_document(mapped.directory / "native-fixture.json")
        self.assertEqual(mapped_fixture["rng_mapping"], "fsdp-mesh-reseed-v1")
        runner.wait(timeout=8)
        final = read_document(runner.current.directory / "native-fixture.json")
        self.assertEqual(final["configuration"], self.candidate)
        self.assertEqual(final["schedule_step"], 40)
        self.assertEqual(final["rng_mapping"], "preserve")
        self.assertEqual(final["actor_loads"], 1)

    def test_stale_plan_is_rejected_without_stopping_new_instance(self):
        runner = self.runner()
        plan = self.plan()
        runner.restart()
        source = runner.current
        with self.assertRaises(ValueError):
            runner.apply(plan)
        self.assertFalse((source.directory / "stop_request.json").exists())
        self.assertIsNone(source.process.poll())

    def test_megatron_mapping_then_restart_uses_portable_manifest(self):
        self.baseline["actor"]["backend"] = "megatron:d4p1t1"
        self.candidate["actor"]["backend"] = "megatron:d1p2t2"
        ctx = context_for(self.baseline, self.candidate)
        ctx["training_checkpoint"] = {
            "source_ref": "fixture:cpu-engine-contract",
            "format": "areal-megatron-fully-reshardable-v1",
            "model_schema": {"source": "fixture:model", "target": "fixture:model"},
            "optimizer_schema": {
                "source": "fixture:optimizer",
                "target": "fixture:optimizer",
            },
        }
        runner = self.runner()
        result = self.validate(context=ctx)
        self.assertTrue(result.passed, result.to_dict())
        plan = build_apply_plan(
            self.baseline, self.candidate, result, execution=self.execution
        )
        runner.apply(plan)
        mapped = runner.current
        runner.restart()
        middle = read_document(mapped.directory / "native-fixture.json")
        self.assertEqual(middle["rng_mapping"], "megatron-model-parallel-reseed-v1")
        runner.wait(timeout=8)
        final = read_document(runner.current.directory / "native-fixture.json")
        self.assertEqual(final["configuration"], self.candidate)
        self.assertEqual(final["schedule_step"], 40)
        self.assertEqual(final["rng_mapping"], "preserve")
        manifests = list((runner.directory / "snapshots").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 2)
        for path in manifests:
            self.assertEqual(
                read_document(path)["checkpoints"]["actor"],
                "areal-megatron-fully-reshardable-v1",
            )

    def test_corrupt_target_never_confirms_and_preserves_source_snapshot(self):
        runner = self.runner()
        plan = self.plan()
        original_launch = runner._launch

        def launch(*args, **kwargs):
            kwargs["target_file"].write_text("{}")
            return original_launch(*args, **kwargs)

        with patch.object(runner, "_launch", launch), self.assertRaises(RestartError):
            runner.apply(plan)
        self.assertTrue(runner.current.process.group_gone())
        self.assertEqual(runner._configuration_id, configuration_id(self.baseline))
        events = HistoryStore(runner.directory / "events.jsonl").read(
            run_id="apply-test"
        )
        kinds = [e["kind"] for e in events]
        self.assertIn("apply.failed", kinds)
        self.assertNotIn("apply.confirmed", kinds)
        self.assertTrue(list((runner.directory / "snapshots").glob("*/manifest.json")))


if __name__ == "__main__":
    unittest.main()
