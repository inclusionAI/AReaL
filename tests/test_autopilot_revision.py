# SPDX-License-Identifier: Apache-2.0

"""Code revision, independent reference scoring and atomic activation."""

import json
import sys
from dataclasses import asdict, replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.autopilot.revising import AFFINE_SOURCE, CACHE_SOURCE, demo_inputs

from areal.autopilot.contracts import WorkloadStats
from areal.autopilot.elastic.policy.basic.cost_model.config import (
    ActorConfig,
    ParallelLayout,
)
from areal.autopilot.revision import (
    AcceptanceCriteria,
    CommandRevisionAgent,
    PolicyRevisionCycle,
    Proposal,
    RevisionEvaluator,
    RevisionLoop,
    RevisionStore,
    apply_accepted_revision,
    load_accepted_model,
)
from areal.autopilot.revision.contracts import BASELINE_SOURCE, digest, source_digest
from areal.autopilot.revision.execution import CandidateRunner
from areal.autopilot.revision.joint_evaluator import (
    JointOptimizerCriteria,
    JointOptimizerEvaluator,
)
from areal.autopilot.revision.model_adapters import ACTOR_BOUNDARY
from areal.autopilot.revision.model_evaluator import (
    HistoricalCase,
    HistoricalInput,
    HistoricalModelEvaluator,
    HistoricalPrediction,
    HistorySpan,
    ModelAcceptanceCriteria,
)
from areal.autopilot.revision.model_execution import HistoricalSourceEvaluator


@pytest.fixture
def runner():
    # Keep these CPU-only subprocess tests independent of AReaL's eager optional
    # infrastructure imports. Full-runtime validation also runs the default worker.
    root = str(Path(__file__).resolve().parents[1])
    bootstrap = f"import sys,types,runpy; p=types.ModuleType('areal'); p.__path__=[{root!r}+'/areal']; sys.modules['areal']=p; runpy.run_module('areal.autopilot.revision.worker',run_name='__main__')"
    return CandidateRunner(command=(sys.executable, "-c", bootstrap))


@pytest.fixture
def serial_optimizer_clock(monkeypatch):
    """Keep serial protocol tests independent of subprocess startup jitter.

    These cases assert quality/query acceptance and durable deployment. Parent
    wall-clock acceptance is exercised separately by the optimizer timing tests.
    Only this evaluator's clock is replaced; worker execution and subprocess
    timeout enforcement remain real.
    """
    import areal.autopilot.revision.joint_evaluator as module

    ticks = count()
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: next(ticks) * 0.001)
    )


def evaluator(target, runner, **kwargs):
    profile, cases = demo_inputs(target)
    return RevisionEvaluator(target, profile, cases, runner=runner, **kwargs)


def phase_runtime(root):
    """Real queue/checkpoint protocol with a deterministic measured CPU service."""
    from tests.test_elastic_loop import ProtocolGroup, ProtocolPublisher

    from areal.autopilot.contracts import BatchWorkload, TrainingConfig
    from areal.autopilot.elastic.loop import ElasticLoop
    from areal.autopilot.elastic.policy import BasicPolicy
    from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer
    from areal.autopilot.elastic.policy.basic.simulator import WorkflowSimulator
    from areal.autopilot.elastic.runtime.history import RuntimeHistory
    from areal.autopilot.elastic.runtime.policy import PolicyApplication
    from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
    from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer
    from areal.autopilot.revision.contracts import workflow_from_profile

    now = [0.0]
    profile, _ = demo_inputs("simulator")

    class Group(ProtocolGroup):
        @staticmethod
        def workload(payload):
            return BatchWorkload(1, payload["tokens"] // 2, payload["tokens"])

        def start(self, *, restore=False):
            now[0] += 3 if restore else 4
            super().start(restore=restore)

        def train_trajectory(self, payload):
            now[0] += 5 + payload["tokens"] / 10
            return super().train_trajectory(payload)

        def checkpoint(self, *, data_cursor):
            now[0] += 1
            return super().checkpoint(data_cursor=data_cursor)

        def stop(self):
            now[0] += int(self.actor is not None)
            super().stop()

    class Publisher(ProtocolPublisher):
        def publish(self, checkpoint):
            now[0] += 1
            return super().publish(checkpoint)

    group = Group(root / "checkpoints")
    group.compatibility = profile["compatibility"]
    group.training_config, group.pool_devices = TrainingConfig(1), [0]
    queue = BatchQueue(root / "queue")
    application = PolicyApplication(
        root / "policies",
        compatibility=group.compatibility,
        execution_context={
            "queue_id": queue.queue_id,
            "checkpoint_root": str(group.store.root.resolve()),
            "inference_gpus": 1,
        },
    )
    history = RuntimeHistory(
        root / "runtime.json",
        context={**application.execution_context, "compatibility": group.compatibility},
    )
    policy = BasicPolicy(
        WorkflowSimulator(workflow_from_profile(profile)),
        optimizer=BasicOptimizer(horizon=1, max_candidates=8),
    )
    loop = ElasticLoop(
        QueuedTrainer(group, queue),
        Publisher(),
        policy=policy,
        policy_application=application,
        runtime_history=history,
        clock=lambda: now[0],
    )
    loop.start()
    return loop, group, queue, now, profile


def measured_update(loop, queue, now, tokens):
    with loop.episode() as version:
        queue.enqueue(
            str(version),
            {"policy_version": version, "value": version + 1, "tokens": tokens},
            policy_version=version,
        )
    assert loop.tick() == "ACTIVE"
    now[0] += 31
    assert loop.tick() == "SUSPENDED"


def accepted_bundle(root, runner):
    from areal.autopilot.revision.online import LifecycleRevisionCycle, runtime_problems

    loop, _, queue, now, profile = phase_runtime(root / "calibration")
    try:
        for tokens in (100, 400, 200, 300):
            measured_update(loop, queue, now, tokens)
        evaluation, cases = runtime_problems(
            profile,
            loop.runtime_history.snapshot(4),
            settings={"horizon": 1, "max_candidates": 8},
            runner=runner,
        )
    finally:
        loop.close()

    class Agent:
        def propose(self, request):
            return Proposal(
                AFFINE_SOURCE
                if request.target == "simulator"
                else BASELINE_SOURCE["optimizer"]
            )

    result = LifecycleRevisionCycle(evaluation, cases, root / "cycle").run(
        Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1
    )
    assert result["model"]["accepted"] and not result["optimizer"]["accepted"]
    return Path(result["bundle"]), evaluation


def test_live_policy_activation_preserves_training_history_and_recovers_on_restart(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.policy import PolicyRequest

    bundle, _ = accepted_bundle(tmp_path / "revision", runner)
    loop, group, queue, now, _ = phase_runtime(tmp_path / "execution")
    try:
        measured_update(loop, queue, now, 200)
        previous_history = loop.history
        snapshot = previous_history.snapshot()
        prepared = loop.prepare_policy(bundle)
        before = (group.step, group.starts[:], group.releases, group.saves)
        request = PolicyRequest("activate", prepared.bundle_id, 0, 1)
        receipt = loop.apply_policy(request, prepared)
        assert (
            loop.history is previous_history
            and loop._policy.history is previous_history
        )
        assert loop.history.snapshot() == snapshot and loop.policy_generation == 1
        assert (group.step, group.starts, group.releases, group.saves) == before
        measured_update(loop, queue, now, 300)
        assert loop.apply_policy(request, prepared) == receipt
        assert loop.policy_generation == 1 and group.updates == 2
        persisted = loop.history.snapshot()
    finally:
        loop.close()
    restored, recovered, _, _, _ = phase_runtime(tmp_path / "execution")
    try:
        assert (
            restored.policy_bundle_id == prepared.bundle_id
            and restored.policy_generation == 1
        )
        assert recovered.step == 2 and recovered.updates == 0
        assert restored.history.snapshot().workloads == persisted.workloads
        assert (
            restored.history.snapshot().measurements[: len(persisted.measurements)]
            == persisted.measurements
        )
    finally:
        restored.close()


@pytest.mark.parametrize("after_replace", [False, True])
def test_policy_pointer_failure_recovers_the_durable_choice_without_training_replay(
    runner, tmp_path, monkeypatch, after_replace
):
    import areal.autopilot.elastic.runtime.policy as module

    bundle, _ = accepted_bundle(tmp_path / "revision", runner)
    loop, group, queue, now, _ = phase_runtime(tmp_path / "execution")
    try:
        measured_update(loop, queue, now, 200)
        prepared = loop.prepare_policy(bundle)
        request = module.PolicyRequest("activate", prepared.bundle_id, 0, 1)
        original = module.atomic_json

        def interrupt(path, value):
            if path.name == "active.json" and value["generation"] == 1:
                if after_replace:
                    original(path, value)
                raise OSError("policy pointer interruption")
            original(path, value)

        monkeypatch.setattr(module, "atomic_json", interrupt)
        with pytest.raises(OSError, match="pointer interruption"):
            loop.apply_policy(request, prepared)
        assert loop.state == "FAILED" and not loop.publisher.ready
        monkeypatch.setattr(module, "atomic_json", original)
        loop.recover_policy()
        assert loop.publisher.ready and group.updates == group.step == 1
        assert loop.policy_generation == int(after_replace)
        assert (loop.policy_bundle_id == prepared.bundle_id) == after_replace
        measured_update(loop, queue, now, 300)
        assert group.updates == group.step == 2
    finally:
        loop.close()


def test_policy_rejects_stale_requests_changed_sources_and_episode_boundary(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.policy import (
        PolicyBoundaryBusy,
        PolicyRequest,
        validate_bundle,
    )

    bundle, _ = accepted_bundle(tmp_path / "revision", runner)
    loop, group, queue, now, _ = phase_runtime(tmp_path / "execution")
    try:
        prepared = loop.prepare_policy(bundle)
        with pytest.raises(ValueError, match="Stale"):
            loop.apply_policy(
                PolicyRequest("stale", prepared.bundle_id, 0, 1), prepared
            )
        with loop.episode():
            with pytest.raises(PolicyBoundaryBusy):
                loop.apply_policy(
                    PolicyRequest("busy", prepared.bundle_id, 0, 0), prepared
                )
        value = json.loads(bundle.read_text())
        value["components"]["simulator"]["source"] += "\n# changed"
        value["bundle_id"] = digest(
            {k: v for k, v in value.items() if k != "bundle_id"}
        )
        with pytest.raises(ValueError, match="source changed"):
            validate_bundle(value)
        assert (
            loop.publisher.ready and group.starts == [] and loop.policy_generation == 0
        )
    finally:
        loop.close()


def test_runtime_revision_cycle_freezes_new_model_then_applies_both_components(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.policy import PolicyRequest
    from areal.autopilot.revision.online import LifecycleRevisionCycle, runtime_problems

    loop, group, queue, now, profile = phase_runtime(tmp_path / "execution")
    requests = []

    class Agent:
        def propose(self, request):
            requests.append(request)
            return Proposal(
                AFFINE_SOURCE if request.target == "simulator" else CACHE_SOURCE
            )

    try:
        for tokens in (100, 400, 200, 300):
            measured_update(loop, queue, now, tokens)
        window = loop.runtime_history.snapshot(4)
        model, cases = runtime_problems(
            profile, window, settings={"horizon": 1, "max_candidates": 8}, runner=runner
        )
        cycle = LifecycleRevisionCycle(model, cases, tmp_path / "cycle")
        result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
        assert result["model"]["accepted"] and result["optimizer"]["accepted"]
        bundle = Path(result["bundle"])
        value = json.loads(bundle.read_text())
        assert value["components"]["simulator"]["source"] == AFFINE_SOURCE
        assert value["components"]["optimizer"]["source"] == CACHE_SOURCE
        assert (
            value["evaluation"]["context"]["deployment_context"]["simulator_source"]
            == AFFINE_SOURCE
        )
        assert all(
            "observed" not in str(request.calibration.get("validation", {}))
            for request in requests
        )
        prepared = loop.prepare_policy(bundle)
        loop.apply_policy(PolicyRequest("cycle", prepared.bundle_id, 0, 4), prepared)
        measured_update(loop, queue, now, 250)
        assert group.updates == group.step == 5 and loop.policy_generation == 1
        count = len(requests)
        assert (
            cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
            == result
        )
        assert len(requests) == count
    finally:
        loop.close()


def test_runtime_revision_background_budget_and_restart_do_not_repeat_agent_calls(
    runner, tmp_path
):
    import time

    from areal.autopilot.revision.online import RuntimeRevisionLoop

    loop, group, queue, now, profile = phase_runtime(tmp_path / "execution")
    calls = []

    class Agent:
        def propose(self, request):
            calls.append(request.target)
            return Proposal(
                AFFINE_SOURCE if request.target == "simulator" else CACHE_SOURCE
            )

    arguments = dict(
        profile=profile,
        settings={"horizon": 1, "max_candidates": 8},
        model_agent=Agent(),
        optimizer_agent=Agent(),
        max_windows=1,
        max_model_rounds=1,
        max_optimizer_rounds=1,
        runner=runner,
    )
    revision = RuntimeRevisionLoop(tmp_path / "online", **arguments)
    try:
        for tokens in (100, 400, 200, 300):
            measured_update(loop, queue, now, tokens)
        revision.tick(loop)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "online").glob("window-*/outcome.json")):
            assert time.monotonic() < deadline
            revision.tick(loop)
            time.sleep(0.02)
        assert loop.policy_generation == 1 and calls == ["simulator", "optimizer"]
        measured_update(loop, queue, now, 250)
        revision.tick(loop)
        assert loop.policy_generation == 1 and len(calls) == 2
    finally:
        revision.close()
    recovered = RuntimeRevisionLoop(tmp_path / "online", **arguments)
    try:
        recovered.tick(loop)
        assert len(calls) == 2 and recovered.future is None
    finally:
        recovered.close()
        loop.close()


def phase_evaluator(runner):
    """Measured train service times with public inputs and private timing labels."""
    profile, cases = demo_inputs("simulator")
    rows = []
    for case in cases[:-1]:
        count = case["plan"][0]["config"]["world_size"]
        row = json.loads(json.dumps(case))
        row["phase"] = "train"
        row["observation"]["training_config"] = row["plan"][0]["config"]
        row["observed"]["seconds"] = 5 + row["observation"]["next_batch"][
            "sequence_tokens"
        ] / (10 * count)
        rows.append(row)
    return RevisionEvaluator("simulator", profile, rows, runner=runner)


@pytest.mark.parametrize("kind", ["stateful", "extra_cost"])
def test_phase_gate_rejects_unscored_cost_and_stateful_estimates(
    runner, tmp_path, kind
):
    source = """from dataclasses import replace
from areal.autopilot.elastic.policy.basic.simulator import Estimate, WorkflowSimulator
class Train:
    def __init__(self): self.calls = 0
    def estimate(self, context):
        self.calls += 1
        return Estimate(SECONDS, extra_gpu_seconds=EXTRA)
def build_simulator(workflow):
    return WorkflowSimulator(replace(workflow, train=Train()))
""".replace("SECONDS", "self.calls" if kind == "stateful" else "10").replace(
        "EXTRA", "1" if kind == "extra_cost" else "0"
    )
    report = phase_evaluator(runner).evaluate(source, tmp_path)
    assert not report.valid
    assert (
        "same instance" in str(report.errors)
        if kind == "stateful"
        else "service seconds" in str(report.errors)
    )


def test_phase_handoff_keeps_accounting_fixed_and_reference_queries_frozen(
    runner, tmp_path
):
    from areal.autopilot.revision.phase import build_phase_simulator

    model = phase_evaluator(runner)
    source = AFFINE_SOURCE.replace(
        "return WorkflowSimulator(replace(workflow, train=AffineTrain(workflow.train)))",
        """candidate = WorkflowSimulator(replace(workflow, train=AffineTrain(workflow.train), inference_gpus=0))
    candidate.evaluate = lambda *args: (_ for _ in ()).throw(RuntimeError("unverified evaluate"))
    return candidate""",
    )
    report = model.evaluate(source, tmp_path / "model")
    assert report.valid
    profile, cases = demo_inputs("optimizer")
    reference = {
        "cases": json.loads(model._public_json),
        "prediction_sha256": report.metrics["prediction_sha256"],
    }
    search = RevisionEvaluator(
        "optimizer",
        profile,
        cases,
        runner=runner,
        reference_simulator_source=source,
        phase_reference=reference,
    )
    fixed = build_phase_simulator(source, profile)
    assert fixed.workflow.inference_gpus == 1
    # Neither candidate queries nor final scoring call the mutable parent again.
    search._reference.evaluate = lambda *args: (_ for _ in ()).throw(
        AssertionError("reference was not frozen")
    )
    assert search.evaluate(CACHE_SOURCE, tmp_path / "optimizer").valid


def test_phase_handoff_rejects_parent_worker_environment_drift(
    runner, tmp_path, monkeypatch
):
    model = phase_evaluator(runner)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    source = AFFINE_SOURCE.replace(
        "return Estimate(max(0,",
        'return Estimate(float(bool(__import__("os").environ.get("CUDA_VISIBLE_DEVICES"))) + max(0,',
    )
    report = model.evaluate(source, tmp_path / "model")
    assert report.valid
    profile, cases = demo_inputs("optimizer")
    with pytest.raises(ValueError, match="accepted worker predictions"):
        RevisionEvaluator(
            "optimizer",
            profile,
            cases,
            runner=runner,
            reference_simulator_source=source,
            phase_reference={
                "cases": json.loads(model._public_json),
                "prediction_sha256": report.metrics["prediction_sha256"],
            },
        )


def test_model_bundle_cannot_skip_optimizer_revalidation(runner, tmp_path):
    from areal.autopilot.elastic.runtime.policy import (
        export_accepted_policy,
        validate_bundle,
    )

    bundle, model = accepted_bundle(tmp_path, runner)
    with pytest.raises(ValueError, match="optimizer revalidation"):
        export_accepted_policy(tmp_path / "cycle/model", model, tmp_path / "unsafe")
    value = json.loads(bundle.read_text())
    value["optimizer_verification"] = None
    value["bundle_id"] = digest({k: v for k, v in value.items() if k != "bundle_id"})
    with pytest.raises(ValueError, match="missing optimizer revalidation"):
        validate_bundle(value)


def test_policy_recovery_logging_failure_cannot_reopen_other_recovery_entry(
    runner, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from areal.autopilot.elastic.runtime.policy import PolicyRequest

    bundle, _ = accepted_bundle(tmp_path / "revision", runner)
    loop, _, _, _, _ = phase_runtime(tmp_path / "execution")
    original = loop._record

    def fail_record(event, **kwargs):
        if event in {"POLICY_APPLIED", "POLICY_RECOVERED"}:
            raise OSError("journal failure")
        return original(event, **kwargs)

    try:
        prepared = loop.prepare_policy(bundle)
        monkeypatch.setattr(loop, "_record", fail_record)
        with pytest.raises(OSError, match="journal failure"):
            loop.apply_policy(
                PolicyRequest("activation", prepared.bundle_id, 0, 0), prepared
            )
        with pytest.raises(OSError, match="journal failure"):
            loop.recover_policy()
        assert loop.state == "FAILED" and not loop.publisher.ready
        loop.config_application = SimpleNamespace()
        with pytest.raises(RuntimeError, match="pending policy activation"):
            loop.recover_config()
        loop.config_application = None
        with pytest.raises(RuntimeError, match="not running"):
            loop.tick()
        monkeypatch.setattr(loop, "_record", original)
        loop.recover_policy()
        assert loop.publisher.ready and loop.policy_generation == 1
    finally:
        loop.config_application = None
        monkeypatch.setattr(loop, "_record", original)
        loop.close()


def test_second_window_can_keep_model_and_accept_optimizer_with_new_probes(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.policy import PolicyRequest
    from areal.autopilot.revision.online import LifecycleRevisionCycle, runtime_problems

    bundle, _ = accepted_bundle(tmp_path / "first", runner)
    loop, group, queue, now, profile = phase_runtime(tmp_path / "execution")

    class Agent:
        def propose(self, request):
            return Proposal(
                AFFINE_SOURCE if request.target == "simulator" else CACHE_SOURCE
            )

    try:
        for tokens in (100, 400, 200, 300):
            measured_update(loop, queue, now, tokens)
        prepared = loop.prepare_policy(bundle)
        loop.apply_policy(PolicyRequest("first", prepared.bundle_id, 0, 4), prepared)
        measured_update(loop, queue, now, 250)
        model, cases = runtime_problems(
            profile,
            loop.runtime_history.snapshot(5),
            settings={"horizon": 1, "max_candidates": 8},
            runner=runner,
        )
        cycle = LifecycleRevisionCycle(
            model,
            cases,
            tmp_path / "second",
            initial_model_source=AFFINE_SOURCE,
            parent_bundle=bundle,
        )
        result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
        assert not result["model"]["accepted"] and result["optimizer"]["accepted"]
        value = json.loads(Path(result["bundle"]).read_text())
        assert value["model_verification"]["report"]["valid"]
        assert value["model_evaluation"]["acceptance"]["accepted"]
        prepared = loop.prepare_policy(Path(result["bundle"]))
        loop.apply_policy(PolicyRequest("second", prepared.bundle_id, 1, 5), prepared)
        measured_update(loop, queue, now, 260)
        assert loop.policy_generation == 2 and group.step == 6
    finally:
        loop.close()


def test_revision_shutdown_failure_still_cleans_training_and_inference(
    tmp_path, monkeypatch
):
    import argparse
    import asyncio
    import types

    from examples.autopilot.elastic.automatic_swe import run
    from tests.test_elastic_loop import ProtocolGroup

    import areal.autopilot.revision.online as online

    profile, _ = demo_inputs("simulator")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps(profile))
    actions = []

    class Group(ProtocolGroup):
        pool_uuids = ["cpu-protocol"]

        def configure(self, config):
            self.training_config = config

        def close(self):
            actions.append("training_closed")
            self.stop()

    group = Group(tmp_path / "checkpoints")
    group.compatibility = profile["compatibility"]
    group.events = []

    class Service:
        gpu_uuid = "cpu-inference"

        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("injected service startup failure")

        def close(self):
            actions.append("inference_closed")
            return {"closed": True}

    class Revision:
        def __init__(self, *args, **kwargs):
            pass

        def close(self, loop):
            actions.append("revision_closed")
            raise OSError("injected revision shutdown failure")

    replacements = {
        "examples.autopilot.elastic.fixtures.groups": {
            "create_ppo_group": lambda **kwargs: group
        },
        "examples.autopilot.elastic.fixtures.inference_service": {
            "InferenceService": Service
        },
        "examples.autopilot.elastic.validation.probes": {"probe_inference": None},
    }
    for name, attributes in replacements.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(online, "RuntimeRevisionLoop", Revision)
    args = argparse.Namespace(
        output=tmp_path / "output",
        state_root=tmp_path / "state",
        model=tmp_path / "model",
        model_revision="cpu-fixture",
        model_name="fixture",
        devices=[0],
        inference_device=1,
        policy="basic",
        profiles=profile_path,
        initial_train_gpus=1,
        horizon=1,
        cost_scope="total",
        idle_seconds=30,
        rounds=1,
        revision_command=["unused-agent"],
        revision_every_updates=4,
        max_revision_windows=1,
        model_revision_rounds=1,
        optimizer_revision_rounds=1,
    )
    with pytest.raises(OSError, match="revision shutdown failure"):
        asyncio.run(run(args))
    assert actions == ["revision_closed", "training_closed", "inference_closed"]
    report = json.loads((args.output / "report.json").read_text())
    assert report["status"] == "failed" and report["inference_cleanup"]["closed"]
    assert "service startup failure" in report["error"]


@pytest.mark.parametrize(
    "target,source", [("simulator", AFFINE_SOURCE), ("optimizer", CACHE_SOURCE)]
)
def test_component_revisions_improve_independent_score(
    target, source, runner, tmp_path
):
    evaluation = evaluator(target, runner)
    baseline = evaluation.evaluate(BASELINE_SOURCE[target], tmp_path / "baseline")
    candidate = evaluation.evaluate(source, tmp_path / "candidate")
    assert baseline.valid, baseline.errors
    assert candidate.valid, candidate.errors
    assert AcceptanceCriteria().compare(candidate, baseline)[0]
    request = json.loads((tmp_path / "candidate/request.json").read_text())
    assert all("observed" not in case for case in request["cases"])
    if target == "simulator":
        assert baseline.score > 0.01 and candidate.score < 1e-10
    else:
        assert (
            candidate.metrics["mean_gpu_seconds"]
            == baseline.metrics["mean_gpu_seconds"]
        )
        assert (
            candidate.metrics["mean_reference_queries"]
            < baseline.metrics["mean_reference_queries"]
        )


def test_loop_rejects_bad_candidate_retries_feedback_and_loads_accepted(
    runner, tmp_path
):
    evaluation = evaluator("simulator", runner)
    requests = []

    class Agent:
        def propose(self, request):
            requests.append(request)
            if len(requests) == 1:
                return Proposal("invalid python !", "First draft has a syntax error")
            return Proposal(AFFINE_SOURCE, "Fit affine costs from calibration")

    loop = RevisionLoop(Agent(), evaluation, tmp_path / "loop")
    result = loop.run(max_rounds=2)
    assert [x["status"] for x in result["rounds"]] == ["rejected", "accepted"]
    assert any("SyntaxError" in str(x) for x in requests[1].feedback)
    assert result["accepted"] == 1
    path, pointer = RevisionStore(tmp_path / "loop").active()
    assert path.read_text() == AFFINE_SOURCE
    simulator, optimizer = apply_accepted_revision(
        tmp_path / "loop", evaluation.calibration, evaluation.settings
    )
    assert callable(simulator.evaluate) and callable(optimizer.optimize)
    assert pointer["benchmark_sha256"] == evaluation.benchmark_sha256


def test_optimizer_cannot_forge_score_or_reference_query_count(runner, tmp_path):
    source = """from dataclasses import replace
from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer
from areal.autopilot.elastic.policy.basic.simulator import SimulationResult
class Forged:
    def __init__(self, settings): self.base=BasicOptimizer(**settings)
    def optimize(self, obs, simulator, history):
        result=self.base.optimize(obs, simulator, history)
        return replace(result, evaluated=0, prediction=SimulationResult(True))
def build_optimizer(settings): return Forged(settings)
"""
    evaluation = evaluator("optimizer", runner)
    baseline = evaluation.evaluate(BASELINE_SOURCE["optimizer"], tmp_path / "baseline")
    forged = evaluation.evaluate(source, tmp_path / "forged")
    assert forged.valid, forged.errors
    assert (
        forged.metrics["mean_reference_queries"]
        == baseline.metrics["mean_reference_queries"]
        > 0
    )
    assert forged.score == baseline.score
    assert not AcceptanceCriteria().compare(forged, baseline)[0]


def test_fixed_forecast_horizon_prevents_shorter_data_cheat(runner, tmp_path):
    source = BASELINE_SOURCE["optimizer"].replace(
        "BasicOptimizer(**settings)", "BasicOptimizer(**{**settings, 'horizon': 1})"
    )
    report = evaluator("optimizer", runner).evaluate(source, tmp_path / "candidate")
    assert not report.valid
    assert "fixed evaluation workload" in str(
        report.errors
    ) or "fixed data horizon" in str(report.errors)


def test_optimizer_budget_is_enforced_outside_candidate(runner, tmp_path):
    source = """from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer
class Excessive:
    def __init__(self, settings): self.settings=settings
    def optimize(self, obs, simulator, history):
        first=BasicOptimizer(**self.settings).optimize(obs, simulator, history)
        if first.plan:
            try: simulator.evaluate(obs, __import__('areal.autopilot.elastic.policy.basic.simulator',fromlist=['forecast_window']).forecast_window(obs,history,1), first.plan, history)
            except ValueError: pass
        return first
def build_optimizer(settings): return Excessive(settings)
"""
    report = evaluator(
        "optimizer", runner, settings={"horizon": 1, "max_candidates": 6}
    ).evaluate(source, tmp_path / "candidate")
    assert not report.valid and "search budget exceeded" in str(report.errors).lower()


def test_worker_timeout_does_not_activate_candidate(runner, tmp_path):
    report = evaluator("simulator", replace(runner, timeout_seconds=0.6)).evaluate(
        "while True: pass\n", tmp_path / "candidate"
    )
    assert not report.valid and "timed out" in str(report.errors)


def test_rejected_or_duplicate_candidate_preserves_active_pointer(runner, tmp_path):
    class Good:
        def propose(self, request):
            return Proposal(AFFINE_SOURCE)

    evaluation = evaluator("simulator", runner)
    root = tmp_path / "loop"
    RevisionLoop(Good(), evaluation, root).run()
    before = (root / "active.json").read_bytes()
    result = RevisionLoop(Good(), evaluation, root).run(max_rounds=2)
    assert result["accepted"] == 0 and all(
        x["reason"] == "Duplicate candidate" for x in result["rounds"]
    )
    assert (root / "active.json").read_bytes() == before


def test_store_rejects_other_benchmark_and_changed_accepted_source(runner, tmp_path):
    class Agent:
        def propose(self, request):
            return Proposal(AFFINE_SOURCE)

    evaluation = evaluator("simulator", runner)
    root = tmp_path / "loop"
    RevisionLoop(Agent(), evaluation, root).run()
    profile, cases = demo_inputs("simulator")
    cases[0]["observed"]["seconds"] += 1
    other = RevisionEvaluator("simulator", profile, cases, runner=runner)
    with pytest.raises(ValueError, match="another fixed evaluation"):
        RevisionLoop(Agent(), other, root).run()
    path, _ = RevisionStore(root).active()
    path.write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        RevisionStore(root).active()


def test_calibration_and_context_are_independent_copies(runner):
    evaluation = evaluator("simulator", runner)
    before = digest(evaluation.calibration)
    changed = evaluation.calibration
    changed["measurements"][0]["seconds"] = 1000
    changed_settings = evaluation.settings
    changed_settings["max_candidates"] = 1
    assert digest(evaluation.calibration) == before
    assert evaluation.settings["max_candidates"] == 100000


def test_command_agent_roundtrip_uses_structured_json(tmp_path):
    from areal.autopilot.revision.contracts import ProposalRequest

    command = (
        sys.executable,
        "-c",
        "import json,sys; x=json.load(sys.stdin); assert 'observed' not in x; sys.stdout.write(json.dumps({'source':x['incumbent_source'],'rationale':str(x['round'])}))",
    )
    request = ProposalRequest(
        "simulator", 2, AFFINE_SOURCE, "hash", {"measurements": []}, {}, ()
    )
    proposal = CommandRevisionAgent(command).propose(request)
    assert proposal.source == AFFINE_SOURCE and proposal.rationale == "2"


def test_measured_policy_can_construct_with_accepted_component(runner, tmp_path):
    from examples.autopilot.elastic.basic_policy import measured_policy

    class Agent:
        def propose(self, request):
            return Proposal(CACHE_SOURCE)

    evaluation = evaluator("optimizer", runner)
    root = tmp_path / "loop"
    RevisionLoop(Agent(), evaluation, root).run()
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(evaluation.calibration))
    policy = measured_policy(profile, revision_root=root)
    assert type(policy.optimizer).__name__ == "CachedOptimizer"
    changed = dict(evaluation.calibration, inference_gpus=2)
    with pytest.raises(ValueError, match="model/residency/search"):
        apply_accepted_revision(root, changed, evaluation.settings)


def test_simulator_must_predict_known_feasibility(runner, tmp_path):
    source = """from areal.autopilot.elastic.policy.basic.simulator import SimulationResult
class AlwaysOff:
    def evaluate(self,*args): return SimulationResult(False)
def build_simulator(workflow): return AlwaysOff()
"""
    report = evaluator("simulator", runner).evaluate(source, tmp_path / "candidate")
    assert report.valid and report.metrics["feasibility_errors"] == 6
    baseline = evaluator("simulator", runner).evaluate(
        BASELINE_SOURCE["simulator"], tmp_path / "baseline"
    )
    assert not AcceptanceCriteria().compare(report, baseline)[0]


def test_cost_regression_cannot_be_hidden_by_search_speed():
    from areal.autopilot.revision.contracts import EvaluationReport

    baseline = EvaluationReport(
        "optimizer", "a", "fixed", True, 1.05, {"mean_gpu_seconds": 10}
    )
    candidate = EvaluationReport(
        "optimizer", "b", "fixed", True, 1.01, {"mean_gpu_seconds": 11}
    )
    assert not AcceptanceCriteria().compare(candidate, baseline)[0]


def test_failed_pointer_publication_can_resume_without_false_activation(
    runner, tmp_path, monkeypatch
):
    import areal.autopilot.revision.loop as module

    class Agent:
        def propose(self, request):
            return Proposal(AFFINE_SOURCE)

    root = tmp_path / "loop"
    loop = RevisionLoop(Agent(), evaluator("simulator", runner), root)
    original = module.atomic_json

    def fail_pointer(path, value):
        if path.name == "active.json":
            raise OSError("injected pointer failure")
        original(path, value)

    monkeypatch.setattr(module, "atomic_json", fail_pointer)
    with pytest.raises(OSError, match="pointer failure"):
        loop.run()
    assert not (root / "active.json").exists()
    monkeypatch.setattr(module, "atomic_json", original)
    assert loop.run()["accepted"] == 1


def test_demo_respects_custom_agent_and_can_resume_without_new_acceptance(
    monkeypatch, tmp_path
):
    import examples.autopilot.revising as cli

    seen = []

    class Loop:
        def __init__(self, agent, evaluation, root):
            seen.append(agent.command)

        def run(self, **kwargs):
            return {"active_source": "already-accepted.py", "accepted": 0}

    monkeypatch.setattr(cli, "RevisionLoop", Loop)
    monkeypatch.setattr(cli, "RevisionEvaluator", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "revising",
            "--target",
            "optimizer",
            "--demo",
            "--output",
            str(tmp_path),
            "--agent-command",
            "custom-agent",
        ],
    )
    cli.main()
    assert seen == [("custom-agent",)]
    monkeypatch.setattr(
        sys,
        "argv",
        ["revising", "--target", "simulator", "--demo", "--output", str(tmp_path)],
    )
    cli.main()


def test_loop_can_repair_an_inaccurate_incumbent_feasibility_prediction(
    runner, tmp_path
):
    profile, cases = demo_inputs("simulator")
    case = cases[0]
    workload = {
        "samples": 1,
        "effective_tokens": 1000,
        "sequence_tokens": 2000,
        "rollout_seconds": 0,
    }
    case["observation"]["next_batch"] = workload
    case["forecast"][0]["workload"] = workload
    case["observed"] = {"feasible": True, "seconds": 211, "total_gpu_seconds": 422}
    evaluation = RevisionEvaluator("simulator", profile, [case], runner=runner)
    initial = evaluation.evaluate(BASELINE_SOURCE["simulator"], tmp_path / "initial")
    assert initial.valid and initial.metrics["feasibility_errors"] == 1

    class Agent:
        def propose(self, request):
            return Proposal(AFFINE_SOURCE)

    assert RevisionLoop(Agent(), evaluation, tmp_path / "loop").run()["accepted"] == 1


def test_invalid_overflow_predictions_produce_json_safe_feedback(runner, tmp_path):
    from dataclasses import asdict

    source = """from areal.autopilot.elastic.policy.basic.simulator import SimulationResult
class Huge:
    def evaluate(self,obs,forecast,plan,history):
        return SimulationResult(True,1e308,1e308,1e308,1e308,
            sum(x.workload.samples for x in forecast),sum(x.workload.effective_tokens for x in forecast))
def build_simulator(workflow): return Huge()
"""
    report = evaluator("simulator", runner).evaluate(source, tmp_path / "candidate")
    assert not report.valid
    json.dumps(asdict(report), allow_nan=False)


# Historical model evaluation uses measured labels, independent of the legacy
# lifecycle code-generation evaluator above. All fixtures below are synthetic.


def history_case(step, tokens, *, split="validation", measured=None, dp=1, group="all"):
    inputs = HistoricalInput(
        "actor",
        ActorConfig("archon", ParallelLayout(dp=dp), max_tokens_per_mb=256),
        WorkloadStats(4, tokens),
        "synthetic-runtime-model-hardware",
        ACTOR_BOUNDARY,
    )
    return HistoricalCase(
        f"step-{step}",
        HistorySpan("run", step, step),
        split,
        inputs,
        (("train_step_seconds", tokens if measured is None else measured),),
        f"synthetic-trace:{step}",
        group,
    )


class TokenHistoryPredictor:
    def __init__(self, scale=1.0, missing=(), overrides=None):
        self.scale, self.missing, self.overrides = scale, missing, overrides or {}
        self.seen = []

    def predict(self, inputs):
        assert not hasattr(inputs, "measured") and not hasattr(inputs, "sample_id")
        self.seen.append(inputs)
        tokens = inputs.workload.sequence_tokens
        if tokens in self.missing:
            return HistoricalPrediction(reason="outside synthetic model coverage")
        return HistoricalPrediction(
            (("train_step_seconds", self.overrides.get(tokens, tokens * self.scale)),)
        )


def history_evaluator(cases=None, **kwargs):
    return HistoricalModelEvaluator(
        tuple(cases or (history_case(1, 10), history_case(2, 100))),
        primary_metrics=("train_step_seconds",),
        **kwargs,
    )


def test_history_evaluator_hides_labels_skips_final_and_scores_fixed_real_configs():
    """Only validation workloads reach prediction; calibration/final stay separate."""
    cases = (
        history_case(0, 4, split="calibration"),
        history_case(1, 10),
        history_case(2, 100, dp=2),
        history_case(3, 999, split="final_holdout"),
    )
    evaluation = history_evaluator(cases)
    baseline_model, revised_model = TokenHistoryPredictor(1.2), TokenHistoryPredictor()
    before = evaluation.evaluate(baseline_model, "v0", fitted_on=(cases[0].span,))
    after = evaluation.evaluate(revised_model, "v1", fitted_on=(cases[0].span,))
    assert before.valid and after.valid
    assert before.score == pytest.approx(0.2) and after.score == 0
    assert evaluation.compare(after, before).accepted
    assert [x.workload.sequence_tokens for x in revised_model.seen] == [10, 100]
    assert [x.config.parallel.dp for x in revised_model.seen] == [1, 2]
    assert after.reserved_cases == 1
    assert len(evaluation.calibration) == 1
    feedback = after.feedback()
    assert "rows" not in feedback and "measured_sum" not in feedback["metrics"][0]
    assert len([x for x in after.slices if x.group_by == "config_id"]) == 2


def test_history_metrics_use_pooled_wape_nearest_rank_p90_and_missing_label_counts():
    """WAPE is sum absolute error / sum labels, not mean percentage error."""
    cases = tuple(
        replace(
            history_case(i, n),
            measured=(("train_step_seconds", n), ("compute_advantage_seconds", None)),
        )
        for i, n in ((1, 10), (2, 100))
    )
    evaluation = history_evaluator(cases)
    report = evaluation.evaluate(
        TokenHistoryPredictor(overrides={10: 20, 100: 110}), "v0", fitted_on=()
    )
    metrics = {m.metric: m for m in report.metrics}
    train = metrics["train_step_seconds"]
    assert train.wape == pytest.approx(20 / 110)
    assert train.p90_relative_error == 1 and train.covered == train.labelled == 2
    assert metrics["compute_advantage_seconds"].labelled == 0
    assert metrics["compute_advantage_seconds"].wape is None
    assert {r.metric for r in report.rows} == {
        "train_step_seconds",
        "compute_advantage_seconds",
    }


def test_history_rejects_equal_coverage_that_swaps_hard_samples():
    """Identical coverage percentages cannot hide lost sample/metric cells."""
    evaluation = history_evaluator()
    before = evaluation.evaluate(
        TokenHistoryPredictor(2, missing=(100,)), "old", fitted_on=()
    )
    after = evaluation.evaluate(
        TokenHistoryPredictor(missing=(10,)), "new", fitted_on=()
    )
    assert before.metrics[0].covered == after.metrics[0].covered == 1
    decision = evaluation.compare(after, before)
    assert not decision.accepted and decision.lost_cells == decision.added_cells == 1


def test_history_rejects_bad_new_coverage_despite_improvement_on_old_cells():
    """New predictions cannot piggyback on small gains in the old coverage set."""
    evaluation = history_evaluator()
    before = evaluation.evaluate(
        TokenHistoryPredictor(1.2, missing=(100,)), "old", fitted_on=()
    )
    after = evaluation.evaluate(
        TokenHistoryPredictor(overrides={10: 11, 100: 10000}), "new", fitted_on=()
    )
    decision = evaluation.compare(after, before)
    assert not decision.accepted and "New coverage" in decision.reason
    good = evaluation.evaluate(
        TokenHistoryPredictor(overrides={10: 11, 100: 105}), "good", fitted_on=()
    )
    decision = evaluation.compare(good, before)
    assert decision.accepted and decision.candidate_score == pytest.approx(0.1)
    assert decision.retained_cells == decision.added_cells == 1


@pytest.mark.parametrize("by_config", [False, True])
def test_history_slice_regression_cannot_hide_behind_long_trial(by_config):
    """A large total workload cannot hide degradation in a smaller fixed slice."""
    cases = (
        history_case(1, 10, group="small"),
        history_case(2, 100, dp=2 if by_config else 1, group="large"),
    )
    evaluation = history_evaluator(cases)
    before = evaluation.evaluate(TokenHistoryPredictor(1.2), "old", fitted_on=())
    after = evaluation.evaluate(
        TokenHistoryPredictor(overrides={10: 13, 100: 100}), "new", fitted_on=()
    )
    assert after.score < before.score
    assert not evaluation.compare(after, before).accepted


def test_history_zero_labels_are_not_missing_or_silently_excluded():
    """All-zero metric WAPE is undefined; finite MAE still guards acceptance."""
    evaluation = history_evaluator((history_case(1, 10, measured=0),))
    before = evaluation.evaluate(TokenHistoryPredictor(), "old", fitted_on=())
    after = evaluation.evaluate(TokenHistoryPredictor(0), "new", fitted_on=())
    assert before.metrics[0].wape is None and before.metrics[0].mae == 10
    assert before.metrics[0].relative_error_samples == 0 and before.score == 10
    assert after.metrics[0].covered == 1 and after.score == 0
    assert evaluation.compare(after, before).accepted


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_history_invalid_candidate_values_are_failures_not_unsupported(bad):
    """Numerical/output failures cannot improve coverage-scoped accuracy."""
    evaluation = history_evaluator()
    report = evaluation.evaluate(
        TokenHistoryPredictor(overrides={10: bad}), "bad", fitted_on=()
    )
    assert not report.valid and report.errors
    baseline = evaluation.evaluate(TokenHistoryPredictor(), "old", fitted_on=())
    assert not evaluation.compare(report, baseline).accepted


def test_history_empty_incumbent_needs_explicit_bootstrap_accuracy():
    """Undefined baseline error cannot count as an automatic improvement."""
    for threshold, accepted in ((None, False), (0.2, True)):
        evaluation = history_evaluator(
            criteria=ModelAcceptanceCriteria(bootstrap_max_error=threshold)
        )
        before = evaluation.evaluate(
            TokenHistoryPredictor(missing=(10, 100)), "old", fitted_on=()
        )
        after = evaluation.evaluate(TokenHistoryPredictor(1.15), "new", fitted_on=())
        assert before.valid and before.score is None
        assert evaluation.compare(after, before).accepted == accepted


@pytest.mark.parametrize("split", ["calibration", "final_holdout"])
def test_history_split_overlap_uses_original_spans_across_different_phases(split):
    """A renamed sample or overlapping window cannot cross the split boundary."""
    validation = replace(history_case(10, 10), span=HistorySpan("run", 10, 20))
    other = replace(
        history_case(21, 10, split=split),
        span=HistorySpan("run", 19, 21),
        measured=(("compute_advantage_seconds", 1),),
    )
    with pytest.raises(ValueError, match="overlap"):
        history_evaluator((validation, other))


def test_history_fit_provenance_and_measurement_identity_are_checked():
    """Profiles fitted on protected spans and duplicated labels are rejected."""
    cases = (
        history_case(0, 4, split="calibration"),
        history_case(1, 10),
        history_case(2, 20, split="final_holdout"),
    )
    evaluation = history_evaluator(cases)
    for span in (HistorySpan("run", 0, 1), cases[2].span):
        with pytest.raises(ValueError, match="fit overlaps"):
            evaluation.evaluate(TokenHistoryPredictor(), "bad", fitted_on=(span,))
    with pytest.raises(ValueError, match="Duplicate original"):
        history_evaluator((cases[1], replace(cases[1], sample_id="alias")))
    old = evaluation.evaluate(TokenHistoryPredictor(), "old", fitted_on=())
    other = history_evaluator((history_case(1, 10, measured=11),))
    with pytest.raises(ValueError, match="different historical"):
        other.compare(old, old)


def test_history_logical_and_given_packing_are_separate_evaluation_contracts():
    """Observed MB count is only admissible in explicitly conditional evaluation."""
    case = history_case(1, 16)
    with pytest.raises(ValueError, match="Logical"):
        replace(
            case.inputs, workload=replace(case.inputs.workload, train_microbatches=2)
        )
    conditional = replace(
        case,
        inputs=replace(
            case.inputs,
            input_mode="given_packing",
            workload=replace(case.inputs.workload, train_microbatches=2),
        ),
    )
    evaluation = history_evaluator((conditional,))
    report = evaluation.evaluate(TokenHistoryPredictor(), "conditional", fitted_on=())
    assert report.metrics[0].input_mode == "given_packing"
    with pytest.raises(ValueError, match="separate evaluators"):
        history_evaluator((conditional, history_case(2, 20)))
    with pytest.raises(ValueError, match="separate evaluators"):
        history_evaluator(
            (
                case,
                replace(
                    history_case(2, 20),
                    inputs=replace(case.inputs, timing_boundary="rank_mean"),
                ),
            )
        )


def test_history_snapshots_survive_candidate_input_mutation_and_settings_are_readonly():
    """Trusted predictors still get private input copies, never evaluator objects."""
    evaluation = history_evaluator(
        (history_case(0, 4, split="calibration"), history_case(1, 10))
    )

    class Mutating:
        def predict(self, inputs):
            object.__setattr__(inputs.workload, "sequence_tokens", 999)
            return HistoricalPrediction()

    evaluation.evaluate(Mutating(), "mutating", fitted_on=())
    calibration = evaluation.calibration
    object.__setattr__(calibration[0].inputs.workload, "sequence_tokens", 999)
    assert evaluation.calibration[0].inputs.workload.sequence_tokens == 4
    report = evaluation.evaluate(TokenHistoryPredictor(), "unchanged", fitted_on=())
    assert report.score == 0
    with pytest.raises(AttributeError):
        evaluation.criteria = ModelAcceptanceCriteria(max_added_error=100)


def test_history_comparison_rejects_tampered_reports_and_foreign_instances():
    """A matching hash cannot authorize candidate-supplied labels or summaries."""
    evaluation = history_evaluator()
    before = evaluation.evaluate(TokenHistoryPredictor(1.2), "old", fitted_on=())
    after = evaluation.evaluate(TokenHistoryPredictor(2), "bad", fitted_on=())
    for forged in (
        replace(
            after, rows=tuple(replace(r, measured=r.predicted) for r in after.rows)
        ),
        replace(after, score=0),
        replace(after, rows=after.rows[:-1]),
    ):
        with pytest.raises(ValueError, match="unchanged reports"):
            evaluation.compare(forged, before)
    foreign = history_evaluator().evaluate(
        TokenHistoryPredictor(), "good", fitted_on=()
    )
    with pytest.raises(ValueError, match="unchanged reports"):
        evaluation.compare(foreign, before)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_history_new_coverage_checks_small_config_and_workload_slices(bootstrap):
    """A large accurate new group cannot hide a bad smaller new group."""
    cases = (
        history_case(1, 10),
        history_case(2, 1000, dp=2, group="large"),
        history_case(3, 4, measured=1, dp=4, group="small"),
    )
    evaluation = history_evaluator(
        cases, criteria=ModelAcceptanceCriteria(bootstrap_max_error=0.1)
    )
    before = evaluation.evaluate(
        TokenHistoryPredictor(1.2, missing=(4, 1000, 10) if bootstrap else (4, 1000)),
        "old",
        fitted_on=(),
    )
    after = evaluation.evaluate(
        TokenHistoryPredictor(overrides={4: 2}), "new", fitted_on=()
    )
    assert after.score < 0.001
    assert not evaluation.compare(after, before).accepted


def test_history_feedback_suppresses_small_slices_and_absolute_statistics():
    """No MAE/WAPE pair that algebraically reveals a singleton measured label."""
    evaluation = history_evaluator(
        tuple(history_case(i, i + 4, dp=i + 1) for i in range(6))
    )
    report = evaluation.evaluate(TokenHistoryPredictor(1.2), "v0", fitted_on=())
    feedback = report.feedback()
    assert feedback["score"] == pytest.approx(0.2)
    assert feedback["metrics"][0]["wape"] == pytest.approx(0.2)
    for item in (*feedback["metrics"], *feedback["slices"]):
        assert (
            not {"mae", "measured_sum", "absolute_error", "normalized_error"}
            & item.keys()
        )
        if item["group_by"] == "config_id":
            assert item["error_suppressed"] and item["wape"] is None
    small = history_evaluator().evaluate(TokenHistoryPredictor(1.2), "v0", fitted_on=())
    assert small.feedback()["score"] is None


MODEL_SOURCE = """from areal.autopilot.revision.model_evaluator import HistoricalPrediction
class Model:
    def __init__(self, scale): self.scale = scale
    def predict(self, inputs):
        assert not hasattr(inputs, "measured") and not hasattr(inputs, "sample_id")
        return HistoricalPrediction((("train_step_seconds", inputs.workload.sequence_tokens * self.scale),))
def build_model(calibration_cases, settings):
    assert all(c.split == "calibration" for c in calibration_cases)
    return Model(settings["initial_scale"])
"""
FITTED_MODEL_SOURCE = MODEL_SOURCE.replace(
    'settings["initial_scale"]',
    'sum(dict(c.measured)["train_step_seconds"] / c.inputs.workload.sequence_tokens for c in calibration_cases) / len(calibration_cases)',
)


def source_evaluator(runner, *, cases=None, settings=None):
    return HistoricalSourceEvaluator(
        history_evaluator(
            cases
            or (
                history_case(0, 4, split="calibration"),
                history_case(1, 10),
                history_case(2, 100),
                history_case(3, 997, split="final_holdout"),
            )
        ),
        settings=settings or {"initial_scale": 1.4},
        fitted_on=(),
        runner=runner,
    )


def test_historical_source_worker_gets_only_calibration_labels_and_validation_inputs(
    runner, tmp_path
):
    """Factory and prediction use typed inputs; no heldout labels or final work cross the pipe."""
    evaluation = source_evaluator(runner)
    before = evaluation.evaluate(MODEL_SOURCE, tmp_path / "before")
    after = evaluation.evaluate(FITTED_MODEL_SOURCE, tmp_path / "after")
    assert before.valid and after.valid, (before.errors, after.errors)
    assert before.score == pytest.approx(0.4) and after.score == 0
    assert evaluation.compare(after, before)[0]
    request = json.loads((tmp_path / "after/request.json").read_text())
    assert [c["split"] for c in request["calibration"]["cases"]] == ["calibration"]
    assert [c["workload"]["sequence_tokens"] for c in request["cases"]] == [10, 100]
    assert all(
        "measured" not in c and "sample_id" not in c and "span" not in c
        for c in request["cases"]
    )
    detail = json.loads((tmp_path / "after/history-evaluation.json").read_text())
    assert detail["reserved_cases"] == 1 and detail["fitted_on"] == [
        {"run_id": "run", "first_step": 0, "last_step": 0}
    ]
    assert (
        evaluation.feedback(before)["score"] is None
    )  # Small validation groups stay suppressed.
    with pytest.raises(ValueError, match="unchanged issued reports"):
        evaluation.compare(replace(after, score=0.1), before)


def test_historical_revision_loop_rejects_retries_accepts_loads_and_resumes(
    runner, tmp_path
):
    """Reuse durable acceptance, active pointer, feedback retries and duplicate checks."""

    class Agent:
        def __init__(self):
            self.requests = []

        def propose(self, request):
            self.requests.append(request)
            source = (
                "def build_model(calibration_cases, settings): raise ValueError('bad fit')"
                if len(self.requests) == 1
                else FITTED_MODEL_SOURCE
            )
            return Proposal(source)

    agent, evaluation = Agent(), source_evaluator(runner)
    root = tmp_path / "historical-loop"
    result = RevisionLoop(agent, evaluation, root, initial_source=MODEL_SOURCE).run(
        max_rounds=2
    )
    assert result["accepted"] == 1
    assert [r["status"] for r in result["rounds"]] == ["rejected", "accepted"]
    assert result["final_score"] == 0
    assert "build_model" in agent.requests[0].to_dict()["api"]
    assert agent.requests[0].feedback[0]["score"] is None
    assert "bad fit" in str(agent.requests[1].feedback)
    assert "rows" not in str(agent.requests[1].feedback)
    path, pointer = RevisionStore(root).active()
    assert path.read_text() == FITTED_MODEL_SOURCE and pointer["target"] == "model"
    loaded = load_accepted_model(root, evaluation)
    assert (
        dict(loaded.predict(history_case(1, 10).inputs).values)["train_step_seconds"]
        == 10
    )
    pointer_bytes = (root / "active.json").read_bytes()

    class Duplicate:
        def propose(self, request):
            return Proposal(FITTED_MODEL_SOURCE)

    resumed = RevisionLoop(
        Duplicate(), source_evaluator(runner), root, initial_source=MODEL_SOURCE
    ).run(max_rounds=1)
    assert (
        resumed["accepted"] == 0
        and resumed["rounds"][0]["reason"] == "Duplicate candidate"
    )
    assert (root / "active.json").read_bytes() == pointer_bytes
    with pytest.raises(ValueError, match="load_accepted_model"):
        apply_accepted_revision(root, {}, {})
    with pytest.raises(ValueError, match="different historical evaluation context"):
        load_accepted_model(
            root, source_evaluator(runner, settings={"initial_scale": 1.8})
        )


@pytest.mark.parametrize(
    "source",
    [
        "this is not valid Python !!!",
        "def build_model(c, s): return object()",
        "def build_model(c, s): raise RuntimeError('factory failed')",
        MODEL_SOURCE.replace(
            'return HistoricalPrediction((("train_step_seconds", inputs.workload.sequence_tokens * self.scale),))',
            "return {'train_step_seconds': 10}",
        ),
        MODEL_SOURCE.replace(
            "inputs.workload.sequence_tokens * self.scale", 'float("nan")'
        ),
    ],
)
def test_historical_source_failures_remain_invalid_without_losing_the_incumbent(
    runner, tmp_path, source
):
    """Factory/type/nonfinite errors fail evaluation rather than becoming missing coverage."""
    evaluation = source_evaluator(runner)
    before = evaluation.evaluate(MODEL_SOURCE, tmp_path / "before")
    failed = evaluation.evaluate(source, tmp_path / "failed")
    assert before.valid and not failed.valid and failed.errors
    assert not evaluation.compare(failed, before)[0]


def test_historical_factory_timeout_is_bounded_and_not_activated(runner, tmp_path):
    """The existing process deadline also covers model construction and fitting."""
    evaluation = source_evaluator(replace(runner, timeout_seconds=0.6))
    report = evaluation.evaluate("while True: pass", tmp_path / "timeout")
    assert not report.valid and "timed out" in str(report.errors)


@pytest.mark.parametrize(
    "source,error",
    [
        (
            MODEL_SOURCE.replace(
                'settings["initial_scale"]', '__import__("os").getpid()'
            ),
            "not reproducible",
        ),
        (
            MODEL_SOURCE.replace(
                "def predict(self, inputs):",
                "def predict(self, inputs):\n        self.scale += 0.1",
            ),
            "repeatable without updating fitted state",
        ),
    ],
)
def test_historical_factory_and_predictor_must_reproduce(
    runner, tmp_path, source, error
):
    """Neither per-process fitting nor mutable prediction can identify a frozen model."""
    report = source_evaluator(runner).evaluate(source, tmp_path)
    assert not report.valid and error in str(report.errors)


def test_historical_load_and_resume_reject_changed_fit(runner, tmp_path, monkeypatch):
    """The accepted prediction receipt binds the fit as well as the source code."""
    source = MODEL_SOURCE.replace(
        'settings["initial_scale"]',
        'float(__import__("os").environ["AUTOPILOT_TEST_FIT_SCALE"])',
    )

    class Agent:
        def propose(self, request):
            return Proposal(source)

    monkeypatch.setenv("AUTOPILOT_TEST_FIT_SCALE", "1")
    evaluation = source_evaluator(runner)
    result = RevisionLoop(
        Agent(), evaluation, tmp_path, initial_source=MODEL_SOURCE
    ).run(max_rounds=1)
    assert result["accepted"] == 1
    active_bytes = (tmp_path / "active.json").read_bytes()
    baseline_bytes = (tmp_path / "baseline.json").read_bytes()
    monkeypatch.setenv("AUTOPILOT_TEST_FIT_SCALE", "1.1")
    with pytest.raises(ValueError, match="differ from accepted fitted model"):
        load_accepted_model(tmp_path, evaluation)
    with pytest.raises(ValueError, match="changed since acceptance"):
        RevisionLoop(Agent(), evaluation, tmp_path, initial_source=MODEL_SOURCE).run(
            max_rounds=1
        )
    assert (tmp_path / "active.json").read_bytes() == active_bytes
    assert (tmp_path / "baseline.json").read_bytes() == baseline_bytes


def test_historical_loop_freezes_settings_provenance_and_initial_factory(
    runner, tmp_path
):
    """Reusing a store with another fitting/search context requires a fresh experiment."""
    settings = {"initial_scale": 1.4}
    evaluation = source_evaluator(runner, settings=settings)
    settings["initial_scale"] = 100
    assert evaluation.settings["initial_scale"] == 1.4
    with pytest.raises(AttributeError):
        evaluation.history = history_evaluator()
    with pytest.raises(ValueError, match="explicit initial_source"):
        RevisionLoop(None, evaluation, tmp_path / "missing")
    with pytest.raises(ValueError, match="criteria are frozen"):
        RevisionLoop(
            None,
            evaluation,
            tmp_path / "criteria",
            initial_source=MODEL_SOURCE,
            criteria=ModelAcceptanceCriteria(max_added_error=1),
        )
    with pytest.raises(ValueError, match="fit overlaps"):
        HistoricalSourceEvaluator(
            evaluation.history,
            settings={},
            fitted_on=(HistorySpan("run", 1, 1),),
            runner=runner,
        )

    class Duplicate:
        def propose(self, request):
            return Proposal(MODEL_SOURCE)

    root = tmp_path / "store"
    RevisionLoop(Duplicate(), evaluation, root, initial_source=MODEL_SOURCE).run(
        max_rounds=1
    )
    with pytest.raises(ValueError, match="another fixed evaluation context"):
        RevisionLoop(
            Duplicate(), evaluation, root, initial_source=FITTED_MODEL_SOURCE
        ).run(max_rounds=1)
    another_worker = replace(runner, command=(*runner.command[:1], "-V"))
    changed = source_evaluator(another_worker)
    assert changed.benchmark_sha256 != evaluation.benchmark_sha256
    with pytest.raises(ValueError, match="another fixed evaluation context"):
        RevisionLoop(Duplicate(), changed, root, initial_source=MODEL_SOURCE).run(
            max_rounds=1
        )


def test_historical_input_json_roundtrip_preserves_rollout_joint_and_initial_state():
    """Nested actor stats, logical packing views and version counters survive IPC."""
    from areal.autopilot.contracts import BatchWorkload, TrainingConfig
    from areal.autopilot.elastic.policy.basic.cost_model.rollout import (
        RolloutConfig,
        RolloutStage,
        RolloutWorkload,
    )
    from areal.autopilot.elastic.policy.basic.joint import (
        JointBatch,
        JointConfig,
        JointScenario,
    )
    from areal.autopilot.revision.model_evaluator import JOINT_BOUNDARY
    from areal.autopilot.revision.model_execution import historical_input_from_dict

    rollout = RolloutConfig(
        "model", "hardware", "runtime", "sampling", 1, 1, 8, 128, 1024, 2
    )
    workload = RolloutWorkload((RolloutStage(10, 20, 7),), 2, 64)
    stats = WorkloadStats(4, 40, loss_tokens=20)
    config = JointConfig(TrainingConfig(1, "full-profile"), rollout)
    batch = JointBatch(BatchWorkload(4, 20, 40, actor_stats=stats), workload, 1)
    scenario = JointScenario(
        (batch,) * 2,
        config,
        4,
        1,
        8,
        8,
        16,
        ready_batches=1,
        accepted_batches=3,
        provider_version=2,
    )
    inputs = (
        history_case(1, 10).inputs,
        HistoricalInput(
            "rollout", rollout, workload, "synthetic", "service", "given_external"
        ),
        HistoricalInput(
            "joint",
            config,
            scenario,
            "synthetic",
            JOINT_BOUNDARY,
            "given_external",
            True,
            True,
        ),
    )
    for value in inputs:
        restored = historical_input_from_dict(json.loads(json.dumps(asdict(value))))
        assert restored == value


def test_historical_worker_receives_no_unsupported_joint_trace(runner, tmp_path):
    """Boundary applicability is resolved before any candidate code sees a trace."""
    from areal.autopilot.contracts import BatchWorkload, TrainingConfig
    from areal.autopilot.elastic.policy.basic.cost_model.rollout import RolloutConfig
    from areal.autopilot.elastic.policy.basic.joint import (
        JointBatch,
        JointConfig,
        JointScenario,
    )
    from areal.autopilot.revision.model_evaluator import JOINT_BOUNDARY

    config = JointConfig(
        TrainingConfig(1), RolloutConfig("m", "h", "r", "s", 1, 1, 8, 128, 1024, 1)
    )
    scenario = JointScenario(
        (
            JointBatch(
                BatchWorkload(4, 20, 40, actor_stats=WorkloadStats(4, 40)), None, 1
            ),
        ),
        config,
        4,
        1,
        8,
        8,
        16,
        ready_batches=1,
        accepted_batches=1,
    )
    case = HistoricalCase(
        "unaudited",
        HistorySpan("r", 1, 1),
        "validation",
        HistoricalInput(
            "joint", config, scenario, "synthetic", JOINT_BOUNDARY, "given_external"
        ),
        (("seconds", 10),),
        "synthetic",
    )
    evaluation = HistoricalSourceEvaluator(
        HistoricalModelEvaluator((case,), primary_metrics=("seconds",)),
        settings={"initial_scale": 1},
        fitted_on=(),
        runner=runner,
    )
    report = evaluation.evaluate(MODEL_SOURCE, tmp_path / "uncovered")
    assert report.valid and report.score is None
    assert json.loads((tmp_path / "uncovered/request.json").read_text())["cases"] == []


def test_historical_source_rejects_reference_queries_even_if_worker_catches_error(
    tmp_path,
):
    """Reference access violations are recorded by the parent independently of outputs."""

    class QueryingRunner:
        command = ()
        timeout_seconds = 1
        max_output_bytes = 1024

        def run(self, source, request, directory, simulate):
            try:
                simulate({"type": "simulate", "case": 0})
            except ValueError:
                pass
            return {
                "rows": [
                    {
                        "prediction": {
                            "values": [
                                ["train_step_seconds", c["workload"]["sequence_tokens"]]
                            ]
                        }
                    }
                    for c in request["cases"]
                ]
            }

    report = source_evaluator(QueryingRunner()).evaluate(MODEL_SOURCE, tmp_path)
    assert not report.valid and "cannot query evaluation references" in str(
        report.errors
    )


def test_historical_source_protected_context_changes_invalidate_execution(
    runner, monkeypatch, tmp_path
):
    """Factory/library mutations cannot reuse historical acceptance context."""
    evaluation = source_evaluator(runner)
    monkeypatch.setattr(evaluation, "_code_fingerprint", lambda: ("changed",))
    report = evaluation.evaluate(MODEL_SOURCE, tmp_path)
    assert not report.valid and "factory/worker code changed" in str(report.errors)
    assert not (tmp_path / "request.json").exists()


JOINT_HEURISTIC_SOURCE = """from areal.autopilot.elastic.policy.basic.joint import JointSearchResult
class Optimizer:
    def __init__(self, settings): self.candidates = settings["candidates"]
    def optimize(self, scenario, simulator):
        config = self.candidates[-1]
        # Deliberately bogus claims: evaluator uses only the selected config.
        return JointSearchResult(config, None, (), 0, 999999, "unverified claims")
def build_joint_optimizer(settings): return Optimizer(settings)
"""
JOINT_WEAK_SOURCE = BASELINE_SOURCE["joint_optimizer"].replace(
    "JointOptimizer(**settings)",
    'JointOptimizer(**{**settings, "max_evaluations": 1})',
)


def joint_source_evaluator(runner, **kwargs):
    from tests.test_elastic_lifecycle_optimizer import (
        joint_config,
        joint_scenario,
        joint_simulator,
    )

    configs = joint_config(), joint_config(actor_gpus=2)
    model = joint_simulator(
        configs, actor_times={configs[0].actor: 100, configs[1].actor: 1}
    )
    return (
        JointOptimizerEvaluator(
            model,
            (joint_scenario(configs[0]),),
            configs,
            model_version="basic-test-v0",
            runner=runner,
            **kwargs,
        ),
        model,
        configs,
    )


def test_joint_optimizer_quality_queries_and_model_snapshot_are_independent(
    runner, tmp_path
):
    """Use full-profile cost in parent and count actual queries rather than result claims."""
    evaluation, model, configs = joint_source_evaluator(runner)
    before = evaluation.evaluate(JOINT_WEAK_SOURCE, tmp_path / "weak")
    exhaustive = evaluation.evaluate(
        BASELINE_SOURCE["joint_optimizer"], tmp_path / "full"
    )
    heuristic = evaluation.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "heuristic")
    assert before.valid and exhaustive.valid and heuristic.valid, (
        before.errors,
        exhaustive.errors,
        heuristic.errors,
    )
    assert evaluation.compare(exhaustive, before) == (
        True,
        "Improved fixed-model configuration quality",
    )
    assert evaluation.compare(heuristic, exhaustive) == (
        True,
        "Preserved model quality with fewer queries",
    )
    assert (
        heuristic.metrics["mean_queries"] == 0
        and exhaustive.metrics["mean_queries"] == 2
    )
    assert (
        heuristic.metrics["mean_gpu_seconds"]
        == exhaustive.metrics["mean_gpu_seconds"]
        > 0
    )
    assert heuristic.metrics["mean_optimizer_seconds"] < 999999
    # Later model mutation cannot alter the frozen evaluation projection.
    model.actors.times[configs[1].actor] = 10000
    repeated = evaluation.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "after-mutation")
    assert repeated.valid and repeated.score == heuristic.score
    request = json.loads((tmp_path / "heuristic/request.json").read_text())
    assert "predictions" not in request and "reference_costs" not in request
    assert (
        request["cases"][0]["batches"]
        == json.loads(evaluation._scenarios_json)[0]["batches"]
    )
    assert evaluation.calibration == {"model_version": "basic-test-v0"}
    assert "runs" not in evaluation.feedback(heuristic)
    with pytest.raises(ValueError, match="unchanged issued reports"):
        evaluation.compare(replace(heuristic, score=0), exhaustive)


@pytest.mark.parametrize(
    "attempt,error",
    [
        (
            "simulator.evaluate(replace(scenario, batches=scenario.batches[:1]), config)",
            "changed the frozen scenario",
        ),
        (
            'simulator.evaluate(scenario, replace(config, actor=replace(config.actor, profile_id="outside")))',
            "outside fixed allowlist",
        ),
        (
            "simulator.evaluate(scenario, config)\n            simulator.evaluate(scenario, config)",
            "query budget exceeded",
        ),
        (
            "simulator.index = 9\n            simulator.evaluate(scenario, config)",
            "Invalid joint scenario index",
        ),
    ],
)
def test_joint_optimizer_cannot_hide_query_violations(runner, tmp_path, attempt, error):
    """Caught RPC exceptions remain parent-owned failures and cannot activate a candidate."""
    source = "from dataclasses import replace\n" + JOINT_HEURISTIC_SOURCE.replace(
        "# Deliberately bogus claims: evaluator uses only the selected config.",
        "try:\n            " + attempt + "\n        except Exception: pass",
    )
    evaluation, _, _ = joint_source_evaluator(runner, max_queries=1)
    report = evaluation.evaluate(source, tmp_path)
    assert not report.valid and error in str(report.errors)


def test_joint_optimizer_rechecks_switch_margin_job_bound_and_current_allowlist(
    runner, tmp_path
):
    """Changing optimizer settings cannot relax evaluator-owned feasibility constraints."""
    from tests.test_elastic_lifecycle_optimizer import (
        joint_config,
        joint_scenario,
        joint_simulator,
    )

    old, new = joint_config(label="old"), joint_config(label="new")
    model = joint_simulator((old, new), actor_times={old.actor: 100, new.actor: 96})
    evaluation = JointOptimizerEvaluator(
        model,
        (joint_scenario(old),),
        (new,),
        model_version="margin",
        switch_margin=0.05,
        runner=runner,
    )
    # Current is legal even when absent from the supplied candidate tuple.
    baseline = evaluation.evaluate(
        BASELINE_SOURCE["joint_optimizer"], tmp_path / "baseline"
    )
    candidate = evaluation.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "candidate")
    assert baseline.valid and baseline.metrics["mean_queries"] == 2
    assert not candidate.valid and "fixed constraints" in str(candidate.errors)
    timed = JointOptimizerEvaluator(
        model,
        (joint_scenario(old),),
        (new,),
        model_version="deadline",
        max_job_seconds=206,
        runner=runner,
    )
    old_source = JOINT_HEURISTIC_SOURCE.replace(
        "self.candidates[-1]", "scenario.current"
    )
    assert not timed.evaluate(old_source, tmp_path / "job-bound").valid
    with pytest.raises(ValueError, match="no eligible reference configuration"):
        JointOptimizerEvaluator(
            model,
            (joint_scenario(old),),
            (new,),
            model_version="none",
            max_job_seconds=1,
        )
    with pytest.raises(ValueError, match="at least three"):
        joint_source_evaluator(runner, repetitions=1)


def test_joint_optimizer_runtime_is_parent_measured_and_requires_repeatable_choices(
    runner, tmp_path
):
    """A slow solver cannot hide behind fabricated elapsed_seconds in its result."""
    slow = "import time\n" + JOINT_HEURISTIC_SOURCE.replace(
        "config = self.candidates[-1]",
        "time.sleep(0.04)\n        config = self.candidates[-1]",
    )
    evaluation, _, _ = joint_source_evaluator(runner, max_optimizer_seconds=0.02)
    report = evaluation.evaluate(slow, tmp_path / "timeout")
    assert not report.valid and "parent time budget" in str(report.errors)

    # Independent runs always evaluate the same fixed selections/query counts.
    class ChangingRunner:
        command = ()
        timeout_seconds = 60
        max_output_bytes = 4096

        def __init__(self):
            self.calls = 0

        def run(self, source, request, directory, simulate):
            self.calls += 1
            simulate({"case": 0, "operation": "start"})
            if self.calls > 1:
                simulate(
                    {
                        "case": 0,
                        "operation": "query",
                        "scenario": request["cases"][0],
                        "config": request["settings"]["candidates"][-1],
                    }
                )
            selected = request["settings"]["candidates"][-1]
            simulate({"case": 0, "operation": "finish", "config": selected})
            return {"configs": [selected]}

    changing, _, _ = joint_source_evaluator(ChangingRunner())
    report = changing.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "changing")
    assert not report.valid and "must reproduce" in str(report.errors)


def test_joint_optimizer_revision_accepts_resumes_and_binds_model_context(
    runner, tmp_path
):
    """New joint target uses the existing acceptance store without changing lifecycle targets."""

    class Agent:
        def propose(self, request):
            assert request.target == "joint_optimizer"
            assert "build_joint_optimizer" in request.to_dict()["api"]
            return Proposal(JOINT_HEURISTIC_SOURCE)

    evaluation, model, configs = joint_source_evaluator(runner)
    result = RevisionLoop(
        Agent(), evaluation, tmp_path, initial_source=JOINT_WEAK_SOURCE
    ).run(max_rounds=1)
    assert result["accepted"] == 1 and result["final_score"] == 1
    pointer = (tmp_path / "active.json").read_bytes()
    resumed = RevisionLoop(
        Agent(), evaluation, tmp_path, initial_source=JOINT_WEAK_SOURCE
    ).run(max_rounds=1)
    assert (
        resumed["accepted"] == 0
        and resumed["rounds"][0]["reason"] == "Duplicate candidate"
    )
    assert (tmp_path / "active.json").read_bytes() == pointer
    with pytest.raises(ValueError, match="criteria are frozen"):
        RevisionLoop(
            Agent(),
            evaluation,
            tmp_path,
            criteria=JointOptimizerCriteria(max_cost_regression=0.1),
        )
    model.actors.times[configs[1].actor] = 3
    changed = JointOptimizerEvaluator(
        model,
        (joint_scenario_from_store(evaluation),),
        configs,
        model_version="basic-test-v0",
        runner=runner,
    )
    assert changed.benchmark_sha256 != evaluation.benchmark_sha256
    with pytest.raises(ValueError, match="another fixed evaluation context"):
        RevisionLoop(Agent(), changed, tmp_path, initial_source=JOINT_WEAK_SOURCE).run(
            max_rounds=1
        )
    with pytest.raises(ValueError, match="fixed offline evaluator"):
        apply_accepted_revision(tmp_path, {}, {})


def test_joint_optimizer_resume_rejects_changed_accepted_selections(
    runner, tmp_path, monkeypatch
):
    """Consistent choices within a new run cannot replace different accepted choices."""
    source = JOINT_HEURISTIC_SOURCE.replace(
        "self.candidates[-1]",
        'self.candidates[-1] if __import__("os").environ["AUTOPILOT_TEST_JOINT_CHOICE"] == "1" else scenario.current',
    )

    class Agent:
        def propose(self, request):
            return Proposal(source)

    monkeypatch.setenv("AUTOPILOT_TEST_JOINT_CHOICE", "1")
    evaluation, _, _ = joint_source_evaluator(runner)
    result = RevisionLoop(
        Agent(), evaluation, tmp_path, initial_source=JOINT_WEAK_SOURCE
    ).run(max_rounds=1)
    assert result["accepted"] == 1
    before = [
        (tmp_path / name).read_bytes() for name in ("active.json", "baseline.json")
    ]
    monkeypatch.setenv("AUTOPILOT_TEST_JOINT_CHOICE", "0")
    with pytest.raises(ValueError, match="changed since acceptance"):
        RevisionLoop(
            Agent(), evaluation, tmp_path, initial_source=JOINT_WEAK_SOURCE
        ).run(max_rounds=1)
    assert before == [
        (tmp_path / name).read_bytes() for name in ("active.json", "baseline.json")
    ]


def joint_scenario_from_store(evaluation):
    from areal.autopilot.revision.model_execution import joint_scenario_from_dict

    return joint_scenario_from_dict(json.loads(evaluation._scenarios_json)[0])


def test_joint_optimizer_rejects_per_scenario_regression_and_query_time_tradeoff(
    runner, tmp_path
):
    """Averages cannot hide a damaged scenario; fewer queries cannot hide a slower solve."""
    from tests.test_elastic_lifecycle_optimizer import joint_scenario

    _, model, configs = joint_source_evaluator(runner)
    evaluation = JointOptimizerEvaluator(
        model,
        (
            replace(joint_scenario(configs[0], count=10), current=None),
            replace(joint_scenario(configs[0], count=1, external=1000), current=None),
        ),
        configs,
        model_version="different-bottlenecks",
        runner=runner,
    )
    old = evaluation.evaluate(JOINT_WEAK_SOURCE, tmp_path / "old")
    new = evaluation.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "new")
    assert old.valid and new.valid and new.score < old.score
    assert evaluation.compare(new, old) == (False, "Per-scenario model cost regressed")
    single, _, _ = joint_source_evaluator(runner)
    exhaustive = single.evaluate(
        BASELINE_SOURCE["joint_optimizer"], tmp_path / "exhaustive"
    )
    slow_source = "import time\n" + JOINT_HEURISTIC_SOURCE.replace(
        "config = self.candidates[-1]",
        "time.sleep(0.25)\n        config = self.candidates[-1]",
    )
    slow = single.evaluate(slow_source, tmp_path / "slow")
    fast = single.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "fast")
    assert slow.valid and fast.valid and exhaustive.valid
    assert not single.compare(slow, exhaustive)[0]
    assert single.compare(fast, slow) == (
        True,
        "Preserved model quality with faster solving",
    )
    import_slow = single.evaluate(
        "import time; time.sleep(0.15)\n" + JOINT_HEURISTIC_SOURCE,
        tmp_path / "import-slow",
    )
    assert import_slow.valid
    assert import_slow.metrics["worker_seconds"] > fast.metrics["worker_seconds"] + 0.08
    assert not single.compare(import_slow, exhaustive)[0]


@pytest.mark.parametrize(
    "source",
    [
        "invalid Python !!!",
        "def build_joint_optimizer(settings): return object()",
        JOINT_HEURISTIC_SOURCE.replace(
            "JointSearchResult(config, None", "JointSearchResult(None, None"
        ),
    ],
)
def test_joint_optimizer_invalid_outputs_cannot_be_accepted(runner, tmp_path, source):
    """Syntax/type failures and absent configuration are failures, never zero cost plans."""
    evaluation, _, _ = joint_source_evaluator(runner)
    report = evaluation.evaluate(source, tmp_path)
    assert not report.valid and report.score is None and report.errors


@pytest.mark.parametrize(
    "change,error",
    [
        ({"completed_batches": 0}, "identical workload"),
        ({"config": None}, "different configuration/type"),
    ],
)
def test_joint_optimizer_rejects_invalid_reference_contract(runner, change, error):
    """A broken reference cannot establish an optimum for a different amount of work."""
    _, model, configs = joint_source_evaluator(runner)
    scenario = joint_scenario_from_store(joint_source_evaluator(runner)[0])

    class Broken:
        def evaluate(self, scenario, config):
            return replace(model.evaluate(scenario, config), **change)

    with pytest.raises(ValueError, match=error):
        JointOptimizerEvaluator(Broken(), (scenario,), configs, model_version="broken")


def test_joint_optimizer_zero_reference_cost_does_not_create_free_regret(
    runner, tmp_path
):
    """Zero cost has explicit additive regret; a positive-cost choice is still worse."""
    evaluation, model, configs = joint_source_evaluator(runner)
    scenario = joint_scenario_from_store(evaluation)

    class ZeroReference:
        def evaluate(self, scenario, config):
            value = model.evaluate(scenario, config)
            return (
                replace(value, actor_gpu_seconds=0, rollout_gpu_seconds=0)
                if config == configs[-1]
                else value
            )

    zero = JointOptimizerEvaluator(
        ZeroReference(), (scenario,), configs, model_version="zero", runner=runner
    )
    weak = zero.evaluate(JOINT_WEAK_SOURCE, tmp_path / "weak")
    best = zero.evaluate(JOINT_HEURISTIC_SOURCE, tmp_path / "best")
    assert weak.valid and best.valid and weak.score > best.score == 1
    assert zero.compare(best, weak)[0]


JOINT_EXPORT_SOURCE = """
from areal.autopilot.elastic.policy.basic.joint import JointPrediction
class Joint:
    def __init__(self, scale): self.scale = scale
    def evaluate(self, scenario, config):
        seconds = len(scenario.batches) * self.scale * (100 if config.actor.world_size == 1 else 1)
        return JointPrediction(
            "complete", config, seconds,
            seconds * config.actor.world_size, seconds * config.rollout.world_size,
            len(scenario.batches), sum(b.actor.samples for b in scenario.batches),
            sum(b.actor.effective_tokens for b in scenario.batches),
            sum(b.actor.sequence_tokens for b in scenario.batches),
            source_ids=("synthetic-cycle",),
        )
"""
CYCLE_MODEL_SOURCE = (
    MODEL_SOURCE.replace(
        "self.scale = scale",
        "self.scale = scale; self.joint = Joint(scale)",
    )
    + JOINT_EXPORT_SOURCE
)
CYCLE_FITTED_SOURCE = (
    FITTED_MODEL_SOURCE.replace(
        "self.scale = scale",
        "self.scale = scale; self.joint = Joint(scale)",
    )
    + JOINT_EXPORT_SOURCE
)


class CycleAgent:
    def __init__(self, *sources):
        self.sources, self.requests = iter(sources), []

    def propose(self, request):
        self.requests.append(request)
        value = next(self.sources)
        if isinstance(value, BaseException):
            raise value
        return Proposal(value)


def policy_cycle(runner, root, *, initial=CYCLE_MODEL_SOURCE):
    problem, _, _ = joint_source_evaluator(runner)
    return PolicyRevisionCycle(
        source_evaluator(runner),
        problem,
        root,
        initial_model_source=initial,
        initial_optimizer_source=JOINT_WEAK_SOURCE,
    ), problem


def test_policy_cycle_freezes_accepted_model_rebaselines_search_and_is_idempotent(
    runner, tmp_path
):
    """Both acceptance routes use the effective model; a completed cycle makes no new proposals."""
    cycle, problem = policy_cycle(runner, tmp_path)
    models, optimizers = (
        CycleAgent(CYCLE_FITTED_SOURCE),
        CycleAgent(JOINT_HEURISTIC_SOURCE),
    )
    result = cycle.run(models, optimizers, max_model_rounds=1, max_optimizer_rounds=1)
    assert (
        result["status"] == "complete"
        and result["model"]["accepted"]
        and result["optimizer"]["accepted"]
    )
    assert result["model"]["source_sha256"] == source_digest(CYCLE_FITTED_SOURCE)
    assert result["optimizer_scores"]["benchmark_sha256"] != problem.benchmark_sha256
    assert (
        result["optimizer_scores"]["effective"]
        == 1
        < result["optimizer_scores"]["baseline"]
    )
    baseline = json.loads((tmp_path / "optimizer/baseline.json").read_text())
    assert (
        baseline["metrics"]["mean_gpu_seconds"] == 400
    )  # New fitted model; neither old model nor template cost.
    assert "required_joint_export" in models.requests[0].calibration
    model_context = json.loads((tmp_path / "model/context.json").read_text())
    search_context = json.loads((tmp_path / "optimizer/context.json").read_text())
    assert model_context["deployment_context"]["joint_requests_sha256"]
    assert (
        search_context["deployment_context"]["model_version"]
        == result["handoff"]["model_version"]
    )
    assert result["handoff"]["joint_snapshot_sha256"]
    export_request = json.loads(
        next(tmp_path.glob("export-*/run-0/request.json")).read_text()
    )
    assert all(
        c["split"] == "calibration" for c in export_request["calibration"]["cases"]
    )
    assert [c["workload"]["sequence_tokens"] for c in export_request["cases"]] == [
        10,
        100,
    ]
    assert "measured" not in json.dumps(export_request["joint_cases"])
    cached = (tmp_path / "result.json").read_bytes()
    empty_model, empty_search = CycleAgent(), CycleAgent()
    resumed, _ = policy_cycle(runner, tmp_path)
    assert (
        resumed.run(
            empty_model, empty_search, max_model_rounds=1, max_optimizer_rounds=1
        )
        == result
    )
    assert not empty_model.requests and not empty_search.requests
    assert (tmp_path / "result.json").read_bytes() == cached


def test_policy_cycle_keeps_baselines_when_both_revisions_are_rejected(
    runner, tmp_path
):
    """No accepted source uses the explicitly verified baseline and the same handoff contract."""
    cycle, _ = policy_cycle(runner, tmp_path)
    result = cycle.run(
        CycleAgent(CYCLE_MODEL_SOURCE),
        CycleAgent(JOINT_WEAK_SOURCE),
        max_model_rounds=1,
        max_optimizer_rounds=1,
    )
    assert not result["model"]["accepted"] and not result["optimizer"]["accepted"]
    assert result["model"]["source"] == "initial.py"
    assert not (tmp_path / "model/active.json").exists()
    assert not (tmp_path / "optimizer/active.json").exists()
    assert (
        result["optimizer_scores"]["baseline"]
        == result["optimizer_scores"]["effective"]
    )


@pytest.mark.parametrize(
    "boundary",
    ["model-stage-receipt", "active-pointer", "outcome", "handoff", "result"],
)
def test_policy_cycle_recovers_durable_acceptance_without_repeating_agents(
    runner, tmp_path, monkeypatch, boundary
):
    """Recover each publication boundary while retaining the original total proposal budget."""
    import areal.autopilot.revision.cycle as cycle_module
    import areal.autopilot.revision.loop as loop_module

    module = loop_module if boundary in {"active-pointer", "outcome"} else cycle_module
    original, failed = module.atomic_json, False

    def interrupt(path, value):
        nonlocal failed
        trigger = (
            (
                boundary == "model-stage-receipt"
                and path.name == "stage.json"
                and path.parent.name == "model"
            )
            or (
                boundary == "active-pointer"
                and path.name == "active.json"
                and path.parent.name == "model"
            )
            or (
                boundary == "outcome"
                and path.name == "outcome.json"
                and value.get("status") == "accepted"
            )
            or (boundary == "handoff" and path.name == "handoff.json")
            or (boundary == "result" and path.name == "result.json")
        )
        if trigger and not failed:
            failed = True
            raise OSError("injected cycle publication failure")
        original(path, value)

    monkeypatch.setattr(module, "atomic_json", interrupt)
    cycle, _ = policy_cycle(runner, tmp_path)
    models, searches = (
        CycleAgent(CYCLE_FITTED_SOURCE),
        CycleAgent(JOINT_HEURISTIC_SOURCE),
    )
    with pytest.raises(OSError, match="publication failure"):
        cycle.run(models, searches, max_model_rounds=1, max_optimizer_rounds=1)
    assert failed and len(models.requests) == 1
    monkeypatch.setattr(module, "atomic_json", original)
    resumed, _ = policy_cycle(runner, tmp_path)
    result = resumed.run(models, searches, max_model_rounds=1, max_optimizer_rounds=1)
    assert result["model"]["accepted"] and result["optimizer"]["accepted"]
    assert len(models.requests) == len(searches.requests) == 1
    assert len(list((tmp_path / "model/attempts").glob("round-*"))) == 1


def test_policy_cycle_resume_consumes_started_attempt_and_preserves_completed_model(
    runner, tmp_path
):
    """A mid-agent interruption uses a proposal slot; model-stage work is not repeated."""
    cycle, _ = policy_cycle(runner, tmp_path)
    models = CycleAgent(CYCLE_FITTED_SOURCE)
    with pytest.raises(KeyboardInterrupt):
        cycle.run(
            models,
            CycleAgent(KeyboardInterrupt()),
            max_model_rounds=1,
            max_optimizer_rounds=2,
        )
    model_receipt = (tmp_path / "model/stage.json").read_bytes()
    empty_models, search = CycleAgent(), CycleAgent(JOINT_HEURISTIC_SOURCE)
    result = cycle.run(empty_models, search, max_model_rounds=1, max_optimizer_rounds=2)
    assert (
        result["optimizer"]["accepted"]
        and not empty_models.requests
        and len(search.requests) == 1
    )
    assert len(list((tmp_path / "optimizer/attempts").glob("round-*"))) == 2
    assert (tmp_path / "model/stage.json").read_bytes() == model_receipt


def test_policy_cycle_exhausted_interruption_does_not_repeat_agent_call(
    runner, tmp_path
):
    """A fully used proposal budget completes with the verified baseline after interruption."""
    cycle, _ = policy_cycle(runner, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        cycle.run(
            CycleAgent(KeyboardInterrupt()),
            CycleAgent(),
            max_model_rounds=1,
            max_optimizer_rounds=1,
        )
    model, search = CycleAgent(), CycleAgent(JOINT_WEAK_SOURCE)
    result = cycle.run(model, search, max_model_rounds=1, max_optimizer_rounds=1)
    assert not result["model"]["accepted"] and not model.requests
    assert len(list((tmp_path / "model/attempts").glob("round-*"))) == 1


def test_policy_cycle_baseline_drift_after_interruption_is_rejected(
    runner, tmp_path, monkeypatch
):
    """A new stable factory fit cannot overwrite the original baseline while reusing prior feedback."""
    source = CYCLE_MODEL_SOURCE.replace(
        'settings["initial_scale"]',
        'float(__import__("os").environ["AUTOPILOT_CYCLE_SCALE"])',
    )
    monkeypatch.setenv("AUTOPILOT_CYCLE_SCALE", "1.4")
    cycle, _ = policy_cycle(runner, tmp_path, initial=source)
    with pytest.raises(KeyboardInterrupt):
        cycle.run(
            CycleAgent(KeyboardInterrupt()),
            CycleAgent(),
            max_model_rounds=2,
            max_optimizer_rounds=1,
        )
    baseline = (tmp_path / "model/baseline.json").read_bytes()
    monkeypatch.setenv("AUTOPILOT_CYCLE_SCALE", "1.3")
    models, search = CycleAgent(CYCLE_FITTED_SOURCE), CycleAgent(JOINT_HEURISTIC_SOURCE)
    with pytest.raises(ValueError, match="since baseline evaluation"):
        cycle.run(models, search, max_model_rounds=2, max_optimizer_rounds=1)
    assert not models.requests and not search.requests
    assert (tmp_path / "model/baseline.json").read_bytes() == baseline


@pytest.mark.parametrize(
    "source,error",
    [
        (MODEL_SOURCE, "must export predictor.joint"),
        (
            CYCLE_MODEL_SOURCE.replace(
                "self.scale * (100", 'self.scale * __import__("os").getpid() * (100'
            ),
            "not reproducible",
        ),
        (
            CYCLE_MODEL_SOURCE.replace(
                "len(scenario.batches), sum(b.actor.samples", "0, sum(b.actor.samples"
            ),
            "complete workload",
        ),
    ],
)
def test_policy_cycle_invalid_initial_joint_export_stops_before_optimizer(
    runner, tmp_path, source, error
):
    """Historical Actor agreement cannot substitute for a usable reproducible joint model."""
    cycle, _ = policy_cycle(runner, tmp_path, initial=source)
    models, search = CycleAgent(), CycleAgent()
    with pytest.raises(ValueError, match=error):
        cycle.run(models, search, max_model_rounds=1, max_optimizer_rounds=1)
    assert (
        not models.requests
        and not search.requests
        and not (tmp_path / "optimizer").exists()
    )


def test_policy_cycle_missing_joint_candidate_is_rejected_then_revised(
    runner, tmp_path
):
    """The cycle supplies the export contract to model revision and retries bad factories."""
    cycle, _ = policy_cycle(runner, tmp_path)
    model = CycleAgent(FITTED_MODEL_SOURCE, CYCLE_FITTED_SOURCE)
    result = cycle.run(
        model,
        CycleAgent(JOINT_HEURISTIC_SOURCE),
        max_model_rounds=2,
        max_optimizer_rounds=1,
    )
    assert result["model"]["accepted"] and len(model.requests) == 2
    assert "must export predictor.joint" in str(model.requests[1].feedback)


def test_policy_cycle_context_and_completed_source_cannot_be_replaced(runner, tmp_path):
    """Resume refuses changed inputs or source, rather than following a new active model."""
    cycle, _ = policy_cycle(runner, tmp_path)
    result = cycle.run(
        CycleAgent(CYCLE_FITTED_SOURCE),
        CycleAgent(JOINT_HEURISTIC_SOURCE),
        max_model_rounds=1,
        max_optimizer_rounds=1,
    )
    with pytest.raises(ValueError, match="context.json differs"):
        cycle.run(
            CycleAgent(), CycleAgent(), max_model_rounds=2, max_optimizer_rounds=1
        )
    source = tmp_path / "model" / result["model"]["source"]
    source.write_text(source.read_text() + "\n# changed after handoff\n")
    with pytest.raises(ValueError, match="source is missing or changed"):
        cycle.run(
            CycleAgent(), CycleAgent(), max_model_rounds=1, max_optimizer_rounds=1
        )


def test_policy_cycle_initial_source_publication_is_atomic_and_never_rewrites(
    runner, tmp_path, monkeypatch
):
    """An interrupted source publication is retryable; completed inputs are read-only."""
    import areal.autopilot.revision.cycle as module

    original = module.os.replace

    def fail_initial(src, dst):
        if Path(dst).name == "initial.py":
            raise OSError("initial source publication failure")
        original(src, dst)

    cycle, _ = policy_cycle(runner, tmp_path)
    models, searches = (
        CycleAgent(CYCLE_FITTED_SOURCE),
        CycleAgent(JOINT_HEURISTIC_SOURCE),
    )
    monkeypatch.setattr(module.os, "replace", fail_initial)
    with pytest.raises(OSError, match="initial source publication"):
        cycle.run(models, searches, max_model_rounds=1, max_optimizer_rounds=1)
    assert not (tmp_path / "model/initial.py").exists() and not models.requests
    monkeypatch.setattr(module.os, "replace", original)
    cycle.run(models, searches, max_model_rounds=1, max_optimizer_rounds=1)
    paths = [tmp_path / name / "initial.py" for name in ("model", "optimizer")]
    before = [(p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths]
    monkeypatch.setattr(module.os, "replace", fail_initial)
    cycle.run(CycleAgent(), CycleAgent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert before == [
        (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths
    ]


def test_policy_cycle_joint_export_cannot_mutate_historical_fit(runner, tmp_path):
    """A repeatable joint projection must leave its owner's historical parameters frozen."""
    source = CYCLE_MODEL_SOURCE.replace(
        "self.joint = Joint(scale)", "self.joint = Joint(scale, self)"
    )
    source = source.replace(
        "class Joint:\n    def __init__(self, scale): self.scale = scale",
        "class Joint:\n    def __init__(self, scale, owner): self.scale = scale; self.owner = owner",
    ).replace(
        "def evaluate(self, scenario, config):",
        "def evaluate(self, scenario, config):\n        self.owner.scale = 999.0",
    )
    cycle, _ = policy_cycle(runner, tmp_path, initial=source)
    model, optimizer = CycleAgent(), CycleAgent()
    with pytest.raises(ValueError, match="changed fitted historical model state"):
        cycle.run(model, optimizer, max_model_rounds=1, max_optimizer_rounds=1)
    assert not model.requests and not optimizer.requests


def serial_revision_inputs(*, updates=6, compatibility=None):
    """Known serial timings; validation targets never enter the source profile."""
    from tests.test_elastic_basic_policy import WORK, observation, serial_joint_policy

    policy, current, target = serial_joint_policy()
    profile = {
        "format_version": "serial-joint-v1",
        "compatibility": compatibility or {"fixture": "joint"},
        "initial": asdict(current),
        "candidates": [asdict(current), asdict(target)],
        "phases": [asdict(p) for p in policy.simulator.phases],
        "episodes": [asdict(e) for e in policy.simulator.episodes],
        "transitions": [asdict(t) for t in policy.simulator.transitions.profiles],
    }
    settings = asdict(policy.optimizer)
    window = {"updates": [], "boundaries": [], "through_version": updates}
    policy.history.add_workload(WORK)
    for version in range(1, updates + 1):
        # Repeated actual configuration makes both timing and search gates defined.
        policy.decide_joint(
            observation(model_version=version - 1, serving_version=version - 1),
            current,
            4,
        )
        row = {
            "weights_version": version,
            "joint_config": asdict(current),
            "observation": asdict(
                observation(model_version=version - 1, serving_version=version - 1)
            ),
            "history": asdict(policy.history.snapshot()),
            "joint_window": asdict(policy.last_window),
            "measurements": [
                {
                    "phase": p.phase,
                    "config": asdict(current.actor),
                    "workload": asdict(WORK),
                    "seconds": p.seconds / 2,
                }
                for p in policy.simulator.phases
                if p.config == current
            ],
            "episode": {
                "rollout": asdict(current.rollout),
                "workload": asdict(replace(WORK, rollout_seconds=10)),
                "seconds": 10,
            },
        }
        window["updates"].append(row)
        policy.record_committed(row)
    return profile, settings, window


SERIAL_TIMING_SOURCE = """from dataclasses import replace
from areal.autopilot.elastic.policy.basic.online import serial_model_from_profile, SerialTimingEstimate
class Model:
    def __init__(self, profile): self.base = serial_model_from_profile(profile)
    def predict_timing(self, context):
        value = self.base.predict_timing(replace(context, phases=(), episodes=()))
        return SerialTimingEstimate(None if value.seconds is None else value.seconds / 2)
def build_serial_model(profile): return Model(profile)
"""

SERIAL_SEARCH_SOURCE = """from areal.autopilot.elastic.policy.basic.joint import JointSearchResult
class Search:
    def __init__(self, settings): self.config = settings['candidates'][-1]
    def optimize(self, scenario, simulator):
        p = simulator.evaluate(scenario, self.config)
        return JointSearchResult(self.config, p, (), 0, 0, 'fixture')
def build_joint_optimizer(settings): return Search(settings)
"""


def accepted_serial_bundle(root, runner, *, updates=6, compatibility=None):
    from areal.autopilot.revision.serial import (
        SerialRevisionCycle,
        runtime_joint_problems,
    )

    profile, settings, window = serial_revision_inputs(
        updates=updates, compatibility=compatibility
    )
    model, scenarios = runtime_joint_problems(
        profile, window, settings=settings, runner=runner
    )

    class Agent:
        def propose(self, request):
            return Proposal(
                SERIAL_TIMING_SOURCE
                if request.target == "serial_model"
                else SERIAL_SEARCH_SOURCE
            )

    cycle = SerialRevisionCycle(model, scenarios, root)
    result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert result["model"]["accepted"] and result["optimizer"]["accepted"]
    return Path(result["bundle"]), model, scenarios, profile, settings, window


def test_serial_model_labels_stay_private_and_episode_source_config_is_preserved(
    runner, tmp_path
):
    from areal.autopilot.revision.serial import runtime_joint_problems

    profile, settings, window = serial_revision_inputs()
    # Final update consumes under a different rollout; generation used the source.
    window["updates"][-1]["joint_config"] = profile["candidates"][-1]
    model, _ = runtime_joint_problems(profile, window, settings=settings, runner=runner)
    cases = json.loads(model._public_json)
    assert all(
        c["workload"]["rollout_seconds"] == 0 and c["workload"]["actor_stats"] is None
        for c in cases
    )
    assert all(
        e["source_id"]
        not in {"committed-episode-v4", "committed-episode-v5", "committed-episode-v6"}
        for c in cases
        for e in c["episodes"]
    )
    assert cases[-1]["config"] == profile["initial"]
    assert len(model.calibration["runtime_calibration"]) == 3
    baseline = model.evaluate(BASELINE_SOURCE["serial_model"], tmp_path / "baseline")
    improved = model.evaluate(SERIAL_TIMING_SOURCE, tmp_path / "improved")
    assert baseline.valid and improved.valid and model.compare(improved, baseline)[0]
    assert improved.score == 0
    assert model.feedback(improved)["score"] is None
    assert all(
        g["wape"] is None
        for g in model.feedback(improved)["metrics"]["groups"].values()
    )
    request = json.loads((tmp_path / "improved/request.json").read_text())
    assert "runtime_calibration" not in request["profile"]
    assert all("seconds" not in c for c in request["cases"])


@pytest.mark.parametrize("mode", ["missing", "stateful", "nonfinite", "parent-drift"])
def test_serial_timing_rejects_coverage_loss_state_and_worker_drift(
    runner, tmp_path, mode, monkeypatch
):
    from areal.autopilot.revision.serial import runtime_joint_problems

    if mode == "parent-drift":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-only-test-sentinel")
    profile, settings, window = serial_revision_inputs()
    model, _ = runtime_joint_problems(profile, window, settings=settings, runner=runner)
    baseline = model.evaluate(BASELINE_SOURCE["serial_model"], tmp_path / "baseline")
    body = {
        "missing": "return SerialTimingEstimate(None)",
        "stateful": "self.n += 1; return SerialTimingEstimate(self.n)",
        "nonfinite": "return SerialTimingEstimate(float('nan'))",
        "parent-drift": "import os; return SerialTimingEstimate(1 if os.environ.get('CUDA_VISIBLE_DEVICES') == '' else 2)",
    }[mode]
    source = (
        "from areal.autopilot.elastic.policy.basic.online import SerialTimingEstimate\nclass Model:\n    n=0\n    def predict_timing(self, context):\n        "
        + body
        + "\ndef build_serial_model(profile): return Model()\n"
    )
    report = model.evaluate(source, tmp_path / mode)
    assert not model.compare(report, baseline)[0]
    if mode != "missing":
        assert not report.valid


@pytest.mark.usefixtures("serial_optimizer_clock")
def test_serial_cycle_freezes_model_revalidates_search_and_deploys_executable_predictor(
    runner, tmp_path
):
    from tests.test_elastic_basic_policy import WORK

    from areal.autopilot.elastic.policy.basic.online import SerialTimingInput
    from areal.autopilot.elastic.runtime.serial_policy import (
        build_serial_policy,
        validate_serial_bundle,
    )

    bundle, evaluation, scenarios, profile, settings, window = accepted_serial_bundle(
        tmp_path / "cycle", runner
    )
    value = json.loads(bundle.read_text())
    assert set(value["accepted_components"]) == {"serial_model", "joint_optimizer"}
    assert value["evaluations"]["joint_optimizer"]["context"]["deployment_context"][
        "model_version"
    ].startswith("serial-model:")
    policy = build_serial_policy(value)
    current = scenarios[-1].current
    # Not a finite prediction lookup: new train workload is evaluated by code.
    unseen = SerialTimingInput(current, "train", replace(WORK, sequence_tokens=200))
    assert policy.simulator.estimate_timing(unseen).seconds == 1
    before = policy.simulator.evaluate(scenarios[-1], current)
    selected = policy.optimizer.optimize(scenarios[-1], policy.simulator)
    assert selected.config == policy.optimizer.candidates[-1]
    assert before.actor_gpu_seconds > 0 and before.rollout_gpu_seconds > 0
    # Hashing alone cannot replace evidence: mutate a contract and rehash.
    for key, changed in (("horizon", 3), ("idle_seconds", 1)):
        modified = json.loads(bundle.read_text())
        modified[key] = changed
        modified["bundle_id"] = digest(
            {k: v for k, v in modified.items() if k != "bundle_id"}
        )
        with pytest.raises(ValueError, match="revalidated"):
            validate_serial_bundle(modified)
    value["settings"]["switch_margin"] = 0
    value["bundle_id"] = digest({k: v for k, v in value.items() if k != "bundle_id"})
    with pytest.raises(ValueError, match="acceptance|settings|revalidation|contract"):
        validate_serial_bundle(value)


@pytest.mark.usefixtures("serial_optimizer_clock")
def test_serial_model_only_revision_revalidates_unchanged_optimizer_and_resumes(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.serial_policy import build_serial_policy
    from areal.autopilot.revision.serial import (
        SerialRevisionCycle,
        runtime_joint_problems,
    )

    profile, settings, window = serial_revision_inputs()
    model, scenarios = runtime_joint_problems(
        profile, window, settings=settings, runner=runner
    )
    calls = []

    class Agent:
        def propose(self, request):
            calls.append(request.target)
            return Proposal(
                SERIAL_TIMING_SOURCE
                if request.target == "serial_model"
                else BASELINE_SOURCE["joint_optimizer"]
            )

    cycle = SerialRevisionCycle(model, scenarios, tmp_path / "cycle")
    result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert result["model"]["accepted"] and not result["optimizer"]["accepted"]
    assert (
        cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
        == result
    )
    assert calls == ["serial_model", "joint_optimizer"]
    value = json.loads(Path(result["bundle"]).read_text())
    assert value["evaluations"]["joint_optimizer"]["report"]["valid"]
    build_serial_policy(value)


@pytest.fixture
def serial_execution(tmp_path):
    from tests.test_elastic_recovery import joint_execution

    yield from joint_execution.__wrapped__(tmp_path)


@pytest.mark.parametrize("pointer_fault", [False, True])
@pytest.mark.usefixtures("serial_optimizer_clock")
def test_joint_policy_activation_hydrates_history_preserves_config_and_recovers(
    runner, tmp_path, serial_execution, monkeypatch, pointer_fault
):
    from tests.test_elastic_recovery import online_episode

    import areal.autopilot.elastic.runtime.policy as storage
    from areal.autopilot.elastic.policy.basic.online import (
        OnlineJointPolicy,
        serial_model_from_profile,
    )
    from areal.autopilot.elastic.runtime.history import RuntimeHistory
    from areal.autopilot.elastic.runtime.policy import PolicyApplication, PolicyRequest
    from areal.autopilot.elastic.runtime.serial_policy import serial_policy_identity
    from areal.autopilot.revision.serial import joint_settings

    loop, executor, configs, _, _ = serial_execution
    bundle, _, _, profile, settings, _ = accepted_serial_bundle(
        tmp_path / "revision", runner, compatibility=loop.trainer.group.compatibility
    )
    loop._policy = OnlineJointPolicy(
        serial_model_from_profile(profile), joint_settings(settings)
    )
    loop.history = loop._policy.history
    loop.runtime_history = RuntimeHistory(
        tmp_path / "observations.json",
        context={
            "queue_id": loop.trainer.queue.queue_id,
            "checkpoint_root": str(loop.trainer.group.store.root.resolve()),
            "compatibility": loop.trainer.group.compatibility,
        },
    )
    loop.policy_application = PolicyApplication(
        tmp_path / "policy-application",
        compatibility=profile["compatibility"],
        execution_context={
            "queue_id": loop.trainer.queue.queue_id,
            "checkpoint_root": str(loop.trainer.group.store.root.resolve()),
            "scope": "serial-joint-timing-v1",
            "joint_executor": executor.context_id,
            "max_total_gpus": 4,
            "joint_policy_identity": serial_policy_identity(
                profile, settings, horizon=2, idle_seconds=30
            ),
        },
    )
    for _ in range(2):
        online_episode(loop, executor)
        loop.tick()
    old_history = loop.history
    snapshot = old_history.snapshot()
    phases, episodes = loop._policy.phases[:], loop._policy.episodes[:]
    cursors = loop._policy.through_version, loop._policy._boundary_sequence
    before = loop.trainer.version, configs.state()["generation"], executor.observe()
    prepared = loop.prepare_policy(bundle)
    request = PolicyRequest(
        "joint-activate", prepared.bundle_id, 0, loop.trainer.version
    )
    if pointer_fault:
        write = storage.atomic_json

        def interrupt(path, value):
            write(path, value)
            if path.name == "active.json":
                raise OSError("Lost joint policy pointer acknowledgement")

        monkeypatch.setattr(storage, "atomic_json", interrupt)
        with pytest.raises(OSError, match="acknowledgement"):
            loop.apply_policy(request, prepared)
        assert loop.state == "FAILED" and not loop.publisher.ready
        monkeypatch.setattr(storage, "atomic_json", write)
        loop.recover_policy()
    else:
        with loop.episode():
            from areal.autopilot.elastic.runtime.policy import PolicyBoundaryBusy

            with pytest.raises(PolicyBoundaryBusy):
                loop.apply_policy(request, prepared)
        assert loop.policy_generation == 0
        loop.apply_policy(request, prepared)
    assert loop.policy_generation == 1 and loop.history is old_history
    assert loop.history.snapshot() == snapshot
    assert loop._policy.history is old_history
    assert (loop._policy.phases, loop._policy.episodes) == (phases, episodes)
    assert (loop._policy.through_version, loop._policy._boundary_sequence) == cursors
    assert (
        loop.trainer.version,
        configs.state()["generation"],
        executor.observe(),
    ) == before
    online_episode(loop, executor)
    loop.tick()
    assert loop.trainer.version == before[0] + 1
    assert (
        loop.runtime_history.snapshot(loop.trainer.version)["updates"][-1][
            "policy_generation"
        ]
        == 1
    )
    assert loop._policy.last_window.observation.model_version == before[0]
    assert loop._policy.through_version == loop.trainer.version


def test_joint_runtime_optimizer_rejects_swallowed_query_violations(runner, tmp_path):
    from areal.autopilot.elastic.runtime.serial_policy import VerifiedJointOptimizer
    from areal.autopilot.revision.serial import (
        build_serial_model,
        runtime_joint_problems,
    )

    profile, settings, window = serial_revision_inputs()
    model, scenarios = runtime_joint_problems(
        profile, window, settings=settings, runner=runner
    )
    source = SERIAL_SEARCH_SOURCE.replace(
        "p = simulator.evaluate(scenario, self.config)",
        "try: simulator.evaluate(scenario, None)\n        except ValueError: pass\n        p = simulator.evaluate(scenario, self.config)",
    )
    optimizer = VerifiedJointOptimizer(source, model.settings)
    with pytest.raises(ValueError, match="violated fixed contract"):
        optimizer.optimize(
            scenarios[0], build_serial_model(BASELINE_SOURCE["serial_model"], profile)
        )


@pytest.mark.usefixtures("serial_optimizer_clock")
def test_joint_next_window_retains_model_acceptance_and_revalidates_under_new_window(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.serial_policy import build_serial_policy
    from areal.autopilot.revision.serial import (
        SerialRevisionCycle,
        runtime_joint_problems,
    )

    parent, _, _, profile, settings, _ = accepted_serial_bundle(
        tmp_path / "first", runner
    )
    _, _, window = serial_revision_inputs(updates=8)
    model, scenarios = runtime_joint_problems(
        profile, window, settings=settings, runner=runner
    )
    no_queries = SERIAL_SEARCH_SOURCE.replace(
        "p = simulator.evaluate(scenario, self.config)", "p = None"
    )

    class Agent:
        def propose(self, request):
            return Proposal(
                SERIAL_TIMING_SOURCE if request.target == "serial_model" else no_queries
            )

    cycle = SerialRevisionCycle(
        model,
        scenarios,
        tmp_path / "second",
        initial_model_source=SERIAL_TIMING_SOURCE,
        initial_optimizer_source=SERIAL_SEARCH_SOURCE,
        parent_bundle=parent,
    )
    result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert not result["model"]["accepted"] and result["optimizer"]["accepted"]
    value, prior = (
        json.loads(Path(result["bundle"]).read_text()),
        json.loads(parent.read_text()),
    )
    assert value["acceptances"]["serial_model"] == prior["acceptances"]["serial_model"]
    assert (
        value["evaluations"]["serial_model"]["context"]["benchmark_sha256"]
        != prior["evaluations"]["serial_model"]["context"]["benchmark_sha256"]
    )
    build_serial_policy(value)


def test_serial_interrupted_baseline_resume_rejects_changed_prediction_fingerprint(
    runner, tmp_path, monkeypatch
):
    from areal.autopilot.revision.serial import runtime_joint_problems

    profile, settings, window = serial_revision_inputs()
    model, _ = runtime_joint_problems(profile, window, settings=settings, runner=runner)
    source = """import os
from areal.autopilot.elastic.policy.basic.online import serial_model_from_profile, SerialTimingEstimate
class Model:
    def __init__(self,p): self.base=serial_model_from_profile(p)
    def predict_timing(self,c):
        v=self.base.predict_timing(c)
        return SerialTimingEstimate(None if v.seconds is None else v.seconds+float(os.environ.get('SERIAL_SHIFT','0')))
def build_serial_model(profile): return Model(profile)
"""

    class Agent:
        def propose(self, request):
            raise KeyboardInterrupt("interrupted after durable baseline")

    loop = RevisionLoop(Agent(), model, tmp_path / "revision", initial_source=source)
    with pytest.raises(KeyboardInterrupt):
        loop.run(max_rounds=1)
    before = (tmp_path / "revision/baseline.json").read_text()
    monkeypatch.setenv("SERIAL_SHIFT", "1")
    with pytest.raises(ValueError, match="changed"):
        loop.run(max_rounds=1)
    assert (tmp_path / "revision/baseline.json").read_text() == before


def test_joint_runtime_optimizer_cannot_block_ready_work_with_empty_selection(
    runner, tmp_path
):
    from areal.autopilot.elastic.runtime.serial_policy import VerifiedJointOptimizer
    from areal.autopilot.revision.serial import (
        build_serial_model,
        runtime_joint_problems,
    )

    profile, settings, window = serial_revision_inputs()
    model, scenarios = runtime_joint_problems(
        profile, window, settings=settings, runner=runner
    )
    source = SERIAL_SEARCH_SOURCE.replace(
        "p = simulator.evaluate(scenario, self.config)", "p = None"
    ).replace("JointSearchResult(self.config", "JointSearchResult(None")
    with pytest.raises(ValueError, match="omitted an eligible selection"):
        VerifiedJointOptimizer(source, model.settings).optimize(
            scenarios[0], build_serial_model(BASELINE_SOURCE["serial_model"], profile)
        )


@pytest.mark.usefixtures("serial_optimizer_clock")
def test_joint_background_revision_activates_and_then_automatically_switches_config(
    runner, tmp_path, serial_execution
):
    import time

    from tests.test_elastic_recovery import online_episode

    from areal.autopilot.elastic.policy.basic.online import (
        OnlineJointPolicy,
        serial_model_from_profile,
    )
    from areal.autopilot.elastic.runtime.history import RuntimeHistory
    from areal.autopilot.elastic.runtime.policy import PolicyApplication
    from areal.autopilot.elastic.runtime.serial_policy import serial_policy_identity
    from areal.autopilot.revision.online import RuntimeRevisionLoop
    from areal.autopilot.revision.serial import joint_settings

    loop, executor, configs, _, _ = serial_execution
    profile, settings, _ = serial_revision_inputs(
        compatibility=loop.trainer.group.compatibility
    )
    for phase in profile["phases"]:
        if phase["phase"] == "publish":
            phase["seconds"] = 1_000_000
    loop._policy = OnlineJointPolicy(
        serial_model_from_profile(profile), joint_settings(settings)
    )
    loop.history = loop._policy.history
    loop.runtime_history = RuntimeHistory(
        tmp_path / "history.json",
        context={
            "queue_id": loop.trainer.queue.queue_id,
            "checkpoint_root": str(loop.trainer.group.store.root.resolve()),
            "compatibility": profile["compatibility"],
        },
    )
    loop.policy_application = PolicyApplication(
        tmp_path / "policies",
        compatibility=profile["compatibility"],
        execution_context={
            "queue_id": loop.trainer.queue.queue_id,
            "checkpoint_root": str(loop.trainer.group.store.root.resolve()),
            "scope": "serial-joint-timing-v1",
            "joint_executor": executor.context_id,
            "max_total_gpus": 4,
            "joint_policy_identity": serial_policy_identity(
                profile, settings, horizon=2, idle_seconds=30
            ),
        },
    )
    source = """from areal.autopilot.elastic.policy.basic.online import serial_model_from_profile, SerialTimingEstimate
class Model:
    def __init__(self,p): self.base=serial_model_from_profile(p)
    def predict_timing(self,c): return SerialTimingEstimate(0) if c.phase=='publish' else self.base.predict_timing(c)
def build_serial_model(profile): return Model(profile)
"""
    calls = []

    class Agent:
        def propose(self, request):
            calls.append(request.target)
            return Proposal(
                source if request.target == "serial_model" else SERIAL_SEARCH_SOURCE
            )

    arguments = dict(
        profile=profile,
        settings=settings,
        scope="serial-joint-timing-v1",
        model_agent=Agent(),
        optimizer_agent=Agent(),
        max_windows=1,
        max_model_rounds=1,
        max_optimizer_rounds=1,
        runner=runner,
    )
    revision = RuntimeRevisionLoop(tmp_path / "online", **arguments)
    try:
        for _ in range(4):
            online_episode(loop, executor)
            loop.tick()
        assert configs.state()["generation"] == 0
        revision.tick(loop)
        deadline = time.monotonic() + 30
        while loop.policy_generation == 0:
            assert time.monotonic() < deadline
            revision.tick(loop)
            time.sleep(0.02)
        assert calls == ["serial_model", "joint_optimizer"]
        assert configs.state()["generation"] == 0
        assert loop._policy.through_version == 5 and len(loop._policy.episodes) == 4
        online_episode(loop, executor)
        loop.tick()
        assert configs.state()["generation"] == 1
        assert executor.observe().config == loop._policy.optimizer.candidates[-1]
    finally:
        revision.close()
    recovered = RuntimeRevisionLoop(tmp_path / "online", **arguments)
    try:
        recovered.tick(loop)
        assert recovered.future is None and len(calls) == 2
    finally:
        recovered.close()
