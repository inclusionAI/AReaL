# SPDX-License-Identifier: Apache-2.0

"""Hand-calculated workflow costs, causal forecasting and joint action search."""

from dataclasses import replace

import pytest

from areal.elastic.contracts import BatchWorkload, ElasticObservation, TrainingConfig
from areal.elastic.policy.basic import BasicPolicy, IdleTimeoutPolicy
from areal.elastic.policy.basic.optimizer import BasicOptimizer
from areal.elastic.policy.basic.replay import (
    ActorPhase,
    ActorReplayStep,
    GroupArrival,
    replay_actor,
)
from areal.elastic.policy.basic.simulator import (
    ComponentContext,
    ConstantComponent,
    Estimate,
    ForecastUpdate,
    HistorySnapshot,
    HistoryWindow,
    MeasuredComponent,
    Measurement,
    ParallelComponent,
    PlanStep,
    SequenceComponent,
    WorkflowModel,
    WorkflowSimulator,
    forecast_window,
)

WORK = BatchWorkload(2, 50, 100, 20)


def observation(**changes):
    base = ElasticObservation(
        "ACTIVE",
        1,
        1,
        True,
        True,
        0,
        True,
        False,
        True,
        10,
        training_config=TrainingConfig(2),
        next_batch=WORK,
        max_train_gpus=5,
        has_checkpoint=True,
    )
    return replace(base, **changes)


def model(train=10, restore=5):
    return WorkflowModel(
        ConstantComponent(train),
        ConstantComponent(5),
        ConstantComponent(restore),
        ConstantComponent(2),
        ConstantComponent(1),
        ConstantComponent(3),
        inference_gpus=1,
    )


def serial_joint_policy(*, seconds=20, switch=1, max_seconds=None):
    from tests.test_elastic_recovery import rollout_config

    from areal.elastic.policy.basic.cost_model.transition import (
        JointTransitionModel,
        TransitionPhase,
        TransitionProfile,
    )
    from areal.elastic.policy.basic.joint import JointConfig
    from areal.elastic.policy.basic.online import (
        EpisodeTiming,
        JointPhaseTiming,
        OnlineJointPolicy,
        SerialJointModel,
    )
    from areal.elastic.policy.basic.optimizer import JointOptimizer

    current = JointConfig(TrainingConfig(2), rollout_config())
    target = JointConfig(TrainingConfig(1), replace(current.rollout, replicas=2))
    transitions = JointTransitionModel(
        tuple(
            TransitionProfile(
                "reconfigure", a, b, (TransitionPhase(switch, 2, 2),), "fixture-switch"
            )
            for a, b in ((current, target), (target, current))
        )
    )
    workflow = model(train=1, restore=5)
    phases = tuple(
        JointPhaseTiming(
            c,
            phase,
            getattr(workflow, phase).seconds,
            100 if phase == "train" else 0,
            "seed-phase",
        )
        for c in (current, target)
        for phase in ("start", "restore", "train", "checkpoint", "publish", "release")
    )
    predictor = SerialJointModel(
        phases,
        (
            EpisodeTiming(current.rollout, WORK, seconds, "seed-current"),
            EpisodeTiming(target.rollout, WORK, 1, "seed-target"),
        ),
        transitions,
        idle_seconds=5,
    )
    return (
        OnlineJointPolicy(
            predictor, JointOptimizer((current, target), max_seconds=max_seconds)
        ),
        current,
        target,
    )


def test_serial_joint_cost_charges_only_future_rollout_and_idle_release():
    """Hand accounting includes directed switches, overlap and terminal release."""
    policy, current, target = serial_joint_policy()
    policy.history.add_workload(WORK)
    decision = policy.decide_joint(observation(), current, 4)
    window = policy.last_window
    incumbent = policy.simulator.evaluate(window, current)
    # First update 1+2+1=4; future wait20 overlaps release after5 (release3),
    # restore5, second update4, terminal release3: wall36, Actor48, Rollout36.
    assert (
        incumbent.seconds,
        incumbent.actor_gpu_seconds,
        incumbent.rollout_gpu_seconds,
    ) == (36, 48, 36)
    alternative = policy.simulator.evaluate(window, target)
    # Switch1 (2+2 GPUs), update4, future wait1, update4, terminal release3.
    assert (
        alternative.seconds,
        alternative.actor_gpu_seconds,
        alternative.rollout_gpu_seconds,
    ) == (13, 14, 26)
    assert decision.target == target
    assert len(window.batches) == 2 and window.batches[0].actor is WORK


@pytest.mark.parametrize(
    "change",
    ["missing-rollout", "missing-transition", "pool", "deadline", "expensive-switch"],
)
def test_online_joint_search_respects_coverage_capacity_deadline_and_switch_cost(
    change,
):
    policy, current, target = serial_joint_policy(
        switch=100 if change == "expensive-switch" else 1,
        max_seconds=1 if change == "deadline" else None,
    )
    if change == "missing-rollout":
        policy.simulator.episodes = policy.simulator.episodes[:1]
    if change == "missing-transition":
        from areal.elastic.policy.basic.cost_model.transition import (
            JointTransitionModel,
        )

        policy.simulator.transitions = JointTransitionModel(())
    policy.history.add_workload(WORK)
    decision = policy.decide_joint(observation(), current, 3 if change == "pool" else 4)
    if change == "pool":
        # Both actual candidates fit three GPUs; a four-GPU transition does not.
        assert decision.target == current
    elif change == "deadline":
        assert not decision.train and policy.last_search.config is None
    else:
        assert decision.target == current


def test_online_joint_cold_start_and_uncommitted_or_busy_work_do_not_search():
    policy, current, target = serial_joint_policy()
    assert policy.decide_joint(observation(), current, 4) == "train"
    assert policy.last_search is None
    policy.history.add_workload(WORK)
    for sample in (
        observation(gate_available=False),
        observation(committed=False),
        observation(pending_publication=True),
    ):
        policy.decide_joint(sample, current, 4)
        assert policy.last_search is None
    policy.through_version = 2
    with pytest.raises(ValueError, match="newer"):
        policy.decide_joint(observation(), current, 4)


def test_online_joint_opaque_rollout_profiles_do_not_scale_to_different_workloads():
    policy, current, target = serial_joint_policy()
    policy.history.add_workload(replace(WORK, sequence_tokens=101))
    decision = policy.decide_joint(observation(), current, 4)
    assert decision == "train"  # Bootstrap current; no unsupported switch.
    assert all(
        row.prediction.status == "unsupported" for row in policy.last_search.candidates
    )


def test_serial_joint_phase_feedback_never_crosses_rollout_configuration():
    """Same Actor count does not make publication timings portable across layouts."""
    from areal.elastic.policy.basic.online import JointPhaseTiming

    policy, current, target = serial_joint_policy()
    policy.history.add_workload(WORK)
    policy.decide_joint(observation(), current, 4)
    missing = replace(target, actor=current.actor)
    result = policy.simulator.evaluate(
        replace(policy.last_window, current=missing), missing
    )
    assert (
        result.status == "unsupported"
        and "complete joint configuration" in result.reason
    )
    before = policy.simulator.evaluate(policy.last_window, target)
    policy.phases.append(
        JointPhaseTiming(current, "publish", 1000, 100, "new-source-measurement")
    )
    policy.decide_joint(observation(), current, 4)
    after = policy.simulator.evaluate(policy.last_window, target)
    assert after.total_gpu_seconds == before.total_gpu_seconds
    policy.phases.append(
        JointPhaseTiming(target, "publish", 100, 100, "new-target-measurement")
    )
    policy.decide_joint(observation(), current, 4)
    assert (
        policy.simulator.evaluate(policy.last_window, target).seconds > before.seconds
    )


def test_composition_counts_shared_residency_once():
    """Parallel extra resources add, while shared training residency uses max time."""
    phase = SequenceComponent(
        (
            ParallelComponent((ConstantComponent(4, 8), ConstantComponent(7, 3))),
            ConstantComponent(2, 1),
        )
    )
    context = ComponentContext(TrainingConfig(2), WORK, HistorySnapshot())
    assert phase.estimate(context) == Estimate(9, 12)
    simulator = WorkflowSimulator(replace(model(), train=phase))
    result = simulator.evaluate(
        observation(),
        (ForecastUpdate(WORK, 0),),
        (PlanStep(TrainingConfig(2)),),
        HistorySnapshot(),
    )
    assert result.seconds == 15  # train9 + checkpoint2 + publish1 + release3
    assert result.training_gpu_seconds == 30
    assert result.extra_gpu_seconds == 12
    assert result.total_gpu_seconds == 57
    assert result.samples == 2 and result.effective_tokens == 50


def test_release_overlaps_rollout_but_restore_waits_for_data():
    simulator = WorkflowSimulator(model())
    forecast = (ForecastUpdate(WORK, 20),)
    obs = observation(batch_ready=False, next_batch=None)
    keep = simulator.evaluate(
        obs, forecast, (PlanStep(TrainingConfig(2)),), HistorySnapshot()
    )
    release = simulator.evaluate(
        obs, forecast, (PlanStep(TrainingConfig(2), True),), HistorySnapshot()
    )
    assert keep.seconds == 36 and keep.training_gpu_seconds == 72
    assert release.seconds == 41 and release.training_gpu_seconds == 48
    assert release.total_gpu_seconds == 89 < keep.total_gpu_seconds == 108


def test_history_snapshot_and_forecast_do_not_read_future_updates():
    window = HistoryWindow(2)
    window.add_workload(WORK)
    snapshot = window.snapshot()
    window.add_workload(replace(WORK, sequence_tokens=500, rollout_seconds=100))
    forecast = forecast_window(
        observation(
            batch_ready=False, next_batch=None, episode_active=True, episode_elapsed=5
        ),
        snapshot,
        2,
    )
    assert [x.wait_seconds for x in forecast] == [15, 20]
    assert all(x.workload.sequence_tokens == 100 for x in forecast)
    window.add_workload(replace(WORK, rollout_seconds=10))
    assert len(window.snapshot().workloads) == 2


def test_measured_components_reject_unknown_counts_and_distant_workloads():
    component = MeasuredComponent(
        "train", (Measurement("train", 3, 10, 100),), scale_tokens=True
    )
    context = ComponentContext(
        TrainingConfig(3), replace(WORK, sequence_tokens=200), HistorySnapshot()
    )
    assert component.estimate(context).seconds == 20
    assert not component.estimate(replace(context, config=TrainingConfig(2))).feasible
    assert not component.estimate(
        replace(context, workload=replace(WORK, sequence_tokens=1000))
    ).feasible


def test_optimizer_searches_every_count_and_selects_three_gpus():
    profiles = tuple(
        Measurement("train", n, seconds, 100)
        for n, seconds in enumerate((40, 22, 9, 8, 10), 1)
    )
    simulator = WorkflowSimulator(
        replace(model(), train=MeasuredComponent("train", profiles, scale_tokens=True))
    )
    optimizer = BasicOptimizer(horizon=2, switch_margin=0)
    result = optimizer.optimize(
        observation(training_active=False), simulator, HistorySnapshot((WORK,))
    )
    assert result.decision.action == "launch"
    assert result.decision.config == TrainingConfig(3)
    assert result.decision.train
    assert result.prediction.samples == 4
    assert result.evaluated == 100
    # A shorter bounded horizon still considers all five counts.
    bounded = replace(optimizer, max_candidates=10).optimize(
        observation(training_active=False), simulator, HistorySnapshot((WORK,))
    )
    assert bounded.evaluated == 10 and bounded.decision.config.world_size == 3


def test_optimizer_releases_for_long_gap_and_keeps_for_short_gap():
    policy = BasicPolicy(
        WorkflowSimulator(model()), optimizer=BasicOptimizer(horizon=1, switch_margin=0)
    )
    policy.record_workload(replace(WORK, rollout_seconds=100))
    obs = observation(batch_ready=False, next_batch=None, max_train_gpus=2)
    assert policy.decide(obs).action == "release"
    short = BasicPolicy(
        WorkflowSimulator(model(restore=100)), optimizer=BasicOptimizer(horizon=1)
    )
    short.record_workload(replace(WORK, rollout_seconds=1))
    decision = short.decide(obs)
    assert decision.action == "keep" and not decision.train


def test_cold_fallback_makes_progress_and_preserves_publication_guard():
    missing = MeasuredComponent("unmeasured")
    policy = BasicPolicy(
        WorkflowSimulator(replace(model(), train=missing)),
        fallback=IdleTimeoutPolicy(5),
    )
    decision = policy.decide(observation(training_active=False))
    assert (
        decision.action == "launch"
        and decision.config.world_size == 2
        and decision.train
    )
    assert not policy.decide(observation(pending_publication=True)).train
    assert (
        policy.decide(
            observation(batch_ready=False, next_batch=None, idle_duration=6)
        ).action
        == "release"
    )


def test_custom_workflow_component_controls_feasibility():
    class MemoryBound:
        def estimate(self, context):
            return Estimate(
                10, feasible=context.config.world_size >= 3, reason="Memory limit"
            )

    simulator = WorkflowSimulator(replace(model(), train=MemoryBound()))
    result = BasicOptimizer(horizon=1).optimize(
        observation(training_active=False), simulator, HistorySnapshot()
    )
    assert result.decision.config.world_size == 3
    assert "Memory limit" in result.rejected


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_config_rejects_invalid_counts(value):
    with pytest.raises(ValueError):
        TrainingConfig(value)


def test_rollout_is_a_composable_workflow_component():
    workflow = replace(
        model(),
        rollout=ParallelComponent((ConstantComponent(8, 4), ConstantComponent(5, 7))),
    )
    result = WorkflowSimulator(workflow).evaluate(
        observation(),
        (ForecastUpdate(WORK, 20),),
        (PlanStep(TrainingConfig(2)),),
        HistorySnapshot(),
    )
    assert result.seconds == 24  # rollout8 + train10 + save2 + publish1 + release3
    assert result.training_gpu_seconds == 48 and result.extra_gpu_seconds == 11


def test_example_uses_runtime_inference_residency_override(tmp_path):
    import json

    from examples.autopilot.elastic.basic_policy import measured_policy

    profile = tmp_path / "profiles.json"
    profile.write_text(
        json.dumps({"format_version": 1, "measurements": [], "inference_gpus": 0})
    )
    policy = measured_policy(profile, inference_gpus=1)
    assert policy.simulator.workflow.inference_gpus == 1


def test_forecast_uses_all_window_statistics():
    history = HistorySnapshot(
        (BatchWorkload(1, 10, 20, 10), BatchWorkload(3, 30, 60, 30))
    )
    future = forecast_window(
        observation(next_batch=None, batch_ready=False), history, 2
    )
    assert all(x.workload == BatchWorkload(2, 20, 40, 20) for x in future)
    assert [x.wait_seconds for x in future] == [20, 20]


def test_fallback_does_not_override_known_feasibility_or_time_constraints():
    class Impossible:
        def estimate(self, context):
            return Estimate(feasible=False, reason="Memory limit")

    policy = BasicPolicy(WorkflowSimulator(replace(model(), train=Impossible())))
    assert not policy.decide(observation()).train
    bounded = BasicPolicy(
        WorkflowSimulator(model()), optimizer=BasicOptimizer(max_seconds=1)
    )
    assert not bounded.decide(observation()).train


def test_joint_runtime_state_preserves_observations_older_than_bounded_update_archive():
    from dataclasses import asdict

    from tests.test_autopilot_revision import serial_revision_inputs

    policy, _, _ = serial_joint_policy()
    _, _, window = serial_revision_inputs(updates=4)
    template = window["updates"][0]
    template["measurements"] = [
        p
        for p in template["measurements"]
        if p["phase"] in {"train", "checkpoint", "publish"}
    ]
    rows = []
    for version in range(1, 601):
        row = {**template, "weights_version": version}
        rows.append(row)
        policy.record_committed(row)
    assert len(policy.phases) == 1536
    restored, _, _ = serial_joint_policy()
    bounded = {"updates": rows[-256:], "boundaries": []}
    restored.restore_runtime(bounded)
    assert len(restored.phases) == 768
    state = policy.runtime_state()
    restored.restore_joint_state(state, history=policy.history)
    for row in bounded["updates"]:
        restored.record_committed(row)
    assert restored.runtime_state() == state
    assert asdict(restored.history.snapshot()) == asdict(policy.history.snapshot())


def test_actor_replay_keeps_absolute_arrivals_while_actor_speed_changes():
    """Arrivals overlap Actor work; a slower Actor removes wait rather than shifting Rollout."""
    arrivals = (
        GroupArrival("a", 10, 4),
        GroupArrival("b", 20, 4),
        GroupArrival("prefetch", 12, 4),
    )
    steps = tuple(
        ActorReplayStep((g,), (ActorPhase("train", 5),), 100, 50) for g in ("a", "b")
    )
    fast = replay_actor(arrivals, steps, actor_gpus=4)
    slow = replay_actor(
        arrivals,
        (replace(steps[0], phases=(ActorPhase("train", 15),)), steps[1]),
        actor_gpus=4,
    )
    assert fast["seconds"] == 25 and slow["seconds"] == 30
    assert fast["phase_seconds"]["wait"] == 15
    assert slow["phase_seconds"]["wait"] == 10
    assert slow["steps"][1]["actor_start"] == 25
    assert slow["steps"][1]["queue_wait_seconds"] == {"b": 5}
    for result in (fast, slow):
        assert [
            (e["group_id"], e["time"])
            for e in result["events"]
            if e["kind"] == "group_ready"
        ] == [("a", 10), ("prefetch", 12), ("b", 20)]
        assert result["samples"] == 8 and result["groups"] == 2
        assert result["unused_arrived_groups"] == 1
        assert result["actor_gpu_hours"] == pytest.approx(4 * result["seconds"] / 3600)
        assert sum(result["phase_seconds"].values()) == pytest.approx(result["seconds"])


def test_actor_replay_relative_control_save_and_future_prefetch_accounting():
    """Relative overhead and final save hold Actor GPUs; unarrived work is excluded."""
    arrivals = (GroupArrival("a", 0, 4), GroupArrival("future", 100, 4))
    step = ActorReplayStep(
        ("a",),
        (ActorPhase("train", 2), ActorPhase("save", 5)),
        100,
        40,
        before_seconds=3,
        after_seconds=1,
    )
    result = replay_actor(arrivals, (step,), actor_gpus=4)
    assert result["seconds"] == 11
    assert result["phase_seconds"] == {"controller": 4, "train": 2, "save": 5}
    assert result["future_groups"] == 1 and result["unused_arrived_groups"] == 0
    assert result["effective_tokens"] == 40 and result["sequence_tokens"] == 100
    assert result["actor_gpu_hours"] == pytest.approx(44 / 3600)
    assert all(e["time"] <= 11 for e in result["events"])


def test_actor_replay_unsupported_work_has_no_full_window_score():
    """A covered prefix cannot masquerade as a cheaper complete workload."""
    arrivals = (GroupArrival("a", 0, 4), GroupArrival("b", 1, 4))
    steps = (
        ActorReplayStep(("a",), (ActorPhase("train", 2),), 100, 50),
        ActorReplayStep(("b",), (ActorPhase("train", None, "unsupported"),), 100, 50),
    )
    result = replay_actor(arrivals, steps, actor_gpus=4)
    assert result["status"] == "unsupported" and result["unsupported_steps"] == [2]
    assert result["seconds"] is None and result["actor_gpu_hours"] is None


@pytest.mark.parametrize(
    "case", ["duplicate_arrival", "duplicate_consumption", "missing_arrival"]
)
def test_actor_replay_rejects_ambiguous_consumption(case):
    """Each trained group has exactly one arrival and one consumption."""
    arrival = GroupArrival("a", 1, 4)
    step = ActorReplayStep(("a",), (ActorPhase("train", 2),), 100, 50)
    arrivals, steps = (arrival,), (step,)
    if case == "duplicate_arrival":
        arrivals = (arrival, arrival)
    elif case == "duplicate_consumption":
        steps = (step, step)
    else:
        arrivals = ()
    with pytest.raises(ValueError):
        replay_actor(arrivals, steps, actor_gpus=4)


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
def test_actor_replay_rejects_invalid_clock_values(seconds):
    """Invalid clock values cannot create negative or free GPU intervals."""
    with pytest.raises(ValueError):
        GroupArrival("a", seconds, 4)
    with pytest.raises(ValueError):
        ActorPhase("train", seconds)
