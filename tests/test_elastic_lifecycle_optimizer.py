# SPDX-License-Identifier: Apache-2.0

"""Actor-model integration, full profile identity and exhaustive lifecycle oracles."""

import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_elastic_basic_policy import model, observation

from areal.autopilot.domains.elastic.contracts import observation_from_dict
from areal.autopilot.domains.elastic.model_adapters import (
    JOINT_BOUNDARY,
    ROLLOUT_BOUNDARY,
    BasicHistoryPredictor,
)
from areal.autopilot.domains.elastic.model_evaluator import (
    HistoricalCase,
    HistoricalInput,
    HistoricalModelEvaluator,
    HistoricalPrediction,
    HistorySpan,
)
from areal.elastic.contracts import BatchWorkload, TrainingConfig
from areal.elastic.policy.basic import (
    ActorCatalog,
    JointBatch,
    JointConfig,
    JointOptimizer,
    JointPolicy,
    JointScenario,
    JointSimulator,
    JointTransitionModel,
    LifecycleCosts,
    LifecycleOptimizer,
    RolloutConfig,
    RolloutCostModel,
    RolloutProfile,
    RolloutStage,
    RolloutWorkload,
    TransitionPhase,
    TransitionProfile,
    actor_workload,
    build_actor_policy,
)
from areal.elastic.policy.basic.actor_cost import (
    ActorPrediction,
    ActorTrainComponent,
    CompatibleRestore,
)
from areal.elastic.policy.basic.cost_model import (
    ActorConfig,
    ActorCostModel,
    ParallelLayout,
    WorkloadStats,
    hardware_profile,
    qwen3_30b_a3b,
    trial_config,
)
from areal.elastic.policy.basic.simulator import (
    ComponentContext,
    ConstantComponent,
    ForecastUpdate,
    HistorySnapshot,
    MeasuredComponent,
    Measurement,
    PlanStep,
    WorkflowSimulator,
    forecast_window,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/autopilot/elastic/lifecycle_scenario.json"
STATS = WorkloadStats(128, 262144, max_seq_len=8192, min_seq_len=512, loss_tokens=65536)
WORK = actor_workload(STATS, samples=32, rollout_seconds=120)


@pytest.fixture
def catalog():
    actor_model = ActorCostModel(
        qwen3_30b_a3b(), hardware_profile("h200_sxm_reference")
    )
    trials = json.loads(EXAMPLE.read_text())["trials"]
    return ActorCatalog(
        actor_model,
        {
            label: replace(
                trial_config(label), recompute_logprob=True, gradient_checkpointing=True
            )
            for label in trials
        },
    )


def lifecycle(restore=12):
    return LifecycleCosts(
        ConstantComponent(15),
        ConstantComponent(restore),
        ConstantComponent(3),
        ConstantComponent(2),
        ConstantComponent(2),
        8,
    )


def test_full_profiles_deduplicate_aliases_without_collapsing_layouts(catalog):
    """Nine trial labels become four distinct layouts, all on 32 GPUs."""
    assert len(catalog.configurations) == 4
    assert {config.world_size for config in catalog.configurations} == {32}
    assert catalog.labels["S031"] == catalog.labels["S013"] == catalog.labels["S030"]
    assert catalog.labels["S029"] != catalog.labels["S031"]
    assert len(set(catalog.configurations)) == 4
    assert all(config != TrainingConfig(32) for config in catalog.configurations)
    with pytest.raises(ValueError, match="Unknown full actor"):
        catalog.actor_config(TrainingConfig(32))


def test_candidate_catalog_preserves_training_semantics(catalog):
    """A cheaper forecast cannot silently reduce PPO optimizer updates."""
    actor = catalog.actor_config(catalog.configurations[0])
    with pytest.raises(ValueError, match="training semantics"):
        ActorCatalog(
            catalog.model, {"one": actor, "two": replace(actor, ppo_n_minibatches=2)}
        )


def test_actor_component_and_simulator_charge_residency_once(catalog):
    """The three real actor stages feed train; lifecycle GPU residency is shared."""
    config = catalog.labels["S029"]
    expected = catalog.model.predict(catalog.actor_config(config), STATS)
    context = ComponentContext(config, WORK, HistorySnapshot())
    estimate = ActorTrainComponent(catalog).estimate(context)
    assert estimate.seconds == pytest.approx(sum(expected.values()))
    assert estimate.extra_gpu_seconds == 0
    simulator = WorkflowSimulator(lifecycle().workflow(catalog))
    result = simulator.evaluate(
        observation(
            training_active=False,
            training_config=None,
            has_checkpoint=False,
            next_batch=WORK,
            max_train_gpus=32,
        ),
        (ForecastUpdate(WORK, 0),),
        (PlanStep(config),),
        HistorySnapshot(),
    )
    assert result.seconds == pytest.approx(15 + estimate.seconds + 3 + 2 + 2)
    assert result.training_gpu_seconds == pytest.approx(32 * result.seconds)
    assert result.inference_gpu_seconds == pytest.approx(8 * result.seconds)
    assert result.samples == 32 and result.samples != STATS.n_seqs
    assert result.effective_tokens == 65536
    assert all(profile == config.profile_id for _, profile, _ in result.profile_events)


def test_missing_actor_stats_never_substitute_sample_count_or_cold_fallback(catalog):
    """Legacy summaries cannot be mistaken for actual actor sequence counts."""
    policy = build_actor_policy(catalog, lifecycle(), horizon=1)
    decision = policy.decide(
        observation(
            training_active=False,
            training_config=None,
            max_train_gpus=32,
            has_checkpoint=False,
        )
    )
    assert decision.action == "keep"
    assert policy.last_search.prediction is None
    assert not policy.last_search.fallback_allowed
    assert any("n_seqs" in reason for reason in policy.last_search.rejected)
    with pytest.raises(ValueError, match="effective_tokens"):
        actor_workload(WorkloadStats(128, 262144), samples=32)


def test_actor_static_memory_lower_bound_rejects_impossible_candidate(catalog):
    """A single GPU cannot hold the full Qwen Adam training state."""
    small = ActorCatalog(
        catalog.model, {"one": ActorConfig("archon", ParallelLayout())}
    )
    result = ActorTrainComponent(small).estimate(
        ComponentContext(small.configurations[0], WORK, HistorySnapshot())
    )
    assert not result.feasible and "memory" in result.reason


def test_restore_requires_explicit_directed_full_profile_compatibility(catalog):
    """Equal card counts and cross-backend checkpoints imply no compatibility."""
    source, target = catalog.labels["S031"], catalog.labels["S029"]
    context = ComponentContext(target, WORK, HistorySnapshot(), source)
    component = CompatibleRestore(catalog, ConstantComponent(12))
    assert not component.estimate(context).feasible
    assert not component.estimate(replace(context, source_config=None)).feasible
    assert component.estimate(replace(context, source_config=target)).seconds == 12
    declared = replace(component, allowed_transitions=frozenset({(source, target)}))
    assert declared.estimate(context).feasible
    assert not declared.estimate(
        replace(context, config=source, source_config=target)
    ).feasible


def test_measurements_do_not_leak_between_same_card_profiles(catalog):
    """Both recorder and estimator carry complete profile identity."""
    source, target = catalog.configurations[:2]
    policy = build_actor_policy(catalog, lifecycle())
    policy.record_phase("restore", source, 7)
    history = policy.history.snapshot()
    component = MeasuredComponent("restore")
    context = ComponentContext(source, WORK, history)
    assert component.estimate(context).seconds == 7
    assert not component.estimate(replace(context, config=target)).feasible
    assert history.measurements[0].profile_id == source.profile_id


def test_forecast_preserves_observed_actor_batch_and_is_causal():
    """No averaging of sequence geometry and no unseen future workload access."""
    history = HistorySnapshot((WORK, BatchWorkload(4, 10, 20, 60)))
    obs = observation(
        batch_ready=False, next_batch=None, episode_active=True, episode_elapsed=20
    )
    forecast = forecast_window(obs, history, 3, preserve_workload=True)
    assert [step.wait_seconds for step in forecast] == [70, 90, 90]
    assert all(step.workload.actor_stats == STATS for step in forecast)
    assert all(step.workload.samples == 32 for step in forecast)
    ready = replace(obs, next_batch=WORK, batch_ready=True)
    assert (
        forecast_window(ready, HistorySnapshot(), 2, preserve_workload=True)[
            0
        ].wait_seconds
        == 0
    )
    restored = observation_from_dict(
        json.loads(
            json.dumps(
                asdict(replace(ready, checkpoint_config=TrainingConfig(32, "full")))
            )
        )
    )
    assert restored == replace(ready, checkpoint_config=TrainingConfig(32, "full"))


@pytest.mark.parametrize("scope", ["total", "training"])
@pytest.mark.parametrize("deadline", [None, 37, 70])
def test_lifecycle_dynamic_program_matches_exhaustive_fixed_horizon(scope, deadline):
    """A full plan oracle verifies DP pruning, terminal cost and time feasibility."""
    configs = (
        TrainingConfig(1, "slow"),
        TrainingConfig(2, "fast"),
        TrainingConfig(2, "other"),
    )
    measurements = tuple(
        Measurement("train", config.world_size, seconds, 100, config.profile_id)
        for config, seconds in zip(configs, (12, 5, 7), strict=True)
    )
    simulator = WorkflowSimulator(
        replace(
            model(train=1, restore=4), train=MeasuredComponent("train", measurements)
        )
    )
    history = HistorySnapshot((BatchWorkload(2, 50, 100, 4),))
    obs = observation(training_config=configs[0], max_train_gpus=2)
    optimizer = LifecycleOptimizer(
        configs, horizon=3, switch_margin=0, cost_scope=scope, max_seconds=deadline
    )
    forecast = forecast_window(obs, history, 3, preserve_workload=True)
    feasible = []
    options = tuple(
        PlanStep(config, release) for config in configs for release in (False, True)
    )
    for plan in itertools.product(options, repeat=3):
        result = simulator.evaluate(obs, forecast, plan, history)
        if result.feasible and (deadline is None or result.seconds <= deadline):
            cost = (
                result.total_gpu_seconds
                if scope == "total"
                else result.training_gpu_seconds
            )
            feasible.append((cost, result.seconds))
    search = optimizer.optimize(obs, simulator, history)
    if not feasible:
        assert search.prediction is None and search.decision.action == "keep"
    else:
        expected = min(feasible)
        prediction = search.prediction
        assert prediction is not None
        cost = (
            prediction.total_gpu_seconds
            if scope == "total"
            else prediction.training_gpu_seconds
        )
        assert (cost, prediction.seconds) == pytest.approx(expected)
        replay = simulator.evaluate(obs, forecast, search.plan, history)
        assert replay == prediction
        assert prediction.samples == 6 and prediction.effective_tokens == 150
    assert search.evaluated <= optimizer.max_evaluations


def test_named_profiles_enable_same_world_size_configuration_decision():
    """Two 32-GPU profiles must still produce a reconfigure action."""
    slow, fast = TrainingConfig(32, "slow"), TrainingConfig(32, "fast")
    measurements = (
        Measurement("train", 32, 100, profile_id="slow"),
        Measurement("train", 32, 1, profile_id="fast"),
    )
    simulator = WorkflowSimulator(
        replace(model(restore=1), train=MeasuredComponent("train", measurements))
    )
    search = LifecycleOptimizer((slow, fast), horizon=1).optimize(
        observation(training_config=slow, max_train_gpus=32),
        simulator,
        HistorySnapshot(),
    )
    assert search.decision.action == "reconfigure" and search.decision.config == fast


@pytest.mark.parametrize("margin,expected", [(0, "reconfigure"), (0.05, "keep")])
def test_lifecycle_hysteresis_requires_enough_first_action_benefit(margin, expected):
    """A 1% forecast improvement is not sufficient for a 5% switching threshold."""
    current, target = TrainingConfig(1, "current"), TrainingConfig(1, "target")
    measured = MeasuredComponent(
        "train",
        (
            Measurement("train", 1, 100, profile_id="current"),
            Measurement("train", 1, 99, profile_id="target"),
        ),
    )
    simulator = WorkflowSimulator(
        replace(model(restore=0), train=measured, release=ConstantComponent(0))
    )
    search = LifecycleOptimizer(
        (current, target), horizon=1, switch_margin=margin
    ).optimize(observation(training_config=current), simulator, HistorySnapshot())
    assert search.decision.action == expected


def test_deadline_preserves_faster_prefix_even_when_it_costs_more():
    """A cheap slow prefix must not eliminate a fast prefix needed by a later deadline."""
    slow, fast, final = (
        TrainingConfig(1, "slow"),
        TrainingConfig(4, "fast"),
        TrainingConfig(1, "final"),
    )

    class Timings:
        def estimate(self, context):
            from areal.elastic.policy.basic.simulator import Estimate

            duration = {"slow": 8, "fast": 3, "final": 5}[context.config.profile_id]
            # Last update can only use the terminal profile; first two may use all.
            if context.workload.samples == 3 and context.config != final:
                return Estimate(feasible=False, reason="Terminal profile required")
            return Estimate(duration)

    # Exercise the DP's Pareto pruning using three different known workloads.
    import areal.elastic.policy.basic.optimizer as optimizer_module

    forecast = tuple(
        ForecastUpdate(BatchWorkload(samples, 50, 100), 0) for samples in (1, 2, 3)
    )
    # Controlled forecast isolates deadline pruning; causal forecasting is tested separately.
    from unittest.mock import patch

    simulator = WorkflowSimulator(
        replace(
            model(),
            train=Timings(),
            start=ConstantComponent(),
            restore=ConstantComponent(),
            checkpoint=ConstantComponent(),
            publish=ConstantComponent(),
            release=ConstantComponent(),
            inference_gpus=0,
        )
    )
    obs = observation(
        training_active=False,
        has_checkpoint=False,
        training_config=None,
        max_train_gpus=4,
    )
    optimizer = LifecycleOptimizer(
        (slow, fast, final),
        horizon=3,
        cost_scope="training",
        max_seconds=12,
        switch_margin=0,
    )
    with patch.object(optimizer_module, "forecast_window", return_value=forecast):
        search = optimizer.optimize(obs, simulator, HistorySnapshot())
    oracle = []
    for targets in itertools.product((slow, fast, final), repeat=3):
        prediction = simulator.evaluate(
            obs,
            forecast,
            tuple(PlanStep(config) for config in targets),
            HistorySnapshot(),
        )
        if prediction.feasible and prediction.seconds <= 12:
            oracle.append((prediction.training_gpu_seconds, prediction.seconds))
    assert search.prediction is not None
    assert (search.prediction.training_gpu_seconds, search.prediction.seconds) == min(
        oracle
    )
    assert search.plan[0].config == fast and search.prediction.seconds == 11


@pytest.mark.parametrize("wait,expected", [(120, "release"), (1, "keep")])
def test_integrated_policy_releases_long_gap_and_keeps_short_gap(
    catalog, wait, expected
):
    """The real actor model participates in resource lifecycle decisions."""
    policy = build_actor_policy(catalog, lifecycle(), horizon=2)
    policy.record_workload(replace(WORK, rollout_seconds=wait))
    decision = policy.decide(
        observation(
            training_config=catalog.labels["S029"],
            next_batch=None,
            batch_ready=False,
            episode_active=True,
            max_train_gpus=32,
        )
    )
    assert decision.action == expected
    assert policy.last_search.prediction.samples == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"pending_publication": True},
        {"committed": False},
        {"gate_available": False},
        {"next_batch": None},
    ],
)
def test_guarded_lifecycle_search_consumes_no_work(changes, catalog):
    """Uncommitted, unpublished and inadmissible work cannot trigger a switch."""
    policy = build_actor_policy(catalog, lifecycle())
    obs = observation(
        training_config=catalog.labels["S029"], next_batch=WORK, max_train_gpus=32
    )
    assert policy.decide(replace(obs, **changes)).action == "keep"
    assert policy.last_search.evaluated == 0


def test_bounded_search_retains_all_profiles_and_shortens_lookahead(catalog):
    """A small search budget reduces depth instead of dropping layouts."""
    policy = build_actor_policy(catalog, lifecycle(), horizon=10, max_evaluations=8)
    decision = policy.decide(
        observation(
            training_active=False,
            training_config=None,
            has_checkpoint=False,
            next_batch=WORK,
            max_train_gpus=32,
        )
    )
    assert decision.action == "launch"
    assert policy.last_search.evaluated == 8 and len(policy.last_search.plan) == 1
    with pytest.raises(ValueError, match="every lifecycle candidate"):
        build_actor_policy(catalog, lifecycle(), max_evaluations=7).decide(
            observation(next_batch=WORK, max_train_gpus=32)
        )


# The joint model has an independent inference input and no Expert dependency.
def joint_config(actor_gpus=1, rollout_gpus=1, *, label="", inflight=2):
    return JointConfig(
        TrainingConfig(actor_gpus, label or f"actor-{actor_gpus}"),
        RolloutConfig(
            "test-model",
            "test-gpu",
            "test-runtime",
            "fixed-sampling",
            1,
            rollout_gpus,
            8,
            1024,
            4096,
            inflight,
        ),
    )


def joint_scenario(config, *, count=2, external=0, staleness=1, **kwargs):
    stats = WorkloadStats(2, 20, loss_tokens=10, sequence_lengths=(10, 10))
    batch = JointBatch(
        actor_workload(stats, samples=2, rollout_seconds=9999),
        RolloutWorkload((RolloutStage(0, 100, external, parallel_requests=2),), 2, 100),
        1,
    )
    return JointScenario((batch,) * count, config, 2, staleness, 64, 64, 128, **kwargs)


class JointActorTimes:
    """Explicit timing oracle isolates event scheduling from actor formulas."""

    def __init__(self, times):
        self.times = times
        self.inputs = []

    def actor_config(self, config):
        if config not in self.times:
            raise ValueError("Unknown full actor profile")
        return config

    def predict(self, config, stats):
        self.inputs.append((config, stats))
        return ActorPrediction((("train_step", self.times[config]),), False, ())


def joint_simulator(
    configs,
    *,
    actor_times=None,
    rates=None,
    publish=0,
    checkpoint=0,
    start=0,
    switch=0,
    shutdown=0,
):
    configs = tuple(dict.fromkeys(configs))
    actors = JointActorTimes(actor_times or {c.actor: 1 for c in configs})
    rollouts = tuple(dict.fromkeys(c.rollout for c in configs))
    profiles = tuple(
        RolloutProfile(
            r,
            10,
            (rates or {}).get(r, 10),
            1,
            100,
            1,
            10000,
            4096,
            "synthetic-rates",
        )
        for r in rollouts
    )
    transitions = []
    for c in configs:
        for action, src, dst, seconds in (
            ("publish", c, c, publish),
            ("checkpoint", c, c, checkpoint),
            ("start", None, c, start),
            ("shutdown", c, None, shutdown),
        ):
            transitions.append(
                TransitionProfile(
                    action,
                    src,
                    dst,
                    (
                        TransitionPhase(
                            seconds, c.actor.world_size, c.rollout.world_size
                        ),
                    ),
                    "synthetic-transition",
                )
            )
        for src in configs:
            if src != c:
                transitions.append(
                    TransitionProfile(
                        "reconfigure",
                        src,
                        c,
                        (
                            TransitionPhase(
                                switch, c.actor.world_size, c.rollout.world_size
                            ),
                        ),
                        "synthetic-switch",
                    )
                )
    return JointSimulator(
        actors, RolloutCostModel(profiles), JointTransitionModel(tuple(transitions))
    )


def event_times(result, kind):
    return [e.time for e in result.events if e.kind == kind]


def test_joint_shared_inference_and_external_wait_overlap():
    """External waits release inference capacity; token work still shares one lane."""
    c = joint_config()
    sim = joint_simulator((c,))
    result = sim.evaluate(joint_scenario(c, external=100), c)
    assert result.status == "complete"
    assert event_times(result, "rollout_service_start") == [0, 10]
    assert event_times(result, "batch_ready") == [110, 120]
    assert result.seconds == 121
    assert result.actor_gpu_seconds == result.rollout_gpu_seconds == 121
    assert (
        result.completed_batches,
        result.samples,
        result.effective_tokens,
        result.sequence_tokens,
    ) == (2, 4, 20, 40)
    assert "synthetic-rates" in result.source_ids
    assert any("assumed" in note for note in result.assumptions)


def test_joint_staleness_waits_for_publication_not_dequeue():
    """A consumed batch never subtracts from the cumulative accepted counter."""
    c = joint_config()
    sim = joint_simulator((c,), publish=5)
    result = sim.evaluate(joint_scenario(c, staleness=0), c)
    assert event_times(result, "rollout_admitted") == [0, 16]
    assert event_times(result, "provider_version") == [16, 32]
    starts = [e for e in result.events if e.kind == "actor_start"]
    assert [e.accepted_batches for e in starts] == [1, 2]
    assert result.seconds == 32


def test_joint_publish_waits_for_gpu_but_external_work_continues():
    """A publication barrier pauses shared inference admission and service."""
    c = joint_config()
    sim = joint_simulator((c,), publish=5)
    result = sim.evaluate(joint_scenario(c), c)
    assert event_times(result, "actor_end") == [11, 26]
    assert event_times(result, "publish_start") == [20, 26]
    assert result.seconds == 31
    assert result.queue_wait_seconds == 5


def test_joint_multistage_fifo_resumes_after_environment_wait():
    """Turn stages rejoin the shared queue instead of reserving a GPU during wait."""
    c = joint_config()
    scenario = joint_scenario(c)
    rollout = RolloutWorkload((RolloutStage(0, 10, 10), RolloutStage(0, 10)), 2, 10)
    scenario = replace(
        scenario, batches=tuple(replace(b, rollout=rollout) for b in scenario.batches)
    )
    result = joint_simulator((c,)).evaluate(scenario, c)
    assert event_times(result, "rollout_service_start") == [0, 1, 11, 12]
    assert event_times(result, "batch_ready") == [12, 13]
    assert result.seconds == 14


def test_joint_checkpoint_frequency_uses_absolute_committed_versions():
    """Checkpoint cost is periodic, including at the final matching version."""
    c = joint_config()
    scenario = joint_scenario(
        c,
        count=3,
        staleness=0,
        provider_version=1,
        accepted_batches=1,
        checkpoint_every=2,
    )
    result = joint_simulator((c,), checkpoint=7).evaluate(scenario, c)
    assert event_times(result, "checkpoint_start") == [11, 40]
    assert result.seconds == 47
    assert result.total_gpu_seconds == 94


def test_joint_transition_phases_charge_unique_residency_and_cleanup():
    """Startup/shutdown allocations replace steady residency instead of adding it."""
    c = joint_config(2, 2)
    sim = joint_simulator((c,))
    profiles = tuple(
        p for p in sim.transitions.profiles if p.action not in {"start", "shutdown"}
    ) + (
        TransitionProfile(
            "start",
            None,
            c,
            (TransitionPhase(2, 1, 1), TransitionPhase(3, 2, 2)),
            "start",
        ),
        TransitionProfile("shutdown", c, None, (TransitionPhase(5, 2, 1),), "shutdown"),
    )
    sim.transitions = JointTransitionModel(profiles)
    result = sim.evaluate(replace(joint_scenario(c, count=1), current=None), c)
    assert result.seconds == 21
    assert result.actor_gpu_seconds == 40
    assert result.rollout_gpu_seconds == 35
    assert result.total_gpu_seconds == 75


@pytest.mark.parametrize(
    "action", ["start", "publish", "checkpoint", "shutdown", "reconfigure"]
)
def test_joint_missing_transition_is_unsupported_not_free(action):
    """Every directed lifecycle component must exist, including explicit zero costs."""
    old, c = joint_config(label="old"), joint_config()
    sim = joint_simulator((old, c))
    sim.transitions = JointTransitionModel(
        tuple(p for p in sim.transitions.profiles if p.action != action)
    )
    scenario = joint_scenario(c, count=1, checkpoint_every=1)
    scenario = replace(
        scenario,
        current=None if action == "start" else old if action == "reconfigure" else c,
    )
    result = sim.evaluate(scenario, c)
    assert result.status == "unsupported"
    assert action in result.reason


def test_joint_rejects_temporary_allocation_above_pool_limit():
    """A feasible target cannot hide an infeasible transition peak."""
    old, c = joint_config(label="old"), joint_config()
    sim = joint_simulator((old, c))
    sim.transitions = JointTransitionModel(
        tuple(
            replace(p, phases=(TransitionPhase(1, 2, 1),))
            if p.action == "reconfigure"
            else p
            for p in sim.transitions.profiles
        )
    )
    scenario = replace(joint_scenario(old), max_actor_gpus=1)
    assert sim.evaluate(scenario, c).status == "infeasible"


def test_joint_search_changes_both_sides_and_reports_complete_candidates():
    """Different actor and inference bottlenecks favor different joint layouts."""
    configs = tuple(joint_config(a, r) for a, r in itertools.product((1, 2), repeat=2))
    scenario = joint_scenario(configs[0], count=1)
    for actor_time, rates, expected in (
        (
            {configs[0].actor: 100, configs[2].actor: 1},
            {configs[1].rollout: 10},
            configs[2],
        ),
        (
            {configs[0].actor: 1, configs[2].actor: 1},
            {configs[0].rollout: 1, configs[1].rollout: 100},
            configs[1],
        ),
        (
            {configs[0].actor: 100, configs[2].actor: 1},
            {configs[0].rollout: 1, configs[1].rollout: 100},
            configs[3],
        ),
    ):
        policy = JointPolicy(
            joint_simulator(configs, actor_times=actor_time, rates=rates),
            JointOptimizer(configs),
        )
        result = policy.plan(scenario)
        assert result is policy.last_search
        assert result.config == expected
        assert result.exhaustive and len(result.candidates) == 4
        assert {item.prediction.samples for item in result.candidates} == {2}
        assert all(item.eligible for item in result.candidates)
        assert result.elapsed_seconds >= 0


def test_joint_external_wait_does_not_scale_with_gpu_count():
    """When environment latency dominates, extra inference GPUs can cost more."""
    old = joint_config()
    c = replace(old, rollout=replace(old.rollout, replicas=2))
    sim = joint_simulator((old, c))
    scenario = joint_scenario(old, count=1, external=100)
    first, second = (sim.evaluate(scenario, x) for x in (old, c))
    assert first.seconds == 111 and second.seconds == 106
    assert JointOptimizer((c,)).optimize(scenario, sim).config == old


def test_joint_single_request_cannot_use_extra_replicas():
    """Autoregressive request latency does not divide by unused replicas."""
    c = joint_config()
    extra = replace(c, rollout=replace(c.rollout, replicas=8))
    sim = joint_simulator((c, extra))
    # Even if the batch contains several sequential requests, only explicit
    # within-stage parallelism licenses replica speedup.
    for requests in (1, 8):
        workload = RolloutWorkload((RolloutStage(0, 100),), requests, 100)
        first = sim.rollout.predict(c.rollout, workload)
        second = sim.rollout.predict(extra.rollout, workload)
        assert first.stages == second.stages == ((10, 0),)
    parallel = RolloutWorkload((RolloutStage(0, 100, parallel_requests=2),), 8, 100)
    assert sim.rollout.predict(extra.rollout, parallel).stages == ((5, 0),)
    with pytest.raises(ValueError, match="Stage parallel requests"):
        RolloutWorkload((RolloutStage(0, 100, parallel_requests=2),), 1, 100)


def test_joint_budget_keeps_current_first_without_shortening_window():
    """Budget truncates candidate enumeration, not the amount of compared work."""
    old, c = joint_config(), joint_config(2, 2)
    result = JointOptimizer((c,), max_evaluations=1).optimize(
        joint_scenario(old, count=4), joint_simulator((old, c))
    )
    assert result.config == old
    assert not result.exhaustive and result.unevaluated == 1
    assert result.prediction.completed_batches == 4
    assert len(result.candidates) == 1


def test_joint_switch_cost_and_margin_can_keep_current():
    """Both transition overhead and minimum improvement apply to total cost."""
    old, c = joint_config(label="old"), joint_config()
    scenario = joint_scenario(old, count=1)
    sim = joint_simulator((old, c), actor_times={old.actor: 10, c.actor: 9})
    assert JointOptimizer((c,), switch_margin=0).optimize(scenario, sim).config == c
    assert JointOptimizer((c,), switch_margin=0.1).optimize(scenario, sim).config == old
    sim = joint_simulator((old, c), actor_times={old.actor: 10, c.actor: 1}, switch=10)
    assert JointOptimizer((c,)).optimize(scenario, sim).config == old


def test_joint_time_bound_includes_terminal_cleanup():
    """A candidate completing training in time can still violate its total deadline."""
    c = joint_config()
    sim = joint_simulator((c,), shutdown=10)
    result = JointOptimizer((c,), max_seconds=15).optimize(
        joint_scenario(c, count=1), sim
    )
    assert result.config is None and result.prediction is None
    assert result.candidates[0].prediction.seconds == 21
    assert "time bound" in result.candidates[0].reason


@pytest.mark.parametrize("change", ["requests", "tokens", "kv", "identity"])
def test_joint_rollout_profile_rejects_unknown_or_uncovered_work(change):
    """No rate extrapolation across profile identity, capacity or workload scope."""
    c = joint_config()
    sim = joint_simulator((c,))
    profile = sim.rollout.profiles[0]
    if change == "requests":
        profile = replace(profile, min_requests=3)
    elif change == "tokens":
        profile = replace(profile, max_tokens=50)
    elif change == "kv":
        c = replace(c, rollout=replace(c.rollout, kv_capacity_tokens=50))
    else:
        c = replace(c, rollout=replace(c.rollout, max_batch_tokens=2048))
    sim.rollout = RolloutCostModel((profile,))
    result = sim.evaluate(joint_scenario(c, count=1), c)
    assert result.status == "unsupported"


def test_joint_optimizer_rejects_fewer_work_items_from_revised_model():
    """A model that claims completion of a smaller batch cannot win search."""
    c = joint_config()
    scenario = joint_scenario(c)
    predicted = joint_simulator((c,)).evaluate(scenario, c)
    bad = SimpleNamespace(evaluate=lambda *_: replace(predicted, samples=2))
    result = JointOptimizer((c,)).optimize(scenario, bad)
    assert result.config is None
    assert "identical workload" in result.candidates[0].reason


def test_joint_same_card_profiles_keep_separate_identity():
    """Different runtime or batch limits cannot reuse an exact inference profile."""
    c = joint_config()
    other = replace(c, rollout=replace(c.rollout, runtime_id="another-runtime"))
    assert c.world_size == other.world_size and c != other
    assert c.rollout.profile_id != other.rollout.profile_id
    with pytest.raises(ValueError, match="sampling semantics"):
        JointOptimizer(
            (c, replace(c, rollout=replace(c.rollout, sampling_id="changed")))
        )


def test_joint_freezes_surrogate_and_reuses_rank_stats_only_at_source():
    """Candidate token capacity cannot alter work; source packing never leaks."""
    source, target = TrainingConfig(2, "source"), TrainingConfig(2, "target")
    stats = WorkloadStats(4, 100, loss_tokens=50, train_microbatches=3)
    batch = JointBatch(actor_workload(stats, samples=2), None, 1, source_actor=source)
    assert batch.inferred_lengths
    copied = replace(batch, groups=1)
    assert copied.inferred_lengths
    assert copied.stats_for(target) == batch.stats_for(target)
    assert (
        batch.stats_for(source).sequence_lengths
        == batch.stats_for(target).sequence_lengths
    )
    assert batch.stats_for(source).train_microbatches == 3
    assert batch.stats_for(target).train_microbatches is None
    ranked = WorkloadStats(
        4, 100, loss_tokens=50, per_dp_sequence_lengths=((10, 20), (30, 40))
    )
    with pytest.raises(ValueError, match="original trajectory order"):
        JointBatch(actor_workload(ranked, samples=2), None, 1, source_actor=source)
    logical = replace(
        ranked,
        per_dp_sequence_lengths=None,
        sequence_lengths=(30, 10, 40, 20),
        trajectory_group_sizes=(2, 2),
    )
    batch = JointBatch(
        actor_workload(ranked, samples=2),
        None,
        1,
        source_actor=source,
        logical_stats=logical,
    )
    assert (
        batch.stats_for(source).per_dp_sequence_lengths
        == ranked.per_dp_sequence_lengths
    )
    assert batch.stats_for(target).per_dp_sequence_lengths is None
    assert batch.stats_for(target).sequence_lengths == (30, 10, 40, 20)
    with pytest.raises(ValueError, match="observed rank sequence lengths"):
        JointBatch(
            actor_workload(ranked, samples=2),
            None,
            1,
            source_actor=source,
            logical_stats=replace(logical, sequence_lengths=(25, 25, 25, 25)),
        )


def test_joint_ready_work_uses_explicit_cost_boundary_and_real_actor_catalog(catalog):
    """Already-ready batches need no invented rollout cost or duration."""
    c = replace(joint_config(), actor=catalog.configurations[0])
    batch = JointBatch(WORK, None, 4)
    scenario = JointScenario(
        (batch,), c, 8, 2, 32, 1, 33, ready_batches=1, accepted_batches=1
    )
    sim = joint_simulator((c,))
    sim.actors, sim.rollout = catalog, RolloutCostModel(())
    result = sim.evaluate(scenario, c)
    assert result.status == "complete"
    expected = catalog.predict(c.actor, batch.stats_for(c.actor)).seconds
    assert result.seconds == pytest.approx(expected)
    assert result.samples == 32 and result.sequence_tokens == 262144
    assert not event_times(result, "rollout_admitted")
    assert any("surrogate" in note for note in result.assumptions)


def test_joint_initial_state_and_work_contracts_reject_inconsistent_inputs():
    """A drained committed snapshot uses counters with a common batch origin."""
    c = joint_config()
    scenario = joint_scenario(c)
    with pytest.raises(ValueError, match="inconsistent"):
        replace(scenario, provider_version=2)
    with pytest.raises(ValueError, match="samples/group"):
        replace(scenario, samples_per_group=3)
    with pytest.raises(ValueError, match="Future batches"):
        replace(scenario, batches=(replace(scenario.batches[0], rollout=None),))
    with pytest.raises(ValueError, match="source_actor"):
        JointBatch(
            actor_workload(replace(STATS, train_microbatches=3), samples=2), None, 1
        )


def test_history_joint_adapter_accounts_for_fixed_work_and_measured_cleanup():
    """A measured joint window must include the simulator's shutdown boundary."""
    config = joint_config()
    simulator = joint_simulator((config,), publish=2, shutdown=3)
    inputs = HistoricalInput(
        "joint",
        config,
        joint_scenario(config, external=100),
        "synthetic",
        JOINT_BOUNDARY,
        "given_external",
        True,
        True,
    )
    predictor = BasicHistoryPredictor("synthetic", joint=simulator)
    values = dict(predictor.predict(inputs).values)
    assert values["seconds"] == 126
    assert values["total_gpu_seconds"] == 252
    assert values["actor_gpu_seconds"] == values["rollout_gpu_seconds"] == 126
    assert not predictor.predict(replace(inputs, execution_id="other-runtime")).values
    assert not predictor.predict(replace(inputs, shutdown_included=None)).values
    too_small = replace(inputs, workload=replace(inputs.workload, max_total_gpus=1))
    assert not predictor.predict(too_small).values


@pytest.mark.parametrize(
    "changed",
    [
        {"initial_state_verified": None},
        {"shutdown_included": False},
        {"timing_boundary": "continuous_window_without_cleanup"},
    ],
)
def test_history_evaluator_owns_joint_trace_applicability(changed):
    """Revised predictors cannot bypass missing initial state or timing boundaries."""
    config = joint_config()
    inputs = HistoricalInput(
        "joint",
        config,
        joint_scenario(config),
        "synthetic",
        JOINT_BOUNDARY,
        "given_external",
        True,
        True,
    )
    case = HistoricalCase(
        "joint-window",
        HistorySpan("run", 1, 2),
        "validation",
        replace(inputs, **changed),
        (("seconds", 21),),
        "synthetic",
    )
    evaluator = HistoricalModelEvaluator((case,), primary_metrics=("seconds",))

    class MustNotRun:
        def predict(self, inputs):
            raise AssertionError("An unaudited trace reached the predictor")

    report = evaluator.evaluate(MustNotRun(), "v0", fitted_on=())
    assert report.valid and report.metrics[0].covered == 0
    assert "verified committed/drained" in report.rows[0].reason


def test_history_rollout_adapter_and_acceptance_do_not_hide_error_with_known_wait():
    """Long given waits cannot serve as the acceptance target or dilute service error."""
    config = joint_config()
    simulator = joint_simulator((config,))
    workload = joint_scenario(config, external=1000).batches[0].rollout
    inputs = HistoricalInput(
        "rollout",
        config.rollout,
        workload,
        "synthetic",
        ROLLOUT_BOUNDARY,
        "given_external",
    )
    predictor = BasicHistoryPredictor("synthetic", rollout=simulator.rollout)
    assert dict(predictor.predict(inputs).values) == {
        "inference_seconds": 10,
        "external_seconds": 1000,
        "rollout_seconds": 1010,
    }
    assert not predictor.predict(
        replace(inputs, timing_boundary="timeperf/rollout")
    ).values
    assert not predictor.predict(
        replace(inputs, config=replace(inputs.config, tp=2))
    ).values
    case = HistoricalCase(
        "rollout",
        HistorySpan("run", 1, 1),
        "validation",
        inputs,
        (
            ("inference_seconds", 10),
            ("external_seconds", 1000),
            ("rollout_seconds", 1010),
        ),
        "synthetic",
    )
    with pytest.raises(ValueError, match="Known external waits"):
        HistoricalModelEvaluator((case,), primary_metrics=("external_seconds",))
    with pytest.raises(ValueError, match="also requires inference"):
        HistoricalModelEvaluator((case,), primary_metrics=("rollout_seconds",))
    primary = ("inference_seconds", "rollout_seconds")
    with pytest.raises(ValueError, match="independent inference label"):
        HistoricalModelEvaluator(
            (replace(case, measured=(("rollout_seconds", 1010),)),),
            primary_metrics=primary,
        )
    evaluator = HistoricalModelEvaluator((case,), primary_metrics=primary)
    baseline = evaluator.evaluate(predictor, "v0", fitted_on=())

    class BadInference:
        def predict(self, inputs):
            return HistoricalPrediction(
                (
                    ("inference_seconds", 20),
                    ("external_seconds", 1000),
                    ("rollout_seconds", 1020),
                )
            )

    candidate = evaluator.evaluate(BadInference(), "bad", fitted_on=())
    assert candidate.score > 0.5  # Total error alone would be less than one percent.
    assert not evaluator.compare(candidate, baseline).accepted

    class MissingInference:
        def predict(self, inputs):
            return HistoricalPrediction((("rollout_seconds", 1000),))

    missing = evaluator.evaluate(MissingInference(), "missing", fitted_on=())
    assert not missing.valid
    assert all(m.covered == 0 for m in missing.metrics)
    assert "paired inference prediction" in missing.errors[0]
    assert not evaluator.compare(missing, baseline).accepted
    with pytest.raises(ValueError, match="input mode"):
        replace(inputs, input_mode="logical")
