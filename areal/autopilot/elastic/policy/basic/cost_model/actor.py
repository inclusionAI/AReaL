# SPDX-License-Identifier: Apache-2.0

"""Actor stage predictor composed from packing, operators, engines and scheduling."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from .components import (
    Advantages,
    Context,
    Efficiency,
    Operators,
    Optimizer,
    Schedule,
    Transformer,
)
from .config import ActorConfig, Hardware, ModelShape
from .packing import PackingPlan, plan_workload
from .workload import WorkloadStats

STAGES = ("recompute_logp", "compute_advantage", "train_step")


@dataclass(frozen=True)
class ActorComponents:
    """Replace a single component without replacing the predictor or its inputs."""

    pack: Callable[..., PackingPlan] = plan_workload
    transformer: Transformer = field(default_factory=Transformer)
    optimizer: Optimizer = field(default_factory=Optimizer)
    schedule: Schedule = field(default_factory=Schedule)
    advantages: Advantages = field(default_factory=Advantages)


def layer_partition(config: ActorConfig, model: ModelShape) -> tuple[int, ...]:
    if config.pipeline_layers is not None:
        if sum(config.pipeline_layers) != model.layers:
            raise ValueError("pipeline_layers must sum to model.layers")
        return config.pipeline_layers
    # Homogeneous layers with embedding/head parameter balancing. Minimax DP,
    # matching the objective of Megatron's parameter-balanced stage allocation.
    pp = config.parallel.pp
    if pp > model.layers:
        raise ValueError("Pipeline stages exceed model layers")
    # Match the backend's partition HEURISTIC (including its hidden^2 output
    # projection estimate), not the true compute/parameter count used below.
    layer_params = (
        model.hidden_size
        * model.head_dim
        * (model.attention_heads + 2 * model.kv_heads)
        + model.hidden_size**2
        + 2 * model.hidden_size
    )
    layer_params += (
        model.router_parameters_per_layer
        + model.expert_parameters_per_layer / config.parallel.ep
    )
    table = {(0, 0): (0.0, ())}
    for stage in range(pp):
        next_table = {}
        for (_, used), (cost, partition) in table.items():
            for count in range(1, model.layers - used - (pp - stage - 1) + 1):
                head = model.head_parameters * ((stage == 0) + (stage == pp - 1))
                candidate = (
                    max(cost, count * layer_params + head),
                    partition + (count,),
                )
                key = (stage + 1, used + count)
                if key not in next_table or candidate[0] < next_table[key][0]:
                    next_table[key] = candidate
        table = next_table
    return table[(pp, model.layers)][1]


@dataclass(frozen=True)
class ActorCostModel:
    model: ModelShape
    hardware: Hardware
    efficiency: Efficiency = field(default_factory=Efficiency)
    components: ActorComponents = field(default_factory=ActorComponents)
    group_gbps: dict[str, float] = field(default_factory=dict)
    # Public fit only rescales aggregate train_step. It does not identify phases.
    train_scale: float = 1.0
    calibration_note: str = "Analytical priors; no matching public fit applied"

    def __post_init__(self) -> None:
        if not math.isfinite(self.train_scale) or self.train_scale <= 0:
            raise ValueError("train_scale must be finite and positive")

    def _run(
        self, config: ActorConfig, workload: WorkloadStats, *, tail: bool = False
    ) -> dict:
        if self.model.experts % config.parallel.ep:
            raise ValueError("EP must divide model expert count")
        if self.model.attention_heads % config.parallel.tp:
            raise ValueError("TP must divide attention heads")
        plan = self.components.pack(config, workload, self.model.hidden_size, tail=tail)
        op = Operators(
            self.hardware,
            self.efficiency,
            self.group_gbps,
            stage_world_size=config.parallel.world_size // config.parallel.pp,
            cp_rank_span=config.parallel.tp * config.parallel.cp,
        )
        partition = layer_partition(config, self.model)
        contexts = [
            Context(config, self.model, op, count, i == 0, i == len(partition) - 1)
            for i, count in enumerate(partition)
        ]

        def schedule(ranks, training):
            services = [
                [
                    [
                        self.components.transformer.estimate(
                            replace(ctx, microbatch_index=index), mb, training=training
                        )
                        for ctx in contexts
                    ]
                    for index, mb in enumerate(rank)
                ]
                for rank in ranks
            ]
            return self.components.schedule.estimate(services)

        logp_detail = schedule(plan.logprob, False) if plan.logprob else {}
        train_detail = {
            key: 0.0
            for key in (
                "compute",
                "communication_exposed",
                "pipeline_bubble",
                "optimizer",
                "gradient_sync_exposed",
                "parameter_gather",
            )
        }
        for ranks in plan.train:
            detail = schedule(ranks, True)
            optimizer = [self.components.optimizer.estimate(ctx) for ctx in contexts]
            # One optimizer step per PPO minibatch. Megatron syncs at update;
            # Archon FSDP sync is inside each backward microbatch service.
            candidates = []
            for item in optimizer:
                sync = item["gradient_sync_raw"]
                if config.overlap_grad_reduce:
                    sync = max(sync - 0.5 * detail["compute"], 0)
                gather = item["parameter_gather"]
                candidates.append(
                    (item["optimizer"] + sync + gather, item["optimizer"], sync, gather)
                )
            _, opt, sync, gather = max(candidates)
            for key, value in detail.items():
                train_detail[key] += value
            train_detail["optimizer"] += opt
            train_detail["gradient_sync_exposed"] += sync
            train_detail["parameter_gather"] += gather
        stages = {
            "recompute_logp": sum(logp_detail.values()),
            "compute_advantage": self.components.advantages.estimate(
                config, workload.sequence_tokens, op
            ),
            "train_step": sum(train_detail.values()) * self.train_scale,
        }
        if any(not math.isfinite(v) or v < 0 for v in stages.values()):
            raise ValueError("Component returned invalid stage time")
        p = config.parallel
        static = []
        for ctx in contexts:
            dense, expert = ctx.owned_parameters
            sharded = dense / (p.dp * p.cp) + expert / p.expert_dp
            # A lower bound excluding activations, temporary gathers and fragmentation.
            if config.backend == "archon":
                # FP32 parameter/gradient plus two FP32 moments. Temporary BF16
                # allgathers are excluded from this lower bound.
                stored = sharded * 16
            elif config.distributed_optimizer:
                stored = (dense + expert) * 2 + sharded * 16
            else:
                stored = (dense + expert) * 18
            static.append(stored / 1e9)
        return {
            "stages_seconds": stages,
            "total_actor_seconds": sum(stages.values()),
            "analytical_details_seconds": {
                "recompute_logp": logp_detail,
                "train_step_before_aggregate_calibration": train_detail,
            },
            "engine_phases_before_calibration_seconds": {
                "forward_backward_schedule": sum(
                    train_detail[key]
                    for key in ("compute", "communication_exposed", "pipeline_bubble")
                ),
                "optimizer_step_and_sync": sum(
                    train_detail[key]
                    for key in (
                        "optimizer",
                        "gradient_sync_exposed",
                        "parameter_gather",
                    )
                ),
            },
            "packing": {
                "exact_lengths": plan.exact_lengths,
                "exact_dispatch": plan.exact_dispatch,
                "train_microbatches_per_rank": [
                    len(update[0]) for update in plan.train
                ],
                "logprob_microbatches_per_rank": len(plan.logprob[0])
                if plan.logprob
                else 0,
                "global_train_compute_token_slots": plan.train_compute_tokens,
                "global_sequence_tokens": workload.sequence_tokens,
                "global_loss_tokens": workload.loss_tokens,
            },
            "pipeline_layers": partition,
            "communication_placement": {
                "ranks_per_pp_stage": op.stage_world_size,
                "cp_rank_span": op.cp_rank_span,
                "assumption": "Contiguous ranks per node and backend default mesh; DP may stay inside one PP-stage node; override bandwidth for other placements",
            },
            "static_memory_lower_bound_gb_per_pipeline_stage": static,
            "static_memory_exceeds_capacity": max(static) > self.hardware.memory_gb,
            "assumptions": list(plan.assumptions),
        }

    def predict(
        self, config: ActorConfig, workload_stats: WorkloadStats
    ) -> dict[str, float]:
        """Small policy-facing API; seconds for the three AReaL actor phases."""
        return self._run(config, workload_stats)["stages_seconds"]

    def estimate(
        self,
        config: ActorConfig,
        workload_stats: WorkloadStats,
        *,
        sensitivity: bool = True,
    ) -> dict:
        """Auditable report with scenario sensitivity, not a statistical interval."""
        report = self._run(config, workload_stats)
        report["config"] = asdict(config)
        report["workload_stats"] = asdict(workload_stats)
        report["hardware"] = asdict(self.hardware)
        report["communication_placement"]["group_gbps_overrides"] = self.group_gbps
        report["efficiency_priors"] = asdict(self.efficiency)
        report["calibration"] = {
            "train_scale": self.train_scale,
            "note": self.calibration_note,
            "independent_stage_validation": False,
        }
        report["assumptions"] += [
            "Uniform expected MoE routing with explicit imbalance prior; no per-expert counters required",
            "Reference group placement unless group_gbps supplied; GPU model alone does not specify network",
            "Forward/backward and overlap are analytical; default AReaL traces do not identify each kernel",
            "Steady state only: excludes initialization, compile warmup, checkpoint, refit and rollout",
            "Static memory is a lower bound, not an OOM guarantee",
        ]
        if config.backend == "archon":
            report["assumptions"].append(
                "No public Archon actor timing labels: backend transfer is unvalidated"
            )
            report["assumptions"].append(
                "Current SequentialRunner FSDP gradient reduce-scatter runs per engine microbatch; parameter AG and RS share an overlap budget"
            )
            report["assumptions"].append(
                "Parameter gather cache modeled within a phase; cross-phase retained unsharded weights can reduce the first gather"
            )
        if config.parallel.pp > 1:
            report["assumptions"].append(
                "PP fill/drain uses a service-time envelope; no exact 1F1B launch trace assumed"
            )
        if config.gradient_checkpointing:
            report["assumptions"].append(config.checkpoint_note)
        report["assumptions"].append(
            "Expert dispatcher="
            + config.expert_dispatcher
            + "; source config, not inferred from GPU count"
        )
        if sensitivity:
            scenarios = {"nominal": report["stages_seconds"]}
            errors = {}
            variants = {
                "slower_effective_rates": replace(
                    self.efficiency,
                    dense=self.efficiency.dense * 0.75,
                    moe=self.efficiency.moe * 0.75,
                    attention=self.efficiency.attention * 0.75,
                    network=self.efficiency.network * 0.75,
                    fsdp_overlap=0,
                    ep_overlap=0,
                ),
                "faster_effective_rates": replace(
                    self.efficiency,
                    dense=min(self.efficiency.dense * 1.25, 1),
                    moe=min(self.efficiency.moe * 1.25, 1),
                    attention=min(self.efficiency.attention * 1.25, 1),
                    network=min(self.efficiency.network * 1.25, 1),
                ),
            }
            for name, efficiency in variants.items():
                scenarios[name] = replace(self, efficiency=efficiency)._run(
                    config, workload_stats
                )["stages_seconds"]
            if not report["packing"]["exact_lengths"]:
                try:
                    scenarios["bounded_tail_lengths"] = self._run(
                        config, workload_stats, tail=True
                    )["stages_seconds"]
                except ValueError as exc:
                    errors["bounded_tail_lengths"] = str(exc)
            report["sensitivity"] = {
                "kind": "Scenario envelope, NOT a confidence interval or guaranteed error bound",
                "scenarios_seconds": scenarios,
                "unavailable_scenarios": errors,
                "range_seconds": {
                    stage: [
                        min(v[stage] for v in scenarios.values()),
                        max(v[stage] for v in scenarios.values()),
                    ]
                    for stage in STAGES
                },
            }
        return report
