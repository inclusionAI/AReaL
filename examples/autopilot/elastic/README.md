# Elastic training with AReaL v2

Current research scope and company-side next steps: [Tau2 plan](../PLAN.md). The SWE
path below remains an infrastructure regression fixture.

A single-node prototype on AReaL v2.1.0: real [SWE episodes](swe/README.md) enter a
durable queue, wake FSDP workers, update the policy, and publish weights to a managed
SGLang service. Policies include idle timeout, fixed-Rollout Basic and serial joint
Actor/Rollout planning with background component revision.

[M2 implementation snapshot](milestones/m2/README.md) records the joint policy loop,
separate model/optimizer revisions, deployment, evidence and limits at `2ae0135c`.
[M1](milestones/m1/README.md) and its
[2026-10-18 rerun](milestones/m1/rerun-20261018.json) retain earlier SWE lifecycle
acceptance, measured timings and recovery evidence.

## R10 fixed Actor replay

`replay_r10.py` freezes an external Expert/R10 numerical archive and runs locally on
CPU. It preserves the original 16 batches/64 groups, absolute runner completion times,
relative controller overhead and final-save duration. Actor holds four GPUs throughout
the observed main-loop window; startup/teardown are excluded. Arrivals continue during
Actor execution. Historical batch waits and absolute Actor start times do not drive the
simulation.

```bash
python examples/autopilot/elastic/replay_r10.py \
  --reference-root /path/to/expert_designed --output /path/to/new-replay
```

The reference directory must contain `trials/r10` and its profiles; its source tree must
include the associated `autopilot/contracts.py`. No training dependencies or model
weights are loaded. `--reference-python` selects a Python/NumPy environment accepted by
the archive. The output directory must be new; it retains the frozen reference, per-file
hashes, numeric inputs and full event/interval report.

Measured timings provide an internal queue/clock/accounting check. The simulation uses
frozen Expert logp/train/publication predictions plus a separately identified structural
advantage prior. The formal comparison is measured execution versus simulation. Strict
mode preserves unknown steps and omits a full-window score. `--diagnostic-fallback`
enables assumed logp/train fallback; it never substitutes an unknown structural
advantage prediction. By default, steps 1–5 use measured timings to establish state;
steps 2–5 independently calibrate advantage and weight publication, then the frozen
models predict steps 6–16. Set both `--advantage-calibration none` and
`--publication-calibration none` for a full-window prior replay.
`--advantage-model legacy-affine --legacy-publication` reproduces the
[previous baseline](validation/r10-actor-replay-20261020.json).

### Structural advantage model

[`advantage.py`](../../../areal/autopilot/elastic/advantage.py) dispatches atomic task
groups to DP ranks, pads each rank to its local maximum length, then estimates:

```text
T = RPC + host serialization + trainer stats
    + max_rank(batching/preparation + token GAE + normalization + transfers + audit)
T_calibrated = scale × T
GAE iterations = local padded length - 1
GAE element work = local batch size × iterations
```

The eager recurrence has about 12 pointwise expressions per iteration, including masked
positions. Launch/iteration overhead and memory traffic are separate terms. This tensor
shape is independent of training microbatch packing and parameter offload. R10 executes
advantage on CUDA; group reward normalization has no all-reduce. Masked advantage
normalization with null mean/std still has tensor passes and a host synchronization.
Pure DP has no model-peer broadcast payload; RPC fanout and input/output transfers
remain.

[`r10-advantage-structural-v1.json`](profiles/r10-advantage-structural-v1.json)
separates recipe, operation counts and timing rates.
`--advantage-profile /path/to/profile.json` selects a revision; each run freezes and
hashes it with predictor source. Per-step output includes rank assignments, shapes,
iterations, bytes, component times and critical rank. Individual timing rates remain
**assumed**. Default calibration fits one whole-stage scale by least squares on steps
2–5: `scale = Σ(prior × measured) / Σprior²`. The four aggregate timings cannot identify
individual component rates. `advantage-calibration.json` freezes the selected labels,
profile hash, factor and observed width range. Prediction accepts no timing labels.
Component times are allocated by the same scale, not individually fitted. Width
extrapolation remains assumed; different execution/device/DP requires its own
calibration. Counts are source-derived approximations, not measured kernels. The
additive transport prior assumes full materialization without overlap/cache reuse. The
recipe assumes nonempty loss masks; compiled/turn GAE and other normalization modes
remain unsupported. DP1/2 responsiveness has unit checks; cross-config timing accuracy
remains unvalidated.

[Prefix calibration validation](validation/r10-advantage-prefix-calibration-20261020.json),
on the same eight covered development-validation steps:

| Phase             | Uncalibrated WAPE | Calibrated WAPE | Absolute error / measured three-phase total |
| ----------------- | ----------------- | --------------- | ------------------------------------------- |
| recompute_logp    | 1.11%             | 1.11%           | 0.0694%                                     |
| compute_advantage | 41.97%            | 5.64%           | 0.0974%                                     |
| train_step        | 5.57%             | 5.57%           | 5.1212%                                     |

All eleven later steps have an advantage estimate (WAPE 4.56%); step 9 extrapolates the
calibration width range. Strict logp/train steps 9/11/16 remain unsupported. Actor-total
WAPE on the common eight steps is 5.07%; phase errors can cancel. The
[uncalibrated record](validation/r10-structured-advantage-20261020.json) is retained.
These are development-validation results, not an independent test. Internal measured
replay still matches recorded 1809.492s within 0.029s per boundary. With diagnostic
logp/train fallback, the predicted suffix is 1235.088s versus measured replay 1230.908s.
Runner completion remains an availability proxy; controller residuals and saves are
conditional inputs. No config search, revision acceptance, staleness/admission feedback
or training-quality validation is included.

### Structural weight publication

The Expert
[`publication.py`](../../../areal/autopilot/elastic/policy/expert_designed/cost_model/publication.py)
models the complete steady-state `update_weights` boundary:

```text
pause → cleanup → max_rank(staging + gather + conversion + send/receive preparation)
      → network rounds → max_rank(receive writeback) → barrier → resume/version
```

The default [R10 profile](profiles/r10-publication-structural-v1.json) contains actual
metadata-derived traffic: 122.18 GB across 79,520 sends, including replica fanout. Each
Actor rank stages 15.27 GB in 5,043 logical tensors; 3,168 unique send copies move 10.07
GB of logical read/write traffic. Copies are deduplicated per version; network fanout is
retained. R10 receive slices need no temporary copy.

Staging shares a node's explicit host bandwidth; network rounds share sender/receiver
capacity and, across nodes, NIC capacity. All ranks contribute preparation, including
Megatron ranks that gather/convert without sending. Effective rates, launch costs and
control costs remain **unmeasured priors**. Bulk preparation and network bottleneck
rounds approximate scheduling; they do not reconstruct R10's manual NCCL operation
order. Prefix calibration replaces the whole duration and leaves component times raw.
The output records the raw model, fitted scale and separate `update_weights` error.
Connection/plan/process-group initialization is excluded from this steady-state cost.

R10 development validation (steps 6–16, 11/11 covered): raw prior 1.606s has WAPE
81.79%; prefix scale 5.3858 yields 8.651s and WAPE 3.45%, identical to the old constant
baseline. This validates the new execution path, not cross-config accuracy or
independently identified component rates.
[Record](validation/r10-publication-20261020.json).

Use `--publication-profile /path/to/profile.json` for explicit rank work and traffic.
Profiles bind the full Actor/Rollout recipe, model, runtime and hardware; a target needs
its own plan and rates. Missing counters, rates or receive coverage yield unknown.
FSDP/Megatron support dense separation AWEX. Archon uses explicit `xccl_broadcast`
bucket traffic plus per-bucket RPC acknowledgements; this is a logical star-broadcast
approximation and does not add an Archon v2 AWEX adapter. Delta, disk and colocate
publication remain unsupported. `PublicationEstimate.transition(source, target)` plugs
the result into the existing asynchronous simulator's publish/version events.

### Megatron, Archon and H200 targets

`--actor-target` selects an explicit Expert recipe, keeping the original logical batch
and group stream. The frozen Expert redispatches and repacks for the target; no Basic
model or source FSDP calibration is used. The bundled recipes select FP32 Adam without
CPU offload. ETP, interleaved/virtual pipeline and Archon expert TP without EP require
additional components and are rejected.

```bash
python examples/autopilot/elastic/replay_r10.py \
  --reference-root areal/autopilot/elastic/policy/expert_designed \
  --actor-target examples/autopilot/elastic/profiles/r10-archon-dp4-prior.json \
  --hardware h200-sxm-reference --system h200-8gpu-reference \
  --output /path/to/new-archon-h200-replay
```

Use `r10-megatron-dp4-prior.json` for Megatron. Omitting `--actor-target` keeps the
source FSDP recipe on the selected hardware, with fresh priors. Hardware and machine
topology are separate Catalog profiles. H200 SXM uses
[NVIDIA specifications](https://www.nvidia.com/en-us/data-center/h200/): 141 GB, 4800
GB/s HBM and 989.5 dense BF16 TFLOP/s (published sparse 1979 / 2). The 8-GPU topology,
bandwidth efficiencies, latencies and CPU/RPC rates remain scenario priors, not company
measurements. H200 NVL is not covered by this preset.

Counterfactual runs reject prefix calibration and FSDP diagnostic fallback. They report
per-step packing, memory lower bounds and assumed timings, with accuracy marked
`not_scored` until target measurements exist. Proven memory infeasibility blocks cost
scoring; a passing static lower bound still excludes activations and temporary buffers.
At DP4, Qwen3-30B's FP32 Adam lower bounds are 183.19 GB/GPU for Megatron and 122.13 GB
for Archon; the H20 reference has 102.63 GB and the H200 preset 141 GB.

Full-window cost requires a target-bound `--publication-profile` or explicit
`auxiliary.update_weights`, plus `auxiliary.save` and `auxiliary.final_save`. Auxiliary
durations each require `seconds` and a `source`; missing values remain unsupported.
Structural publication and an explicit publication duration are mutually exclusive.
Targets never fit publication using source timing labels. For TP/CP/PP, also provide
`advantage_broadcast.bytes_per_second` and `latency_seconds`. Each participating worker
executes the full DP-local GAE tensor; the model adds worker fanout and broadcast costs.

[Contract validation](validation/r10-backends-h200-20261020.json) covers both backends,
H200, source regression and explicit Actor8 scenarios. Targets currently cover one Actor
node. Actor8 plus the fixed four-GPU Rollout trace is a twelve-GPU resource
counterfactual, without pool-placement or training-quality validation. GPU·h measures
Actor allocation time; no cross-hardware price conversion is included. Conditional
controller residuals and fixed Rollout arrivals retain the original experiment limits.

## Policy and revision loops

![Policy loop and separate cost-model and optimizer revision loops](figures/elastic-loops-compact.svg)

`M` is the cost model and `O` the optimizer. Ready data → joint search → config
application → committed observations → replanning forms the policy loop. Model revision
uses historical measurements; optimizer revision uses the frozen effective model and
starts by revalidating the current optimizer. Independent acceptance records lead to one
atomic bundle activation. This serial joint path and the fixed-Rollout lifecycle path
have separate bundle scopes; both have CPU and H20 execution evidence. Current checks
use controlled inputs and proposals, so they establish execution rather than accuracy or
workflow benefits.

## Code map

```text
areal/autopilot/
├── contracts.py             # Shared CPU observations and lifecycle decisions
├── elastic/                 # Training execution loop
│   ├── loop.py              # Decide → execute → checkpoint → publish
│   ├── policy/
│   │   ├── basic/           # online.py serial joint path; offline/lifecycle models
│   │   └── expert_designed/ # Independent frozen cost model, simulator and R10 archive
│   └── runtime/
│       ├── config.py        # Durable requests, actual-config receipts and recovery
│       ├── policy.py        # Accepted component bundles and atomic policy activation
│       ├── serial_policy.py # Joint executable bundles and runtime search guards
│       ├── history.py       # Committed updates, actual inputs and measured phases
│       ├── rollout.py       # Replaceable inference generations and verified weights
│       ├── processes.py     # Owned processes and GPU occupancy
│       ├── training/
│       │   ├── trainer.py    # Consume → train → commit → acknowledge
│       │   ├── workers.py    # Start, restore and release workers; v2 integration
│       │   └── service.py    # Persistent CPU service and suspended HTTP responses
│       └── state/
│           ├── batch_queue.py        # Durable input and consumption cursor
│           └── checkpoint.py         # Atomic commit, runtime state and weight checks
└── revision/                # Bounded component revision and evaluation
    ├── loop.py              # Propose → independent evaluate → accept or retry
    ├── agent.py             # Agent protocol and external JSON command adapter
    ├── evaluator.py         # Held-out measurements or frozen reference predictions
    ├── online.py            # Background runtime windows and activation
    ├── serial.py            # Joint timing model revision → frozen model handoff
    └── joint_evaluator.py   # Joint optimizer quality/query/time evaluation
```

`ElasticLoop` executes a constructor-injected scheduling policy. `IdleTimeoutPolicy`
preserves the data-triggered wake and idle-release baseline. Policies receive immutable
CPU observations and return actions; the loop owns consistency guards and GPU work.
`QueuedTrainer` validates and consumes trajectories; `TrainGroup` owns worker lifecycle
and checkpoints. Resource scheduling rules are separate from the model's RL policy.

The runtime contains training execution and durable state. SGLang service ownership,
debug weight verification and the SWE trajectory envelope live in examples. The loop
receives a publisher from its caller to coordinate model versions without owning an
inference backend. Existing class/function names are retained at the new module paths.

AReaL already exports trajectories and saves/loads model and optimizer state. The local
queue adds durable consumption tied to checkpoint commit; `state/checkpoint.py` adds
atomic generations and scheduler/RNG recovery around engine save/load. In this checkout,
the general `RecoverHandler` still rejects v2 `GatewayTrainController`, so these
recovery guarantees cannot yet be replaced by that handler.

`BasicPolicy` uses a bounded history window and `BasicOptimizer` to jointly select
resource ownership and a pure-DP configuration. `ResourceDecision.train` separately
controls consuming ready data. The original string decisions and `IdleTimeoutPolicy`
remain supported. The [revision loop](../../../areal/autopilot/revision/README.md) can
evaluate and select one component module independently.

## Configuration application

`ConfigApplication` records intent before execution and commits observed configuration
receipts. Requests bind a complete `JointConfig`, expected config generation and weight
version, request ID and scheduling policy identity. Duplicate IDs return their original
receipt; stale or unsupported requests fail before worker changes. The journal is bound
to the queue, checkpoint store, worker pool and model compatibility.

Install `JointConfigExecutor(trainer, rollout, publisher)` with a managed `RolloutGroup`
and pass the application to `ElasticLoop(config_application=...)`. Both
`apply_config(request)` and automatic Actor decisions use the same admission gate and
journal. At a committed boundary, execution preserves the checkpoint, data cursor and
weight version. A partial transition recovers its source config; uncertain cleanup
leaves admission closed. `recover_config()` reconciles application failures without
replaying training updates.

The joint adapter supports v2 FSDP pure-DP Actor size and real SGLang replica/TP changes
on one shared GPU pool. Sequence, prefill-token and KV-token capacities map to native
`max_running_requests`, `max_prefill_tokens` and `max_total_tokens`; startup reads the
server settings back. Rollout takes the pool prefix and Actor the following devices.
Conflicting generations stop before replacement, reload the same committed weights, and
reopen admission only after the durable receipt and every proxy's resume check. Clients
resolve the group's gateway and key at each episode because restarting Rollout replaces
its service endpoints.

The current boundary requires removed sessions, a whole-episode gate and one inflight
batch. Model/runtime/sampling identities stay fixed. Async staleness, hybrid training
backends remain separate work. `ActorConfigExecutor` remains available for callers with
externally managed, fixed Rollout resources. `JointPolicy.plan` remains the asynchronous
offline interface; `OnlineJointPolicy` uses the serial runtime model and returns joint
actions to this same application path. Actor-only search deducts current Rollout GPUs
from its capacity limit.

Add `--tiny --config-application` to the validation command below to check real Actor
changes, duplicate requests and interruption after target activation. The diagnostic
publisher does not test inference or training quality. The
[2026-10-19 validation](validation/config-application-20261019.json) records nine exact
model/Adam recovery checks, interrupted application rollback and resource cleanup.

Use `--tiny --joint-config-application` for real SGLang generation, replica/TP and
capacity changes, joint failure recovery and exact Actor model/Adam restoration on a
four-GPU pool. TP verification compares rank-zero shard probes and checks reload/version
on every backend; it does not compare every tensor on every TP rank. GPU validation
records
[four real configuration changes](validation/joint-config-application-20261019.json),
five exact model/Adam checks, seven inference probes and interrupted receipt recovery.
Dynamic Actor capacity, continued training and resource cleanup passed on the final
source. Current status is maintained in [PROGRESS.md](../../../PROGRESS.md).

## Current integration target

Use the existing Qwen3-0.6B, FSDP and SGLang fixture. The
[M2 joint validation](milestones/m2/README.md#%E5%B7%B2%E9%AA%8C%E8%AF%81%E4%B8%8E%E6%9C%AA%E9%AA%8C%E8%AF%81)
covers revision deployment, subsequent automatic config application and continued
training. The earlier fixed-Rollout runtime history, model→optimizer revision and policy
activation have CPU tests and a
[Qwen3-0.6B H20 check](validation/runtime-revision-20261019.json): nine real PPO
updates, two exact model/Adam comparisons, five inference probes, interrupted activation
and coordinator-object recovery. Activation preserves history, workers and weights. The
fixture uses synthetic trajectories, deterministic JSON proposals and temporary
checkpoints. Online joint planning uses a separate serial model, described below.

All GPU assertions passed. The harness then hit `EBUSY` deleting its Docker bind-mount
root (exit 1); corrected contents cleanup was independently replayed (exit 0), followed
by host-directory removal and GPU/lock checks. The report preserves both outcomes.

Adaptation to other workflows/configs, including native SWE/Tau2 with Archon/Megatron,
comes after this end-to-end check. The unfinished native entrypoint prototype has been
removed from the active checkout. Small-model checks establish execution correctness;
training quality and cost-efficiency gains require separate experiments.

## Automatic joint planning

Use
`automatic_swe.py --policy joint --profiles /path/to/joint-profiles.json --horizon 2`
with the existing fixture arguments. `--inference-device` followed by `--devices` forms
one shared placement pool. A ready batch triggers bounded `JointOptimizer` search; only
its first joint configuration is applied, then the batch trains and publishes. The next
decision uses the actual configuration and committed historical observations. Idle
release remains active. No empty queue starts Actor workers.

Profiles use `format_version: "serial-joint-v1"` and the exact training `compatibility`:

| Field                   | Contents                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `initial`, `candidates` | Complete `JointConfig` objects; the allowlist includes the initial config                                       |
| `phases`                | `config`, `phase`, `seconds`, `sequence_tokens`, `source_id`; each phase is scoped to the complete joint config |
| `episodes`              | `rollout`, `workload`, `seconds`, `source_id`; elapsed time includes the whole episode/export/enqueue           |
| `transitions`           | Directed `TransitionProfile` objects, including restart/reload and temporary role residency                     |
| `search`                | Optional evaluation budget, predicted time bound and switch margin                                              |

The SWE fixture retains its 16,384-token context; supplied Rollout token/KV limits must
cover it. Other workflows and native backends are outside this adapter's scope.

The first cold update retains the initial config. Later windows contain the real ready
batch plus repeated past workload counts; unknown profiles reject a candidate, and no
profile-based switch infers TP speedup. Actor train timing permits bounded local token
scaling; whole-episode timing requires equal observed sample/token counts. Neither
approximation predicts reward or proves portability to a different workload. Terminal
Actor release, ongoing Rollout residency, idle timeout and directed switch cost are
included. No eligible supported prediction within a time bound keeps execution paused.

`runtime-history.json` stores actual joint configs, episode timing and phase
measurements only after checkpoint/publication. Idle releases are separate boundary
observations. History and config receipts recover together; no recovered update is
re-executed. Different Rollout layouts never share publication profiles.
`--policy joint` uses its own executable `serial-joint-policy-v1` revision bundle. It
preserves the complete effective joint history and both progress cursors across
activation/restart. Fixed-Rollout bundles remain a separate execution scope.

## Runtime revision

For `automatic_swe.py`, add these arguments to the measured Basic or joint fixture
command; place `--revision-command` last:

```text
--policy basic --profiles /path/to/measured-profiles.json --rounds 5
--revision-every-updates 4 --max-revision-windows 3
--model-revision-rounds 2 --optimizer-revision-rounds 2
--revision-command /path/to/python /path/to/json-agent-wrapper.py
```

The wrapper reads a `ProposalRequest` JSON on stdin and writes a `Proposal` JSON on
stdout; the existing `CommandRevisionAgent` enforces its timeout. No agent is invoked
without this explicit command. At least four measured updates are needed; five rounds
allow training after a revision becomes ready, without guaranteeing the agent finishes
before the last round. `--policy-bundle /path/to/bundle.json` loads a previously
accepted executable bundle. Both options require measured Basic profiles and exclude the
legacy `--revision-root` constructor loader. For joint planning, use `--policy joint`
with its `serial-joint-v1` profile; the revision flags and command protocol are shared.

State lives under `policy-application/`, `runtime-history.json` and
`runtime-revisions/window-*/`: immutable inputs, separate model/optimizer evaluations,
acceptance proofs, budgets and activation outcomes. Episodes and updates continue while
CPU revision runs; a busy boundary delays activation. Failures preserve the incumbent or
close admission for explicit recovery.

Joint revisions use `SerialModelEvaluator` → `SerialRevisionCycle` →
`JointOptimizerEvaluator`. Timing labels are actual phases and whole episodes under
their source full configurations; the first 60% of whole updates supply calibration.
Validation timing inputs omit elapsed-time copies and use calibration history only.
Small feedback groups suppress numerical errors. Timing error/coverage are checked per
phase/config; the optimizer is revalidated on the newly frozen executable model before
acceptance. Its public probes use calibration planning windows. GPU accounting,
transition profiles, allowlists, horizon and idle timeout remain fixed. This checks
conditional timings and model-based search, not full-window accuracy or reward.

The [joint H20 check](validation/joint-runtime-revision-20261019.json) exercises both
component rejections/acceptances, activation at v5, a subsequent automatic Actor 2→1 /
Rollout 1→2 transition, exact model/Adam preservation, eight inference probes and
coordinator/worker recovery through seven real PPO updates. Linux CPU regression: 481
passed, one CUDA RNG skip. All owned jobs/tmpfs were cleaned; the original control plane
remained running. Proposals, seed profiles and PPO trajectories are controlled fixtures,
so this establishes execution rather than workflow benefits.

## Basic policy draft

The [actor-model lifecycle policy](../../../areal/autopilot/elastic/policy/README.md)
connects `ActorCostModel → WorkflowSimulator → LifecycleOptimizer` with explicit full
profiles. It supports offline comparison of hybrid layouts, including distinct layouts
on the same card count. Its
[synthetic validation](validation/lifecycle-policy-20261018.json) does not establish
H200 accuracy or GPU execution of hybrid profiles. The existing pure-DP GPU baseline
below remains available.

The [2026-10-18 validation](validation/basic-policy-20261018.json) records the
tiny-model resize matrix, final-source checks and one diagnostic forecast. It is not a
quality or cost-savings benchmark.

- GPU candidates cover every integer from 1 through the configured pool size. Missing
  measurements reject a candidate; there is no power-of-two restriction or assumed
  linear GPU speedup. Search shortens its horizon if needed to cover the whole pool
  within its candidate budget.
- `WorkflowModel` composes rollout, startup/restore, train, checkpoint, publication and
  release components. Each phase accepts custom components or nested `SequenceComponent`
  / `ParallelComponent`. Children report elapsed time and additional GPU work; the
  simulator charges shared training residency once. Workflow authors must ensure
  parallel children can share their declared resources.
- The known queue head uses its actual workload. Future updates use mean sample/token
  counts and rollout duration from completed episodes in the history window; active
  episode remaining time uses conditional historical durations. No future trace is read.
- Per-card-count phase measurements seed the model; completed runtime phases update it.
  Training estimates scale locally with sequence tokens, with a bounded profile range.
  This simple model does not capture attention length distributions or contention.
- Plans include idle residency, release/rollout overlap, startup/restore and terminal
  cleanup. The objective is total GPU seconds by default, with training-only cost
  available. Actual SWE always charges its one resident inference GPU. A 5% switching
  margin reduces churn; only the first action is executed before replanning.
- Unknown measurements use the existing configuration and idle timeout to bootstrap.
  Known infeasibility or a configured time bound is not bypassed by fallback. Without
  admitted work, the idle timeout remains a fallback for eventual release.
- New opt-in pure-DP checkpoints support resizing. Native DCP restores the full model
  and Adam state. Same-size RNG recovery is exact; cross-size recovery uses a versioned,
  deterministic per-rank reseed. A private loading view leaves the committed source
  unchanged. A verified failed target rolls back to the previous layout; uncertain
  teardown retains the consumer fence.
- Committed global samples and loss-mask tokens are bound to the checkpoint cursor. Rank
  replicas and rolled-back replay do not multiply progress. The fixture counts one whole
  multi-turn episode as one sample.

Calibrate and check the runtime with all directed GPU-count changes:

```bash
python -m examples.autopilot.elastic.validation.basic_policy \
  --model /path/to/model --model-revision PINNED_REVISION \
  --output /path/to/new-validation --devices 4 5 6 7 --all-pairs
```

Add `--tiny` for a small random Qwen3 infrastructure check. The command emits
`profiles.json`, exact model/Adam restoration checks, loop decisions and a report. Its
publication is a diagnostic CPU receipt; it does not benchmark inference or reward. For
the SWE entry point below, add `--policy basic --profiles /path/to/profiles.json` and
optionally `--initial-train-gpus`, `--horizon`, `--cost-scope`. Profiles must match the
model/training contract; tiny-model profiles cannot be used for the real model.
Publication estimates from the diagnostic calibration need real SGLang calibration
before using them for performance conclusions.

Reward preservation must be checked against resident training at equal cumulative
samples, with effective tokens also reported. Neither cost predictions nor synthetic PPO
recovery checks establish reward preservation or cost savings. The reward tolerance and
quality baseline remain to be specified. Prewarming and changes to training algorithms
remain outside this draft. Component revision evaluations run on CPU; the optional
runtime bridge above connects their accepted results to subsequent decisions.

Revision targets the simulator and joint optimizer. `revision/evaluator.py` defines two
evaluation scopes:

| Candidate | Fixed reference                                              | Evaluation focus                                                                                               |
| --------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Simulator | Held-out runtime measurements                                | Prediction error and feasibility accuracy                                                                      |
| Optimizer | Reference simulator, workload, constraints and search budget | Joint action/configuration quality, GPU time, waiting time, training progress, feasibility and search overhead |

The implemented revision evaluations use measured labels for simulator revisions and
fixed-reference predictions for optimizer revisions; they do not run training. See the
[revision workflow](../../../areal/autopilot/revision/README.md) for commands and
`--revision-root` deployment.

The evaluator controls reference data and versions outside candidate edits. A simulator
candidate cannot validate itself using its own predictions; each comparison changes one
component while holding the other fixed. Lifecycle execution and consistency guards are
fixed across all candidates.

Core modules do not import examples. Under this example directory:

- `automatic_swe.py`: automatic SWE lifecycle validation entry point.
- `fixtures/groups.py`: PPO and fixed-batch LM validation groups; engines and
  diagnostics live alongside them in `fixtures/`.
- `fixtures/inference_service.py` and `fixtures/weight_publication.py`: local SGLang
  lifecycle and verified reload for the example, supplied to the elastic loop.
- `validation/`: isolated PPO and LM recovery checks plus shared inference probes.
- `swe/`: agent, durable rollout producer, sandbox cleanup and deployment setup;
  `trajectory.py` adapts already-exported AReaL trajectories to the fixture queue
  schema.
- `milestones/m1/`: draft acceptance and historical evidence summary.

## Behavior and boundaries

- An empty queue never allocates training GPUs. CPU preparation acquires the consumer
  fence and reconciles the checkpoint cursor before looking for pending work.
- One current-policy episode triggers one update and checkpoint. After successful
  publication, an idle timeout releases workers; it does not write another checkpoint.
  Clean release can continue while the next agent episode holds admission.
- The producer holds `loop.episode()` through grading, export, outbox and durable
  enqueue. The same gate serializes training/publication; a second outstanding episode
  is rejected until consumption/publication completes. Consumption requires matching
  behavior and training versions: version staleness is 0. Rollout overlaps clean idle
  release, while training and publication wait for the episode gate. External producers
  or requests bypassing it are unsupported.
- Publication failures keep admission closed and retry the same committed artifact.
  Uncertain update/checkpoint outcomes require stop and recovery, never a live retry.
- A new inference incarnation reloads the latest artifact. Only a completed publication
  reopens admission; a receipt alone is insufficient.

The correctness fixture uses clipped policy gradient with fixed baseline 0.5, unchanged
binary task rewards, no reward/advantage normalization, no KL penalty and no weight
decay. Every rank computes the complete small batch; one episode is counted once. This
is not default GRPO, a throughput benchmark, or evidence of improved SWE quality.

## Run

Use AReaL's CUDA runtime and pinned Qwen3-0.6B revision below. Prepare the sandbox and
separate agent/SDK environments using the [SWE instructions](swe/README.md).

```bash
export AREAL_CACHE_DIR=/path/to/project/cache/areal
export SWANLAB_SAVE_DIR=/path/to/project/cache/swanlab
python -m examples.autopilot.elastic.automatic_swe \
  --model /path/to/models/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --state-root /path/to/elastic/state --output /path/to/elastic/validation \
  --fixture /path/to/swe/fixture.json --environment autopilot-swe-bash@0.1.0 \
  --agent-python /path/to/agent-venv/bin/python \
  --sandbox-python /path/to/client-venv/bin/python \
  --kubectl /path/to/kubectl --kubeconfig /path/to/kubeconfig \
  --devices 4 5 6 7 --inference-device 0 --idle-seconds 5 --rounds 2
```

Output must be new; state may resume a compatible, drained queue. Run ordinary Python
without outer torchrun. Docker requires `--init --network host`, exposed GPUs, shared
memory, matching path mounts and writable caches; set `LOGNAME` for an unregistered UID.
Five seconds is a validation timeout, not an optimized resource policy.

Other fixtures:

- `python -m examples.autopilot.elastic.validation.ppo_cycle --help`: consume an
  existing version-0 trajectory, publish, release and restore; requires a fresh training
  state.
- `python -m examples.autopilot.elastic.validation.train_lifecycle --controller v2 --frontend-lifecycle persistent --batch-source queue --queue-faults ...`:
  six synthetic LM batches, continuous/restored comparison and
  checkpoint/acknowledgement fault windows.

## Recovery and publication

- SQLite WAL/FULL plus a non-expiring consumer `flock`; local filesystem only. Stable
  IDs and payload/version digests make enqueue retries idempotent. Retain payloads.
- A checkpoint atomically binds DCP model/Adam, scheduler/version/per-rank RNG, HF
  weights/tokenizer/checksums and consumed queue prefix. Acknowledge only after commit.
  Recovery replays beyond the checkpoint or repairs a lost acknowledgement.
- Preserve configured embedding/output-head sharing before FSDP and after restore,
  including Transformers metadata. Verify any duplicate HF aliases are identical.
  Checkpoints without the corrected weight-tying marker are rejected.
- Reload requires SGLang semantic success, exact loaded embedding and down-projection
  comparisons against HF, then matching proxy versions. The initial release's
  down-projection-only check missed a tied-weight mismatch; its PPO checkpoint is
  superseded and must not be restored.
- The zero-GPU training router/gateway/standby retain addresses, PIDs and key across
  worker generations. Recovery is private until verified; suspension returns HTTP 503.
  Verified release checks owned process exit and independent CUDA allocation.
- The lifecycle thread completes its current tick before shutdown joins it and cleans up
  services. `QueuedTrainer.suspend()` retains the queue fence while GPUs are off.

## Checks and storage

```bash
python -m pytest -q tests/test_elastic_*.py tests/test_autopilot_swe_fixture.py \
  tests/test_train_controller.py tests/test_fsdp_sft_batch.py
pre-commit run --all-files
```

Use the AReaL Linux runtime for these checks; the CUDA RNG case skips without a GPU.
Real SWE validation additionally requires the model, sandbox and GPU pools above.

A corrected checkpoint including HF is about 9.30 GB. Keep two committed generations,
room for the next staging write and the 10 GiB free-space floor. Weights and checkpoints
stay on the GPU machine. See [M1](milestones/m1/README.md) for the evidence and
remaining gaps.
