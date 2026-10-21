# Elastic training with AReaL v2.1.0

The execution entry is native `PPOTrainer`. With `actor_lifecycle.mode=on_demand`,
Rollout and CPU data services survive Actor worker release and restart. PPO computation,
checkpointing and publication stay in the native trainer. Current verification and GPU
blockers: [PROGRESS](../../../PROGRESS.md); research scope: [plan](../PLAN.md).

The former serial `ElasticLoop`, `QueuedTrainer`, `TrainGroup`, fixture engines and SWE
runner have been removed. Their durable state mechanisms are shared under
`areal/elastic/state` and `control`; raw history and models also belong to AReaL.
AutoPilot provides generic revision plus an Elastic adapter. Historical
[M1](../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/milestones/m1/README.md)
and
[M2](../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/milestones/m2/README.md)
records describe the retired executor, not GPU acceptance of the current path.

## Policy and revision loops

```text
Agent Service → resident Rollout → durable episode inbox → complete eligible batch
  → PPOTrainer / OnDemandActor → native PPO update → full checkpoint + cursor
  → AWEX publication + all-replica acknowledgement → idle release / next batch
                         ↑                         ↓
              NativePolicyRuntime ← committed RuntimeHistory
                         ↑                         ↓
                 accepted bundle ← model gate → frozen model → optimizer gate
```

`NativePolicyRuntime` supplies observations, optional Basic decisions, phase history and
atomic accepted-bundle activation. It never creates workers or performs PPO. The native
executor retains batch, version, checkpoint, publication, minimum-residence and release
safety checks regardless of the selected policy.

## Code map

| Responsibility                                                       | Current module                            |
| -------------------------------------------------------------------- | ----------------------------------------- |
| PPO execution and Actor lifecycle                                    | `areal/trainer/{rl_trainer,on_demand}.py` |
| Durable inbox, queue and checkpoint                                  | `areal/elastic/state/`                    |
| Process ownership, native control and activation/config transactions | `areal/elastic/control/`                  |
| Raw measured history                                                 | `areal/elastic/telemetry/`                |
| Basic models, simulator and search                                   | `areal/elastic/policy/basic/`             |
| Independent optional Expert reference                                | `areal/elastic/policy/reference/`         |
| Generic proposal, evaluation orchestration, acceptance and store     | `areal/autopilot/revision/`               |
| Elastic evaluators, worker, bundle verification and native bridge    | `areal/autopilot/domains/elastic/`        |
| Real online application                                              | `examples/hermes/{train,hermes}.py`       |

## Run and policy integration

Use the normal Hermes training entry with a verified local model, explicit disjoint GPU
pools, sufficient checkpoint space and a new persistent `state_root`. Default lifecycle
mode remains `resident`; on-demand requires v2 online mode and supported native
backends. See [ActorLifecycleConfig](../../../areal/api/cli_args.py) for capability
validation.

The existing `examples/hermes/train.py` accepts `--elastic-policy` for baseline history,
`--elastic-profile path.json` for measured Basic, optional
`--elastic-settings path.json` and `--elastic-bundle path.json` for accepted deployment.
These flags require `actor_lifecycle.mode=on_demand`; normal config arguments continue
to pass unchanged. Without these flags, the native lifecycle also works without
AutoPilot.

A custom native entry can attach timeout-baseline history without changing PPO:

```python
from areal import PPOTrainer
from areal.autopilot.domains.elastic.native import NativePolicyRuntime

runtime = NativePolicyRuntime()
with PPOTrainer(config, elastic_policy=runtime) as trainer:
    trainer.train()
```

A measured Basic run supplies `profile=` and `settings={"horizon": 1}`. The generated
`native-profile.json` identifies the model/training semantics, runtime sources, CPU
hooks, tokenizer, hardware, sampling and lifecycle options. Collect and audit
measurements, retain that identity, then use a fresh state root for the measured run.
Queue/checkpoint ownership binds the individual run separately. A helper-less checkpoint
cannot be upgraded in place to a policy-managed run.

For bounded automatic revision, construct `RuntimeRevisionLoop` with the same profile,
`settings=runtime.settings`, `scope="native-async-v1"`, `horizon=1` and matching
`idle_seconds`, then attach it as `runtime.revision` before `trainer.train()`. Explicit
proposal agents and durable window/round budgets are required; see the
[Elastic revision contract](../../../areal/autopilot/domains/elastic/README.md).

Accepted bundles can be preloaded with `NativePolicyRuntime(bundle=path)`, or prepared
and applied with `PolicyRequest` at a committed, fully published boundary. Both
components are independently evaluated; loading checks scope, source/proof hashes,
execution identity, settings and idle timeout. An uncertain activation closes admission
until recovery follows the durable pointer. History survives policy replacement.

## Behavior and boundaries

- One fixed Actor layout and one fixed Rollout placement per run. Native configuration
  transactions accept the identical configuration only; topology changes fail before
  mutation. Serial joint models and accepted bundles remain offline references.
- A complete eligible batch wakes Actor; no predictive prewarm. Default release requires
  idle ≥60s and residence ≥120s, no active update and confirmed published weights.
  Policy decisions cannot weaken those guards. Production starting point: B=4, S=2;
  lifecycle smoke: S=0, idle=5s, residence=0.
- Full checkpoint plus consumed cursor commits before publication. Lost publication
  acknowledgement repairs committed weights without repeating PPO. Unknown owners or
  communication outcomes remain fenced rather than being assumed cleaned up.
- Native phase measurements include actual CPU hooks and distinguish release boundaries
  from committed updates. Missing phase durations remain unknown. Fixed-layout, one-step
  prediction and revision do not establish asynchronous makespan, GPU-hour savings,
  cross-layout accuracy or training quality.
- CPU tests exercise real durable stores and protocol logic with fake workers/model
  endpoints. Actual FSDP/Megatron restore, Agent Service → GSM8K → Hermes training,
  asynchronous overlap and resource cleanup still require GPU acceptance.

## R10 fixed Actor replay

`replay_r10.py` remains an offline tool. Historical observations, figures, measured
reports and the complete frozen Expert reference live in the
[local archive](../../../../artifacts/repository-archive-20261021/README.md). The source
tree ships model code and small profiles; it does not bundle the R10 dataset.

```bash
python examples/autopilot/elastic/replay_r10.py \
  --reference-root /path/to/external/expert_designed --output /path/to/new-replay
```

The external reference must contain `trials/r10`, matching profiles, Expert numerical
sources and its associated `autopilot/contracts.py`. The archive preserves this layout
and all file hashes. Replay freezes those inputs and requires a new output directory; no
fitting or downloads occur while loading archived coefficients.

For current-code dataset tests, set `AREAL_TEST_EXPERT_CATALOG` to a derived catalog
whose exact source and numerical environment passed a calibration compatibility audit,
and run the normal pytest command. The original frozen catalog remains for original-code
replay. With no external dataset only archive-dependent tests skip; an invalid explicit
path fails. Native runtime/model tests need no archive.

[Historical R10 methodology and results](../../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/README.md#r10-fixed-actor-replay)
retain their original calibration, assumptions and coverage limits. These records do not
establish asynchronous cost savings, cross-layout accuracy or training quality.
